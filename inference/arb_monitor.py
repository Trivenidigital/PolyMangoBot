"""
Arbitrage Monitor
=================

Continuous monitoring for structural arbitrage opportunities.

Features:
- Periodic polling for market data
- Opportunity persistence tracking
- Alert generation
- Integration with trading engine
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from inference.engine import InferenceEngine
from inference.models import (
    ArbSignal,
    ArbState,
    PolymarketMarket,
    ViolationType,
)

logger = logging.getLogger("PolyMangoBot.inference.arb_monitor")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ArbMonitorConfig:
    """Configuration for the arbitrage monitor"""
    # Polling settings
    poll_interval_seconds: float = 45.0  # Default 45 second polling
    fast_poll_interval_seconds: float = 10.0  # Fast poll for high-value opps

    # Opportunity tracking
    min_persistence_seconds: float = 30.0  # Min time before acting
    max_tracking_seconds: float = 3600.0  # Max time to track an opportunity
    stale_threshold_seconds: float = 120.0  # Mark stale if not seen

    # Alert thresholds
    alert_edge_threshold_pct: float = 2.0  # Alert for edges >= 2%
    alert_liquidity_threshold: float = 5000.0  # Alert for high liquidity

    # Execution settings
    auto_execute: bool = False  # Auto-execute opportunities
    execute_edge_threshold_pct: float = 1.0  # Min edge for auto-execute
    execute_persistence_seconds: float = 60.0  # Min persistence for auto-execute


# =============================================================================
# OPPORTUNITY TRACKER
# =============================================================================

class OpportunityTracker:
    """
    Tracks arbitrage opportunities over time.

    Monitors persistence, price movements, and determines
    when opportunities are stable enough to execute.
    """

    def __init__(self, config: ArbMonitorConfig):
        self.config = config
        self._active: dict[str, ArbState] = {}
        self._history: list[ArbState] = []

    def update(self, signal: ArbSignal) -> ArbState:
        """
        Update tracking state with a new signal.

        Args:
            signal: The detected ArbSignal

        Returns:
            Updated ArbState
        """
        key = f"{signal.family_id}:{signal.subtype}"
        now = datetime.now()

        if key in self._active:
            # Update existing
            state = self._active[key]
            state.last_seen = now
            state.price_snapshots.append({
                "timestamp": now.isoformat(),
                "edge": signal.realizable_edge,
                "worst_pnl": signal.worst_case_pnl
            })
        else:
            # Create new
            state = ArbState(
                family_id=signal.family_id,
                violation_type=ViolationType(signal.subtype),
                first_seen=now,
                last_seen=now,
                price_snapshots=[{
                    "timestamp": now.isoformat(),
                    "edge": signal.realizable_edge,
                    "worst_pnl": signal.worst_case_pnl
                }]
            )
            self._active[key] = state

        return state

    def is_persistent(self, state: ArbState) -> bool:
        """Check if opportunity has been persistent long enough."""
        return state.duration_seconds >= self.config.min_persistence_seconds

    def is_stable(self, state: ArbState) -> bool:
        """
        Check if opportunity prices are stable.

        Looks for consistent edge across recent snapshots.
        """
        if len(state.price_snapshots) < 2:
            return False

        # Get last 3 edge values
        recent = state.price_snapshots[-3:]
        edges = [s.get("edge", 0) for s in recent]

        if not edges:
            return False

        # Check variance
        avg_edge: float = sum(edges) / len(edges)
        variance: float = sum((e - avg_edge) ** 2 for e in edges) / len(edges)

        # Stable if variance is low (< 0.1)
        return bool(variance < 0.1)

    def should_alert(self, signal: ArbSignal, state: ArbState) -> bool:
        """Determine if we should generate an alert."""
        if state.alert_sent:
            return False

        if signal.realizable_edge >= self.config.alert_edge_threshold_pct:
            return True

        return signal.min_liquidity >= self.config.alert_liquidity_threshold

    def should_execute(self, signal: ArbSignal, state: ArbState) -> bool:
        """Determine if we should auto-execute."""
        if not self.config.auto_execute:
            return False

        if signal.realizable_edge < self.config.execute_edge_threshold_pct:
            return False

        if state.duration_seconds < self.config.execute_persistence_seconds:
            return False

        return self.is_stable(state)

    def cleanup_stale(self):
        """Remove stale opportunities."""
        now = datetime.now()
        stale_keys = []

        for key, state in self._active.items():
            age = (now - state.last_seen).total_seconds()
            if age > self.config.stale_threshold_seconds:
                stale_keys.append(key)
                self._history.append(state)

        for key in stale_keys:
            del self._active[key]

        if stale_keys:
            logger.info(f"Cleaned up {len(stale_keys)} stale opportunities")

    def get_active(self) -> list[ArbState]:
        """Get all active opportunities."""
        return list(self._active.values())

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "active_count": len(self._active),
            "history_count": len(self._history),
            "persistent_count": sum(
                1 for s in self._active.values()
                if self.is_persistent(s)
            )
        }


# =============================================================================
# ARB MONITOR
# =============================================================================

class ArbMonitor:
    """
    Continuous monitoring for structural arbitrage.

    Polls market data, runs inference engine, tracks opportunities,
    and triggers execution when appropriate.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[ArbMonitorConfig] = None,
        market_fetcher: Optional[Callable[[], list[PolymarketMarket]]] = None
    ):
        self.engine = engine
        self.config = config or ArbMonitorConfig()
        self._market_fetcher = market_fetcher

        self.tracker = OpportunityTracker(self.config)

        # Callbacks
        self._on_signal: Optional[Callable[[ArbSignal], None]] = None
        self._on_alert: Optional[Callable[[ArbSignal, ArbState], None]] = None
        self._on_execute: Optional[Callable[[ArbSignal], bool]] = None

        # State
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_poll_time: float = 0.0

        # Statistics
        self._stats = {
            "polls": 0,
            "signals_found": 0,
            "alerts_sent": 0,
            "executions": 0,
            "total_run_time": 0.0,
        }

        logger.info(
            f"ArbMonitor initialized: poll_interval={self.config.poll_interval_seconds}s"
        )

    def set_market_fetcher(
        self,
        fetcher: Callable[[], list[PolymarketMarket]]
    ) -> None:
        """Set the market data fetcher function."""
        self._market_fetcher = fetcher

    def on_signal(self, callback: Callable[[ArbSignal], None]) -> None:
        """Register callback for new signals."""
        self._on_signal = callback

    def on_alert(self, callback: Callable[[ArbSignal, ArbState], None]) -> None:
        """Register callback for alerts."""
        self._on_alert = callback

    def on_execute(self, callback: Callable[[ArbSignal], bool]) -> None:
        """Register callback for execution."""
        self._on_execute = callback

    async def start(self):
        """Start the monitoring loop."""
        if self._is_running:
            logger.warning("Monitor already running")
            return

        if not self._market_fetcher:
            raise ValueError("Market fetcher not set")

        self._is_running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("ArbMonitor started")

    async def stop(self):
        """Stop the monitoring loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("ArbMonitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._is_running:
            try:
                start_time = time.time()

                # Fetch markets
                markets = await self._fetch_markets()

                if markets:
                    # Run inference
                    signals = self.engine.process_markets(markets)
                    self._stats["polls"] += 1

                    # Process signals
                    for signal in signals:
                        await self._process_signal(signal)

                    # Cleanup stale opportunities
                    self.tracker.cleanup_stale()

                # Calculate poll interval
                poll_interval = self._get_poll_interval()
                elapsed = time.time() - start_time
                self._stats["total_run_time"] += elapsed

                sleep_time = max(0, poll_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    async def _fetch_markets(self) -> list[PolymarketMarket]:
        """Fetch market data."""
        if not self._market_fetcher:
            logger.warning("No market fetcher configured")
            return []
        try:
            if asyncio.iscoroutinefunction(self._market_fetcher):
                markets: list[PolymarketMarket] = await self._market_fetcher()
                return markets
            else:
                return self._market_fetcher()  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def _process_signal(self, signal: ArbSignal) -> None:
        """Process a detected signal."""
        self._stats["signals_found"] += 1

        # Update tracker
        state = self.tracker.update(signal)

        # Notify callback
        if self._on_signal:
            try:
                if asyncio.iscoroutinefunction(self._on_signal):
                    await self._on_signal(signal)
                else:
                    self._on_signal(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        # Check for alert
        if self.tracker.should_alert(signal, state):
            state.alert_sent = True
            self._stats["alerts_sent"] += 1

            logger.info(
                f"ALERT: {signal.subtype} opportunity in {signal.family_id} "
                f"edge={signal.realizable_edge:.2f}%"
            )

            if self._on_alert:
                try:
                    if asyncio.iscoroutinefunction(self._on_alert):
                        await self._on_alert(signal, state)
                    else:
                        self._on_alert(signal, state)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        # Check for auto-execute
        if self.tracker.should_execute(signal, state):
            await self._execute_signal(signal)

    async def _execute_signal(self, signal: ArbSignal) -> None:
        """Execute a signal."""
        logger.info(
            f"Executing: {signal.subtype} in {signal.family_id} "
            f"edge={signal.realizable_edge:.2f}%"
        )

        if self._on_execute:
            try:
                if asyncio.iscoroutinefunction(self._on_execute):
                    success = await self._on_execute(signal)
                else:
                    success = self._on_execute(signal)

                if success:
                    self._stats["executions"] += 1
            except Exception as e:
                logger.error(f"Execution callback error: {e}")

    def _get_poll_interval(self) -> float:
        """
        Determine poll interval based on active opportunities.

        Use faster polling when high-value opportunities are active.
        """
        active = self.tracker.get_active()

        for state in active:
            snapshots = state.price_snapshots
            if snapshots:
                last_edge = snapshots[-1].get("edge", 0)
                if last_edge >= self.config.alert_edge_threshold_pct:
                    return self.config.fast_poll_interval_seconds

        return self.config.poll_interval_seconds

    def poll_once(self, markets: list[PolymarketMarket]) -> list[ArbSignal]:
        """
        Run a single poll cycle (for testing/manual use).

        Args:
            markets: List of markets to process

        Returns:
            List of detected signals
        """
        self._stats["polls"] += 1
        signals = self.engine.process_markets(markets)

        for signal in signals:
            state = self.tracker.update(signal)
            self._stats["signals_found"] += 1

            if self.tracker.should_alert(signal, state):
                state.alert_sent = True
                self._stats["alerts_sent"] += 1

        self.tracker.cleanup_stale()
        return signals

    def get_active_opportunities(self) -> list[dict[str, Any]]:
        """Get summary of active opportunities."""
        result = []
        for state in self.tracker.get_active():
            last_snapshot = state.price_snapshots[-1] if state.price_snapshots else {}
            result.append({
                "family_id": state.family_id,
                "type": state.violation_type.value,
                "first_seen": state.first_seen.isoformat(),
                "duration_seconds": state.duration_seconds,
                "last_edge": last_snapshot.get("edge", 0),
                "is_persistent": self.tracker.is_persistent(state),
                "is_stable": self.tracker.is_stable(state),
                "alert_sent": state.alert_sent,
            })
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            **self._stats,
            "is_running": self._is_running,
            "tracker": self.tracker.get_stats(),
            "engine": self.engine.get_stats(),
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_monitor(
    engine: Optional[InferenceEngine] = None,
    poll_interval: float = 45.0,
    auto_execute: bool = False
) -> ArbMonitor:
    """
    Factory function to create an ArbMonitor.

    Args:
        engine: InferenceEngine to use (creates default if None)
        poll_interval: Polling interval in seconds
        auto_execute: Whether to auto-execute opportunities

    Returns:
        Configured ArbMonitor
    """
    if engine is None:
        from inference.engine import create_engine
        engine = create_engine()

    config = ArbMonitorConfig(
        poll_interval_seconds=poll_interval,
        auto_execute=auto_execute
    )

    return ArbMonitor(engine, config)
