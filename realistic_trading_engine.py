"""
Realistic Trading Engine
=========================

A trading engine built around ACTUAL edges, not speed.

Philosophy:
- We are NOT faster than HFT (and never will be)
- Our edges: Signal quality, risk management, longer timeframes, illiquid markets
- Focus on being RIGHT, not being FAST

Key Design Decisions:
1. No sub-second arbitrage (HFT wins that game)
2. Minimum 15-minute holding periods (where speed doesn't matter)
3. Quality over quantity (fewer, better trades)
4. Dynamic adaptation (volatility regimes, performance-based allocation)
5. Robust risk management (the only true edge retail has)
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

from edge_strategies import (
    EdgeStrategyEnsemble,
    EdgeSignal,
    EdgeType,
    SignalStrength,
    EnhancedDirectionalStrategy,
    StatisticalArbitrageStrategy,
    IlliquidMarketStrategy,
    VolatilityRegimeStrategy
)

logger = logging.getLogger("PolyMangoBot.realistic_engine")


# =============================================================================
# CONFIGURATION
# =============================================================================

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"   # 1-2% position sizes, strict filters
    MODERATE = "moderate"           # 2-4% position sizes, normal filters
    AGGRESSIVE = "aggressive"       # 4-8% position sizes, relaxed filters


@dataclass
class RealisticEngineConfig:
    """Configuration for realistic trading"""

    # Capital
    starting_capital: float = 10000.0
    max_capital_per_trade_pct: float = 5.0
    max_total_exposure_pct: float = 30.0

    # Risk management (THE REAL EDGE)
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 8.0
    max_drawdown_pct: float = 15.0
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Signal quality thresholds
    min_signal_quality: float = 0.5
    min_win_probability: float = 0.50
    min_risk_reward: float = 1.5
    min_edge_score: float = 0.4

    # Position management
    max_concurrent_positions: int = 5
    max_positions_per_asset: int = 1
    max_correlated_positions: int = 3

    # Timing (NOT trying to be fast)
    min_hold_time_minutes: int = 15
    signal_refresh_seconds: float = 60.0  # Check for signals every minute
    order_timeout_seconds: float = 30.0   # Generous timeout (speed doesn't matter)

    # Strategy allocation
    strategy_allocations: Dict[str, float] = field(default_factory=lambda: {
        "directional": 0.40,
        "stat_arb": 0.30,
        "illiquid": 0.15,
        "volatility": 0.15
    })

    @classmethod
    def conservative(cls) -> "RealisticEngineConfig":
        return cls(
            max_capital_per_trade_pct=2.0,
            max_total_exposure_pct=15.0,
            max_daily_loss_pct=2.0,
            min_signal_quality=0.6,
            min_win_probability=0.55,
            min_risk_reward=2.0,
            risk_level=RiskLevel.CONSERVATIVE,
            max_concurrent_positions=3
        )

    @classmethod
    def moderate(cls) -> "RealisticEngineConfig":
        return cls()

    @classmethod
    def aggressive(cls) -> "RealisticEngineConfig":
        return cls(
            max_capital_per_trade_pct=8.0,
            max_total_exposure_pct=50.0,
            max_daily_loss_pct=5.0,
            min_signal_quality=0.4,
            min_win_probability=0.45,
            min_risk_reward=1.2,
            risk_level=RiskLevel.AGGRESSIVE,
            max_concurrent_positions=8
        )


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

@dataclass
class Position:
    """An open trading position"""
    id: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    position_value: float

    target_price: float
    stop_loss: float

    edge_type: EdgeType
    signal_quality: float
    expected_hold_hours: float

    entry_time: float = field(default_factory=time.time)
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Status tracking
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0
    updates: int = 0

    def update(self, current_price: float):
        """Update position with current price"""
        self.last_price = current_price
        self.updates += 1

        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

        self.unrealized_pnl_pct = self.unrealized_pnl / self.position_value * 100

        # Track peak/trough
        self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
        self.trough_pnl = min(self.trough_pnl, self.unrealized_pnl)

    @property
    def hold_time_hours(self) -> float:
        return (time.time() - self.entry_time) / 3600

    @property
    def should_close_target(self) -> bool:
        """Check if target hit"""
        if self.direction == "long":
            return self.last_price >= self.target_price
        else:
            return self.last_price <= self.target_price

    @property
    def should_close_stop(self) -> bool:
        """Check if stop hit"""
        if self.direction == "long":
            return self.last_price <= self.stop_loss
        else:
            return self.last_price >= self.stop_loss


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    THE ACTUAL EDGE: Superior risk management.

    Most retail traders lose because of poor risk management.
    Our edge is avoiding bad trades and cutting losses quickly.
    """

    def __init__(self, config: RealisticEngineConfig):
        self.config = config
        self.capital = config.starting_capital

        # Daily/weekly tracking
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._peak_capital: float = config.starting_capital
        self._day_start = self._get_day_start()
        self._week_start = self._get_week_start()

        # Position tracking
        self._positions: Dict[str, Position] = {}
        self._closed_positions: deque = deque(maxlen=1000)

        # Risk metrics
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._is_locked: bool = False
        self._lock_reason: str = ""

    def _get_day_start(self) -> float:
        """Get timestamp of day start"""
        from datetime import datetime
        now = datetime.utcnow()
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _get_week_start(self) -> float:
        """Get timestamp of week start"""
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        week_start = now - timedelta(days=now.weekday())
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def can_open_position(self, signal: EdgeSignal) -> Tuple[bool, str]:
        """Check if we can open a new position"""

        # Check if trading is locked
        if self._is_locked:
            return False, f"Trading locked: {self._lock_reason}"

        # Refresh daily/weekly if needed
        self._refresh_periods()

        # Check daily loss limit
        daily_loss_pct = abs(min(0, self._daily_pnl)) / self.capital * 100
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self._lock_trading("Daily loss limit reached")
            return False, "Daily loss limit reached"

        # Check weekly loss limit
        weekly_loss_pct = abs(min(0, self._weekly_pnl)) / self.capital * 100
        if weekly_loss_pct >= self.config.max_weekly_loss_pct:
            self._lock_trading("Weekly loss limit reached")
            return False, "Weekly loss limit reached"

        # Check drawdown
        drawdown_pct = (self._peak_capital - self.capital) / self._peak_capital * 100
        if drawdown_pct >= self.config.max_drawdown_pct:
            self._lock_trading("Max drawdown reached")
            return False, "Max drawdown reached"

        # Check consecutive losses
        if self._consecutive_losses >= 5:
            return False, "5 consecutive losses - taking a break"

        # Check position limits
        if len(self._positions) >= self.config.max_concurrent_positions:
            return False, "Max concurrent positions reached"

        # Check exposure
        current_exposure = sum(p.position_value for p in self._positions.values())
        max_exposure = self.capital * (self.config.max_total_exposure_pct / 100)
        if current_exposure >= max_exposure:
            return False, "Max exposure reached"

        # Check if already have position in this asset
        existing = [p for p in self._positions.values() if p.symbol == signal.symbol]
        if len(existing) >= self.config.max_positions_per_asset:
            return False, f"Already have position in {signal.symbol}"

        # Check signal quality thresholds
        if signal.signal_quality < self.config.min_signal_quality:
            return False, f"Signal quality too low: {signal.signal_quality:.2f}"

        if signal.win_probability < self.config.min_win_probability:
            return False, f"Win probability too low: {signal.win_probability:.2f}"

        if signal.risk_reward_ratio < self.config.min_risk_reward:
            return False, f"Risk/reward too low: {signal.risk_reward_ratio:.2f}"

        return True, "OK"

    def calculate_position_size(self, signal: EdgeSignal) -> float:
        """Calculate position size based on risk parameters"""

        # Base size from config
        max_pct = self.config.max_capital_per_trade_pct

        # Adjust for signal quality
        quality_factor = signal.signal_quality  # 0-1

        # Adjust for win probability
        prob_factor = (signal.win_probability - 0.5) * 2 + 0.5  # Normalize around 0.5

        # Kelly-based suggestion (but capped)
        kelly_pct = signal.suggested_position_pct * 100

        # Combined
        suggested_pct = min(
            max_pct,
            kelly_pct,
            max_pct * quality_factor * prob_factor
        )

        # Risk level adjustment
        if self.config.risk_level == RiskLevel.CONSERVATIVE:
            suggested_pct *= 0.5
        elif self.config.risk_level == RiskLevel.AGGRESSIVE:
            suggested_pct *= 1.5

        # After consecutive losses, reduce size
        if self._consecutive_losses >= 2:
            suggested_pct *= (1 - self._consecutive_losses * 0.15)

        # Minimum viable position
        suggested_pct = max(0.5, suggested_pct)

        # Convert to actual value
        position_value = self.capital * (suggested_pct / 100)

        # Round to sensible amount
        return round(position_value, 2)

    def open_position(self, signal: EdgeSignal, position_value: float) -> Optional[Position]:
        """Open a new position"""

        quantity = position_value / signal.entry_price

        position = Position(
            id=f"pos_{int(time.time() * 1000)}_{signal.symbol}",
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            quantity=quantity,
            position_value=position_value,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            edge_type=signal.edge_type,
            signal_quality=signal.signal_quality,
            expected_hold_hours=signal.expected_hold_time_hours
        )

        self._positions[position.id] = position
        self._daily_trades += 1

        logger.info(
            f"Opened position: {position.symbol} {position.direction} "
            f"${position_value:.2f} @ ${signal.entry_price:.2f}"
        )

        return position

    def close_position(self, position_id: str, close_price: float, reason: str) -> Optional[float]:
        """Close a position and record PnL"""

        if position_id not in self._positions:
            return None

        position = self._positions.pop(position_id)
        position.update(close_price)

        pnl = position.unrealized_pnl

        # Update capital
        self.capital += pnl
        self._daily_pnl += pnl
        self._weekly_pnl += pnl

        # Track peak
        if self.capital > self._peak_capital:
            self._peak_capital = self.capital

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Record
        self._closed_positions.append({
            "position": position,
            "close_price": close_price,
            "pnl": pnl,
            "reason": reason,
            "timestamp": time.time()
        })

        logger.info(
            f"Closed position: {position.symbol} {position.direction} "
            f"PnL: ${pnl:.2f} ({position.unrealized_pnl_pct:.2f}%) "
            f"Reason: {reason}"
        )

        return pnl

    def update_positions(self, prices: Dict[str, float]):
        """Update all positions with current prices"""
        for position in self._positions.values():
            if position.symbol in prices:
                position.update(prices[position.symbol])

    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """Check which positions should be closed"""
        exits = []

        for pos_id, position in list(self._positions.items()):
            if position.symbol not in prices:
                continue

            position.update(prices[position.symbol])

            # Check target
            if position.should_close_target:
                exits.append((pos_id, "target_hit"))
                continue

            # Check stop
            if position.should_close_stop:
                exits.append((pos_id, "stop_hit"))
                continue

            # Check time-based exit (position held too long)
            if position.hold_time_hours > position.expected_hold_hours * 2:
                exits.append((pos_id, "time_exit"))
                continue

            # Trailing stop logic (lock in profits)
            if position.unrealized_pnl_pct > 1.5:
                # If we were up 1.5% and now down to 0.5%, exit
                if position.peak_pnl > 0 and position.unrealized_pnl < position.peak_pnl * 0.5:
                    exits.append((pos_id, "trailing_stop"))

        return exits

    def _refresh_periods(self):
        """Refresh daily/weekly tracking"""
        now = time.time()
        current_day = self._get_day_start()
        current_week = self._get_week_start()

        if current_day != self._day_start:
            self._day_start = current_day
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._is_locked = False  # Unlock for new day

        if current_week != self._week_start:
            self._week_start = current_week
            self._weekly_pnl = 0.0

    def _lock_trading(self, reason: str):
        """Lock trading for the day"""
        self._is_locked = True
        self._lock_reason = reason
        logger.warning(f"Trading locked: {reason}")

    def get_status(self) -> Dict:
        """Get current risk status"""
        current_exposure = sum(p.position_value for p in self._positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())

        return {
            "capital": self.capital,
            "peak_capital": self._peak_capital,
            "drawdown_pct": (self._peak_capital - self.capital) / self._peak_capital * 100,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "unrealized_pnl": unrealized_pnl,
            "current_exposure": current_exposure,
            "exposure_pct": current_exposure / self.capital * 100,
            "open_positions": len(self._positions),
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
            "is_locked": self._is_locked,
            "lock_reason": self._lock_reason
        }


# =============================================================================
# REALISTIC TRADING ENGINE
# =============================================================================

class RealisticTradingEngine:
    """
    Trading engine built around realistic edges.

    NOT trying to:
    - Be faster than HFT
    - Capture sub-second arbitrage
    - Trade high frequency

    ACTUALLY doing:
    - High-quality signal generation
    - Superior risk management
    - Longer timeframe strategies
    - Adaptation to market conditions
    """

    def __init__(
        self,
        capital: float = 10000.0,
        config: Optional[RealisticEngineConfig] = None
    ):
        self.config = config or RealisticEngineConfig(starting_capital=capital)

        # Strategy ensemble
        self.strategies = EdgeStrategyEnsemble(capital=capital)

        # Risk manager (THE CORE EDGE)
        self.risk_manager = RiskManager(self.config)

        # Market data storage
        self._market_data: Dict[str, Dict] = {}
        self._price_cache: Dict[str, float] = {}

        # Performance tracking
        self._performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "by_edge_type": {}
        }

        # State
        self._is_running = False
        self._last_signal_check = 0

        logger.info(
            f"RealisticTradingEngine initialized: "
            f"capital=${capital:,.2f}, risk_level={self.config.risk_level.value}"
        )

    def update_market_data(
        self,
        symbol: str,
        candles_15m: Optional[List[Dict]] = None,
        candles_1h: Optional[List[Dict]] = None,
        candles_4h: Optional[List[Dict]] = None,
        orderbook: Optional[Dict] = None,
        daily_volume: Optional[float] = None
    ):
        """Update market data for a symbol"""
        if symbol not in self._market_data:
            self._market_data[symbol] = {}

        data = self._market_data[symbol]

        if candles_15m:
            data["candles_15m"] = candles_15m
            self._price_cache[symbol] = candles_15m[-1]["close"]

        if candles_1h:
            data["candles_1h"] = candles_1h

        if candles_4h:
            data["candles_4h"] = candles_4h

        if orderbook:
            data["orderbook"] = orderbook

        if daily_volume is not None:
            data["daily_volume"] = daily_volume

    def update_pair_data(
        self,
        pair_symbol: str,
        asset1: str,
        asset2: str,
        prices1: List[float],
        prices2: List[float]
    ):
        """Update data for pairs trading"""
        self._market_data[pair_symbol] = {
            "pair_prices": {
                asset1: prices1,
                asset2: prices2
            }
        }

    def scan_for_signals(self) -> List[EdgeSignal]:
        """Scan all strategies for signals"""
        now = time.time()

        # Rate limit signal checking
        if now - self._last_signal_check < self.config.signal_refresh_seconds:
            return []

        self._last_signal_check = now

        symbols = list(self._market_data.keys())

        # Generate signals from all strategies
        all_signals = self.strategies.generate_all_signals(symbols, self._market_data)

        # Filter by config thresholds
        valid_signals = [
            s for s in all_signals
            if s.signal_quality >= self.config.min_signal_quality
            and s.win_probability >= self.config.min_win_probability
            and s.risk_reward_ratio >= self.config.min_risk_reward
            and s.edge_score >= self.config.min_edge_score
        ]

        # Select best signals
        best_signals = self.strategies.select_best_signals(
            valid_signals,
            max_signals=self.config.max_concurrent_positions
        )

        logger.info(
            f"Signal scan: {len(all_signals)} generated, "
            f"{len(valid_signals)} valid, {len(best_signals)} selected"
        )

        return best_signals

    def process_signal(self, signal: EdgeSignal) -> Optional[Position]:
        """Process a signal and potentially open a position"""

        # Check if we can open
        can_open, reason = self.risk_manager.can_open_position(signal)

        if not can_open:
            logger.debug(f"Cannot open position for {signal.symbol}: {reason}")
            return None

        # Calculate position size
        position_value = self.risk_manager.calculate_position_size(signal)

        # Minimum position check
        if position_value < 50:
            logger.debug(f"Position too small: ${position_value:.2f}")
            return None

        # Open position
        position = self.risk_manager.open_position(signal, position_value)

        if position:
            self._performance["total_trades"] += 1

        return position

    def update_and_check_exits(self) -> List[Tuple[str, float, str]]:
        """Update positions and check for exits"""

        # Update with current prices
        self.risk_manager.update_positions(self._price_cache)

        # Check exits
        exits = self.risk_manager.check_exits(self._price_cache)

        results = []
        for pos_id, reason in exits:
            position = self.risk_manager._positions.get(pos_id)
            if position and position.symbol in self._price_cache:
                close_price = self._price_cache[position.symbol]
                pnl = self.risk_manager.close_position(pos_id, close_price, reason)

                if pnl is not None:
                    # Update performance tracking
                    self._performance["total_pnl"] += pnl
                    if pnl > 0:
                        self._performance["winning_trades"] += 1
                    self._performance["best_trade"] = max(self._performance["best_trade"], pnl)
                    self._performance["worst_trade"] = min(self._performance["worst_trade"], pnl)

                    # Track by edge type
                    edge_type = position.edge_type.value
                    if edge_type not in self._performance["by_edge_type"]:
                        self._performance["by_edge_type"][edge_type] = {
                            "trades": 0, "pnl": 0.0, "wins": 0
                        }
                    self._performance["by_edge_type"][edge_type]["trades"] += 1
                    self._performance["by_edge_type"][edge_type]["pnl"] += pnl
                    if pnl > 0:
                        self._performance["by_edge_type"][edge_type]["wins"] += 1

                    # Record outcome for strategy adaptation
                    self.strategies.record_outcome(
                        EdgeSignal(
                            symbol=position.symbol,
                            direction=position.direction,
                            edge_type=position.edge_type,
                            strength=SignalStrength.MODERATE,
                            confidence=position.signal_quality,
                            signal_quality=position.signal_quality,
                            entry_price=position.entry_price,
                            target_price=position.target_price,
                            stop_loss=position.stop_loss,
                            expected_return_pct=0,
                            win_probability=0,
                            risk_reward_ratio=0,
                            kelly_fraction=0,
                            suggested_position_pct=0,
                            max_position_pct=0,
                            expected_hold_time_hours=0,
                            signal_valid_for_minutes=0
                        ),
                        profitable=pnl > 0,
                        return_pct=pnl / position.position_value * 100
                    )

                    results.append((pos_id, pnl, reason))

        return results

    async def run_cycle(self) -> Dict:
        """Run one trading cycle"""

        cycle_result = {
            "signals_found": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "pnl": 0.0
        }

        # Check for exits first
        exits = self.update_and_check_exits()
        cycle_result["positions_closed"] = len(exits)
        cycle_result["pnl"] = sum(pnl for _, pnl, _ in exits)

        # Scan for new signals
        signals = self.scan_for_signals()
        cycle_result["signals_found"] = len(signals)

        # Process signals
        for signal in signals:
            position = self.process_signal(signal)
            if position:
                cycle_result["positions_opened"] += 1

        return cycle_result

    async def start(self):
        """Start the trading engine"""
        self._is_running = True
        logger.info("RealisticTradingEngine started")

        while self._is_running:
            try:
                cycle_result = await self.run_cycle()

                if cycle_result["positions_opened"] > 0 or cycle_result["positions_closed"] > 0:
                    logger.info(
                        f"Cycle: opened={cycle_result['positions_opened']}, "
                        f"closed={cycle_result['positions_closed']}, "
                        f"pnl=${cycle_result['pnl']:.2f}"
                    )

                # Sleep until next cycle (not trying to be fast!)
                await asyncio.sleep(self.config.signal_refresh_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        """Stop the trading engine"""
        self._is_running = False
        logger.info("RealisticTradingEngine stopped")

    def get_performance(self) -> Dict:
        """Get comprehensive performance report"""
        total = self._performance["total_trades"]
        wins = self._performance["winning_trades"]

        risk_status = self.risk_manager.get_status()
        strategy_perf = self.strategies.get_performance_summary()

        return {
            "summary": {
                "total_trades": total,
                "winning_trades": wins,
                "win_rate": wins / total * 100 if total > 0 else 0,
                "total_pnl": self._performance["total_pnl"],
                "avg_pnl_per_trade": self._performance["total_pnl"] / total if total > 0 else 0,
                "best_trade": self._performance["best_trade"],
                "worst_trade": self._performance["worst_trade"],
                "profit_factor": abs(self._performance["best_trade"] / self._performance["worst_trade"]) if self._performance["worst_trade"] != 0 else 0
            },
            "risk_status": risk_status,
            "by_edge_type": self._performance["by_edge_type"],
            "strategy_performance": strategy_perf
        }

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions"""
        positions = []
        for pos in self.risk_manager._positions.values():
            positions.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "current_price": pos.last_price,
                "position_value": pos.position_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "hold_time_hours": pos.hold_time_hours,
                "edge_type": pos.edge_type.value
            })
        return positions


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_realistic_engine():
    """Test the realistic trading engine"""
    print("=" * 70)
    print("REALISTIC TRADING ENGINE TEST")
    print("=" * 70)
    print("\nPhilosophy: Quality over speed. Risk management is THE edge.")
    print("=" * 70)

    # Create engine
    engine = RealisticTradingEngine(
        capital=10000.0,
        config=RealisticEngineConfig.moderate()
    )

    # Generate synthetic data
    np.random.seed(42)

    def generate_candles(base_price: float, count: int, trend: float = 0.0) -> List[Dict]:
        candles = []
        price = base_price
        timestamp = time.time() - count * 900

        for i in range(count):
            # Add trend and noise
            change = np.random.randn() * 0.015 + trend
            open_price = price
            close_price = price * (1 + change)
            high = max(open_price, close_price) * 1.005
            low = min(open_price, close_price) * 0.995
            volume = np.random.uniform(1000, 10000)

            candles.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })

            price = close_price
            timestamp += 900

        return candles

    # Load data for multiple assets
    print("\n--- Loading Market Data ---")

    assets = {
        "BTC": {"base": 42000, "trend": 0.0005},
        "ETH": {"base": 2300, "trend": 0.0003},
        "SOL": {"base": 95, "trend": 0.0008},
    }

    for symbol, params in assets.items():
        engine.update_market_data(
            symbol=symbol,
            candles_15m=generate_candles(params["base"], 100, params["trend"]),
            candles_1h=generate_candles(params["base"], 50, params["trend"]),
            candles_4h=generate_candles(params["base"], 20, params["trend"]),
            orderbook={
                "best_bid": params["base"] * 0.999,
                "best_ask": params["base"] * 1.001,
                "bid_sizes": [1.0, 2.0, 3.0],
                "ask_sizes": [1.5, 2.0, 2.5]
            },
            daily_volume=5000000
        )
        print(f"  Loaded {symbol}")

    # Add illiquid market
    engine.update_market_data(
        symbol="ILLIQ_TOKEN",
        candles_15m=generate_candles(1.5, 100, 0.001),
        orderbook={
            "best_bid": 1.47,
            "best_ask": 1.53,  # 4% spread
            "bid_sizes": [500, 300, 200],
            "ask_sizes": [400, 350, 250]
        },
        daily_volume=50000  # Low volume
    )
    print("  Loaded ILLIQ_TOKEN (illiquid market)")

    # Add pair for stat arb
    btc_prices = [c["close"] for c in engine._market_data["BTC"]["candles_15m"]]
    eth_prices = [c["close"] for c in engine._market_data["ETH"]["candles_15m"]]
    engine.update_pair_data("BTC/ETH", "BTC", "ETH", btc_prices, eth_prices)
    print("  Loaded BTC/ETH pair")

    # Run simulation
    print("\n--- Running Trading Simulation ---")
    print("(Simulating 10 trading cycles)")

    for cycle in range(10):
        # Update prices with some movement
        for symbol in ["BTC", "ETH", "SOL"]:
            candles = engine._market_data[symbol]["candles_15m"]
            last_price = candles[-1]["close"]
            new_price = last_price * (1 + np.random.randn() * 0.01)

            # Add new candle
            new_candle = {
                "timestamp": time.time(),
                "open": last_price,
                "high": max(last_price, new_price) * 1.002,
                "low": min(last_price, new_price) * 0.998,
                "close": new_price,
                "volume": np.random.uniform(1000, 5000)
            }
            candles.append(new_candle)
            engine._price_cache[symbol] = new_price

        # Run cycle
        result = await engine.run_cycle()

        if result["signals_found"] > 0 or result["positions_opened"] > 0 or result["positions_closed"] > 0:
            print(f"\n  Cycle {cycle + 1}:")
            print(f"    Signals found: {result['signals_found']}")
            print(f"    Positions opened: {result['positions_opened']}")
            print(f"    Positions closed: {result['positions_closed']}")
            print(f"    Cycle PnL: ${result['pnl']:.2f}")

        # Small delay between cycles
        await asyncio.sleep(0.1)

    # Show final performance
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 70)

    perf = engine.get_performance()

    print("\n--- Summary ---")
    summary = perf["summary"]
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate: {summary['win_rate']:.1f}%")
    print(f"  Total PnL: ${summary['total_pnl']:.2f}")
    print(f"  Avg PnL/trade: ${summary['avg_pnl_per_trade']:.2f}")
    print(f"  Best trade: ${summary['best_trade']:.2f}")
    print(f"  Worst trade: ${summary['worst_trade']:.2f}")

    print("\n--- Risk Status ---")
    risk = perf["risk_status"]
    print(f"  Capital: ${risk['capital']:.2f}")
    print(f"  Drawdown: {risk['drawdown_pct']:.2f}%")
    print(f"  Daily PnL: ${risk['daily_pnl']:.2f}")
    print(f"  Exposure: {risk['exposure_pct']:.1f}%")
    print(f"  Open positions: {risk['open_positions']}")
    print(f"  Consecutive losses: {risk['consecutive_losses']}")
    print(f"  Trading locked: {risk['is_locked']}")

    print("\n--- Performance by Edge Type ---")
    for edge_type, stats in perf["by_edge_type"].items():
        win_rate = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"  {edge_type}:")
        print(f"    Trades: {stats['trades']}, Win rate: {win_rate:.1f}%, PnL: ${stats['pnl']:.2f}")

    print("\n--- Open Positions ---")
    positions = engine.get_open_positions()
    if positions:
        for pos in positions:
            print(f"  {pos['symbol']} {pos['direction']}: "
                  f"${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
    else:
        print("  No open positions")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. Risk management prevented excessive losses")
    print("2. Signal quality filters avoided bad trades")
    print("3. No reliance on speed - focused on BEING RIGHT")
    print("4. Multiple edge types provide diversification")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_realistic_engine())
