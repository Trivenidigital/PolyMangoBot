"""
Trade Frequency Target Manager
===============================

Manages trade frequency to achieve target number of trades per period.
Enables high-frequency trading mode with 500+ trades per week.

Features:
1. Configurable trade frequency targets
2. Dynamic threshold adjustment
3. Opportunity prioritization
4. Trade pacing across time periods
5. Performance-based frequency scaling
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("PolyMangoBot.frequency_manager")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class FrequencyMode(Enum):
    """Trade frequency modes"""
    CONSERVATIVE = "conservative"       # 5-10 trades/day
    MODERATE = "moderate"               # 20-50 trades/day
    ACTIVE = "active"                   # 50-100 trades/day
    HIGH_FREQUENCY = "high_frequency"   # 100+ trades/day (500+/week)
    CUSTOM = "custom"                   # User-defined targets


class TradePriority(Enum):
    """Trade priority levels"""
    CRITICAL = "critical"       # Must execute
    HIGH = "high"               # Strong preference
    MEDIUM = "medium"           # Normal priority
    LOW = "low"                 # Optional, for frequency targets
    FILL = "fill"               # Just to meet frequency targets


class TimePeriod(Enum):
    """Time periods for frequency targets"""
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800


@dataclass
class FrequencyTarget:
    """Trade frequency target configuration"""
    # Target trades
    trades_per_day: int = 100
    trades_per_hour: int = 10
    trades_per_minute: int = 2

    # Flexibility
    min_trades_per_day: int = 50       # Minimum acceptable
    max_trades_per_day: int = 200      # Maximum allowed

    # Distribution
    prefer_even_distribution: bool = True  # Spread trades evenly
    allow_burst_trading: bool = True       # Allow trade bursts

    # Time windows (hours where trading is preferred)
    active_hours: List[int] = field(default_factory=lambda: list(range(24)))

    # Threshold adjustments
    can_lower_thresholds: bool = True      # Allow lowering profit thresholds
    min_profit_floor_pct: float = 0.01     # Absolute minimum profit
    threshold_adjustment_rate: float = 0.1 # How fast to adjust

    @classmethod
    def conservative(cls) -> "FrequencyTarget":
        return cls(
            trades_per_day=10,
            trades_per_hour=2,
            trades_per_minute=1,
            min_trades_per_day=5,
            max_trades_per_day=20,
            can_lower_thresholds=False
        )

    @classmethod
    def moderate(cls) -> "FrequencyTarget":
        return cls(
            trades_per_day=50,
            trades_per_hour=5,
            trades_per_minute=1,
            min_trades_per_day=30,
            max_trades_per_day=80
        )

    @classmethod
    def active(cls) -> "FrequencyTarget":
        return cls(
            trades_per_day=100,
            trades_per_hour=10,
            trades_per_minute=2,
            min_trades_per_day=70,
            max_trades_per_day=150
        )

    @classmethod
    def high_frequency(cls) -> "FrequencyTarget":
        """Target: 500+ trades per week (~72+ per day)"""
        return cls(
            trades_per_day=100,
            trades_per_hour=15,
            trades_per_minute=3,
            min_trades_per_day=72,
            max_trades_per_day=200,
            allow_burst_trading=True,
            can_lower_thresholds=True,
            threshold_adjustment_rate=0.15
        )


@dataclass
class TradeSlot:
    """A slot for a potential trade"""
    timestamp: float
    priority: TradePriority
    source: str  # "arbitrage", "directional", "micro_arb"
    opportunity_id: Optional[str] = None
    expected_profit_pct: float = 0.0
    executed: bool = False
    execution_time: Optional[float] = None


@dataclass
class FrequencyStats:
    """Statistics for frequency management"""
    # Counts
    trades_today: int = 0
    trades_this_hour: int = 0
    trades_this_minute: int = 0
    trades_this_week: int = 0

    # Rates
    current_rate_per_hour: float = 0.0
    avg_rate_per_hour: float = 0.0

    # Target tracking
    target_trades_remaining_today: int = 0
    hours_remaining_today: float = 24.0
    required_rate_to_meet_target: float = 0.0

    # Threshold adjustments
    current_threshold_multiplier: float = 1.0
    thresholds_lowered_count: int = 0

    # Performance
    trades_by_source: Dict[str, int] = field(default_factory=dict)
    trades_by_hour: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "trades_today": self.trades_today,
            "trades_this_hour": self.trades_this_hour,
            "trades_this_week": self.trades_this_week,
            "current_rate_per_hour": self.current_rate_per_hour,
            "target_remaining": self.target_trades_remaining_today,
            "required_rate": self.required_rate_to_meet_target,
            "threshold_multiplier": self.current_threshold_multiplier,
            "trades_by_source": self.trades_by_source
        }


@dataclass
class DynamicThresholds:
    """Dynamic trading thresholds"""
    min_spread_pct: float = 0.3
    min_profit_pct: float = 0.1
    min_confidence: float = 0.6
    min_liquidity_usd: float = 1000.0
    max_slippage_pct: float = 0.2

    # Original values (for resetting)
    original_min_spread_pct: float = 0.3
    original_min_profit_pct: float = 0.1
    original_min_confidence: float = 0.6

    def apply_multiplier(self, multiplier: float, floor_pct: float = 0.01):
        """Apply threshold adjustment multiplier"""
        # Lower thresholds = more opportunities
        self.min_spread_pct = max(floor_pct, self.original_min_spread_pct * multiplier)
        self.min_profit_pct = max(floor_pct, self.original_min_profit_pct * multiplier)
        self.min_confidence = max(0.4, self.original_min_confidence * multiplier)

    def reset(self):
        """Reset to original values"""
        self.min_spread_pct = self.original_min_spread_pct
        self.min_profit_pct = self.original_min_profit_pct
        self.min_confidence = self.original_min_confidence


# =============================================================================
# TRADE PACER
# =============================================================================

class TradePacer:
    """
    Manages trade pacing to achieve frequency targets.

    Ensures trades are distributed evenly across time periods
    while allowing flexibility for burst trading when opportunities arise.
    """

    def __init__(self, target: FrequencyTarget):
        self.target = target

        # Time tracking
        self._day_start: float = self._get_day_start()
        self._hour_start: float = self._get_hour_start()
        self._minute_start: float = self._get_minute_start()

        # Trade tracking
        self._trades_today: deque = deque(maxlen=1000)
        self._trades_this_hour: deque = deque(maxlen=200)
        self._trades_this_minute: deque = deque(maxlen=20)

        # Burst tracking
        self._burst_mode: bool = False
        self._burst_start: Optional[float] = None
        self._burst_trades: int = 0

    def _get_day_start(self) -> float:
        """Get timestamp of current day start (UTC)"""
        now = datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start.timestamp()

    def _get_hour_start(self) -> float:
        """Get timestamp of current hour start"""
        now = datetime.utcnow()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        return hour_start.timestamp()

    def _get_minute_start(self) -> float:
        """Get timestamp of current minute start"""
        now = datetime.utcnow()
        minute_start = now.replace(second=0, microsecond=0)
        return minute_start.timestamp()

    def _refresh_periods(self):
        """Refresh time period tracking"""
        now = time.time()

        # Check for new day
        current_day_start = self._get_day_start()
        if current_day_start != self._day_start:
            self._day_start = current_day_start
            self._trades_today.clear()

        # Check for new hour
        current_hour_start = self._get_hour_start()
        if current_hour_start != self._hour_start:
            self._hour_start = current_hour_start
            self._trades_this_hour.clear()

        # Check for new minute
        current_minute_start = self._get_minute_start()
        if current_minute_start != self._minute_start:
            self._minute_start = current_minute_start
            self._trades_this_minute.clear()

    def record_trade(self, trade_slot: TradeSlot):
        """Record a completed trade"""
        self._refresh_periods()

        now = time.time()
        self._trades_today.append(trade_slot)
        self._trades_this_hour.append(trade_slot)
        self._trades_this_minute.append(trade_slot)

        # Update burst tracking
        if self._burst_mode:
            self._burst_trades += 1

    def can_trade_now(self) -> Tuple[bool, str]:
        """Check if we can execute a trade now"""
        self._refresh_periods()

        # Check minute limit
        if len(self._trades_this_minute) >= self.target.trades_per_minute:
            if not self._burst_mode:
                return False, f"Minute limit reached ({self.target.trades_per_minute})"

        # Check hour limit (soft limit)
        if len(self._trades_this_hour) >= self.target.trades_per_hour * 1.5:
            return False, f"Hour limit exceeded"

        # Check day limit
        if len(self._trades_today) >= self.target.max_trades_per_day:
            return False, f"Day limit reached ({self.target.max_trades_per_day})"

        # Check if current hour is active
        current_hour = datetime.utcnow().hour
        if current_hour not in self.target.active_hours:
            return False, f"Outside active hours"

        return True, "OK"

    def get_urgency_score(self) -> float:
        """
        Calculate urgency to trade (0-1).
        Higher = need more trades to meet target.
        """
        self._refresh_periods()

        # Calculate progress
        trades_done = len(self._trades_today)
        target = self.target.trades_per_day

        # Hours remaining in day
        now = datetime.utcnow()
        hours_remaining = 24 - now.hour - (now.minute / 60)
        hours_remaining = max(0.1, hours_remaining)  # Avoid division by zero

        # Required rate to meet target
        trades_remaining = target - trades_done
        required_rate = trades_remaining / hours_remaining

        # Current rate
        current_rate = len(self._trades_this_hour)

        # Calculate urgency
        if trades_remaining <= 0:
            return 0.0  # Already met target

        if current_rate >= required_rate:
            return 0.3  # On track

        # Behind schedule
        rate_deficit = (required_rate - current_rate) / required_rate
        urgency = min(1.0, 0.5 + rate_deficit * 0.5)

        return urgency

    def should_lower_thresholds(self) -> Tuple[bool, float]:
        """
        Determine if thresholds should be lowered to meet targets.

        Returns:
            (should_lower: bool, suggested_multiplier: float)
        """
        if not self.target.can_lower_thresholds:
            return False, 1.0

        urgency = self.get_urgency_score()

        if urgency < 0.5:
            return False, 1.0

        if urgency < 0.7:
            # Slight adjustment
            return True, 0.9

        if urgency < 0.85:
            # Moderate adjustment
            return True, 0.75

        # Significant adjustment
        return True, 0.6

    def enter_burst_mode(self, reason: str):
        """Enter burst trading mode"""
        if not self.target.allow_burst_trading:
            return

        self._burst_mode = True
        self._burst_start = time.time()
        self._burst_trades = 0

        logger.info(f"Entering burst mode: {reason}")

    def exit_burst_mode(self):
        """Exit burst trading mode"""
        if self._burst_mode:
            duration = time.time() - self._burst_start if self._burst_start else 0
            logger.info(
                f"Exiting burst mode: {self._burst_trades} trades in {duration:.1f}s"
            )

        self._burst_mode = False
        self._burst_start = None

    def get_recommended_wait_time(self) -> float:
        """Get recommended time to wait before next trade (seconds)"""
        self._refresh_periods()

        # If we've hit minute limit, wait until next minute
        if len(self._trades_this_minute) >= self.target.trades_per_minute:
            seconds_in_minute = time.time() - self._minute_start
            return max(0, 60 - seconds_in_minute + 0.1)

        # Calculate optimal spacing
        trades_remaining_in_hour = self.target.trades_per_hour - len(self._trades_this_hour)
        if trades_remaining_in_hour <= 0:
            seconds_in_hour = time.time() - self._hour_start
            return max(0, 3600 - seconds_in_hour + 0.1)

        # Spread remaining trades across remaining time in hour
        seconds_in_hour = time.time() - self._hour_start
        seconds_remaining = 3600 - seconds_in_hour

        if trades_remaining_in_hour > 0:
            optimal_spacing = seconds_remaining / trades_remaining_in_hour
            return min(optimal_spacing, 30)  # Cap at 30 seconds

        return 0

    def get_stats(self) -> Dict:
        """Get pacing statistics"""
        self._refresh_periods()

        now = datetime.utcnow()
        hours_remaining = max(0.1, 24 - now.hour - (now.minute / 60))

        trades_today = len(self._trades_today)
        target = self.target.trades_per_day

        return {
            "trades_today": trades_today,
            "trades_this_hour": len(self._trades_this_hour),
            "trades_this_minute": len(self._trades_this_minute),
            "target_per_day": target,
            "progress_pct": (trades_today / target) * 100 if target > 0 else 0,
            "hours_remaining": hours_remaining,
            "trades_remaining": max(0, target - trades_today),
            "required_rate_per_hour": max(0, target - trades_today) / hours_remaining,
            "current_rate_per_hour": len(self._trades_this_hour),
            "urgency_score": self.get_urgency_score(),
            "burst_mode": self._burst_mode
        }


# =============================================================================
# OPPORTUNITY PRIORITIZER
# =============================================================================

class OpportunityPrioritizer:
    """
    Prioritizes trading opportunities based on quality and frequency targets.
    """

    def __init__(self, pacer: TradePacer):
        self.pacer = pacer

        # Priority weights
        self.weights = {
            "profit": 0.3,
            "confidence": 0.25,
            "liquidity": 0.15,
            "freshness": 0.15,
            "urgency_alignment": 0.15
        }

    def prioritize(
        self,
        opportunities: List[Dict],
        max_count: int = 10
    ) -> List[Tuple[Dict, TradePriority, float]]:
        """
        Prioritize opportunities.

        Returns list of (opportunity, priority, score) tuples.
        """
        if not opportunities:
            return []

        urgency = self.pacer.get_urgency_score()
        scored = []

        for opp in opportunities:
            score = self._calculate_score(opp, urgency)
            priority = self._determine_priority(opp, score, urgency)
            scored.append((opp, priority, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        return scored[:max_count]

    def _calculate_score(self, opp: Dict, urgency: float) -> float:
        """Calculate opportunity score"""
        score = 0.0

        # Profit contribution
        profit = opp.get("estimated_profit_pct", 0) or opp.get("expected_return_pct", 0)
        profit_score = min(1.0, profit / 1.0)  # Normalize to 1% = 1.0
        score += profit_score * self.weights["profit"]

        # Confidence contribution
        confidence = opp.get("confidence", 0.5)
        score += confidence * self.weights["confidence"]

        # Liquidity contribution
        liquidity = opp.get("liquidity", 1000)
        liquidity_score = min(1.0, liquidity / 10000)  # 10K = 1.0
        score += liquidity_score * self.weights["liquidity"]

        # Freshness contribution
        age_ms = opp.get("price_age_ms", 500)
        freshness_score = max(0, 1 - (age_ms / 1000))
        score += freshness_score * self.weights["freshness"]

        # Urgency alignment (if urgent, prefer more opportunities)
        if urgency > 0.5:
            # When urgent, boost lower-quality opportunities
            urgency_boost = urgency * 0.2
            score += urgency_boost * self.weights["urgency_alignment"]

        return min(1.0, score)

    def _determine_priority(
        self,
        opp: Dict,
        score: float,
        urgency: float
    ) -> TradePriority:
        """Determine priority level"""
        profit = opp.get("estimated_profit_pct", 0) or opp.get("expected_return_pct", 0)

        # High profit always high priority
        if profit > 0.5 and score > 0.7:
            return TradePriority.CRITICAL

        if score > 0.6:
            return TradePriority.HIGH

        if score > 0.4:
            return TradePriority.MEDIUM

        # When urgent, accept lower quality
        if urgency > 0.7 and score > 0.3:
            return TradePriority.LOW

        if urgency > 0.85 and score > 0.2:
            return TradePriority.FILL

        return TradePriority.LOW


# =============================================================================
# FREQUENCY MANAGER
# =============================================================================

class TradeFrequencyManager:
    """
    Main manager for trade frequency targeting.

    Coordinates:
    - Trade pacing
    - Threshold adjustment
    - Opportunity prioritization
    - Performance tracking
    """

    def __init__(
        self,
        mode: FrequencyMode = FrequencyMode.ACTIVE,
        target: Optional[FrequencyTarget] = None
    ):
        self.mode = mode

        # Set target based on mode or use custom
        if target:
            self.target = target
        elif mode == FrequencyMode.CONSERVATIVE:
            self.target = FrequencyTarget.conservative()
        elif mode == FrequencyMode.MODERATE:
            self.target = FrequencyTarget.moderate()
        elif mode == FrequencyMode.HIGH_FREQUENCY:
            self.target = FrequencyTarget.high_frequency()
        else:
            self.target = FrequencyTarget.active()

        self.pacer = TradePacer(self.target)
        self.prioritizer = OpportunityPrioritizer(self.pacer)

        # Dynamic thresholds
        self.thresholds = DynamicThresholds()
        self._original_thresholds = DynamicThresholds()

        # Statistics
        self._stats = FrequencyStats()
        self._trade_history: deque = deque(maxlen=5000)

        # Callbacks
        self._threshold_change_callbacks: List[Callable] = []

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed now"""
        return self.pacer.can_trade_now()

    def record_trade(
        self,
        source: str,
        opportunity_id: Optional[str] = None,
        profit_pct: float = 0.0
    ):
        """Record a completed trade"""
        slot = TradeSlot(
            timestamp=time.time(),
            priority=TradePriority.MEDIUM,
            source=source,
            opportunity_id=opportunity_id,
            expected_profit_pct=profit_pct,
            executed=True,
            execution_time=time.time()
        )

        self.pacer.record_trade(slot)
        self._trade_history.append(slot)

        # Update stats
        self._stats.trades_today += 1
        self._stats.trades_this_hour += 1
        self._stats.trades_this_minute += 1

        # Track by source
        if source not in self._stats.trades_by_source:
            self._stats.trades_by_source[source] = 0
        self._stats.trades_by_source[source] += 1

        # Track by hour
        hour = datetime.utcnow().hour
        if hour not in self._stats.trades_by_hour:
            self._stats.trades_by_hour[hour] = 0
        self._stats.trades_by_hour[hour] += 1

        # Check if we should adjust thresholds
        self._check_threshold_adjustment()

    def prioritize_opportunities(
        self,
        opportunities: List[Dict],
        max_count: int = 10
    ) -> List[Tuple[Dict, TradePriority, float]]:
        """Prioritize opportunities based on quality and frequency needs"""
        return self.prioritizer.prioritize(opportunities, max_count)

    def get_current_thresholds(self) -> DynamicThresholds:
        """Get current trading thresholds"""
        return self.thresholds

    def _check_threshold_adjustment(self):
        """Check if thresholds need adjustment"""
        should_lower, multiplier = self.pacer.should_lower_thresholds()

        if should_lower:
            old_multiplier = self._stats.current_threshold_multiplier

            # Gradual adjustment
            new_multiplier = old_multiplier + (
                (multiplier - old_multiplier) * self.target.threshold_adjustment_rate
            )

            if abs(new_multiplier - old_multiplier) > 0.01:
                self._apply_threshold_multiplier(new_multiplier)
        else:
            # Gradually restore thresholds
            if self._stats.current_threshold_multiplier < 1.0:
                new_multiplier = min(
                    1.0,
                    self._stats.current_threshold_multiplier + 0.05
                )
                self._apply_threshold_multiplier(new_multiplier)

    def _apply_threshold_multiplier(self, multiplier: float):
        """Apply threshold multiplier"""
        old_multiplier = self._stats.current_threshold_multiplier
        self._stats.current_threshold_multiplier = multiplier

        self.thresholds.apply_multiplier(multiplier, self.target.min_profit_floor_pct)

        if multiplier < old_multiplier:
            self._stats.thresholds_lowered_count += 1

        logger.info(
            f"Threshold adjustment: {old_multiplier:.2f} -> {multiplier:.2f} "
            f"(min_spread: {self.thresholds.min_spread_pct:.3f}%, "
            f"min_profit: {self.thresholds.min_profit_pct:.3f}%)"
        )

        # Notify callbacks
        for callback in self._threshold_change_callbacks:
            try:
                callback(self.thresholds)
            except Exception as e:
                logger.error(f"Threshold callback error: {e}")

    def on_threshold_change(self, callback: Callable):
        """Register callback for threshold changes"""
        self._threshold_change_callbacks.append(callback)

    def get_urgency(self) -> float:
        """Get current trading urgency (0-1)"""
        return self.pacer.get_urgency_score()

    def get_recommended_wait(self) -> float:
        """Get recommended wait time before next trade"""
        return self.pacer.get_recommended_wait_time()

    def enter_burst_mode(self, reason: str = "Manual"):
        """Enter burst trading mode"""
        self.pacer.enter_burst_mode(reason)

    def exit_burst_mode(self):
        """Exit burst trading mode"""
        self.pacer.exit_burst_mode()

    def set_mode(self, mode: FrequencyMode):
        """Change frequency mode"""
        self.mode = mode

        if mode == FrequencyMode.CONSERVATIVE:
            self.target = FrequencyTarget.conservative()
        elif mode == FrequencyMode.MODERATE:
            self.target = FrequencyTarget.moderate()
        elif mode == FrequencyMode.HIGH_FREQUENCY:
            self.target = FrequencyTarget.high_frequency()
        else:
            self.target = FrequencyTarget.active()

        self.pacer.target = self.target

        # Reset thresholds
        self.thresholds.reset()
        self._stats.current_threshold_multiplier = 1.0

        logger.info(f"Frequency mode changed to: {mode.value}")

    def set_custom_target(self, trades_per_day: int):
        """Set custom trades per day target"""
        self.mode = FrequencyMode.CUSTOM
        self.target = FrequencyTarget(
            trades_per_day=trades_per_day,
            trades_per_hour=max(1, trades_per_day // 24),
            trades_per_minute=max(1, trades_per_day // (24 * 60)),
            min_trades_per_day=int(trades_per_day * 0.7),
            max_trades_per_day=int(trades_per_day * 1.5),
            can_lower_thresholds=True
        )
        self.pacer.target = self.target

        logger.info(f"Custom target set: {trades_per_day} trades/day")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        pacer_stats = self.pacer.get_stats()

        return {
            "mode": self.mode.value,
            "target_per_day": self.target.trades_per_day,
            "target_per_week": self.target.trades_per_day * 7,
            **pacer_stats,
            "thresholds": {
                "min_spread_pct": self.thresholds.min_spread_pct,
                "min_profit_pct": self.thresholds.min_profit_pct,
                "min_confidence": self.thresholds.min_confidence,
                "multiplier": self._stats.current_threshold_multiplier
            },
            "trades_by_source": self._stats.trades_by_source,
            "trades_by_hour": self._stats.trades_by_hour,
            "thresholds_lowered_count": self._stats.thresholds_lowered_count
        }

    def get_weekly_projection(self) -> Dict:
        """Get projected weekly performance"""
        stats = self.pacer.get_stats()

        current_daily_rate = stats["trades_today"]
        hours_elapsed = 24 - stats["hours_remaining"]

        if hours_elapsed > 0:
            projected_daily = current_daily_rate / hours_elapsed * 24
        else:
            projected_daily = self.target.trades_per_day

        projected_weekly = projected_daily * 7

        return {
            "projected_daily": projected_daily,
            "projected_weekly": projected_weekly,
            "target_weekly": self.target.trades_per_day * 7,
            "on_track": projected_weekly >= self.target.trades_per_day * 7 * 0.9,
            "efficiency": projected_weekly / (self.target.trades_per_day * 7) * 100
        }


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_frequency_manager():
    """Test frequency management"""
    print("=" * 70)
    print("TRADE FREQUENCY MANAGER TEST")
    print("=" * 70)

    # Test different modes
    for mode in [FrequencyMode.CONSERVATIVE, FrequencyMode.ACTIVE, FrequencyMode.HIGH_FREQUENCY]:
        print(f"\n--- Testing {mode.value.upper()} mode ---")

        manager = TradeFrequencyManager(mode=mode)

        print(f"  Target: {manager.target.trades_per_day} trades/day")
        print(f"  Weekly target: {manager.target.trades_per_day * 7} trades/week")
        print(f"  Per hour: {manager.target.trades_per_hour}")
        print(f"  Per minute: {manager.target.trades_per_minute}")

        # Simulate some trades
        for i in range(10):
            can_trade, reason = manager.can_trade()
            if can_trade:
                manager.record_trade(
                    source="test",
                    profit_pct=np.random.uniform(0.05, 0.3)
                )
            await asyncio.sleep(0.01)

        stats = manager.get_stats()
        print(f"  Trades recorded: {stats['trades_today']}")
        print(f"  Progress: {stats['progress_pct']:.1f}%")
        print(f"  Urgency: {stats['urgency_score']:.2f}")
        print(f"  Threshold multiplier: {stats['thresholds']['multiplier']:.2f}")

    # Test high-frequency mode in detail
    print("\n--- HIGH FREQUENCY MODE DETAIL ---")
    manager = TradeFrequencyManager(mode=FrequencyMode.HIGH_FREQUENCY)

    # Simulate a day's trading
    print("\nSimulating trades...")
    trade_count = 0

    for hour in range(24):
        for minute in range(60):
            can_trade, _ = manager.can_trade()
            if can_trade and np.random.random() < 0.3:  # 30% chance of opportunity
                manager.record_trade(
                    source=np.random.choice(["arbitrage", "directional", "micro_arb"]),
                    profit_pct=np.random.uniform(0.01, 0.5)
                )
                trade_count += 1

            # Check urgency periodically
            if minute == 30:
                urgency = manager.get_urgency()
                if urgency > 0.7:
                    manager.enter_burst_mode("Behind schedule")
                elif urgency < 0.4:
                    manager.exit_burst_mode()

    print(f"\n  Total trades simulated: {trade_count}")

    stats = manager.get_stats()
    print(f"  Trades by source: {stats['trades_by_source']}")

    projection = manager.get_weekly_projection()
    print(f"  Projected weekly: {projection['projected_weekly']:.0f}")
    print(f"  On track: {projection['on_track']}")
    print(f"  Efficiency: {projection['efficiency']:.1f}%")

    # Test priority system
    print("\n--- OPPORTUNITY PRIORITIZATION ---")

    opportunities = [
        {"id": "1", "estimated_profit_pct": 0.5, "confidence": 0.8, "liquidity": 5000, "price_age_ms": 100},
        {"id": "2", "estimated_profit_pct": 0.1, "confidence": 0.9, "liquidity": 20000, "price_age_ms": 200},
        {"id": "3", "estimated_profit_pct": 0.3, "confidence": 0.6, "liquidity": 3000, "price_age_ms": 400},
        {"id": "4", "estimated_profit_pct": 0.05, "confidence": 0.5, "liquidity": 1000, "price_age_ms": 800},
    ]

    prioritized = manager.prioritize_opportunities(opportunities)

    print("  Prioritized opportunities:")
    for opp, priority, score in prioritized:
        print(f"    {opp['id']}: {priority.value} (score: {score:.2f}, profit: {opp['estimated_profit_pct']}%)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_frequency_manager())
