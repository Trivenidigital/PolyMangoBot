"""
Advanced Dynamic Fee and Slippage Estimator
Real-time cost estimation using market microstructure and ML
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger("PolyMangoBot.fees")


class MarketCondition(Enum):
    """Current market condition assessment"""
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"


@dataclass
class CostEstimate:
    """Comprehensive cost estimation result"""
    venue: str
    symbol: str

    # Fee components
    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.0
    expected_fee_pct: float = 0.0  # Weighted average based on expected fill type

    # Slippage components
    base_slippage_pct: float = 0.0
    volume_impact_pct: float = 0.0
    volatility_impact_pct: float = 0.0
    time_decay_pct: float = 0.0  # Slippage increases with time
    total_slippage_pct: float = 0.0

    # Network/execution costs
    network_latency_ms: float = 0.0
    latency_cost_pct: float = 0.0  # Cost of price movement during latency

    # Total cost
    total_cost_pct: float = 0.0
    total_cost_bps: float = 0.0  # In basis points

    # Confidence metrics
    confidence: float = 0.0
    data_quality: float = 0.0
    market_condition: MarketCondition = MarketCondition.NORMAL

    # Bounds
    best_case_cost_pct: float = 0.0
    worst_case_cost_pct: float = 0.0

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            "venue": self.venue,
            "symbol": self.symbol,
            "total_cost_pct": self.total_cost_pct,
            "total_cost_bps": self.total_cost_bps,
            "expected_fee_pct": self.expected_fee_pct,
            "total_slippage_pct": self.total_slippage_pct,
            "confidence": self.confidence,
            "market_condition": self.market_condition.value,
        }


@dataclass
class VenueFeeSchedule:
    """Fee schedule for a venue"""
    venue: str

    # Tiered fee structure (volume -> (maker_fee, taker_fee))
    fee_tiers: Dict[float, Tuple[float, float]] = field(default_factory=dict)

    # Special fees
    withdrawal_fee: float = 0.0
    deposit_fee: float = 0.0

    # Rebates
    maker_rebate: float = 0.0
    volume_rebate_tiers: Dict[float, float] = field(default_factory=dict)

    # Last update time
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())


class AdvancedFeeEstimator:
    """
    Advanced fee estimation with ML-based slippage prediction.

    Features:
    - Tiered fee calculation based on volume
    - Real-time slippage estimation from order book
    - Volatility-adjusted cost prediction
    - Latency impact modeling
    - Historical cost tracking for model calibration
    """

    # Default fee schedules
    DEFAULT_FEES = {
        "polymarket": {
            0: (0.0, 0.02),      # No maker fee, 2% taker (prediction market)
        },
        "kraken": {
            0: (0.0016, 0.0026),        # <$50k: 0.16% maker, 0.26% taker
            50000: (0.0014, 0.0024),    # $50k-$100k
            100000: (0.0012, 0.0022),   # $100k-$500k
            500000: (0.0010, 0.0020),   # $500k-$1M
            1000000: (0.0008, 0.0018),  # $1M-$5M
            5000000: (0.0006, 0.0016),  # $5M-$10M
            10000000: (0.0004, 0.0014), # $10M-$25M
            25000000: (0.0002, 0.0012), # >$25M
        },
        "coinbase": {
            0: (0.004, 0.006),          # <$10k: 0.4% maker, 0.6% taker
            10000: (0.0035, 0.0055),    # $10k-$50k
            50000: (0.0025, 0.0035),    # $50k-$100k
            100000: (0.002, 0.003),     # $100k-$1M
        },
    }

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size

        # Fee schedules per venue
        self.fee_schedules: Dict[str, VenueFeeSchedule] = {}
        self._initialize_fee_schedules()

        # Historical data for learning
        self.cost_history: deque = deque(maxlen=history_size)
        self.slippage_history: Dict[str, deque] = {}  # venue:symbol -> slippages

        # Market condition tracking
        self.volatility_history: Dict[str, deque] = {}  # symbol -> volatilities
        self.spread_history: Dict[str, deque] = {}

        # Learned parameters
        self.slippage_model_params: Dict[str, Dict] = {}

        # Volume tracking for tiered fees
        self.rolling_volume: Dict[str, float] = {}  # venue -> 30-day volume

    def _initialize_fee_schedules(self):
        """Initialize fee schedules from defaults"""
        for venue, tiers in self.DEFAULT_FEES.items():
            self.fee_schedules[venue] = VenueFeeSchedule(
                venue=venue,
                fee_tiers=tiers
            )

    def update_volume(self, venue: str, volume: float):
        """Update rolling 30-day volume for tiered fees"""
        self.rolling_volume[venue] = volume

    def get_fee_rates(
        self,
        venue: str,
        volume_30d: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Get maker and taker fee rates for a venue.

        Returns:
            (maker_fee_pct, taker_fee_pct)
        """
        if venue not in self.fee_schedules:
            return 0.002, 0.003  # Default conservative estimate

        volume = volume_30d or self.rolling_volume.get(venue, 0)
        schedule = self.fee_schedules[venue]

        # Find applicable tier
        applicable_tier = 0
        for tier_volume in sorted(schedule.fee_tiers.keys(), reverse=True):
            if volume >= tier_volume:
                applicable_tier = tier_volume
                break

        maker, taker = schedule.fee_tiers.get(applicable_tier, (0.002, 0.003))
        return maker, taker

    def estimate_base_slippage(
        self,
        order_book: List[Tuple[float, float]],
        quantity: float,
        side: str
    ) -> float:
        """
        Estimate base slippage from order book depth.
        """
        if not order_book:
            return 0.5  # Default 0.5% for empty book

        best_price = order_book[0][0]
        filled = 0.0
        cost = 0.0

        for price, qty in order_book:
            fill_qty = min(qty, quantity - filled)
            cost += fill_qty * price
            filled += fill_qty

            if filled >= quantity:
                break

        if filled == 0:
            return 1.0  # Can't fill

        avg_price = cost / filled
        slippage_pct = abs(avg_price - best_price) / best_price * 100

        return slippage_pct

    def estimate_volume_impact(
        self,
        quantity_usd: float,
        market_volume_24h: float,
        venue: str
    ) -> float:
        """
        Estimate market impact based on order size vs market volume.

        Uses square-root market impact model:
        Impact = k * sqrt(Q / V)

        where Q = order size, V = market volume, k = impact coefficient
        """
        if market_volume_24h <= 0:
            return 0.5  # Default for unknown volume

        # Impact coefficient varies by venue
        impact_coefficients = {
            "polymarket": 0.15,  # Higher for prediction markets (less liquid)
            "kraken": 0.05,
            "coinbase": 0.06,
        }
        k = impact_coefficients.get(venue, 0.08)

        # Square-root impact model
        volume_ratio = quantity_usd / market_volume_24h
        impact_pct = k * math.sqrt(volume_ratio) * 100

        # Cap at reasonable maximum
        return min(impact_pct, 2.0)

    def estimate_volatility_impact(
        self,
        symbol: str,
        volatility: Optional[float] = None
    ) -> Tuple[float, MarketCondition]:
        """
        Estimate additional slippage due to market volatility.

        Returns:
            (volatility_impact_pct, market_condition)
        """
        if volatility is None:
            volatility = self._get_recent_volatility(symbol)

        # Classify market condition
        if volatility < 0.5:
            condition = MarketCondition.CALM
            impact = 0.02
        elif volatility < 1.0:
            condition = MarketCondition.NORMAL
            impact = 0.05
        elif volatility < 2.0:
            condition = MarketCondition.VOLATILE
            impact = 0.15
        else:
            condition = MarketCondition.EXTREME
            impact = 0.35

        return impact, condition

    def _get_recent_volatility(self, symbol: str) -> float:
        """Get recent volatility from history"""
        if symbol not in self.volatility_history:
            return 1.0  # Default moderate volatility

        history = list(self.volatility_history[symbol])
        if len(history) < 2:
            return 1.0

        return statistics.mean(history[-20:])

    def estimate_latency_cost(
        self,
        latency_ms: float,
        volatility: float
    ) -> float:
        """
        Estimate cost of price movement during network latency.

        Price can move during the time it takes to execute.
        Higher volatility = higher latency cost.
        """
        # Approximate price movement per millisecond
        # Based on average tick frequency and size
        movement_per_ms = volatility * 0.0001  # 0.01% per ms at vol=1

        latency_cost = latency_ms * movement_per_ms
        return latency_cost

    def estimate_time_decay(
        self,
        expected_execution_time_ms: float,
        spread_bps: float
    ) -> float:
        """
        Estimate additional cost as spreads may widen during execution.

        Longer execution = higher probability of adverse spread movement.
        """
        # Base decay rate (% per second)
        decay_rate = 0.01

        # Adjust for current spread (wider spreads are less stable)
        spread_factor = 1.0 + (spread_bps / 100)

        time_decay = (expected_execution_time_ms / 1000) * decay_rate * spread_factor
        return time_decay

    def estimate_total_cost(
        self,
        venue: str,
        symbol: str,
        quantity_usd: float,
        order_book: List[Tuple[float, float]],
        side: str,
        market_volume_24h: float = 1000000,
        volatility: Optional[float] = None,
        latency_ms: float = 100,
        volume_30d: Optional[float] = None,
        is_maker: bool = False
    ) -> CostEstimate:
        """
        Estimate total transaction cost comprehensively.

        Args:
            venue: Trading venue
            symbol: Trading pair
            quantity_usd: Order size in USD
            order_book: Current order book
            side: "buy" or "sell"
            market_volume_24h: 24h market volume in USD
            volatility: Current volatility (0-1 scale, 1 = normal)
            latency_ms: Expected network latency
            volume_30d: 30-day trading volume for fee tier
            is_maker: Whether order will likely be maker
        """
        estimate = CostEstimate(
            venue=venue,
            symbol=symbol
        )

        # 1. Fee estimation
        maker_fee, taker_fee = self.get_fee_rates(venue, volume_30d)
        estimate.maker_fee_pct = maker_fee * 100
        estimate.taker_fee_pct = taker_fee * 100

        # Expected fee based on order type
        if is_maker:
            estimate.expected_fee_pct = estimate.maker_fee_pct
        else:
            # Assume mostly taker for market/aggressive orders
            estimate.expected_fee_pct = estimate.taker_fee_pct * 0.8 + estimate.maker_fee_pct * 0.2

        # 2. Base slippage from order book
        estimate.base_slippage_pct = self.estimate_base_slippage(order_book, quantity_usd, side)

        # 3. Volume impact
        estimate.volume_impact_pct = self.estimate_volume_impact(
            quantity_usd, market_volume_24h, venue
        )

        # 4. Volatility impact
        vol_impact, condition = self.estimate_volatility_impact(symbol, volatility)
        estimate.volatility_impact_pct = vol_impact
        estimate.market_condition = condition

        # 5. Latency cost
        effective_vol = volatility if volatility else self._get_recent_volatility(symbol)
        estimate.latency_cost_pct = self.estimate_latency_cost(latency_ms, effective_vol)
        estimate.network_latency_ms = latency_ms

        # 6. Time decay
        execution_time = latency_ms * 2  # Round trip estimate
        spread_bps = self._get_current_spread_bps(venue, symbol)
        estimate.time_decay_pct = self.estimate_time_decay(execution_time, spread_bps)

        # 7. Total slippage
        estimate.total_slippage_pct = (
            estimate.base_slippage_pct +
            estimate.volume_impact_pct +
            estimate.volatility_impact_pct +
            estimate.latency_cost_pct +
            estimate.time_decay_pct
        )

        # 8. Total cost
        estimate.total_cost_pct = estimate.expected_fee_pct + estimate.total_slippage_pct
        estimate.total_cost_bps = estimate.total_cost_pct * 100

        # 9. Confidence estimation
        estimate.confidence = self._calculate_confidence(venue, symbol, order_book)
        estimate.data_quality = self._assess_data_quality(order_book, market_volume_24h)

        # 10. Cost bounds
        estimate.best_case_cost_pct = estimate.expected_fee_pct + estimate.base_slippage_pct * 0.5
        estimate.worst_case_cost_pct = estimate.total_cost_pct * 2  # Double for worst case

        return estimate

    def _get_current_spread_bps(self, venue: str, symbol: str) -> float:
        """Get current spread in basis points"""
        key = f"{venue}:{symbol}"
        if key not in self.spread_history:
            return 20  # Default 20 bps

        history = list(self.spread_history[key])
        if not history:
            return 20

        return history[-1]

    def _calculate_confidence(
        self,
        venue: str,
        symbol: str,
        order_book: List[Tuple[float, float]]
    ) -> float:
        """Calculate confidence in the cost estimate"""
        confidence = 0.7  # Base confidence

        # Adjust for data availability
        key = f"{venue}:{symbol}"

        if key in self.slippage_history and len(self.slippage_history[key]) > 50:
            confidence += 0.1  # More historical data

        if len(order_book) >= 10:
            confidence += 0.1  # Deep order book

        if venue in ["kraken", "coinbase"]:
            confidence += 0.05  # Known fee structure

        return min(confidence, 0.95)

    def _assess_data_quality(
        self,
        order_book: List[Tuple[float, float]],
        market_volume: float
    ) -> float:
        """Assess quality of input data"""
        quality = 0.5

        if len(order_book) >= 10:
            quality += 0.2
        elif len(order_book) >= 5:
            quality += 0.1

        if market_volume > 1000000:
            quality += 0.2
        elif market_volume > 100000:
            quality += 0.1

        return min(quality, 1.0)

    def record_actual_cost(
        self,
        venue: str,
        symbol: str,
        estimated_cost: CostEstimate,
        actual_slippage_pct: float,
        actual_fee_pct: float
    ):
        """
        Record actual execution costs for model calibration.
        """
        self.cost_history.append({
            "venue": venue,
            "symbol": symbol,
            "estimated_total": estimated_cost.total_cost_pct,
            "actual_total": actual_slippage_pct + actual_fee_pct,
            "estimated_slippage": estimated_cost.total_slippage_pct,
            "actual_slippage": actual_slippage_pct,
            "timestamp": datetime.now().timestamp()
        })

        # Update slippage history
        key = f"{venue}:{symbol}"
        if key not in self.slippage_history:
            self.slippage_history[key] = deque(maxlen=100)
        self.slippage_history[key].append(actual_slippage_pct)

    def update_volatility(self, symbol: str, volatility: float):
        """Update volatility observation"""
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=100)
        self.volatility_history[symbol].append(volatility)

    def update_spread(self, venue: str, symbol: str, spread_bps: float):
        """Update spread observation"""
        key = f"{venue}:{symbol}"
        if key not in self.spread_history:
            self.spread_history[key] = deque(maxlen=100)
        self.spread_history[key].append(spread_bps)

    def get_model_accuracy(self) -> Dict:
        """Get model accuracy statistics"""
        if len(self.cost_history) < 10:
            return {"accuracy": 0, "samples": len(self.cost_history)}

        errors = []
        for record in self.cost_history:
            error = abs(record["estimated_total"] - record["actual_total"])
            errors.append(error)

        return {
            "mae": statistics.mean(errors),  # Mean absolute error
            "rmse": math.sqrt(statistics.mean(e**2 for e in errors)),
            "max_error": max(errors),
            "samples": len(self.cost_history),
            "avg_underestimate": statistics.mean(
                record["actual_total"] - record["estimated_total"]
                for record in self.cost_history
            )
        }


# Convenience function for quick estimation
def estimate_transaction_cost(
    venue: str,
    symbol: str,
    quantity_usd: float,
    order_book: List[Tuple[float, float]],
    side: str = "buy"
) -> float:
    """Quick cost estimation - returns total cost percentage"""
    estimator = AdvancedFeeEstimator()
    result = estimator.estimate_total_cost(
        venue=venue,
        symbol=symbol,
        quantity_usd=quantity_usd,
        order_book=order_book,
        side=side
    )
    return result.total_cost_pct


# Test
def test_fee_estimator():
    """Test the fee estimator"""
    estimator = AdvancedFeeEstimator()

    # Mock order book
    order_book = [
        (42500, 1.0),
        (42510, 2.0),
        (42520, 3.0),
        (42530, 4.0),
        (42540, 5.0),
    ]

    # Estimate costs
    estimate = estimator.estimate_total_cost(
        venue="kraken",
        symbol="BTC",
        quantity_usd=10000,
        order_book=order_book,
        side="buy",
        market_volume_24h=50000000,
        volatility=1.0,
        latency_ms=100
    )

    print("Cost Estimate:")
    print(f"  Venue: {estimate.venue}")
    print(f"  Expected fee: {estimate.expected_fee_pct:.3f}%")
    print(f"  Base slippage: {estimate.base_slippage_pct:.3f}%")
    print(f"  Volume impact: {estimate.volume_impact_pct:.3f}%")
    print(f"  Volatility impact: {estimate.volatility_impact_pct:.3f}%")
    print(f"  Total slippage: {estimate.total_slippage_pct:.3f}%")
    print(f"  TOTAL COST: {estimate.total_cost_pct:.3f}% ({estimate.total_cost_bps:.1f} bps)")
    print(f"  Confidence: {estimate.confidence:.1%}")
    print(f"  Market condition: {estimate.market_condition.value}")


if __name__ == "__main__":
    test_fee_estimator()
