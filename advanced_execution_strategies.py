"""
Advanced Execution Strategies Module
=====================================

Sophisticated order execution strategies for optimal trade execution:

1. Adaptive Order Types:
   - Dynamic selection between market/limit/post-only/IOC/FOK
   - Price-responsive order type switching
   - Queue position optimization

2. Optimal Execution Timing:
   - Spread analysis for entry timing
   - Queue position estimation
   - Market impact prediction

3. Split Order Execution:
   - TWAP (Time-Weighted Average Price)
   - VWAP (Volume-Weighted Average Price)
   - Iceberg orders
   - Adaptive splitting based on liquidity
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger("PolyMangoBot.execution_strategies")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class OrderType(Enum):
    """Supported order types"""
    MARKET = "market"           # Immediate execution at best available price
    LIMIT = "limit"             # Execute at specified price or better
    POST_ONLY = "post_only"     # Only add liquidity (maker only)
    IOC = "ioc"                 # Immediate or Cancel - fill what you can, cancel rest
    FOK = "fok"                 # Fill or Kill - all or nothing
    ICEBERG = "iceberg"         # Show only portion of total size


class ExecutionUrgency(Enum):
    """Execution urgency levels"""
    LOW = "low"                 # Can wait for best price
    MEDIUM = "medium"           # Balance speed and price
    HIGH = "high"               # Need execution soon
    CRITICAL = "critical"       # Execute immediately at any price


class SplitStrategy(Enum):
    """Order splitting strategies"""
    NONE = "none"               # Single order
    TWAP = "twap"               # Time-weighted splits
    VWAP = "vwap"               # Volume-weighted splits
    ADAPTIVE = "adaptive"       # Adapt to market conditions
    ICEBERG = "iceberg"         # Hidden size with visible portion


@dataclass
class MarketConditions:
    """Current market conditions for execution decisions"""
    spread_bps: float = 0.0
    spread_percentile: float = 0.5
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    volatility: float = 0.0
    toxicity: float = 0.0
    recent_volume: float = 0.0
    price_trend: float = 0.0  # Positive = upward
    timestamp: float = field(default_factory=time.time)

    @property
    def is_favorable(self) -> bool:
        """Check if conditions are favorable for trading"""
        return (
            self.spread_percentile < 0.5 and
            self.toxicity < 0.4 and
            self.bid_depth > 0 and
            self.ask_depth > 0
        )


@dataclass
class ExecutionPlan:
    """Plan for executing an order"""
    order_type: OrderType
    urgency: ExecutionUrgency
    split_strategy: SplitStrategy
    total_quantity: float
    price: float
    slices: List[Dict] = field(default_factory=list)
    estimated_fill_price: float = 0.0
    estimated_slippage_bps: float = 0.0
    estimated_duration_ms: float = 0.0
    confidence: float = 0.5
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "order_type": self.order_type.value,
            "urgency": self.urgency.value,
            "split_strategy": self.split_strategy.value,
            "total_quantity": self.total_quantity,
            "price": self.price,
            "num_slices": len(self.slices),
            "estimated_fill_price": self.estimated_fill_price,
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "estimated_duration_ms": self.estimated_duration_ms,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class SliceResult:
    """Result of executing a single slice"""
    slice_id: int
    quantity_requested: float
    quantity_filled: float
    avg_fill_price: float
    fees: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Complete execution result"""
    plan: ExecutionPlan
    slice_results: List[SliceResult] = field(default_factory=list)
    total_filled: float = 0.0
    total_fees: float = 0.0
    vwap: float = 0.0
    total_latency_ms: float = 0.0
    success: bool = False
    slippage_bps: float = 0.0

    def calculate_metrics(self):
        """Calculate aggregate metrics from slices"""
        if not self.slice_results:
            return

        self.total_filled = sum(s.quantity_filled for s in self.slice_results)
        self.total_fees = sum(s.fees for s in self.slice_results)
        self.total_latency_ms = sum(s.latency_ms for s in self.slice_results)

        # Calculate VWAP
        total_value = sum(
            s.quantity_filled * s.avg_fill_price
            for s in self.slice_results if s.quantity_filled > 0
        )
        if self.total_filled > 0:
            self.vwap = total_value / self.total_filled

        # Calculate slippage
        if self.plan.price > 0 and self.vwap > 0:
            self.slippage_bps = ((self.vwap - self.plan.price) / self.plan.price) * 10000

        # Success if filled majority
        fill_rate = self.total_filled / self.plan.total_quantity if self.plan.total_quantity > 0 else 0
        self.success = fill_rate >= 0.95


# =============================================================================
# ADAPTIVE ORDER TYPE SELECTOR
# =============================================================================

class AdaptiveOrderTypeSelector:
    """
    Selects optimal order type based on market conditions and urgency.

    Decision factors:
    - Spread width (wide spread -> limit/post-only)
    - Urgency (high urgency -> market/IOC)
    - Order size vs liquidity (large -> split/iceberg)
    - Toxicity (high toxicity -> post-only/limit)
    """

    def __init__(self):
        # Thresholds
        self.wide_spread_threshold_bps = 20
        self.large_order_pct_of_depth = 0.3
        self.high_toxicity_threshold = 0.5

    def select_order_type(
        self,
        side: str,
        quantity: float,
        conditions: MarketConditions,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> Tuple[OrderType, List[str]]:
        """
        Select optimal order type based on conditions.

        Returns:
            (OrderType, list of reasoning strings)
        """
        reasons = []

        # Critical urgency always uses market orders
        if urgency == ExecutionUrgency.CRITICAL:
            reasons.append("Critical urgency - using market order for immediate execution")
            return OrderType.MARKET, reasons

        # Check spread
        spread_is_wide = conditions.spread_bps > self.wide_spread_threshold_bps

        # Check order size relative to available liquidity
        available_depth = conditions.ask_depth if side == "buy" else conditions.bid_depth
        is_large_order = quantity > available_depth * self.large_order_pct_of_depth if available_depth > 0 else False

        # Check toxicity
        is_toxic = conditions.toxicity > self.high_toxicity_threshold

        # Decision logic
        if urgency == ExecutionUrgency.HIGH:
            if spread_is_wide:
                reasons.append(f"High urgency with wide spread ({conditions.spread_bps:.1f}bps) - using IOC to limit slippage")
                return OrderType.IOC, reasons
            else:
                reasons.append("High urgency with tight spread - using market order")
                return OrderType.MARKET, reasons

        if is_toxic:
            reasons.append(f"High toxicity ({conditions.toxicity:.2f}) - using post-only to avoid adverse selection")
            return OrderType.POST_ONLY, reasons

        if spread_is_wide:
            if is_large_order:
                reasons.append(f"Wide spread + large order - using iceberg to hide size")
                return OrderType.ICEBERG, reasons
            else:
                reasons.append(f"Wide spread ({conditions.spread_bps:.1f}bps) - using limit order inside spread")
                return OrderType.LIMIT, reasons

        if is_large_order:
            reasons.append(f"Large order ({quantity:.2f} > {available_depth * self.large_order_pct_of_depth:.2f}) - using iceberg")
            return OrderType.ICEBERG, reasons

        # Default: favorable conditions
        if conditions.spread_percentile < 0.3:
            reasons.append(f"Tight spread (percentile: {conditions.spread_percentile:.1%}) - using limit order at best")
            return OrderType.LIMIT, reasons

        if urgency == ExecutionUrgency.LOW:
            reasons.append("Low urgency - using post-only for maker rebates")
            return OrderType.POST_ONLY, reasons

        reasons.append("Normal conditions - using limit order")
        return OrderType.LIMIT, reasons


# =============================================================================
# OPTIMAL EXECUTION TIMING
# =============================================================================

class ExecutionTimingOptimizer:
    """
    Optimizes execution timing based on market microstructure.

    Features:
    - Queue position estimation
    - Spread analysis for entry timing
    - Market impact prediction
    """

    def __init__(self, history_size: int = 500):
        self.history_size = history_size

        # Historical data
        self._spread_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._fill_rate_history: Dict[str, deque] = {}

    def estimate_queue_position(
        self,
        side: str,
        price: float,
        quantity: float,
        order_book: List[Tuple[float, float]]
    ) -> Dict:
        """
        Estimate queue position if placing limit order at price.

        Returns dict with:
        - position: estimated queue position
        - ahead_quantity: total quantity ahead in queue
        - estimated_wait_time_ms: estimated time to fill
        - fill_probability: probability of getting filled
        """
        if not order_book:
            return {
                "position": 0,
                "ahead_quantity": 0,
                "estimated_wait_time_ms": 0,
                "fill_probability": 0.5
            }

        best_price = order_book[0][0]
        ahead_quantity = 0.0
        position = 0

        # For buys, count orders at same or better (higher) price
        # For sells, count orders at same or better (lower) price
        for book_price, book_qty in order_book:
            if side == "buy":
                if book_price >= price:
                    ahead_quantity += book_qty
                    position += 1
            else:
                if book_price <= price:
                    ahead_quantity += book_qty
                    position += 1

        # Estimate wait time based on typical fill rate
        # Assume average fill rate of 100 units per second
        avg_fill_rate = 100.0
        estimated_wait_ms = (ahead_quantity / avg_fill_rate) * 1000

        # Fill probability decreases with queue depth
        fill_probability = max(0.1, 1.0 - (position * 0.1))

        return {
            "position": position,
            "ahead_quantity": ahead_quantity,
            "estimated_wait_time_ms": estimated_wait_ms,
            "fill_probability": fill_probability
        }

    def should_cross_spread(
        self,
        side: str,
        conditions: MarketConditions,
        urgency: ExecutionUrgency,
        target_fill_time_ms: float = 1000
    ) -> Tuple[bool, str]:
        """
        Decide whether to cross the spread (take liquidity) or join the queue.

        Returns:
            (should_cross: bool, reasoning: str)
        """
        # Always cross for critical urgency
        if urgency == ExecutionUrgency.CRITICAL:
            return True, "Critical urgency - crossing spread for immediate fill"

        # Favorable to cross if spread is tight
        if conditions.spread_bps < 5:
            if urgency in [ExecutionUrgency.HIGH, ExecutionUrgency.MEDIUM]:
                return True, f"Tight spread ({conditions.spread_bps:.1f}bps) - crossing is low cost"

        # Don't cross if spread is wide and not urgent
        if conditions.spread_bps > 30:
            if urgency == ExecutionUrgency.LOW:
                return False, f"Wide spread ({conditions.spread_bps:.1f}bps) with low urgency - joining queue"

        # Consider toxicity
        if conditions.toxicity > 0.5:
            return False, f"High toxicity ({conditions.toxicity:.2f}) - better to provide liquidity"

        # Consider spread percentile
        if conditions.spread_percentile > 0.7:
            return False, f"Spread is wide vs history (percentile: {conditions.spread_percentile:.1%}) - wait for tighter spread"

        if conditions.spread_percentile < 0.3:
            return True, f"Spread is tight vs history (percentile: {conditions.spread_percentile:.1%}) - good time to cross"

        # Default based on urgency
        if urgency == ExecutionUrgency.HIGH:
            return True, "High urgency - defaulting to cross spread"

        return False, "Medium urgency - defaulting to join queue"

    def calculate_optimal_limit_price(
        self,
        side: str,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        conditions: MarketConditions,
        urgency: ExecutionUrgency
    ) -> Tuple[float, str]:
        """
        Calculate optimal limit price for the order.

        Returns:
            (optimal_price, reasoning)
        """
        spread = best_ask - best_bid

        if urgency == ExecutionUrgency.CRITICAL:
            # Cross the spread
            if side == "buy":
                price = best_ask
                reason = "Critical: buying at ask for immediate fill"
            else:
                price = best_bid
                reason = "Critical: selling at bid for immediate fill"
            return price, reason

        if urgency == ExecutionUrgency.HIGH:
            # Aggressive but not market
            if side == "buy":
                price = best_ask - (spread * 0.25)  # 25% inside spread
                reason = f"High urgency: buying {spread * 0.25:.4f} inside spread"
            else:
                price = best_bid + (spread * 0.25)
                reason = f"High urgency: selling {spread * 0.25:.4f} inside spread"
            return price, reason

        if urgency == ExecutionUrgency.LOW:
            # Passive - join the queue
            if side == "buy":
                price = best_bid
                reason = "Low urgency: joining bid queue for maker rebates"
            else:
                price = best_ask
                reason = "Low urgency: joining ask queue for maker rebates"
            return price, reason

        # Medium urgency - place at mid or slightly aggressive
        if conditions.spread_percentile < 0.3:
            # Tight spread - be more aggressive
            if side == "buy":
                price = mid_price + (spread * 0.1)
                reason = "Tight spread: placing slightly above mid"
            else:
                price = mid_price - (spread * 0.1)
                reason = "Tight spread: placing slightly below mid"
        else:
            # Normal spread - place at mid
            price = mid_price
            reason = "Medium urgency: placing at mid price"

        return price, reason

    def estimate_market_impact(
        self,
        side: str,
        quantity: float,
        order_book: List[Tuple[float, float]],
        mid_price: float
    ) -> Dict:
        """
        Estimate market impact of executing quantity.

        Returns dict with:
        - temporary_impact_bps: immediate price impact
        - permanent_impact_bps: estimated permanent price shift
        - total_cost_bps: total execution cost in bps
        """
        if not order_book or quantity <= 0:
            return {
                "temporary_impact_bps": 0,
                "permanent_impact_bps": 0,
                "total_cost_bps": 0
            }

        # Walk the book to calculate fill price
        remaining = quantity
        total_cost = 0.0
        filled = 0.0

        for price, qty in order_book:
            fill_qty = min(remaining, qty)
            total_cost += fill_qty * price
            filled += fill_qty
            remaining -= fill_qty
            if remaining <= 0:
                break

        if filled == 0:
            return {
                "temporary_impact_bps": 1000,  # High impact for no fills
                "permanent_impact_bps": 100,
                "total_cost_bps": 1000
            }

        vwap = total_cost / filled

        # Temporary impact is the difference from mid
        if side == "buy":
            temp_impact_bps = ((vwap - mid_price) / mid_price) * 10000
        else:
            temp_impact_bps = ((mid_price - vwap) / mid_price) * 10000

        # Permanent impact is typically 30-50% of temporary
        perm_impact_bps = temp_impact_bps * 0.4

        # Total cost includes both
        total_cost_bps = temp_impact_bps + (perm_impact_bps * 0.5)  # Discount permanent slightly

        return {
            "temporary_impact_bps": max(0, temp_impact_bps),
            "permanent_impact_bps": max(0, perm_impact_bps),
            "total_cost_bps": max(0, total_cost_bps)
        }


# =============================================================================
# SPLIT ORDER EXECUTION
# =============================================================================

class SplitOrderExecutor:
    """
    Executes orders in multiple slices using various strategies.

    Strategies:
    - TWAP: Time-weighted splits at regular intervals
    - VWAP: Volume-weighted splits based on historical volume
    - Adaptive: Adjust slice size based on market conditions
    - Iceberg: Fixed visible size with hidden remainder
    """

    def __init__(self):
        # Default parameters
        self.min_slice_size = 0.01  # Minimum slice quantity
        self.max_slices = 20
        self.default_interval_ms = 500  # 500ms between slices

        # Historical volume profile (hour -> relative volume)
        self.volume_profile = {
            9: 1.5, 10: 1.3, 11: 1.1, 12: 0.8, 13: 0.9,
            14: 1.0, 15: 1.2, 16: 1.4, 17: 0.7, 18: 0.5
        }

    def create_execution_plan(
        self,
        side: str,
        quantity: float,
        price: float,
        conditions: MarketConditions,
        urgency: ExecutionUrgency,
        strategy: SplitStrategy = SplitStrategy.ADAPTIVE
    ) -> ExecutionPlan:
        """
        Create an execution plan for the order.
        """
        # Select order type
        type_selector = AdaptiveOrderTypeSelector()
        order_type, type_reasons = type_selector.select_order_type(
            side, quantity, conditions, urgency
        )

        plan = ExecutionPlan(
            order_type=order_type,
            urgency=urgency,
            split_strategy=strategy,
            total_quantity=quantity,
            price=price,
            reasoning=type_reasons
        )

        # Determine if splitting is needed
        available_depth = conditions.ask_depth if side == "buy" else conditions.bid_depth
        should_split = self._should_split(quantity, available_depth, urgency, strategy)

        if not should_split:
            plan.split_strategy = SplitStrategy.NONE
            plan.slices = [{
                "slice_id": 0,
                "quantity": quantity,
                "price": price,
                "delay_ms": 0
            }]
            plan.reasoning.append("Order small enough for single execution")
        else:
            # Create slices based on strategy
            if strategy == SplitStrategy.TWAP:
                plan.slices = self._create_twap_slices(quantity, price, urgency)
            elif strategy == SplitStrategy.VWAP:
                plan.slices = self._create_vwap_slices(quantity, price)
            elif strategy == SplitStrategy.ICEBERG:
                plan.slices = self._create_iceberg_slices(quantity, price, available_depth)
            else:  # ADAPTIVE
                plan.slices = self._create_adaptive_slices(
                    quantity, price, conditions, urgency
                )

        # Calculate estimates
        plan.estimated_duration_ms = sum(s.get("delay_ms", 0) for s in plan.slices)
        plan.estimated_fill_price = price
        plan.confidence = self._calculate_plan_confidence(plan, conditions)

        return plan

    def _should_split(
        self,
        quantity: float,
        available_depth: float,
        urgency: ExecutionUrgency,
        strategy: SplitStrategy
    ) -> bool:
        """Determine if order should be split"""
        if strategy == SplitStrategy.NONE:
            return False

        if urgency == ExecutionUrgency.CRITICAL:
            return False  # No time for splitting

        if available_depth <= 0:
            return False  # Can't determine

        # Split if order is >20% of available depth
        depth_ratio = quantity / available_depth
        return depth_ratio > 0.2

    def _create_twap_slices(
        self,
        quantity: float,
        price: float,
        urgency: ExecutionUrgency
    ) -> List[Dict]:
        """Create time-weighted slices"""
        # Determine number of slices based on urgency
        num_slices = {
            ExecutionUrgency.LOW: min(self.max_slices, 10),
            ExecutionUrgency.MEDIUM: min(self.max_slices, 5),
            ExecutionUrgency.HIGH: 3,
            ExecutionUrgency.CRITICAL: 1
        }.get(urgency, 5)

        # Determine interval based on urgency
        interval_ms = {
            ExecutionUrgency.LOW: 2000,
            ExecutionUrgency.MEDIUM: 1000,
            ExecutionUrgency.HIGH: 500,
            ExecutionUrgency.CRITICAL: 0
        }.get(urgency, 1000)

        slice_qty = quantity / num_slices
        if slice_qty < self.min_slice_size:
            num_slices = max(1, int(quantity / self.min_slice_size))
            slice_qty = quantity / num_slices

        slices = []
        for i in range(num_slices):
            slices.append({
                "slice_id": i,
                "quantity": slice_qty,
                "price": price,
                "delay_ms": i * interval_ms
            })

        return slices

    def _create_vwap_slices(
        self,
        quantity: float,
        price: float,
        duration_hours: float = 1.0
    ) -> List[Dict]:
        """Create volume-weighted slices based on historical volume profile"""
        current_hour = datetime.now().hour
        num_intervals = min(self.max_slices, int(duration_hours * 4))  # 15-min intervals

        # Get volume weights for upcoming intervals
        weights = []
        for i in range(num_intervals):
            hour = (current_hour + (i * 15 // 60)) % 24
            weight = self.volume_profile.get(hour, 1.0)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * num_intervals
            total_weight = num_intervals

        slices = []
        interval_ms = int((duration_hours * 3600 * 1000) / num_intervals)

        for i, weight in enumerate(weights):
            slice_qty = quantity * (weight / total_weight)
            if slice_qty >= self.min_slice_size:
                slices.append({
                    "slice_id": i,
                    "quantity": slice_qty,
                    "price": price,
                    "delay_ms": i * interval_ms,
                    "volume_weight": weight / total_weight
                })

        return slices

    def _create_iceberg_slices(
        self,
        quantity: float,
        price: float,
        available_depth: float
    ) -> List[Dict]:
        """Create iceberg slices with small visible portions"""
        # Visible portion should be ~5-10% of available depth
        visible_size = max(self.min_slice_size, available_depth * 0.05)

        num_slices = max(1, int(np.ceil(quantity / visible_size)))
        num_slices = min(num_slices, self.max_slices * 2)  # Allow more slices for iceberg

        slices = []
        remaining = quantity

        for i in range(num_slices):
            slice_qty = min(visible_size, remaining)
            if slice_qty >= self.min_slice_size:
                slices.append({
                    "slice_id": i,
                    "quantity": slice_qty,
                    "price": price,
                    "delay_ms": i * 100,  # Small delay between iceberg refreshes
                    "is_iceberg": True,
                    "visible_qty": slice_qty,
                    "hidden_qty": 0
                })
                remaining -= slice_qty

            if remaining <= 0:
                break

        return slices

    def _create_adaptive_slices(
        self,
        quantity: float,
        price: float,
        conditions: MarketConditions,
        urgency: ExecutionUrgency
    ) -> List[Dict]:
        """Create adaptive slices based on market conditions"""
        # Start with TWAP as base
        base_slices = self._create_twap_slices(quantity, price, urgency)

        # Adjust based on conditions
        if conditions.spread_percentile > 0.7:
            # Wide spread - use fewer, larger slices
            while len(base_slices) > 3 and base_slices:
                # Combine pairs of slices
                combined = []
                for i in range(0, len(base_slices), 2):
                    if i + 1 < len(base_slices):
                        combined.append({
                            "slice_id": len(combined),
                            "quantity": base_slices[i]["quantity"] + base_slices[i+1]["quantity"],
                            "price": price,
                            "delay_ms": base_slices[i]["delay_ms"]
                        })
                    else:
                        combined.append(base_slices[i])
                base_slices = combined

        elif conditions.spread_percentile < 0.3:
            # Tight spread - can be more aggressive, fewer slices
            pass  # Keep base slices

        # Adjust for toxicity
        if conditions.toxicity > 0.5:
            # High toxicity - increase delays
            for slice in base_slices:
                slice["delay_ms"] = int(slice["delay_ms"] * 1.5)

        # Adjust for volatility
        if conditions.volatility > 0.02:  # >2% volatility
            # High volatility - smaller slices
            new_slices = []
            for s in base_slices:
                # Split each slice in half
                half_qty = s["quantity"] / 2
                if half_qty >= self.min_slice_size:
                    new_slices.append({
                        "slice_id": len(new_slices),
                        "quantity": half_qty,
                        "price": price,
                        "delay_ms": s["delay_ms"]
                    })
                    new_slices.append({
                        "slice_id": len(new_slices),
                        "quantity": half_qty,
                        "price": price,
                        "delay_ms": s["delay_ms"] + 200
                    })
                else:
                    new_slices.append(s)
            base_slices = new_slices[:self.max_slices]

        return base_slices

    def _calculate_plan_confidence(
        self,
        plan: ExecutionPlan,
        conditions: MarketConditions
    ) -> float:
        """Calculate confidence in the execution plan"""
        confidence = 0.7  # Base confidence

        # Adjust for spread
        if conditions.spread_percentile < 0.3:
            confidence += 0.1
        elif conditions.spread_percentile > 0.7:
            confidence -= 0.15

        # Adjust for toxicity
        if conditions.toxicity > 0.5:
            confidence -= 0.2
        elif conditions.toxicity < 0.2:
            confidence += 0.05

        # Adjust for number of slices (more slices = lower risk but more complexity)
        if len(plan.slices) > 10:
            confidence -= 0.1
        elif len(plan.slices) <= 3:
            confidence += 0.05

        # Adjust for urgency match
        if plan.urgency == ExecutionUrgency.CRITICAL and len(plan.slices) > 1:
            confidence -= 0.1  # Critical but splitting is risky

        return max(0.1, min(0.95, confidence))


# =============================================================================
# INTEGRATED EXECUTION ENGINE
# =============================================================================

class AdvancedExecutionEngine:
    """
    Integrated execution engine combining all strategies.

    Provides:
    - Automatic strategy selection
    - Execution plan generation
    - Real-time execution with monitoring
    - Performance tracking
    """

    def __init__(self):
        self.type_selector = AdaptiveOrderTypeSelector()
        self.timing_optimizer = ExecutionTimingOptimizer()
        self.split_executor = SplitOrderExecutor()

        # Execution history
        self._execution_history: deque = deque(maxlen=1000)

        # Performance metrics
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_slippage_bps": 0.0,
            "avg_fill_rate": 0.0
        }

    def analyze_execution_options(
        self,
        side: str,
        quantity: float,
        target_price: float,
        order_book: List[Tuple[float, float]],
        conditions: MarketConditions,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> Dict:
        """
        Analyze all execution options and provide recommendations.
        """
        mid_price = order_book[0][0] if order_book else target_price
        best_bid = order_book[0][0] if side == "sell" and order_book else target_price
        best_ask = order_book[0][0] if side == "buy" and order_book else target_price

        # Get order type recommendation
        order_type, type_reasons = self.type_selector.select_order_type(
            side, quantity, conditions, urgency
        )

        # Get timing recommendation
        should_cross, cross_reason = self.timing_optimizer.should_cross_spread(
            side, conditions, urgency
        )

        optimal_price, price_reason = self.timing_optimizer.calculate_optimal_limit_price(
            side, mid_price, best_bid, best_ask, conditions, urgency
        )

        # Estimate market impact
        impact = self.timing_optimizer.estimate_market_impact(
            side, quantity, order_book, mid_price
        )

        # Get queue position estimate
        queue_info = self.timing_optimizer.estimate_queue_position(
            side, optimal_price, quantity, order_book
        )

        # Generate execution plans for different strategies
        plans = {}
        for strategy in [SplitStrategy.NONE, SplitStrategy.TWAP, SplitStrategy.ADAPTIVE]:
            plan = self.split_executor.create_execution_plan(
                side, quantity, optimal_price, conditions, urgency, strategy
            )
            plans[strategy.value] = plan.to_dict()

        return {
            "recommendation": {
                "order_type": order_type.value,
                "should_cross_spread": should_cross,
                "optimal_price": optimal_price,
                "split_strategy": SplitStrategy.ADAPTIVE.value if quantity > conditions.ask_depth * 0.2 else SplitStrategy.NONE.value
            },
            "reasoning": {
                "order_type": type_reasons,
                "spread_crossing": cross_reason,
                "price": price_reason
            },
            "impact_analysis": impact,
            "queue_analysis": queue_info,
            "execution_plans": plans,
            "market_conditions": {
                "spread_bps": conditions.spread_bps,
                "spread_percentile": conditions.spread_percentile,
                "toxicity": conditions.toxicity,
                "is_favorable": conditions.is_favorable
            }
        }

    def create_optimal_plan(
        self,
        side: str,
        quantity: float,
        target_price: float,
        conditions: MarketConditions,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> ExecutionPlan:
        """
        Create the optimal execution plan for the order.
        """
        # Determine if splitting is beneficial
        available_depth = conditions.ask_depth if side == "buy" else conditions.bid_depth
        depth_ratio = quantity / available_depth if available_depth > 0 else 1.0

        # Select split strategy
        if urgency == ExecutionUrgency.CRITICAL:
            strategy = SplitStrategy.NONE
        elif depth_ratio > 0.5:
            strategy = SplitStrategy.ICEBERG
        elif depth_ratio > 0.2:
            strategy = SplitStrategy.ADAPTIVE
        elif urgency == ExecutionUrgency.LOW and depth_ratio > 0.1:
            strategy = SplitStrategy.TWAP
        else:
            strategy = SplitStrategy.NONE

        return self.split_executor.create_execution_plan(
            side, quantity, target_price, conditions, urgency, strategy
        )

    def record_execution(self, result: ExecutionResult):
        """Record execution result for performance tracking"""
        self._execution_history.append(result)

        self._metrics["total_executions"] += 1
        if result.success:
            self._metrics["successful_executions"] += 1

        self._metrics["total_slippage_bps"] += abs(result.slippage_bps)

        # Update average fill rate
        fill_rate = result.total_filled / result.plan.total_quantity if result.plan.total_quantity > 0 else 0
        n = self._metrics["total_executions"]
        self._metrics["avg_fill_rate"] = (
            (self._metrics["avg_fill_rate"] * (n - 1) + fill_rate) / n
        )

    def get_performance_metrics(self) -> Dict:
        """Get execution performance metrics"""
        n = self._metrics["total_executions"]
        return {
            "total_executions": n,
            "success_rate": self._metrics["successful_executions"] / n * 100 if n > 0 else 0,
            "avg_slippage_bps": self._metrics["total_slippage_bps"] / n if n > 0 else 0,
            "avg_fill_rate": self._metrics["avg_fill_rate"] * 100
        }


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_execution_strategies():
    """Test execution strategies"""
    print("=" * 70)
    print("ADVANCED EXECUTION STRATEGIES TEST")
    print("=" * 70)

    # Create test conditions
    conditions = MarketConditions(
        spread_bps=15.0,
        spread_percentile=0.4,
        bid_depth=100.0,
        ask_depth=80.0,
        volatility=0.015,
        toxicity=0.3,
        recent_volume=500.0,
        price_trend=0.001
    )

    # Test order book
    order_book = [
        (100.00, 20.0),
        (100.05, 30.0),
        (100.10, 25.0),
        (100.15, 40.0),
        (100.20, 50.0),
    ]

    engine = AdvancedExecutionEngine()

    print("\n1. ADAPTIVE ORDER TYPE SELECTION")
    print("-" * 50)

    for urgency in ExecutionUrgency:
        order_type, reasons = engine.type_selector.select_order_type(
            "buy", 10.0, conditions, urgency
        )
        print(f"\n  Urgency: {urgency.value}")
        print(f"  Selected: {order_type.value}")
        for reason in reasons:
            print(f"    - {reason}")

    print("\n2. EXECUTION TIMING ANALYSIS")
    print("-" * 50)

    # Queue position
    queue = engine.timing_optimizer.estimate_queue_position(
        "buy", 100.05, 10.0, order_book
    )
    print(f"\n  Queue position at $100.05:")
    print(f"    Position: {queue['position']}")
    print(f"    Ahead quantity: {queue['ahead_quantity']:.2f}")
    print(f"    Est. wait: {queue['estimated_wait_time_ms']:.0f}ms")
    print(f"    Fill probability: {queue['fill_probability']:.1%}")

    # Market impact
    impact = engine.timing_optimizer.estimate_market_impact(
        "buy", 50.0, order_book, 100.02
    )
    print(f"\n  Market impact for buying 50 units:")
    print(f"    Temporary impact: {impact['temporary_impact_bps']:.2f}bps")
    print(f"    Permanent impact: {impact['permanent_impact_bps']:.2f}bps")
    print(f"    Total cost: {impact['total_cost_bps']:.2f}bps")

    print("\n3. SPLIT ORDER STRATEGIES")
    print("-" * 50)

    for strategy in [SplitStrategy.TWAP, SplitStrategy.ADAPTIVE, SplitStrategy.ICEBERG]:
        plan = engine.split_executor.create_execution_plan(
            "buy", 50.0, 100.05, conditions, ExecutionUrgency.MEDIUM, strategy
        )
        print(f"\n  Strategy: {strategy.value}")
        print(f"    Order type: {plan.order_type.value}")
        print(f"    Num slices: {len(plan.slices)}")
        print(f"    Est. duration: {plan.estimated_duration_ms:.0f}ms")
        print(f"    Confidence: {plan.confidence:.1%}")
        if plan.slices:
            print(f"    First slice: {plan.slices[0]}")

    print("\n4. FULL EXECUTION ANALYSIS")
    print("-" * 50)

    analysis = engine.analyze_execution_options(
        side="buy",
        quantity=50.0,
        target_price=100.05,
        order_book=order_book,
        conditions=conditions,
        urgency=ExecutionUrgency.MEDIUM
    )

    print(f"\n  Recommendation:")
    print(f"    Order type: {analysis['recommendation']['order_type']}")
    print(f"    Cross spread: {analysis['recommendation']['should_cross_spread']}")
    print(f"    Optimal price: ${analysis['recommendation']['optimal_price']:.2f}")
    print(f"    Split strategy: {analysis['recommendation']['split_strategy']}")

    print(f"\n  Market Conditions:")
    print(f"    Spread: {analysis['market_conditions']['spread_bps']:.1f}bps")
    print(f"    Favorable: {analysis['market_conditions']['is_favorable']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_execution_strategies()
