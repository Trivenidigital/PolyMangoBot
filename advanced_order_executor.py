"""
Advanced Order Executor
Enterprise-grade atomic execution with:
- Sophisticated rollback mechanisms
- Partial fill handling
- Execution monitoring and alerting
- State machine for order lifecycle
- Audit trail and compliance logging
- Smart order routing
"""

import asyncio
import uuid
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

from exceptions import (
    OrderExecutionError, AtomicExecutionError,
    APITimeoutError, APIError
)
from utils import retry_async, CircuitBreaker

logger = logging.getLogger("PolyMangoBot.advanced_executor")


class ExecutionPhase(Enum):
    """Phases of atomic execution"""
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    PLACING_ORDERS = "placing_orders"
    MONITORING = "monitoring"
    VERIFYING_FILLS = "verifying_fills"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class OrderState(Enum):
    """Individual order states"""
    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class RollbackReason(Enum):
    """Reasons for rollback"""
    PARTIAL_PLACEMENT = "partial_placement"
    PARTIAL_FILL = "partial_fill"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    VALIDATION_FAILED = "validation_failed"
    USER_CANCELLED = "user_cancelled"
    API_ERROR = "api_error"
    PRICE_MOVED = "price_moved"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"


@dataclass
class OrderRecord:
    """Detailed record of an individual order"""
    order_id: str
    client_order_id: str
    venue: str
    market: str
    side: str  # "buy" or "sell"
    order_type: str
    quantity: float
    price: float
    state: OrderState = OrderState.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    created_at: float = field(default_factory=time.time)
    placed_at: Optional[float] = None
    filled_at: Optional[float] = None
    cancelled_at: Optional[float] = None
    error_message: Optional[str] = None
    api_responses: List[Dict] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        """Percentage filled"""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.state in [
            OrderState.FILLED, OrderState.CANCELLED,
            OrderState.REJECTED, OrderState.EXPIRED, OrderState.FAILED
        ]

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "venue": self.venue,
            "market": self.market,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "state": self.state.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "fees": self.fees,
            "fill_rate": self.fill_rate,
            "created_at": self.created_at,
            "placed_at": self.placed_at,
            "filled_at": self.filled_at,
            "cancelled_at": self.cancelled_at,
            "error_message": self.error_message
        }


@dataclass
class AtomicTradeExecution:
    """Complete record of an atomic trade execution"""
    execution_id: str
    market: str
    phase: ExecutionPhase = ExecutionPhase.INITIALIZING
    buy_order: Optional[OrderRecord] = None
    sell_order: Optional[OrderRecord] = None

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Results
    success: bool = False
    expected_profit: float = 0.0
    realized_profit: float = 0.0
    total_fees: float = 0.0
    slippage: float = 0.0

    # Rollback info
    rollback_performed: bool = False
    rollback_reason: Optional[RollbackReason] = None
    rollback_success: bool = False

    # Audit trail
    events: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_event(self, event_type: str, details: Dict = None):
        """Add event to audit trail"""
        self.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "phase": self.phase.value,
            "details": details or {}
        })

    @property
    def execution_time_ms(self) -> float:
        """Total execution time in milliseconds"""
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.phase in [ExecutionPhase.COMPLETED, ExecutionPhase.FAILED]

    def to_dict(self) -> Dict:
        return {
            "execution_id": self.execution_id,
            "market": self.market,
            "phase": self.phase.value,
            "buy_order": self.buy_order.to_dict() if self.buy_order else None,
            "sell_order": self.sell_order.to_dict() if self.sell_order else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "expected_profit": self.expected_profit,
            "realized_profit": self.realized_profit,
            "total_fees": self.total_fees,
            "slippage": self.slippage,
            "rollback_performed": self.rollback_performed,
            "rollback_reason": self.rollback_reason.value if self.rollback_reason else None,
            "rollback_success": self.rollback_success,
            "events": self.events,
            "errors": self.errors
        }


class ExecutionMonitor:
    """Monitors execution metrics and alerts"""

    def __init__(self):
        self._executions: deque = deque(maxlen=1000)
        self._alerts: List[Dict] = []

        # Thresholds
        self.max_execution_time_ms = 5000
        self.max_slippage_pct = 0.5
        self.min_success_rate = 0.8
        self.alert_cooldown_seconds = 300

        self._last_alert_time: Dict[str, float] = {}

    def record_execution(self, execution: AtomicTradeExecution):
        """Record an execution for monitoring"""
        self._executions.append(execution)
        self._check_alerts(execution)

    def _check_alerts(self, execution: AtomicTradeExecution):
        """Check if execution triggers any alerts"""
        current_time = time.time()

        # Slow execution alert
        if execution.execution_time_ms > self.max_execution_time_ms:
            self._maybe_alert(
                "slow_execution",
                f"Execution {execution.execution_id} took {execution.execution_time_ms:.0f}ms",
                {"execution_id": execution.execution_id, "time_ms": execution.execution_time_ms}
            )

        # High slippage alert
        if abs(execution.slippage) > self.max_slippage_pct:
            self._maybe_alert(
                "high_slippage",
                f"Execution {execution.execution_id} had {execution.slippage:.2f}% slippage",
                {"execution_id": execution.execution_id, "slippage": execution.slippage}
            )

        # Failure alert
        if not execution.success:
            self._maybe_alert(
                "execution_failed",
                f"Execution {execution.execution_id} failed: {execution.rollback_reason}",
                {"execution_id": execution.execution_id, "reason": str(execution.rollback_reason)}
            )

    def _maybe_alert(self, alert_type: str, message: str, details: Dict):
        """Send alert if not in cooldown"""
        current_time = time.time()
        last_alert = self._last_alert_time.get(alert_type, 0)

        if current_time - last_alert < self.alert_cooldown_seconds:
            return

        self._last_alert_time[alert_type] = current_time
        alert = {
            "timestamp": current_time,
            "type": alert_type,
            "message": message,
            "details": details
        }
        self._alerts.append(alert)
        logger.warning(f"ALERT [{alert_type}]: {message}")

    def get_stats(self) -> Dict:
        """Get execution statistics"""
        if not self._executions:
            return {"total": 0}

        successful = [e for e in self._executions if e.success]
        failed = [e for e in self._executions if not e.success]

        avg_time = sum(e.execution_time_ms for e in self._executions) / len(self._executions)
        avg_profit = sum(e.realized_profit for e in successful) / len(successful) if successful else 0
        avg_slippage = sum(abs(e.slippage) for e in self._executions) / len(self._executions)

        return {
            "total": len(self._executions),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self._executions) * 100,
            "avg_execution_time_ms": avg_time,
            "avg_profit": avg_profit,
            "avg_slippage_pct": avg_slippage,
            "total_profit": sum(e.realized_profit for e in successful),
            "total_fees": sum(e.total_fees for e in self._executions),
            "recent_alerts": self._alerts[-10:],
            "rollback_reasons": self._get_rollback_stats(failed)
        }

    def _get_rollback_stats(self, failed: List[AtomicTradeExecution]) -> Dict[str, int]:
        """Get breakdown of rollback reasons"""
        reasons = {}
        for e in failed:
            if e.rollback_reason:
                reason = e.rollback_reason.value
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons


class SmartOrderRouter:
    """Intelligent order routing based on venue performance"""

    def __init__(self):
        self._venue_stats: Dict[str, Dict] = {}
        self._latency_history: Dict[str, deque] = {}

    def record_venue_execution(
        self,
        venue: str,
        success: bool,
        latency_ms: float,
        slippage: float
    ):
        """Record venue execution performance"""
        if venue not in self._venue_stats:
            self._venue_stats[venue] = {
                "total": 0,
                "successful": 0,
                "total_latency_ms": 0,
                "total_slippage": 0
            }
            self._latency_history[venue] = deque(maxlen=100)

        stats = self._venue_stats[venue]
        stats["total"] += 1
        if success:
            stats["successful"] += 1
        stats["total_latency_ms"] += latency_ms
        stats["total_slippage"] += abs(slippage)

        self._latency_history[venue].append(latency_ms)

    def get_venue_score(self, venue: str) -> float:
        """Get venue performance score (0-100)"""
        if venue not in self._venue_stats:
            return 50.0  # Default score for unknown venues

        stats = self._venue_stats[venue]
        if stats["total"] == 0:
            return 50.0

        # Components
        success_rate = stats["successful"] / stats["total"]
        avg_latency = stats["total_latency_ms"] / stats["total"]
        avg_slippage = stats["total_slippage"] / stats["total"]

        # Normalize (lower latency and slippage is better)
        latency_score = max(0, 100 - (avg_latency / 50))  # 50ms baseline
        slippage_score = max(0, 100 - (avg_slippage * 100))  # 1% = 0 score

        # Weighted combination
        score = (
            success_rate * 100 * 0.4 +
            latency_score * 0.3 +
            slippage_score * 0.3
        )

        return min(100, max(0, score))

    def recommend_venue_order(self, venues: List[str]) -> List[str]:
        """Recommend order of venues for execution (best first)"""
        scored = [(v, self.get_venue_score(v)) for v in venues]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in scored]


class AdvancedOrderExecutor:
    """
    Advanced order executor with sophisticated rollback and monitoring.

    Features:
    - State machine-based execution flow
    - Partial fill handling
    - Smart rollback decisions
    - Execution monitoring and alerts
    - Audit trail for compliance
    - Smart order routing
    """

    def __init__(
        self,
        api_manager,
        max_concurrent_executions: int = 1,
        default_timeout_ms: int = 10000
    ):
        self.api_manager = api_manager
        self.max_concurrent = max_concurrent_executions
        self.default_timeout_ms = default_timeout_ms

        # Execution control
        self._semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._lock = asyncio.Lock()

        # Components
        self._monitor = ExecutionMonitor()
        self._router = SmartOrderRouter()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # History
        self._executions: List[AtomicTradeExecution] = []
        self._active_executions: Dict[str, AtomicTradeExecution] = {}

        # Configuration
        self.min_fill_rate_for_success = 0.95  # 95% fill = success
        self.partial_fill_threshold = 0.5  # 50% fill = keep, less = rollback
        self.price_tolerance_pct = 0.1  # 0.1% price deviation allowed

    def _get_circuit_breaker(self, venue: str) -> CircuitBreaker:
        """Get or create circuit breaker for venue"""
        if venue not in self._circuit_breakers:
            self._circuit_breakers[venue] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout_seconds=60.0
            )
        return self._circuit_breakers[venue]

    def _get_api(self, venue: str):
        """Get API for venue"""
        if venue.lower() == "polymarket":
            return self.api_manager.polymarket
        elif venue.lower() in self.api_manager.additional_exchanges:
            return self.api_manager.additional_exchanges[venue.lower()]
        else:
            return self.api_manager.exchange

    async def execute_atomic_trade(
        self,
        market: str,
        buy_venue: str,
        buy_price: float,
        buy_quantity: float,
        sell_venue: str,
        sell_price: float,
        sell_quantity: float,
        timeout_ms: Optional[int] = None,
        dry_run: bool = False
    ) -> AtomicTradeExecution:
        """
        Execute an atomic trade with comprehensive handling.

        Returns:
            AtomicTradeExecution with full details of the execution
        """
        execution_id = str(uuid.uuid4())[:8]
        timeout = (timeout_ms or self.default_timeout_ms) / 1000

        execution = AtomicTradeExecution(
            execution_id=execution_id,
            market=market,
            expected_profit=(sell_price - buy_price) * min(buy_quantity, sell_quantity)
        )

        execution.add_event("execution_started", {
            "buy_venue": buy_venue,
            "sell_venue": sell_venue,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_quantity": buy_quantity,
            "sell_quantity": sell_quantity
        })

        logger.info(f"[{execution_id}] Starting atomic trade: {market}")

        async with self._semaphore:
            self._active_executions[execution_id] = execution

            try:
                # Phase 1: Validation
                execution.phase = ExecutionPhase.VALIDATING
                execution.add_event("validation_started")

                validation_ok = await self._validate_execution(
                    execution, buy_venue, sell_venue
                )

                if not validation_ok:
                    execution.phase = ExecutionPhase.FAILED
                    execution.add_event("validation_failed")
                    return execution

                # Create order records
                execution.buy_order = OrderRecord(
                    order_id="",
                    client_order_id=f"{execution_id}_buy",
                    venue=buy_venue,
                    market=market,
                    side="buy",
                    order_type="limit",
                    quantity=buy_quantity,
                    price=buy_price
                )

                execution.sell_order = OrderRecord(
                    order_id="",
                    client_order_id=f"{execution_id}_sell",
                    venue=sell_venue,
                    market=market,
                    side="sell",
                    order_type="limit",
                    quantity=sell_quantity,
                    price=sell_price
                )

                # Phase 2: Place orders
                execution.phase = ExecutionPhase.PLACING_ORDERS
                execution.add_event("placing_orders")

                if dry_run:
                    await self._simulate_placement(execution)
                else:
                    placement_success = await self._place_orders(execution, timeout)

                    if not placement_success:
                        await self._handle_rollback(
                            execution,
                            RollbackReason.PARTIAL_PLACEMENT
                        )
                        return execution

                # Phase 3: Monitor fills
                execution.phase = ExecutionPhase.MONITORING
                execution.add_event("monitoring_fills")

                if not dry_run:
                    await self._monitor_fills(execution, timeout)

                # Phase 4: Verify and complete
                execution.phase = ExecutionPhase.VERIFYING_FILLS
                execution.add_event("verifying_fills")

                await self._verify_and_complete(execution)

                return execution

            except asyncio.TimeoutError:
                logger.error(f"[{execution_id}] Execution timeout")
                await self._handle_rollback(execution, RollbackReason.TIMEOUT)
                return execution

            except Exception as e:
                logger.error(f"[{execution_id}] Execution error: {e}")
                execution.errors.append(str(e))
                await self._handle_rollback(execution, RollbackReason.API_ERROR)
                return execution

            finally:
                execution.completed_at = time.time()
                self._active_executions.pop(execution_id, None)

                async with self._lock:
                    self._executions.append(execution)

                self._monitor.record_execution(execution)

                # Record venue performance
                if execution.buy_order:
                    self._router.record_venue_execution(
                        buy_venue,
                        execution.buy_order.state == OrderState.FILLED,
                        execution.execution_time_ms / 2,
                        0  # TODO: calculate actual slippage
                    )

                if execution.sell_order:
                    self._router.record_venue_execution(
                        sell_venue,
                        execution.sell_order.state == OrderState.FILLED,
                        execution.execution_time_ms / 2,
                        0
                    )

    async def _validate_execution(
        self,
        execution: AtomicTradeExecution,
        buy_venue: str,
        sell_venue: str
    ) -> bool:
        """Validate execution prerequisites"""
        # Check circuit breakers
        buy_cb = self._get_circuit_breaker(buy_venue)
        sell_cb = self._get_circuit_breaker(sell_venue)

        if buy_cb.is_open:
            execution.errors.append(f"Circuit breaker open for {buy_venue}")
            execution.rollback_reason = RollbackReason.CIRCUIT_BREAKER
            return False

        if sell_cb.is_open:
            execution.errors.append(f"Circuit breaker open for {sell_venue}")
            execution.rollback_reason = RollbackReason.CIRCUIT_BREAKER
            return False

        # TODO: Add balance checks, rate limit checks, etc.

        return True

    async def _place_orders(
        self,
        execution: AtomicTradeExecution,
        timeout: float
    ) -> bool:
        """Place both orders simultaneously"""
        buy_order = execution.buy_order
        sell_order = execution.sell_order

        buy_request = {
            "market": buy_order.market,
            "side": buy_order.side,
            "quantity": buy_order.quantity,
            "price": buy_order.price,
            "order_type": buy_order.order_type,
            "client_order_id": buy_order.client_order_id
        }

        sell_request = {
            "market": sell_order.market,
            "side": sell_order.side,
            "quantity": sell_order.quantity,
            "price": sell_order.price,
            "order_type": sell_order.order_type,
            "client_order_id": sell_order.client_order_id
        }

        # Place both orders simultaneously
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    self._place_single_order(buy_order.venue, buy_request),
                    self._place_single_order(sell_order.venue, sell_request),
                    return_exceptions=True
                ),
                timeout=timeout
            )

            buy_result, sell_result = results

            # Process buy result
            if isinstance(buy_result, Exception):
                buy_order.state = OrderState.FAILED
                buy_order.error_message = str(buy_result)
                execution.errors.append(f"Buy order failed: {buy_result}")
            elif buy_result and buy_result.get("order_id"):
                buy_order.order_id = buy_result["order_id"]
                buy_order.state = OrderState.PLACED
                buy_order.placed_at = time.time()
                buy_order.api_responses.append(buy_result)
            else:
                buy_order.state = OrderState.REJECTED
                execution.errors.append("Buy order rejected")

            # Process sell result
            if isinstance(sell_result, Exception):
                sell_order.state = OrderState.FAILED
                sell_order.error_message = str(sell_result)
                execution.errors.append(f"Sell order failed: {sell_result}")
            elif sell_result and sell_result.get("order_id"):
                sell_order.order_id = sell_result["order_id"]
                sell_order.state = OrderState.PLACED
                sell_order.placed_at = time.time()
                sell_order.api_responses.append(sell_result)
            else:
                sell_order.state = OrderState.REJECTED
                execution.errors.append("Sell order rejected")

            # Check if both succeeded
            buy_ok = buy_order.state == OrderState.PLACED
            sell_ok = sell_order.state == OrderState.PLACED

            execution.add_event("orders_placed", {
                "buy_ok": buy_ok,
                "buy_order_id": buy_order.order_id,
                "sell_ok": sell_ok,
                "sell_order_id": sell_order.order_id
            })

            return buy_ok and sell_ok

        except asyncio.TimeoutError:
            execution.errors.append("Order placement timeout")
            return False

    async def _place_single_order(self, venue: str, order: Dict) -> Dict:
        """Place a single order with circuit breaker protection"""
        api = self._get_api(venue)
        cb = self._get_circuit_breaker(venue)

        async def _do_place():
            return await api.place_order(order)

        return await cb.call(_do_place)

    async def _simulate_placement(self, execution: AtomicTradeExecution):
        """Simulate order placement for dry run"""
        execution.buy_order.order_id = f"sim_{execution.execution_id}_buy"
        execution.buy_order.state = OrderState.FILLED
        execution.buy_order.filled_quantity = execution.buy_order.quantity
        execution.buy_order.avg_fill_price = execution.buy_order.price

        execution.sell_order.order_id = f"sim_{execution.execution_id}_sell"
        execution.sell_order.state = OrderState.FILLED
        execution.sell_order.filled_quantity = execution.sell_order.quantity
        execution.sell_order.avg_fill_price = execution.sell_order.price

        execution.add_event("dry_run_simulated")

    async def _monitor_fills(
        self,
        execution: AtomicTradeExecution,
        timeout: float
    ):
        """Monitor orders for fills"""
        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            # Check both orders
            buy_filled = await self._check_order_status(execution.buy_order)
            sell_filled = await self._check_order_status(execution.sell_order)

            # Update states
            if execution.buy_order.filled_quantity >= execution.buy_order.quantity * self.min_fill_rate_for_success:
                execution.buy_order.state = OrderState.FILLED
                execution.buy_order.filled_at = time.time()

            if execution.sell_order.filled_quantity >= execution.sell_order.quantity * self.min_fill_rate_for_success:
                execution.sell_order.state = OrderState.FILLED
                execution.sell_order.filled_at = time.time()

            # Check if both filled
            if (execution.buy_order.state == OrderState.FILLED and
                execution.sell_order.state == OrderState.FILLED):
                execution.add_event("both_orders_filled")
                return

            await asyncio.sleep(check_interval)

        # Timeout - check partial fills
        execution.add_event("fill_monitoring_timeout", {
            "buy_fill_rate": execution.buy_order.fill_rate,
            "sell_fill_rate": execution.sell_order.fill_rate
        })

    async def _check_order_status(self, order: OrderRecord) -> bool:
        """Check order status and update record"""
        if order.is_terminal or not order.order_id:
            return order.state == OrderState.FILLED

        try:
            api = self._get_api(order.venue)
            status = await asyncio.wait_for(
                api.get_order(order.order_id),
                timeout=5.0
            )

            if status:
                order.api_responses.append(status)

                # Update fill info
                if "filled_quantity" in status:
                    order.filled_quantity = float(status["filled_quantity"])
                if "avg_price" in status:
                    order.avg_fill_price = float(status["avg_price"])
                if "fees" in status:
                    order.fees = float(status["fees"])

                # Update state
                api_status = status.get("status", "").lower()
                if api_status in ["filled", "complete"]:
                    order.state = OrderState.FILLED
                elif api_status in ["partially_filled", "partial"]:
                    order.state = OrderState.PARTIALLY_FILLED
                elif api_status in ["cancelled", "canceled"]:
                    order.state = OrderState.CANCELLED
                elif api_status in ["rejected", "failed"]:
                    order.state = OrderState.FAILED

            return order.state == OrderState.FILLED

        except Exception as e:
            logger.warning(f"Failed to check order {order.order_id}: {e}")
            return False

    async def _verify_and_complete(self, execution: AtomicTradeExecution):
        """Verify fills and complete execution"""
        buy_order = execution.buy_order
        sell_order = execution.sell_order

        buy_filled = buy_order.state == OrderState.FILLED
        sell_filled = sell_order.state == OrderState.FILLED

        if buy_filled and sell_filled:
            # Success!
            execution.phase = ExecutionPhase.COMPLETED
            execution.success = True

            # Calculate realized profit
            buy_cost = buy_order.filled_quantity * buy_order.avg_fill_price
            sell_revenue = sell_order.filled_quantity * sell_order.avg_fill_price
            execution.total_fees = buy_order.fees + sell_order.fees
            execution.realized_profit = sell_revenue - buy_cost - execution.total_fees

            # Calculate slippage
            expected_cost = buy_order.quantity * buy_order.price
            expected_revenue = sell_order.quantity * sell_order.price
            if expected_revenue - expected_cost != 0:
                execution.slippage = (
                    (execution.realized_profit - execution.expected_profit) /
                    abs(execution.expected_profit) * 100
                ) if execution.expected_profit != 0 else 0

            execution.add_event("execution_completed", {
                "realized_profit": execution.realized_profit,
                "slippage": execution.slippage
            })

            logger.info(
                f"[{execution.execution_id}] SUCCESS - "
                f"Profit: ${execution.realized_profit:.2f}, "
                f"Slippage: {execution.slippage:.2f}%"
            )

        else:
            # Partial fill - decide what to do
            buy_rate = buy_order.fill_rate
            sell_rate = sell_order.fill_rate

            if buy_rate < self.partial_fill_threshold or sell_rate < self.partial_fill_threshold:
                # Low fill rate - rollback
                await self._handle_rollback(execution, RollbackReason.PARTIAL_FILL)
            else:
                # Acceptable partial fill - complete with what we have
                execution.phase = ExecutionPhase.COMPLETED
                execution.success = True  # Partial success

                # Calculate with partial fills
                buy_cost = buy_order.filled_quantity * buy_order.avg_fill_price
                sell_revenue = sell_order.filled_quantity * sell_order.avg_fill_price
                execution.total_fees = buy_order.fees + sell_order.fees
                execution.realized_profit = sell_revenue - buy_cost - execution.total_fees

                execution.add_event("partial_fill_accepted", {
                    "buy_fill_rate": buy_rate,
                    "sell_fill_rate": sell_rate,
                    "realized_profit": execution.realized_profit
                })

                logger.info(
                    f"[{execution.execution_id}] PARTIAL SUCCESS - "
                    f"Buy: {buy_rate:.1f}%, Sell: {sell_rate:.1f}%, "
                    f"Profit: ${execution.realized_profit:.2f}"
                )

    async def _handle_rollback(
        self,
        execution: AtomicTradeExecution,
        reason: RollbackReason
    ):
        """Handle rollback of failed execution"""
        execution.phase = ExecutionPhase.ROLLING_BACK
        execution.rollback_performed = True
        execution.rollback_reason = reason

        execution.add_event("rollback_started", {"reason": reason.value})

        logger.warning(
            f"[{execution.execution_id}] Rolling back - Reason: {reason.value}"
        )

        # Cancel any active orders
        cancel_tasks = []

        if execution.buy_order and not execution.buy_order.is_terminal:
            cancel_tasks.append(
                self._cancel_order(execution.buy_order)
            )

        if execution.sell_order and not execution.sell_order.is_terminal:
            cancel_tasks.append(
                self._cancel_order(execution.sell_order)
            )

        if cancel_tasks:
            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
            execution.rollback_success = all(
                not isinstance(r, Exception) and r
                for r in results
            )
        else:
            execution.rollback_success = True

        execution.phase = ExecutionPhase.FAILED
        execution.success = False

        execution.add_event("rollback_completed", {
            "success": execution.rollback_success
        })

        logger.info(
            f"[{execution.execution_id}] Rollback {'succeeded' if execution.rollback_success else 'FAILED'}"
        )

    async def _cancel_order(self, order: OrderRecord) -> bool:
        """Cancel an individual order"""
        if not order.order_id:
            return True

        order.state = OrderState.CANCELLING

        try:
            api = self._get_api(order.venue)
            result = await asyncio.wait_for(
                api.cancel_order(order.order_id),
                timeout=10.0
            )

            order.state = OrderState.CANCELLED
            order.cancelled_at = time.time()

            logger.info(f"Cancelled order {order.order_id} on {order.venue}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            order.error_message = str(e)
            return False

    # Public API

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return self._monitor.get_stats()

    def get_venue_scores(self) -> Dict[str, float]:
        """Get performance scores for all venues"""
        venues = set()
        for e in self._executions:
            if e.buy_order:
                venues.add(e.buy_order.venue)
            if e.sell_order:
                venues.add(e.sell_order.venue)

        return {v: self._router.get_venue_score(v) for v in venues}

    def get_active_executions(self) -> List[Dict]:
        """Get currently active executions"""
        return [e.to_dict() for e in self._active_executions.values()]

    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Get recent execution history"""
        return [e.to_dict() for e in self._executions[-limit:]]

    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get circuit breaker status for all venues"""
        return {
            venue: cb.state
            for venue, cb in self._circuit_breakers.items()
        }


# Test function
async def test_advanced_executor():
    """Test the advanced order executor"""
    from api_connectors import APIManager

    manager = APIManager()
    await manager.connect_all()

    executor = AdvancedOrderExecutor(manager)

    print("Testing advanced order executor...\n")

    # Execute a test trade
    execution = await executor.execute_atomic_trade(
        market="BTC",
        buy_venue="kraken",
        buy_price=42500,
        buy_quantity=0.1,
        sell_venue="polymarket",
        sell_price=42700,
        sell_quantity=0.1,
        dry_run=True
    )

    print(f"\nExecution Result:")
    print(f"  ID: {execution.execution_id}")
    print(f"  Phase: {execution.phase.value}")
    print(f"  Success: {execution.success}")
    print(f"  Profit: ${execution.realized_profit:.2f}")
    print(f"  Execution time: {execution.execution_time_ms:.1f}ms")

    print(f"\nExecution Stats:")
    stats = executor.get_execution_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(test_advanced_executor())
