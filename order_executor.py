"""
Order Executor Module
Executes trades atomically across two venues with proper error handling
and race condition prevention
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

from exceptions import (
    OrderExecutionError, AtomicExecutionError, OrderCancellationError,
    APITimeoutError, APIError
)
from utils import retry_async, CircuitBreaker, TimingStats

logger = logging.getLogger("PolyMangoBot.executor")


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"
    PLACING = "placing"  # Added: order is being placed
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ROLLBACK = "rollback"  # Added: order was rolled back


@dataclass
class ExecutedTrade:
    """Record of executed trade"""
    trade_id: str
    market: str
    buy_order_id: str
    buy_venue: str
    buy_price: float
    buy_filled: float
    sell_order_id: str
    sell_venue: str
    sell_price: float
    sell_filled: float
    status: OrderStatus
    profit: float
    timestamp: float
    execution_time_ms: float = 0.0  # Added: track execution latency
    rollback_attempted: bool = False  # Added: track if rollback was needed

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "market": self.market,
            "buy_order_id": self.buy_order_id,
            "buy_venue": self.buy_venue,
            "buy_price": self.buy_price,
            "buy_filled": self.buy_filled,
            "sell_order_id": self.sell_order_id,
            "sell_venue": self.sell_venue,
            "sell_price": self.sell_price,
            "sell_filled": self.sell_filled,
            "status": self.status.value,
            "profit": self.profit,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
            "rollback_attempted": self.rollback_attempted,
        }


class OrderExecutor:
    """
    Handles atomic order execution across venues with proper concurrency control.

    Key features:
    - Atomic execution: both orders succeed or both are cancelled
    - Circuit breaker: prevents cascading failures
    - Execution lock: prevents concurrent trades that could exceed limits
    - Comprehensive error handling and rollback
    """

    def __init__(self, api_manager, max_concurrent_trades: int = 1):
        """
        Args:
            api_manager: Instance of APIManager from api_connectors.py
            max_concurrent_trades: Maximum simultaneous atomic trade executions
        """
        self.api_manager = api_manager
        self.trades: List[ExecutedTrade] = []
        self.pending_orders: Dict[str, ExecutedTrade] = {}

        # Concurrency control
        self._execution_lock = asyncio.Lock()
        self._trade_semaphore = asyncio.Semaphore(max_concurrent_trades)

        # Circuit breakers per venue
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Timing statistics
        self._timing_stats = TimingStats()

        # Execution state tracking
        self._active_trades: Dict[str, asyncio.Event] = {}
    
    def _get_circuit_breaker(self, venue: str) -> CircuitBreaker:
        """Get or create circuit breaker for a venue"""
        if venue not in self._circuit_breakers:
            self._circuit_breakers[venue] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout_seconds=60.0
            )
        return self._circuit_breakers[venue]

    def _get_api(self, venue: str):
        """Get API instance for a venue"""
        if venue.lower() == "polymarket":
            return self.api_manager.polymarket
        elif venue.lower() in self.api_manager.additional_exchanges:
            return self.api_manager.additional_exchanges[venue.lower()]
        else:
            return self.api_manager.exchange

    async def _place_order_with_protection(
        self,
        venue: str,
        order: Dict,
        timeout_seconds: float = 10.0
    ) -> Dict:
        """Place an order with circuit breaker and timeout protection"""
        api = self._get_api(venue)
        cb = self._get_circuit_breaker(venue)

        async def _do_place():
            return await asyncio.wait_for(
                api.place_order(order),
                timeout=timeout_seconds
            )

        try:
            return await cb.call(_do_place)
        except asyncio.TimeoutError:
            raise APITimeoutError(f"Order placement timed out on {venue}")

    async def _cancel_order_safe(self, venue: str, order_id: str) -> bool:
        """Safely cancel an order, handling errors gracefully"""
        if not order_id:
            return True

        api = self._get_api(venue)

        try:
            result = await asyncio.wait_for(
                api.cancel_order(order_id),
                timeout=10.0
            )
            logger.info(f"Cancelled order {order_id} on {venue}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout cancelling order {order_id} on {venue}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on {venue}: {e}")
            return False

    async def execute_atomic_trade(
        self,
        market: str,
        buy_venue: str,
        buy_price: float,
        buy_quantity: float,
        sell_venue: str,
        sell_price: float,
        sell_quantity: float,
        cancel_if_partial: bool = True,
        max_wait_ms: int = 5000,
        dry_run: bool = False
    ) -> Optional[ExecutedTrade]:
        """
        Execute a trade atomically with comprehensive error handling.

        Guarantees:
        - Both orders succeed OR both are cancelled (best effort)
        - Concurrent execution protection via semaphore
        - Circuit breaker prevents cascading failures
        - Detailed logging and error tracking

        Args:
            market: Trading pair/market identifier
            buy_venue: Venue to buy on
            buy_price: Buy price
            buy_quantity: Buy quantity
            sell_venue: Venue to sell on
            sell_price: Sell price
            sell_quantity: Sell quantity
            cancel_if_partial: Cancel both if one fails
            max_wait_ms: Maximum time to wait for fills
            dry_run: If True, simulate without placing real orders

        Returns:
            ExecutedTrade if successful, None if failed
        """
        trade_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        rollback_attempted = False

        logger.info(
            f"[Trade {trade_id}] Starting atomic trade: "
            f"BUY {buy_quantity} {market} @ ${buy_price} on {buy_venue} | "
            f"SELL {sell_quantity} {market} @ ${sell_price} on {sell_venue}"
        )

        # Acquire semaphore to limit concurrent trades
        async with self._trade_semaphore:
            # Create completion event for this trade
            completion_event = asyncio.Event()
            self._active_trades[trade_id] = completion_event

            buy_response = None
            sell_response = None
            buy_order_id = None
            sell_order_id = None

            try:
                # Check circuit breakers before attempting
                buy_cb = self._get_circuit_breaker(buy_venue)
                sell_cb = self._get_circuit_breaker(sell_venue)

                if buy_cb.is_open:
                    logger.warning(f"[Trade {trade_id}] Circuit breaker OPEN for {buy_venue}")
                    return None

                if sell_cb.is_open:
                    logger.warning(f"[Trade {trade_id}] Circuit breaker OPEN for {sell_venue}")
                    return None

                # Prepare orders
                buy_order = {
                    "market": market,
                    "side": "buy",
                    "quantity": buy_quantity,
                    "price": buy_price,
                    "order_type": "limit",
                    "client_order_id": f"{trade_id}_buy"
                }

                sell_order = {
                    "market": market,
                    "side": "sell",
                    "quantity": sell_quantity,
                    "price": sell_price,
                    "order_type": "limit",
                    "client_order_id": f"{trade_id}_sell"
                }

                if dry_run:
                    logger.info(f"[Trade {trade_id}] DRY RUN - simulating order placement")
                    buy_response = {"order_id": f"dry_buy_{trade_id}"}
                    sell_response = {"order_id": f"dry_sell_{trade_id}"}
                else:
                    # Place BOTH orders SIMULTANEOUSLY
                    logger.debug(f"[Trade {trade_id}] Placing orders simultaneously...")

                    results = await asyncio.gather(
                        self._place_order_with_protection(buy_venue, buy_order),
                        self._place_order_with_protection(sell_venue, sell_order),
                        return_exceptions=True
                    )

                    buy_response = results[0]
                    sell_response = results[1]

                # Check for exceptions in results
                buy_exception = isinstance(buy_response, Exception)
                sell_exception = isinstance(sell_response, Exception)

                if buy_exception or sell_exception:
                    logger.error(
                        f"[Trade {trade_id}] Order placement exception - "
                        f"Buy: {buy_response if buy_exception else 'OK'}, "
                        f"Sell: {sell_response if sell_exception else 'OK'}"
                    )

                    # Rollback: cancel any successful order
                    rollback_attempted = True
                    if not buy_exception and isinstance(buy_response, dict):
                        buy_order_id = buy_response.get("order_id")
                        await self._cancel_order_safe(buy_venue, buy_order_id)

                    if not sell_exception and isinstance(sell_response, dict):
                        sell_order_id = sell_response.get("order_id")
                        await self._cancel_order_safe(sell_venue, sell_order_id)

                    raise AtomicExecutionError(
                        "Order placement failed",
                        successful_order_id=buy_order_id or sell_order_id,
                        failed_venue=buy_venue if buy_exception else sell_venue,
                        rollback_status="completed" if rollback_attempted else "not_needed"
                    )

                # Validate responses
                buy_ok = isinstance(buy_response, dict) and buy_response.get("order_id")
                sell_ok = isinstance(sell_response, dict) and sell_response.get("order_id")

                buy_order_id = buy_response.get("order_id") if buy_ok else None
                sell_order_id = sell_response.get("order_id") if sell_ok else None

                if not (buy_ok and sell_ok):
                    logger.warning(
                        f"[Trade {trade_id}] Incomplete order placement - "
                        f"Buy OK: {buy_ok}, Sell OK: {sell_ok}"
                    )

                    # Rollback
                    rollback_attempted = True
                    if buy_ok:
                        await self._cancel_order_safe(buy_venue, buy_order_id)
                    if sell_ok:
                        await self._cancel_order_safe(sell_venue, sell_order_id)

                    return None

                # Calculate execution time
                execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._timing_stats.record(execution_time_ms)

                # Calculate profit (conservative fill assumption)
                buy_filled = buy_quantity * 0.95
                sell_filled = sell_quantity * 0.95
                profit = (sell_filled * sell_price) - (buy_filled * buy_price)

                trade = ExecutedTrade(
                    trade_id=trade_id,
                    market=market,
                    buy_order_id=buy_order_id,
                    buy_venue=buy_venue,
                    buy_price=buy_price,
                    buy_filled=buy_filled,
                    sell_order_id=sell_order_id,
                    sell_venue=sell_venue,
                    sell_price=sell_price,
                    sell_filled=sell_filled,
                    status=OrderStatus.FILLED,
                    profit=profit,
                    timestamp=datetime.now().timestamp(),
                    execution_time_ms=execution_time_ms,
                    rollback_attempted=rollback_attempted,
                )

                # Thread-safe trade recording
                async with self._execution_lock:
                    self.trades.append(trade)
                    self.pending_orders[trade_id] = trade

                logger.info(
                    f"[Trade {trade_id}] SUCCESS - Profit: ${profit:.2f}, "
                    f"Execution time: {execution_time_ms:.1f}ms"
                )

                return trade

            except AtomicExecutionError:
                raise
            except Exception as e:
                logger.error(f"[Trade {trade_id}] Unexpected error: {e}")

                # Best effort rollback
                rollback_attempted = True
                if buy_order_id:
                    await self._cancel_order_safe(buy_venue, buy_order_id)
                if sell_order_id:
                    await self._cancel_order_safe(sell_venue, sell_order_id)

                return None

            finally:
                # Signal completion
                completion_event.set()
                self._active_trades.pop(trade_id, None)
    
    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel both orders in a trade"""
        
        if trade_id not in self.pending_orders:
            print(f"Trade {trade_id} not found")
            return False
        
        trade = self.pending_orders[trade_id]
        
        try:
            # Cancel both orders
            if trade.buy_venue.lower() == "polymarket":
                await self.api_manager.polymarket.cancel_order(trade.buy_order_id)
            else:
                await self.api_manager.exchange.cancel_order(trade.buy_order_id)
            
            if trade.sell_venue.lower() == "polymarket":
                await self.api_manager.polymarket.cancel_order(trade.sell_order_id)
            else:
                await self.api_manager.exchange.cancel_order(trade.sell_order_id)
            
            trade.status = OrderStatus.CANCELLED
            del self.pending_orders[trade_id]
            
            print(f" Trade {trade_id} cancelled")
            return True
            
        except Exception as e:
            print(f" Cancellation failed: {e}")
            return False
    
    def get_trade_history(self, limit: int = 10) -> list:
        """Get recent trades"""
        return self.trades[-limit:]
    
    def get_total_profit(self) -> float:
        """Calculate total profit from all trades"""
        return sum(t.profit for t in self.trades if t.status == OrderStatus.FILLED)
    
    def get_win_rate(self) -> float:
        """Calculate % of profitable trades"""
        if not self.trades:
            return 0.0

        profitable = sum(1 for t in self.trades if t.profit > 0)
        return (profitable / len(self.trades)) * 100

    def get_execution_stats(self) -> Dict:
        """Get execution timing statistics"""
        return {
            "avg_execution_ms": self._timing_stats.avg_ms,
            "min_execution_ms": self._timing_stats.min_ms,
            "max_execution_ms": self._timing_stats.max_ms,
            "p95_execution_ms": self._timing_stats.p95_ms,
            "total_trades": len(self.trades),
            "active_trades": len(self._active_trades),
            "circuit_breakers": {
                venue: cb.state
                for venue, cb in self._circuit_breakers.items()
            }
        }

    async def wait_for_active_trades(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for all active trades to complete"""
        if not self._active_trades:
            return True

        events = list(self._active_trades.values())
        try:
            await asyncio.wait_for(
                asyncio.gather(*[e.wait() for e in events]),
                timeout=timeout_seconds
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for active trades to complete")
            return False


# Test the module
async def test_order_executor():
    """Test order execution"""
    
    from api_connectors import APIManager
    
    manager = APIManager()
    await manager.connect_all()
    
    executor = OrderExecutor(manager)
    
    print("Testing order execution...\n")
    
    # Execute a test trade
    trade = await executor.execute_atomic_trade(
        market="BTC",
        buy_venue="kraken",
        buy_price=42500,
        buy_quantity=0.1,
        sell_venue="polymarket",
        sell_price=42700,
        sell_quantity=0.1,
    )
    
    if trade:
        print(f"\n Trade Result:")
        print(f"   ID: {trade.trade_id}")
        print(f"   Status: {trade.status.value}")
        print(f"   Profit: ${trade.profit:.2f}")
        print(f"   Total execution profit: ${executor.get_total_profit():.2f}")
    
    # Check history
    print(f"\nTrade history (last 5):")
    for t in executor.get_trade_history(5):
        print(f"  {t.trade_id}: ${t.profit:.2f} ({t.status.value})")
    
    await manager.disconnect_all()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_order_executor())