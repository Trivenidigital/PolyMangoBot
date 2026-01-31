"""
WebSocket Manager Module
Real-time price streaming from Polymarket CLOB and exchanges
Replaces HTTP polling with WebSocket connections for sub-millisecond latency
"""

import asyncio
import json
import websockets
import logging
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriceEvent:
    """Real-time price update event"""
    venue: str
    symbol: str
    bid: float
    ask: float
    bid_quantity: float
    ask_quantity: float
    mid_price: float
    timestamp: float
    sequence: int = 0  # For order book reconstruction

    def to_dict(self) -> Dict:
        return {
            'venue': self.venue,
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'bid_quantity': self.bid_quantity,
            'ask_quantity': self.ask_quantity,
            'mid_price': self.mid_price,
            'timestamp': self.timestamp,
            'sequence': self.sequence
        }


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections"""
    polymarket_url: str = "wss://clob.polymarket.com/ws"
    kraken_url: str = "wss://ws.kraken.com"
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0
    heartbeat_interval: float = 30.0
    timeout: float = 60.0
    max_retries: int = 10


class PolymarketWebSocket:
    """WebSocket client for Polymarket CLOB streaming"""

    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        self.ws = None
        self.connected = False
        self.subscriptions: Dict[str, bool] = {}  # market_id -> subscribed
        self.message_queue = asyncio.Queue()
        self.reconnect_attempts = 0
        self.last_heartbeat = time.time()
        self.message_count = 0

    async def connect(self) -> bool:
        """Establish WebSocket connection to Polymarket CLOB"""
        try:
            # websockets v10+ uses open_timeout instead of timeout
            self.ws = await websockets.connect(
                self.config.polymarket_url,
                open_timeout=self.config.timeout,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.timeout / 2
            )
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("[Polymarket WebSocket] Connected successfully")
            return True
        except Exception as e:
            logger.error(f"[Polymarket WebSocket] Connection failed: {e}")
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info("[Polymarket WebSocket] Disconnected")

    async def subscribe(self, market_id: str) -> bool:
        """Subscribe to market price updates"""
        if not self.connected:
            logger.warning(f"[Polymarket WebSocket] Not connected, cannot subscribe to {market_id}")
            return False

        try:
            # Polymarket CLOB subscription message format
            subscribe_msg = {
                "action": "subscribe",
                "channel": "order_book",
                "market_id": market_id
            }

            await self.ws.send(json.dumps(subscribe_msg))
            self.subscriptions[market_id] = True
            logger.info(f"[Polymarket WebSocket] Subscribed to {market_id}")
            return True
        except Exception as e:
            logger.error(f"[Polymarket WebSocket] Subscribe failed: {e}")
            return False

    async def unsubscribe(self, market_id: str) -> bool:
        """Unsubscribe from market"""
        if not self.connected:
            return False

        try:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "channel": "order_book",
                "market_id": market_id
            }

            await self.ws.send(json.dumps(unsubscribe_msg))
            self.subscriptions[market_id] = False
            logger.info(f"[Polymarket WebSocket] Unsubscribed from {market_id}")
            return True
        except Exception as e:
            logger.error(f"[Polymarket WebSocket] Unsubscribe failed: {e}")
            return False

    async def listen(self, callback: Callable) -> None:
        """
        Listen for messages and invoke callback
        Handles reconnection automatically
        """
        while True:
            try:
                if not self.connected:
                    # Reconnect with exponential backoff
                    delay = min(
                        self.config.reconnect_delay * (2 ** self.reconnect_attempts),
                        self.config.max_reconnect_delay
                    )
                    logger.info(f"[Polymarket WebSocket] Reconnecting in {delay}s (attempt {self.reconnect_attempts + 1})")
                    await asyncio.sleep(delay)

                    if self.reconnect_attempts < self.config.max_retries:
                        if await self.connect():
                            # Resubscribe to all markets
                            for market_id in self.subscriptions:
                                await self.subscribe(market_id)
                            self.reconnect_attempts = 0
                        else:
                            self.reconnect_attempts += 1
                    else:
                        logger.error("[Polymarket WebSocket] Max reconnection attempts reached")
                        break
                    continue

                # Receive messages with timeout
                try:
                    msg = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=self.config.timeout
                    )

                    try:
                        data = json.loads(msg)
                        self.message_count += 1

                        # Parse price event from Polymarket format
                        event = self._parse_price_event(data)
                        if event:
                            await callback(event)
                    except json.JSONDecodeError as e:
                        logger.debug(f"[Polymarket WebSocket] Invalid JSON: {e}")

                except asyncio.TimeoutError:
                    logger.warning("[Polymarket WebSocket] No message received (timeout)")
                    self.connected = False

            except websockets.exceptions.ConnectionClosed:
                logger.warning("[Polymarket WebSocket] Connection closed")
                self.connected = False
            except Exception as e:
                logger.error(f"[Polymarket WebSocket] Listen error: {e}")
                self.connected = False
                await asyncio.sleep(1)

    def _parse_price_event(self, data: Dict) -> Optional[PriceEvent]:
        """Parse Polymarket WebSocket message into PriceEvent"""
        try:
            # Handle different message types
            if data.get("type") == "price_update":
                return PriceEvent(
                    venue="polymarket",
                    symbol=data.get("symbol", ""),
                    bid=float(data.get("bid", 0)),
                    ask=float(data.get("ask", 0)),
                    bid_quantity=float(data.get("bid_qty", 0)),
                    ask_quantity=float(data.get("ask_qty", 0)),
                    mid_price=(float(data.get("bid", 0)) + float(data.get("ask", 0))) / 2,
                    timestamp=time.time(),
                    sequence=int(data.get("seq", 0))
                )
            elif data.get("channel") == "order_book":
                # Handle order book snapshot
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if bids and asks:
                    bid_price = float(bids[0][0]) if bids else 0
                    ask_price = float(asks[0][0]) if asks else 0
                    bid_qty = float(bids[0][1]) if bids else 0
                    ask_qty = float(asks[0][1]) if asks else 0

                    return PriceEvent(
                        venue="polymarket",
                        symbol=data.get("market_id", ""),
                        bid=bid_price,
                        ask=ask_price,
                        bid_quantity=bid_qty,
                        ask_quantity=ask_qty,
                        mid_price=(bid_price + ask_price) / 2 if bid_price and ask_price else 0,
                        timestamp=time.time(),
                        sequence=int(data.get("sequence", 0))
                    )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"[Polymarket WebSocket] Parse error: {e}")

        return None


class ExchangeWebSocket:
    """WebSocket client for exchange price feeds (Kraken, etc.)"""

    def __init__(self, exchange_name: str = "kraken", config: WebSocketConfig = None):
        self.exchange_name = exchange_name.lower()
        self.config = config or WebSocketConfig()
        self.ws = None
        self.connected = False
        self.subscriptions: Dict[str, bool] = {}  # symbol -> subscribed
        self.reconnect_attempts = 0
        self.message_count = 0

        # Exchange-specific URLs
        self.urls = {
            "kraken": "wss://ws.kraken.com",
            "coinbase": "wss://ws-feed.exchange.coinbase.com"
        }

    async def connect(self) -> bool:
        """Establish WebSocket connection to exchange"""
        url = self.urls.get(self.exchange_name, self.config.kraken_url)

        try:
            # websockets v10+ uses open_timeout instead of timeout
            self.ws = await websockets.connect(
                url,
                open_timeout=self.config.timeout,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.timeout / 2
            )
            self.connected = True
            self.reconnect_attempts = 0
            logger.info(f"[{self.exchange_name.capitalize()} WebSocket] Connected successfully")
            return True
        except Exception as e:
            logger.error(f"[{self.exchange_name.capitalize()} WebSocket] Connection failed: {e}")
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info(f"[{self.exchange_name.capitalize()} WebSocket] Disconnected")

    async def subscribe(self, symbol: str) -> bool:
        """Subscribe to symbol price feed"""
        if not self.connected:
            return False

        try:
            if self.exchange_name == "kraken":
                subscribe_msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ticker",
                        "symbol": [f"{symbol}USDT"]
                    }
                }
            elif self.exchange_name == "coinbase":
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": [f"{symbol}-USD"],
                    "channels": ["ticker"]
                }
            else:
                return False

            await self.ws.send(json.dumps(subscribe_msg))
            self.subscriptions[symbol] = True
            logger.info(f"[{self.exchange_name.capitalize()} WebSocket] Subscribed to {symbol}")
            return True
        except Exception as e:
            logger.error(f"[{self.exchange_name.capitalize()} WebSocket] Subscribe failed: {e}")
            return False

    async def listen(self, callback: Callable) -> None:
        """Listen for messages and invoke callback"""
        while True:
            try:
                if not self.connected:
                    delay = min(
                        self.config.reconnect_delay * (2 ** self.reconnect_attempts),
                        self.config.max_reconnect_delay
                    )
                    logger.info(f"[{self.exchange_name.capitalize()} WebSocket] Reconnecting in {delay}s")
                    await asyncio.sleep(delay)

                    if self.reconnect_attempts < self.config.max_retries:
                        if await self.connect():
                            for symbol in self.subscriptions:
                                await self.subscribe(symbol)
                            self.reconnect_attempts = 0
                        else:
                            self.reconnect_attempts += 1
                    else:
                        logger.error(f"[{self.exchange_name.capitalize()} WebSocket] Max reconnection attempts reached")
                        break
                    continue

                try:
                    msg = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=self.config.timeout
                    )

                    try:
                        data = json.loads(msg)
                        self.message_count += 1

                        event = self._parse_price_event(data)
                        if event:
                            await callback(event)
                    except json.JSONDecodeError:
                        pass

                except asyncio.TimeoutError:
                    logger.warning(f"[{self.exchange_name.capitalize()} WebSocket] No message received (timeout)")
                    self.connected = False

            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[{self.exchange_name.capitalize()} WebSocket] Connection closed")
                self.connected = False
            except Exception as e:
                logger.error(f"[{self.exchange_name.capitalize()} WebSocket] Listen error: {e}")
                self.connected = False

    def _parse_price_event(self, data: Dict) -> Optional[PriceEvent]:
        """Parse exchange WebSocket message"""
        try:
            if self.exchange_name == "kraken":
                # Kraken ticker format: [channel_id, {...tick_data...}, "ticker", "symbol"]
                if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], dict):
                    tick = data[1]
                    symbol = data[3] if len(data) > 3 else ""

                    # Extract bid/ask - Kraken uses [price, volume] pairs
                    bid = float(tick.get("b", [0])[0]) if "b" in tick else 0
                    ask = float(tick.get("a", [0])[0]) if "a" in tick else 0

                    return PriceEvent(
                        venue="kraken",
                        symbol=symbol.replace("USDT", ""),
                        bid=bid,
                        ask=ask,
                        bid_quantity=float(tick.get("b", [0, 0])[1]) if "b" in tick else 0,
                        ask_quantity=float(tick.get("a", [0, 0])[1]) if "a" in tick else 0,
                        mid_price=(bid + ask) / 2 if bid and ask else 0,
                        timestamp=time.time()
                    )

            elif self.exchange_name == "coinbase":
                # Coinbase ticker format
                if data.get("type") == "ticker":
                    return PriceEvent(
                        venue="coinbase",
                        symbol=data.get("product_id", "").split("-")[0],
                        bid=float(data.get("best_bid", 0)),
                        ask=float(data.get("best_ask", 0)),
                        bid_quantity=float(data.get("best_bid_size", 0)),
                        ask_quantity=float(data.get("best_ask_size", 0)),
                        mid_price=(float(data.get("best_bid", 0)) + float(data.get("best_ask", 0))) / 2,
                        timestamp=time.time()
                    )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"[{self.exchange_name.capitalize()} WebSocket] Parse error: {e}")

        return None


class WebSocketManager:
    """
    Master WebSocket manager
    Orchestrates all WebSocket connections and price streaming
    With proper memory management and graceful shutdown support
    """

    def __init__(self, config: WebSocketConfig = None, max_history_per_symbol: int = 1000):
        self.config = config or WebSocketConfig()
        self.polymarket = PolymarketWebSocket(config)
        self.exchanges: Dict[str, ExchangeWebSocket] = {
            "kraken": ExchangeWebSocket("kraken", config),
            "coinbase": ExchangeWebSocket("coinbase", config)
        }

        # Price event callbacks
        self.price_callbacks: List[Callable] = []

        # Event aggregation with bounded storage (using deque to prevent memory leaks)
        self.latest_prices: Dict[str, PriceEvent] = {}  # venue:symbol -> latest event
        self._max_history = max_history_per_symbol
        # Use deque with maxlen for automatic memory management
        self.price_history: Dict[str, deque] = {}

        # Statistics
        self.total_events = 0
        self.events_per_venue: Dict[str, int] = defaultdict(int)

        # Graceful shutdown support
        self._shutdown_event = asyncio.Event()
        self._is_running = False
        self._listener_tasks: List[asyncio.Task] = []

    def _get_or_create_history(self, symbol: str) -> deque:
        """Get or create bounded history deque for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self._max_history)
        return self.price_history[symbol]

    def register_callback(self, callback: Callable):
        """Register a callback for price events"""
        self.price_callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """Unregister a callback"""
        if callback in self.price_callbacks:
            self.price_callbacks.remove(callback)

    async def _handle_price_event(self, event: PriceEvent):
        """Internal handler for price events with proper error isolation"""
        if self._shutdown_event.is_set():
            return

        self.total_events += 1
        self.events_per_venue[event.venue] += 1

        # Store latest price
        key = f"{event.venue}:{event.symbol}"
        self.latest_prices[key] = event

        # Store in history using bounded deque (automatically evicts old entries)
        history = self._get_or_create_history(event.symbol)
        history.append(event)

        # Invoke all callbacks with error isolation
        for callback in self.price_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Callback error (non-fatal): {e}")

    async def connect_all(self) -> bool:
        """Connect to all WebSocket sources"""
        results = await asyncio.gather(
            self.polymarket.connect(),
            *[ex.connect() for ex in self.exchanges.values()],
            return_exceptions=True
        )

        success = all(r is True for r in results)
        if success:
            logger.info("[WebSocket Manager] All connections established")
        else:
            logger.warning("[WebSocket Manager] Some connections failed")

        return success

    async def disconnect_all(self):
        """Disconnect from all WebSocket sources"""
        tasks = [self.polymarket.disconnect()]
        tasks.extend([ex.disconnect() for ex in self.exchanges.values()])

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("[WebSocket Manager] All disconnected")

    async def subscribe_polymarket(self, market_ids: List[str]) -> bool:
        """Subscribe to Polymarket markets"""
        if not self.polymarket.connected:
            if not await self.polymarket.connect():
                return False

        results = await asyncio.gather(
            *[self.polymarket.subscribe(mid) for mid in market_ids],
            return_exceptions=True
        )

        return all(r is True for r in results)

    async def subscribe_exchange(self, exchange: str, symbols: List[str]) -> bool:
        """Subscribe to exchange symbols"""
        if exchange not in self.exchanges:
            logger.error(f"Unknown exchange: {exchange}")
            return False

        ex_ws = self.exchanges[exchange]
        if not ex_ws.connected:
            if not await ex_ws.connect():
                return False

        results = await asyncio.gather(
            *[ex_ws.subscribe(sym) for sym in symbols],
            return_exceptions=True
        )

        return all(r is True for r in results)

    async def start_streaming(self):
        """Start all WebSocket listeners with graceful shutdown support"""
        if self._is_running:
            logger.warning("WebSocket streaming already running")
            return

        self._is_running = True
        self._shutdown_event.clear()

        await self.connect_all()

        # Start listener tasks
        self._listener_tasks = [
            asyncio.create_task(self.polymarket.listen(self._handle_price_event)),
            *[asyncio.create_task(ex.listen(self._handle_price_event))
              for ex in self.exchanges.values()]
        ]

        try:
            # Wait for shutdown signal or task completion
            done, pending = await asyncio.wait(
                self._listener_tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )

            # Check for exceptions
            for task in done:
                if task.exception():
                    logger.error(f"WebSocket listener error: {task.exception()}")

        except asyncio.CancelledError:
            logger.info("WebSocket streaming cancelled")
        finally:
            self._is_running = False

    async def shutdown(self):
        """Gracefully shutdown all WebSocket connections"""
        logger.info("Initiating WebSocket shutdown...")
        self._shutdown_event.set()

        # Cancel all listener tasks
        for task in self._listener_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)

        # Disconnect all
        await self.disconnect_all()
        self._is_running = False
        logger.info("WebSocket shutdown complete")

    def get_latest_price(self, venue: str, symbol: str) -> Optional[PriceEvent]:
        """Get latest price for a symbol from specific venue"""
        key = f"{venue}:{symbol}"
        return self.latest_prices.get(key)

    def get_all_prices(self, symbol: str) -> Dict[str, PriceEvent]:
        """Get latest prices for all venues for a symbol"""
        result = {}
        for venue in ["polymarket", "kraken", "coinbase"]:
            key = f"{venue}:{symbol}"
            if key in self.latest_prices:
                result[venue] = self.latest_prices[key]
        return result

    def get_price_history(self, symbol: str, limit: Optional[int] = None) -> List[PriceEvent]:
        """Get price history for a symbol (most recent first)"""
        if symbol not in self.price_history:
            return []

        history = list(self.price_history[symbol])
        if limit:
            return history[-limit:]
        return history

    def clear_history(self):
        """Clear all price history (for memory management)"""
        self.price_history.clear()
        logger.info("Price history cleared")

    def get_statistics(self) -> Dict:
        """Get WebSocket statistics"""
        history_memory = sum(len(h) for h in self.price_history.values())

        return {
            "total_events": self.total_events,
            "events_per_venue": dict(self.events_per_venue),
            "polymarket_connected": self.polymarket.connected,
            "polymarket_subscriptions": len(self.polymarket.subscriptions),
            "polymarket_message_count": self.polymarket.message_count,
            "exchange_connections": {
                ex: ws.connected for ex, ws in self.exchanges.items()
            },
            "exchange_message_counts": {
                ex: ws.message_count for ex, ws in self.exchanges.items()
            },
            "is_running": self._is_running,
            "history_entries": history_memory,
            "symbols_tracked": len(self.price_history),
        }


# Test
async def test_websocket_manager():
    """Test WebSocket manager"""
    manager = WebSocketManager()

    # Register callback
    async def price_callback(event: PriceEvent):
        print(f"Price update: {event.venue} {event.symbol} - "
              f"Bid: {event.bid:.4f}, Ask: {event.ask:.4f}, Mid: {event.mid_price:.4f}")

    manager.register_callback(price_callback)

    # Subscribe and stream
    await manager.connect_all()
    await manager.subscribe_polymarket(["0x123", "0x456"])  # Mock market IDs
    await manager.subscribe_exchange("kraken", ["BTC", "ETH"])

    # Run for 10 seconds
    try:
        await asyncio.wait_for(manager.start_streaming(), timeout=10)
    except asyncio.TimeoutError:
        pass

    # Print statistics
    stats = manager.get_statistics()
    print(f"\nWebSocket Statistics: {stats}")

    await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(test_websocket_manager())
