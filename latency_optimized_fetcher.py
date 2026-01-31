"""
Latency-Optimized Data Fetcher
Priority 1 implementation from analysis report.

Features:
- WebSocket-first architecture with REST fallback
- orjson for 3-10x faster JSON parsing
- Parallel order placement
- Connection pre-warming
- Latency profiling and monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from contextlib import contextmanager
from enum import Enum
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

# Use orjson for faster JSON parsing (3-10x faster than stdlib json)
try:
    import orjson
    def json_loads(data):
        return orjson.loads(data)
    def json_dumps(data):
        return orjson.dumps(data).decode('utf-8')
    USING_ORJSON = True
except ImportError:
    import json
    json_loads = json.loads
    json_dumps = json.dumps
    USING_ORJSON = False

logger = logging.getLogger("PolyMangoBot.latency_optimized")


class DataSource(Enum):
    """Data source type"""
    WEBSOCKET = "websocket"
    REST = "rest"
    CACHE = "cache"


@dataclass
class LatencyMetrics:
    """Latency tracking for performance monitoring"""
    component: str
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ns: int):
        self.samples.append(latency_ns)

    @property
    def avg_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples) / 1_000_000

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)] / 1_000_000

    @property
    def p99_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)] / 1_000_000

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "avg_ms": round(self.avg_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "samples": len(self.samples)
        }


class LatencyProfiler:
    """Global latency profiler for all components"""

    def __init__(self):
        self.metrics: Dict[str, LatencyMetrics] = {}
        self._enabled = True

    @contextmanager
    def track(self, component: str):
        """Context manager for tracking latency"""
        if not self._enabled:
            yield
            return

        if component not in self.metrics:
            self.metrics[component] = LatencyMetrics(component=component)

        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self.metrics[component].record(elapsed)

    def record(self, component: str, latency_ns: int):
        """Direct recording of latency"""
        if component not in self.metrics:
            self.metrics[component] = LatencyMetrics(component=component)
        self.metrics[component].record(latency_ns)

    def get_report(self) -> Dict:
        """Get full latency report"""
        return {
            name: metrics.to_dict()
            for name, metrics in self.metrics.items()
        }

    def log_summary(self):
        """Log latency summary"""
        logger.info("=== Latency Summary ===")
        for name, metrics in sorted(self.metrics.items()):
            logger.info(
                f"  {name}: avg={metrics.avg_ms:.2f}ms, "
                f"p95={metrics.p95_ms:.2f}ms, p99={metrics.p99_ms:.2f}ms"
            )


# Global profiler instance
profiler = LatencyProfiler()


@dataclass
class MarketData:
    """Normalized market data from any source"""
    venue: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp_ms: float
    source: DataSource
    sequence: int = 0

    # Full order book (optional)
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0

    @property
    def age_ms(self) -> float:
        return time.time() * 1000 - self.timestamp_ms


class WebSocketDataStream:
    """
    High-performance WebSocket data stream with automatic reconnection.
    """

    def __init__(
        self,
        name: str,
        url: str,
        subscribe_messages: List[Dict],
        parse_func: Callable[[Dict], Optional[MarketData]],
        on_data: Optional[Callable[[MarketData], None]] = None
    ):
        self.name = name
        self.url = url
        self.subscribe_messages = subscribe_messages
        self.parse_func = parse_func
        self.on_data = on_data

        self._ws = None
        self._running = False
        self._connected = False
        self._last_message_time = 0.0
        self._message_count = 0
        self._reconnect_count = 0

        # Data cache
        self._latest_data: Dict[str, MarketData] = {}
        self._data_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        self._running = True

        try:
            with profiler.track(f"ws_connect_{self.name}"):
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=5
                    ),
                    timeout=10.0
                )

            self._connected = True
            logger.info(f"[{self.name}] WebSocket connected to {self.url}")

            # Send subscriptions
            for msg in self.subscribe_messages:
                with profiler.track(f"ws_subscribe_{self.name}"):
                    await self._ws.send(json_dumps(msg))
                    await asyncio.sleep(0.05)  # Small delay between subscriptions

            # Start receive loop
            asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            logger.error(f"[{self.name}] Connection failed: {e}")
            self._connected = False
            return False

    async def _receive_loop(self):
        """Main receive loop with fast parsing"""
        while self._running and self._ws:
            try:
                # Use wait_for with short timeout to stay responsive
                raw_message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=30.0
                )

                receive_time = time.time() * 1000

                # Fast JSON parsing with orjson
                with profiler.track(f"json_parse_{self.name}"):
                    message = json_loads(raw_message)

                self._last_message_time = receive_time
                self._message_count += 1

                # Parse to normalized format
                with profiler.track(f"data_parse_{self.name}"):
                    data = self.parse_func(message)

                if data:
                    data.timestamp_ms = receive_time

                    # Update cache
                    cache_key = f"{data.venue}:{data.symbol}"
                    async with self._data_lock:
                        self._latest_data[cache_key] = data

                    # Callback
                    if self.on_data:
                        with profiler.track(f"callback_{self.name}"):
                            self.on_data(data)

            except asyncio.TimeoutError:
                # No message received - check connection health
                if time.time() * 1000 - self._last_message_time > 60000:
                    logger.warning(f"[{self.name}] No messages for 60s, reconnecting")
                    await self._reconnect()

            except ConnectionClosed as e:
                logger.warning(f"[{self.name}] Connection closed: {e}")
                await self._reconnect()

            except Exception as e:
                logger.error(f"[{self.name}] Receive error: {e}")
                await asyncio.sleep(0.1)

    async def _reconnect(self):
        """Reconnect with exponential backoff"""
        self._connected = False
        self._reconnect_count += 1

        delay = min(30, 2 ** min(self._reconnect_count, 5))
        logger.info(f"[{self.name}] Reconnecting in {delay}s (attempt {self._reconnect_count})")

        await asyncio.sleep(delay)

        if self._running:
            await self.connect()

    async def get_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for a symbol"""
        cache_key = f"{self.name}:{symbol}"
        async with self._data_lock:
            return self._latest_data.get(cache_key)

    async def disconnect(self):
        """Disconnect WebSocket"""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_stats(self) -> Dict:
        return {
            "name": self.name,
            "connected": self._connected,
            "message_count": self._message_count,
            "reconnect_count": self._reconnect_count,
            "last_message_age_ms": time.time() * 1000 - self._last_message_time if self._last_message_time > 0 else -1,
            "cached_symbols": len(self._latest_data)
        }


class RESTFallback:
    """
    REST API fallback when WebSocket is unavailable.
    Optimized with connection pooling and fast parsing.
    """

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create optimized session"""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
                force_close=False,
                ttl_dns_cache=300  # Cache DNS for 5 minutes
            )

            timeout = aiohttp.ClientTimeout(
                total=5,
                connect=2,
                sock_read=3
            )

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )

        return self._session

    async def fetch(self, url: str, venue: str) -> Optional[Dict]:
        """Fetch data from REST API"""
        session = await self._get_session()

        try:
            with profiler.track(f"rest_fetch_{venue}"):
                async with session.get(url) as response:
                    if response.status == 200:
                        raw = await response.read()

                        with profiler.track(f"rest_parse_{venue}"):
                            return json_loads(raw)
                    else:
                        logger.warning(f"REST {venue} returned {response.status}")
                        return None

        except Exception as e:
            logger.error(f"REST fetch error for {venue}: {e}")
            return None

    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()


class LatencyOptimizedDataManager:
    """
    Main data manager with WebSocket-first architecture.

    Features:
    - Primary: WebSocket streams for real-time data
    - Fallback: REST API when WebSocket unavailable
    - Caching: Local cache with staleness detection
    - Profiling: Continuous latency monitoring
    """

    def __init__(self):
        self.ws_streams: Dict[str, WebSocketDataStream] = {}
        self.rest_fallback = RESTFallback()

        # Data cache (unified across sources)
        self._cache: Dict[str, MarketData] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_max_age_ms = 1000  # 1 second

        # Pre-computed features (avoid recomputation)
        self._precomputed: Dict[str, Dict] = {}
        self._precompute_interval = 0.1  # 100ms
        self._last_precompute = 0.0

        # Callbacks
        self._data_callbacks: List[Callable[[MarketData], None]] = []

    def add_data_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for new data"""
        self._data_callbacks.append(callback)

    def _on_ws_data(self, data: MarketData):
        """Handle incoming WebSocket data"""
        # Update cache
        cache_key = f"{data.venue}:{data.symbol}"
        self._cache[cache_key] = data

        # Trigger callbacks
        for callback in self._data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")

    async def setup_kraken_stream(self, symbols: List[str]):
        """Setup Kraken WebSocket stream"""

        def parse_kraken(msg: Dict) -> Optional[MarketData]:
            """Parse Kraken WebSocket message"""
            # Kraken ticker format: [channelID, data, "ticker", "XBT/USD"]
            if not isinstance(msg, list) or len(msg) < 4:
                return None

            if msg[2] != "ticker":
                return None

            try:
                data = msg[1]
                pair = msg[3]

                # Map pair to symbol
                symbol = pair.replace("/", "").replace("XBT", "BTC")

                return MarketData(
                    venue="kraken",
                    symbol=symbol,
                    bid=float(data['b'][0]),
                    ask=float(data['a'][0]),
                    bid_size=float(data['b'][2]),
                    ask_size=float(data['a'][2]),
                    timestamp_ms=time.time() * 1000,
                    source=DataSource.WEBSOCKET
                )
            except (KeyError, IndexError, ValueError) as e:
                return None

        # Build subscription messages
        pairs = []
        for symbol in symbols:
            if symbol == "BTC":
                pairs.append("XBT/USD")
            elif symbol == "ETH":
                pairs.append("ETH/USD")
            else:
                pairs.append(f"{symbol}/USD")

        subscribe_msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ticker"}
        }

        stream = WebSocketDataStream(
            name="kraken",
            url="wss://ws.kraken.com",
            subscribe_messages=[subscribe_msg],
            parse_func=parse_kraken,
            on_data=self._on_ws_data
        )

        self.ws_streams["kraken"] = stream
        return await stream.connect()

    async def setup_coinbase_stream(self, symbols: List[str]):
        """Setup Coinbase WebSocket stream"""

        def parse_coinbase(msg: Dict) -> Optional[MarketData]:
            """Parse Coinbase WebSocket message"""
            if msg.get("type") != "ticker":
                return None

            try:
                product_id = msg.get("product_id", "")
                symbol = product_id.replace("-USD", "").replace("-", "")

                return MarketData(
                    venue="coinbase",
                    symbol=symbol,
                    bid=float(msg.get("best_bid", 0)),
                    ask=float(msg.get("best_ask", 0)),
                    bid_size=float(msg.get("best_bid_size", 0)),
                    ask_size=float(msg.get("best_ask_size", 0)),
                    timestamp_ms=time.time() * 1000,
                    source=DataSource.WEBSOCKET
                )
            except (KeyError, ValueError):
                return None

        # Build subscription
        product_ids = [f"{s}-USD" for s in symbols]

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker"]
        }

        stream = WebSocketDataStream(
            name="coinbase",
            url="wss://ws-feed.exchange.coinbase.com",
            subscribe_messages=[subscribe_msg],
            parse_func=parse_coinbase,
            on_data=self._on_ws_data
        )

        self.ws_streams["coinbase"] = stream
        return await stream.connect()

    async def get_market_data(
        self,
        venue: str,
        symbol: str,
        max_age_ms: float = 1000
    ) -> Optional[MarketData]:
        """
        Get market data with WebSocket-first, REST fallback strategy.
        """
        cache_key = f"{venue}:{symbol}"

        # Check cache first
        with profiler.track("cache_lookup"):
            cached = self._cache.get(cache_key)
            if cached and cached.age_ms < max_age_ms:
                return cached

        # Try WebSocket stream
        stream = self.ws_streams.get(venue)
        if stream and stream.is_connected:
            data = await stream.get_data(symbol)
            if data and data.age_ms < max_age_ms:
                return data

        # Fallback to REST
        with profiler.track(f"rest_fallback_{venue}"):
            return await self._fetch_rest(venue, symbol)

    async def _fetch_rest(self, venue: str, symbol: str) -> Optional[MarketData]:
        """Fetch from REST API"""
        if venue == "kraken":
            pair = "XBTUSD" if symbol == "BTC" else f"{symbol}USD"
            url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"

            data = await self.rest_fallback.fetch(url, venue)
            if data and "result" in data:
                for pair_name, ticker in data["result"].items():
                    return MarketData(
                        venue="kraken",
                        symbol=symbol,
                        bid=float(ticker['b'][0]),
                        ask=float(ticker['a'][0]),
                        bid_size=float(ticker['b'][2]),
                        ask_size=float(ticker['a'][2]),
                        timestamp_ms=time.time() * 1000,
                        source=DataSource.REST
                    )

        elif venue == "coinbase":
            product_id = f"{symbol}-USD"
            url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"

            data = await self.rest_fallback.fetch(url, venue)
            if data:
                return MarketData(
                    venue="coinbase",
                    symbol=symbol,
                    bid=float(data.get('bid', 0)),
                    ask=float(data.get('ask', 0)),
                    bid_size=float(data.get('size', 0)),
                    ask_size=float(data.get('size', 0)),
                    timestamp_ms=time.time() * 1000,
                    source=DataSource.REST
                )

        return None

    async def get_all_venues(
        self,
        symbol: str,
        venues: List[str] = None
    ) -> Dict[str, MarketData]:
        """
        Get market data from all venues in parallel.
        """
        if venues is None:
            venues = list(self.ws_streams.keys())

        with profiler.track("parallel_fetch_all"):
            tasks = [
                self.get_market_data(venue, symbol)
                for venue in venues
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for venue, result in zip(venues, results):
            if isinstance(result, MarketData):
                data[venue] = result
            elif isinstance(result, Exception):
                logger.warning(f"Failed to get {symbol} from {venue}: {result}")

        return data

    async def connect_all(self, symbols: List[str] = None):
        """Connect all WebSocket streams"""
        if symbols is None:
            symbols = ["BTC", "ETH"]

        logger.info(f"Connecting WebSocket streams for {symbols}")
        logger.info(f"Using orjson: {USING_ORJSON}")

        # Connect in parallel
        results = await asyncio.gather(
            self.setup_kraken_stream(symbols),
            self.setup_coinbase_stream(symbols),
            return_exceptions=True
        )

        connected = sum(1 for r in results if r is True)
        logger.info(f"Connected {connected}/{len(results)} WebSocket streams")

    async def disconnect_all(self):
        """Disconnect all streams"""
        for stream in self.ws_streams.values():
            await stream.disconnect()

        await self.rest_fallback.close()

    def get_stats(self) -> Dict:
        """Get comprehensive stats"""
        return {
            "using_orjson": USING_ORJSON,
            "streams": {
                name: stream.get_stats()
                for name, stream in self.ws_streams.items()
            },
            "cache_size": len(self._cache),
            "latency": profiler.get_report()
        }


class ParallelOrderExecutor:
    """
    Parallel order execution for minimal latency.
    """

    def __init__(self, api_manager):
        self.api_manager = api_manager
        self._execution_lock = asyncio.Lock()

    async def execute_parallel(
        self,
        buy_order: Dict,
        sell_order: Dict,
        timeout_ms: float = 5000
    ) -> Tuple[Dict, Dict]:
        """
        Execute buy and sell orders in parallel.

        Returns:
            Tuple of (buy_result, sell_result)
        """
        async with self._execution_lock:
            with profiler.track("parallel_order_execution"):
                # Create tasks
                buy_task = asyncio.create_task(
                    self._place_order(buy_order)
                )
                sell_task = asyncio.create_task(
                    self._place_order(sell_order)
                )

                # Execute with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(buy_task, sell_task, return_exceptions=True),
                        timeout=timeout_ms / 1000
                    )
                    return results[0], results[1]

                except asyncio.TimeoutError:
                    # Cancel pending tasks
                    buy_task.cancel()
                    sell_task.cancel()

                    return (
                        {"error": "timeout", "success": False},
                        {"error": "timeout", "success": False}
                    )

    async def _place_order(self, order: Dict) -> Dict:
        """Place a single order"""
        venue = order.get("venue", "")

        with profiler.track(f"order_place_{venue}"):
            try:
                # Get appropriate API
                if venue == "kraken":
                    api = self.api_manager.exchange
                elif venue == "polymarket":
                    api = self.api_manager.polymarket
                else:
                    api = self.api_manager.additional_exchanges.get(venue)

                if api is None:
                    return {"error": f"Unknown venue: {venue}", "success": False}

                result = await api.place_order(order)
                return {"success": True, "result": result}

            except Exception as e:
                return {"error": str(e), "success": False}


# Test function
async def test_latency_optimized():
    """Test the latency-optimized data manager"""
    manager = LatencyOptimizedDataManager()

    print("Testing Latency-Optimized Data Manager")
    print(f"Using orjson: {USING_ORJSON}")
    print()

    # Connect
    await manager.connect_all(["BTC", "ETH"])

    # Wait for data
    await asyncio.sleep(3)

    # Fetch data
    print("Fetching BTC from all venues...")
    data = await manager.get_all_venues("BTC")

    for venue, market_data in data.items():
        print(f"  {venue}: bid={market_data.bid:.2f}, ask={market_data.ask:.2f}, "
              f"spread={market_data.spread_bps:.2f}bps, source={market_data.source.value}")

    # Print latency report
    print("\nLatency Report:")
    profiler.log_summary()

    # Stats
    print("\nStats:")
    stats = manager.get_stats()
    for stream_name, stream_stats in stats["streams"].items():
        print(f"  {stream_name}: messages={stream_stats['message_count']}, "
              f"connected={stream_stats['connected']}")

    await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(test_latency_optimized())
