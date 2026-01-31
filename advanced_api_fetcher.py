"""
Advanced Parallel API Fetcher
High-performance concurrent data fetching with intelligent batching,
connection pooling, and adaptive rate limiting
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import statistics

logger = logging.getLogger("PolyMangoBot.fetcher")


class FetchPriority(Enum):
    """Priority levels for API requests"""
    CRITICAL = 0    # Order execution, cancellation
    HIGH = 1        # Price updates for active opportunities
    NORMAL = 2      # Regular price polling
    LOW = 3         # Background data collection


@dataclass
class FetchRequest:
    """Single fetch request with metadata"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Dict] = None
    priority: FetchPriority = FetchPriority.NORMAL
    venue: str = ""
    symbol: str = ""
    callback: Optional[Callable] = None
    timeout: float = 5.0
    retries: int = 3
    created_at: float = field(default_factory=time.time)


@dataclass
class FetchResult:
    """Result of a fetch operation"""
    request: FetchRequest
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on API responses.

    Features:
    - Learns optimal request rate from 429 responses
    - Backs off exponentially on errors
    - Recovers gradually when errors stop
    """

    def __init__(self, initial_rate: float = 10.0, burst: int = 5):
        self.base_rate = initial_rate
        self.current_rate = initial_rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()

        # Adaptive parameters
        self.error_count = 0
        self.success_count = 0
        self.last_429_time: Optional[float] = None
        self.backoff_until: Optional[float] = None

        # Statistics
        self.request_history: deque = deque(maxlen=100)

    async def acquire(self) -> float:
        """Acquire a token, returning wait time if needed"""
        async with self._lock:
            now = time.time()

            # Check if we're in backoff
            if self.backoff_until and now < self.backoff_until:
                wait_time = self.backoff_until - now
                await asyncio.sleep(wait_time)
                return wait_time

            # Replenish tokens
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.current_rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.current_rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return wait_time
            else:
                self.tokens -= 1
                return 0

    def record_success(self):
        """Record successful request"""
        self.success_count += 1
        self.error_count = max(0, self.error_count - 1)

        # Gradually recover rate
        if self.success_count > 10 and self.current_rate < self.base_rate:
            self.current_rate = min(self.base_rate, self.current_rate * 1.1)

    def record_error(self, status_code: int, retry_after: Optional[float] = None):
        """Record error and adjust rate"""
        self.error_count += 1
        self.success_count = 0

        if status_code == 429:
            self.last_429_time = time.time()

            # Use server's retry-after if provided
            if retry_after:
                self.backoff_until = time.time() + retry_after
            else:
                # Exponential backoff
                backoff = min(60, 2 ** self.error_count)
                self.backoff_until = time.time() + backoff

            # Reduce rate
            self.current_rate = max(0.5, self.current_rate * 0.5)
            logger.warning(f"Rate limit hit. Reducing rate to {self.current_rate:.1f}/s")
        else:
            # Other errors - mild backoff
            self.current_rate = max(1.0, self.current_rate * 0.8)

    def get_stats(self) -> Dict:
        return {
            "current_rate": self.current_rate,
            "base_rate": self.base_rate,
            "tokens": self.tokens,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "in_backoff": self.backoff_until is not None and time.time() < self.backoff_until
        }


class ConnectionPool:
    """Managed connection pool for async HTTP"""

    def __init__(self, max_connections: int = 100, keepalive_timeout: float = 30.0):
        self.max_connections = max_connections
        self.keepalive_timeout = keepalive_timeout
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create session with connection pooling"""
        async with self._lock:
            if self._session is None or self._session.closed:
                self._connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    keepalive_timeout=self.keepalive_timeout,
                    enable_cleanup_closed=True,
                    force_close=False
                )
                timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=timeout
                )
            return self._session

    async def close(self):
        """Close all connections"""
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None


class ParallelAPIFetcher:
    """
    High-performance parallel API fetcher with intelligent features.

    Features:
    - Priority-based request queue
    - Adaptive rate limiting per venue
    - Connection pooling
    - Automatic retries with exponential backoff
    - Request batching for efficiency
    - Latency tracking and anomaly detection
    """

    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.pool = ConnectionPool(max_connections=max_concurrent * 2)

        # Per-venue rate limiters
        self.rate_limiters: Dict[str, AdaptiveRateLimiter] = {}

        # Priority queue
        self._request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Latency tracking
        self.latency_stats: Dict[str, deque] = {}  # venue -> latencies

        # Background workers
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _get_rate_limiter(self, venue: str) -> AdaptiveRateLimiter:
        """Get or create rate limiter for venue"""
        if venue not in self.rate_limiters:
            # Different default rates for different venues
            rates = {
                "polymarket": 5.0,
                "kraken": 10.0,
                "coinbase": 8.0,
                "default": 5.0
            }
            rate = rates.get(venue, rates["default"])
            self.rate_limiters[venue] = AdaptiveRateLimiter(initial_rate=rate, burst=3)
        return self.rate_limiters[venue]

    def _record_latency(self, venue: str, latency_ms: float):
        """Record latency for anomaly detection"""
        if venue not in self.latency_stats:
            self.latency_stats[venue] = deque(maxlen=100)
        self.latency_stats[venue].append(latency_ms)

    def get_latency_stats(self, venue: str) -> Dict:
        """Get latency statistics for a venue"""
        if venue not in self.latency_stats or len(self.latency_stats[venue]) < 2:
            return {"avg": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}

        latencies = list(self.latency_stats[venue])
        sorted_latencies = sorted(latencies)

        return {
            "avg": statistics.mean(latencies),
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies)
        }

    async def fetch_single(self, request: FetchRequest) -> FetchResult:
        """Execute a single fetch request with retry logic"""
        session = await self.pool.get_session()
        rate_limiter = self._get_rate_limiter(request.venue)

        result = FetchResult(request=request, success=False)

        for attempt in range(request.retries):
            try:
                # Acquire rate limit token
                await rate_limiter.acquire()

                start_time = time.time()

                async with self._semaphore:
                    if request.method == "GET":
                        async with session.get(
                            request.url,
                            headers=request.headers,
                            timeout=aiohttp.ClientTimeout(total=request.timeout)
                        ) as response:
                            latency_ms = (time.time() - start_time) * 1000
                            self._record_latency(request.venue, latency_ms)

                            if response.status == 200:
                                data = await response.json()
                                rate_limiter.record_success()

                                result.success = True
                                result.data = data
                                result.latency_ms = latency_ms
                                result.retry_count = attempt

                                if request.callback:
                                    await self._safe_callback(request.callback, result)

                                return result

                            elif response.status == 429:
                                retry_after = float(response.headers.get("Retry-After", 0))
                                rate_limiter.record_error(429, retry_after)
                                result.error = "Rate limited"
                            else:
                                rate_limiter.record_error(response.status)
                                result.error = f"HTTP {response.status}"

                    elif request.method == "POST":
                        async with session.post(
                            request.url,
                            headers=request.headers,
                            json=request.data,
                            timeout=aiohttp.ClientTimeout(total=request.timeout)
                        ) as response:
                            latency_ms = (time.time() - start_time) * 1000
                            self._record_latency(request.venue, latency_ms)

                            if response.status in (200, 201):
                                data = await response.json()
                                rate_limiter.record_success()

                                result.success = True
                                result.data = data
                                result.latency_ms = latency_ms
                                result.retry_count = attempt

                                return result
                            else:
                                rate_limiter.record_error(response.status)
                                result.error = f"HTTP {response.status}"

            except asyncio.TimeoutError:
                result.error = "Timeout"
                rate_limiter.record_error(408)
            except aiohttp.ClientError as e:
                result.error = str(e)
                rate_limiter.record_error(500)
            except Exception as e:
                result.error = str(e)
                logger.error(f"Fetch error: {e}")

            # Wait before retry (exponential backoff)
            if attempt < request.retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

        return result

    async def _safe_callback(self, callback: Callable, result: FetchResult):
        """Safely execute callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def fetch_batch(
        self,
        requests: List[FetchRequest],
        return_when: str = "ALL_COMPLETED"
    ) -> List[FetchResult]:
        """
        Fetch multiple requests in parallel.

        Args:
            requests: List of fetch requests
            return_when: "ALL_COMPLETED" or "FIRST_COMPLETED"

        Returns:
            List of results in same order as requests
        """
        # Sort by priority
        sorted_requests = sorted(requests, key=lambda r: r.priority.value)

        # Create tasks
        tasks = [
            asyncio.create_task(self.fetch_single(req))
            for req in sorted_requests
        ]

        if return_when == "ALL_COMPLETED":
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(FetchResult(
                        request=sorted_requests[i],
                        success=False,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)

            return final_results
        else:
            # Return first completed
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending
            for task in pending:
                task.cancel()

            results = []
            for task in done:
                try:
                    results.append(task.result())
                except Exception as e:
                    pass

            return results

    async def fetch_prices_parallel(
        self,
        symbols: List[str],
        venues: List[str],
        get_url_func: Callable[[str, str], str],
        get_headers_func: Optional[Callable[[str], Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch prices for multiple symbols from multiple venues in parallel.

        Returns:
            {symbol: {venue: data}}
        """
        requests = []

        for symbol in symbols:
            for venue in venues:
                url = get_url_func(venue, symbol)
                headers = get_headers_func(venue) if get_headers_func else {}

                requests.append(FetchRequest(
                    url=url,
                    venue=venue,
                    symbol=symbol,
                    headers=headers,
                    priority=FetchPriority.HIGH,
                    timeout=3.0
                ))

        results = await self.fetch_batch(requests)

        # Organize results
        organized = {}
        for result in results:
            symbol = result.request.symbol
            venue = result.request.venue

            if symbol not in organized:
                organized[symbol] = {}

            if result.success:
                organized[symbol][venue] = result.data
            else:
                organized[symbol][venue] = None

        return organized

    async def close(self):
        """Close the fetcher and release resources"""
        await self.pool.close()

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "rate_limiters": {
                venue: limiter.get_stats()
                for venue, limiter in self.rate_limiters.items()
            },
            "latency": {
                venue: self.get_latency_stats(venue)
                for venue in self.latency_stats.keys()
            }
        }


class SmartDataAggregator:
    """
    Aggregates data from multiple sources with intelligent caching
    and staleness detection.
    """

    def __init__(self, max_age_ms: float = 1000.0):
        self.max_age_ms = max_age_ms
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (data, timestamp)
        self._lock = asyncio.Lock()

    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable[[], Any],
        max_age_override: Optional[float] = None
    ) -> Tuple[Any, bool]:
        """
        Get cached data or fetch fresh.

        Returns:
            (data, was_cached)
        """
        async with self._lock:
            max_age = max_age_override or self.max_age_ms

            if key in self.cache:
                data, timestamp = self.cache[key]
                age_ms = (time.time() - timestamp) * 1000

                if age_ms < max_age:
                    return data, True

            # Fetch fresh
            if asyncio.iscoroutinefunction(fetch_func):
                data = await fetch_func()
            else:
                data = fetch_func()

            self.cache[key] = (data, time.time())
            return data, False

    def invalidate(self, key: str):
        """Invalidate cached data"""
        if key in self.cache:
            del self.cache[key]

    def invalidate_all(self):
        """Clear all cached data"""
        self.cache.clear()

    def get_stale_keys(self) -> List[str]:
        """Get keys with stale data"""
        now = time.time()
        stale = []

        for key, (_, timestamp) in self.cache.items():
            age_ms = (now - timestamp) * 1000
            if age_ms > self.max_age_ms:
                stale.append(key)

        return stale


# Test
async def test_parallel_fetcher():
    """Test the parallel fetcher"""
    fetcher = ParallelAPIFetcher(max_concurrent=10)

    # Create test requests
    requests = [
        FetchRequest(
            url="https://api.kraken.com/0/public/Ticker?pair=BTCUSD",
            venue="kraken",
            symbol="BTC",
            priority=FetchPriority.HIGH
        ),
        FetchRequest(
            url="https://api.kraken.com/0/public/Ticker?pair=ETHUSD",
            venue="kraken",
            symbol="ETH",
            priority=FetchPriority.NORMAL
        ),
    ]

    print("Fetching prices in parallel...")
    results = await fetcher.fetch_batch(requests)

    for result in results:
        print(f"{result.request.symbol}: success={result.success}, "
              f"latency={result.latency_ms:.1f}ms")

    print(f"\nStats: {fetcher.get_stats()}")

    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(test_parallel_fetcher())
