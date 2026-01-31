"""
Enhanced Parallel API Fetcher
Advanced improvements for high-performance data fetching:

1. Request Batching - Combine multiple requests to same endpoint
2. Predictive Pre-fetching - Anticipate data needs based on patterns
3. Failure-Aware Load Balancing - Route away from failing endpoints
"""

import asyncio
import aiohttp
import logging
import time
import hashlib
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
import heapq

logger = logging.getLogger("PolyMangoBot.enhanced_fetcher")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class FetchPriority(Enum):
    """Priority levels for API requests"""
    CRITICAL = 0    # Order execution, cancellation
    HIGH = 1        # Price updates for active opportunities
    NORMAL = 2      # Regular price polling
    LOW = 3         # Background data collection
    PREFETCH = 4    # Predictive pre-fetch (lowest priority)


class EndpointHealth(Enum):
    """Health status of an endpoint"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


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
    endpoint_type: str = ""  # e.g., "orderbook", "ticker", "trades"
    callback: Optional[Callable] = None
    timeout: float = 5.0
    retries: int = 3
    created_at: float = field(default_factory=time.time)
    batch_key: Optional[str] = None  # Key for batching similar requests
    is_prefetch: bool = False

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value < other.priority.value


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
    from_cache: bool = False
    from_batch: bool = False


@dataclass
class EndpointStats:
    """Statistics for an endpoint"""
    url_pattern: str
    venue: str

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency stats
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    # Error tracking
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

    # Health
    health: EndpointHealth = EndpointHealth.HEALTHY
    circuit_open_until: Optional[float] = None

    def record_success(self, latency_ms: float):
        """Record successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.latencies.append(latency_ms)
        self._update_health()

    def record_failure(self, error: str):
        """Record failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self._update_health()

    def _update_health(self):
        """Update health status based on recent performance"""
        if self.consecutive_failures >= 10:
            self.health = EndpointHealth.DEAD
            # Open circuit for 60 seconds
            self.circuit_open_until = time.time() + 60
        elif self.consecutive_failures >= 5:
            self.health = EndpointHealth.UNHEALTHY
            self.circuit_open_until = time.time() + 30
        elif self.consecutive_failures >= 2:
            self.health = EndpointHealth.DEGRADED
        else:
            self.health = EndpointHealth.HEALTHY
            self.circuit_open_until = None

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open_until is None:
            return False
        if time.time() >= self.circuit_open_until:
            # Circuit timeout expired, allow half-open
            self.circuit_open_until = None
            return False
        return True

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> Dict:
        return {
            "venue": self.venue,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency,
            "p95_latency_ms": self.p95_latency,
            "health": self.health.value,
            "consecutive_failures": self.consecutive_failures,
            "circuit_open": self.is_circuit_open()
        }


# =============================================================================
# REQUEST BATCHING
# =============================================================================

class RequestBatcher:
    """
    Batches similar requests to reduce API calls.

    Features:
    - Groups requests by batch key (venue + endpoint type)
    - Waits for batch window before executing
    - Deduplicates identical requests
    - Returns results to all waiters
    """

    def __init__(
        self,
        batch_window_ms: float = 50.0,      # Wait this long to collect requests
        max_batch_size: int = 10,            # Max requests per batch
        enabled_endpoints: Optional[Set[str]] = None  # Which endpoints support batching
    ):
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.enabled_endpoints = enabled_endpoints or {"orderbook", "ticker", "trades"}

        # Pending batches: batch_key -> list of (request, future)
        self._pending: Dict[str, List[Tuple[FetchRequest, asyncio.Future]]] = defaultdict(list)
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self.batches_executed = 0
        self.requests_batched = 0
        self.requests_deduplicated = 0

    def _get_batch_key(self, request: FetchRequest) -> Optional[str]:
        """Get batch key for a request, or None if not batchable"""
        if request.endpoint_type not in self.enabled_endpoints:
            return None

        # Create key from venue and endpoint type
        return f"{request.venue}:{request.endpoint_type}"

    def _get_dedup_key(self, request: FetchRequest) -> str:
        """Get deduplication key for a request"""
        return hashlib.md5(f"{request.url}:{request.method}".encode()).hexdigest()

    async def add_request(
        self,
        request: FetchRequest,
        execute_func: Callable[[List[FetchRequest]], Any]
    ) -> FetchResult:
        """
        Add a request to batch queue.

        Returns result when batch is executed.
        """
        batch_key = self._get_batch_key(request)

        if batch_key is None:
            # Not batchable, execute immediately
            results = await execute_func([request])
            return results[0] if results else FetchResult(request=request, success=False, error="No result")

        async with self._lock:
            # Create future for this request
            future = asyncio.Future()

            # Check for duplicate in pending batch
            dedup_key = self._get_dedup_key(request)
            for pending_req, pending_future in self._pending[batch_key]:
                if self._get_dedup_key(pending_req) == dedup_key:
                    # Duplicate found, share the same future
                    self.requests_deduplicated += 1
                    # Wait for the original request's result
                    return await pending_future

            # Add to pending batch
            self._pending[batch_key].append((request, future))
            self.requests_batched += 1

            # Start timer if not already running
            if batch_key not in self._batch_timers:
                self._batch_timers[batch_key] = asyncio.create_task(
                    self._batch_timer(batch_key, execute_func)
                )

            # Check if batch is full
            if len(self._pending[batch_key]) >= self.max_batch_size:
                # Execute immediately
                if batch_key in self._batch_timers:
                    self._batch_timers[batch_key].cancel()
                    del self._batch_timers[batch_key]
                await self._execute_batch(batch_key, execute_func)

        # Wait for result
        return await future

    async def _batch_timer(
        self,
        batch_key: str,
        execute_func: Callable[[List[FetchRequest]], Any]
    ):
        """Timer to execute batch after window expires"""
        try:
            await asyncio.sleep(self.batch_window_ms / 1000)
            async with self._lock:
                if batch_key in self._batch_timers:
                    del self._batch_timers[batch_key]
                await self._execute_batch(batch_key, execute_func)
        except asyncio.CancelledError:
            pass

    async def _execute_batch(
        self,
        batch_key: str,
        execute_func: Callable[[List[FetchRequest]], Any]
    ):
        """Execute a batch of requests"""
        if batch_key not in self._pending or not self._pending[batch_key]:
            return

        pending = self._pending[batch_key]
        del self._pending[batch_key]

        self.batches_executed += 1

        # Extract requests
        requests = [req for req, _ in pending]

        try:
            # Execute batch
            results = await execute_func(requests)

            # Match results to futures
            result_map = {self._get_dedup_key(r.request): r for r in results}

            for request, future in pending:
                dedup_key = self._get_dedup_key(request)
                if dedup_key in result_map:
                    result = result_map[dedup_key]
                    result.from_batch = True
                    if not future.done():
                        future.set_result(result)
                else:
                    if not future.done():
                        future.set_result(FetchResult(
                            request=request,
                            success=False,
                            error="No result in batch"
                        ))

        except Exception as e:
            # Fail all futures
            for _, future in pending:
                if not future.done():
                    future.set_result(FetchResult(
                        request=request,
                        success=False,
                        error=str(e)
                    ))

    def get_stats(self) -> Dict:
        return {
            "batches_executed": self.batches_executed,
            "requests_batched": self.requests_batched,
            "requests_deduplicated": self.requests_deduplicated,
            "pending_batches": len(self._pending),
            "efficiency": (
                self.requests_deduplicated / max(1, self.requests_batched)
            )
        }


# =============================================================================
# PREDICTIVE PRE-FETCHING
# =============================================================================

class PredictivePrefetcher:
    """
    Anticipates data needs based on access patterns.

    Features:
    - Tracks access patterns for symbols/venues
    - Predicts next likely requests
    - Pre-fetches data before it's needed
    - Manages prefetch cache
    """

    def __init__(
        self,
        pattern_window: int = 100,          # How many requests to track
        prediction_threshold: float = 0.7,   # Min probability to prefetch
        prefetch_ahead_ms: float = 500.0,    # How far ahead to prefetch
        max_prefetch_queue: int = 20,        # Max pending prefetches
        cache_ttl_ms: float = 2000.0         # How long to cache prefetched data
    ):
        self.pattern_window = pattern_window
        self.prediction_threshold = prediction_threshold
        self.prefetch_ahead_ms = prefetch_ahead_ms
        self.max_prefetch_queue = max_prefetch_queue
        self.cache_ttl_ms = cache_ttl_ms

        # Access pattern tracking
        # Tracks sequences of (venue, symbol, endpoint) accesses
        self._access_history: deque = deque(maxlen=pattern_window)
        self._transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Prefetch cache: key -> (data, timestamp)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = asyncio.Lock()

        # Pending prefetches
        self._pending_prefetches: Set[str] = set()

        # Statistics
        self.prefetches_triggered = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    def _get_access_key(self, request: FetchRequest) -> str:
        """Get key representing an access pattern"""
        return f"{request.venue}:{request.symbol}:{request.endpoint_type}"

    def _get_cache_key(self, request: FetchRequest) -> str:
        """Get cache key for a request"""
        return hashlib.md5(request.url.encode()).hexdigest()

    def record_access(self, request: FetchRequest):
        """Record an access to update patterns"""
        key = self._get_access_key(request)

        # Update transition counts
        if self._access_history:
            prev_key = self._access_history[-1]
            self._transition_counts[prev_key][key] += 1

        self._access_history.append(key)

    def predict_next(self, current_request: FetchRequest) -> List[Tuple[str, float]]:
        """
        Predict next likely requests based on current access.

        Returns list of (access_key, probability) sorted by probability.
        """
        current_key = self._get_access_key(current_request)
        transitions = self._transition_counts.get(current_key, {})

        if not transitions:
            return []

        total = sum(transitions.values())
        predictions = [
            (next_key, count / total)
            for next_key, count in transitions.items()
        ]

        # Sort by probability descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def get_prefetch_requests(
        self,
        current_request: FetchRequest,
        request_builder: Callable[[str, str, str], FetchRequest]
    ) -> List[FetchRequest]:
        """
        Get list of requests to prefetch based on predictions.

        Args:
            current_request: The current request being made
            request_builder: Function to build request from (venue, symbol, endpoint)

        Returns:
            List of requests to prefetch
        """
        predictions = self.predict_next(current_request)
        prefetch_requests = []

        for access_key, probability in predictions:
            if probability < self.prediction_threshold:
                break

            if len(prefetch_requests) >= self.max_prefetch_queue:
                break

            # Skip if already pending or cached
            if access_key in self._pending_prefetches:
                continue

            # Parse access key
            parts = access_key.split(":")
            if len(parts) != 3:
                continue

            venue, symbol, endpoint = parts

            # Check if already cached and fresh
            cache_key = f"{venue}:{symbol}:{endpoint}"
            if cache_key in self._cache:
                _, timestamp = self._cache[cache_key]
                age_ms = (time.time() - timestamp) * 1000
                if age_ms < self.cache_ttl_ms * 0.5:
                    continue  # Still fresh enough

            # Build prefetch request
            try:
                request = request_builder(venue, symbol, endpoint)
                request.priority = FetchPriority.PREFETCH
                request.is_prefetch = True
                prefetch_requests.append(request)
                self._pending_prefetches.add(access_key)
                self.prefetches_triggered += 1
            except Exception as e:
                logger.debug(f"Failed to build prefetch request: {e}")

        return prefetch_requests

    async def cache_result(self, request: FetchRequest, result: FetchResult):
        """Cache a prefetch result"""
        if not result.success:
            return

        cache_key = self._get_cache_key(request)
        access_key = self._get_access_key(request)

        async with self._cache_lock:
            self._cache[cache_key] = (result.data, time.time())

        # Remove from pending
        self._pending_prefetches.discard(access_key)

    async def get_cached(self, request: FetchRequest) -> Optional[FetchResult]:
        """Get cached result if available and fresh"""
        cache_key = self._get_cache_key(request)

        async with self._cache_lock:
            if cache_key not in self._cache:
                self.prefetch_misses += 1
                return None

            data, timestamp = self._cache[cache_key]
            age_ms = (time.time() - timestamp) * 1000

            if age_ms > self.cache_ttl_ms:
                # Stale, remove from cache
                del self._cache[cache_key]
                self.prefetch_misses += 1
                return None

            self.prefetch_hits += 1
            return FetchResult(
                request=request,
                success=True,
                data=data,
                latency_ms=0.0,
                from_cache=True
            )

    async def cleanup_cache(self):
        """Remove stale cache entries"""
        now = time.time()
        async with self._cache_lock:
            stale_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if (now - timestamp) * 1000 > self.cache_ttl_ms
            ]
            for key in stale_keys:
                del self._cache[key]

    def get_stats(self) -> Dict:
        total_lookups = self.prefetch_hits + self.prefetch_misses
        return {
            "prefetches_triggered": self.prefetches_triggered,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "hit_rate": self.prefetch_hits / max(1, total_lookups),
            "cache_size": len(self._cache),
            "pending_prefetches": len(self._pending_prefetches),
            "patterns_tracked": len(self._transition_counts)
        }


# =============================================================================
# FAILURE-AWARE LOAD BALANCING
# =============================================================================

class FailureAwareLoadBalancer:
    """
    Routes requests away from failing endpoints.

    Features:
    - Tracks endpoint health metrics
    - Circuit breaker pattern
    - Weighted routing based on latency and success rate
    - Automatic recovery detection
    """

    def __init__(
        self,
        circuit_open_threshold: int = 5,    # Failures before opening circuit
        circuit_half_open_after_ms: float = 30000,  # Time before half-open
        health_check_interval_ms: float = 10000,    # Health check frequency
        latency_weight: float = 0.3,        # Weight for latency in routing
        success_weight: float = 0.7         # Weight for success rate in routing
    ):
        self.circuit_open_threshold = circuit_open_threshold
        self.circuit_half_open_after_ms = circuit_half_open_after_ms
        self.health_check_interval_ms = health_check_interval_ms
        self.latency_weight = latency_weight
        self.success_weight = success_weight

        # Endpoint stats: endpoint_pattern -> EndpointStats
        self._endpoints: Dict[str, EndpointStats] = {}
        self._lock = asyncio.Lock()

        # Alternative endpoints mapping: venue -> list of base URLs
        self._alternatives: Dict[str, List[str]] = {
            "kraken": [
                "https://api.kraken.com",
                "https://futures.kraken.com"
            ],
            "polymarket": [
                "https://clob.polymarket.com",
                "https://gamma-api.polymarket.com"
            ],
            "coinbase": [
                "https://api.exchange.coinbase.com",
                "https://api.pro.coinbase.com"
            ]
        }

        # Routing decisions
        self._route_cache: Dict[str, Tuple[str, float]] = {}  # venue -> (chosen_url, timestamp)
        self._route_cache_ttl_ms = 5000

    def _get_endpoint_key(self, url: str, venue: str) -> str:
        """Extract endpoint pattern from URL"""
        # Extract base URL pattern (remove query params and specific IDs)
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        path_parts = parsed.path.split("/")

        # Keep first 3 path segments
        pattern_path = "/".join(path_parts[:4]) if len(path_parts) >= 4 else parsed.path

        return f"{parsed.netloc}{pattern_path}"

    async def record_result(self, request: FetchRequest, result: FetchResult):
        """Record result for endpoint health tracking"""
        endpoint_key = self._get_endpoint_key(request.url, request.venue)

        async with self._lock:
            if endpoint_key not in self._endpoints:
                self._endpoints[endpoint_key] = EndpointStats(
                    url_pattern=endpoint_key,
                    venue=request.venue
                )

            endpoint = self._endpoints[endpoint_key]

            if result.success:
                endpoint.record_success(result.latency_ms)
            else:
                endpoint.record_failure(result.error or "Unknown error")

    def should_skip_endpoint(self, request: FetchRequest) -> bool:
        """Check if endpoint should be skipped due to circuit breaker"""
        endpoint_key = self._get_endpoint_key(request.url, request.venue)

        if endpoint_key not in self._endpoints:
            return False

        endpoint = self._endpoints[endpoint_key]
        return endpoint.is_circuit_open()

    def get_endpoint_health(self, request: FetchRequest) -> EndpointHealth:
        """Get health status of endpoint"""
        endpoint_key = self._get_endpoint_key(request.url, request.venue)

        if endpoint_key not in self._endpoints:
            return EndpointHealth.HEALTHY

        return self._endpoints[endpoint_key].health

    async def get_best_endpoint(self, venue: str) -> Optional[str]:
        """Get best available endpoint for a venue based on health metrics"""
        alternatives = self._alternatives.get(venue, [])

        if not alternatives:
            return None

        # Check cache
        if venue in self._route_cache:
            cached_url, timestamp = self._route_cache[venue]
            age_ms = (time.time() - timestamp) * 1000
            if age_ms < self._route_cache_ttl_ms:
                return cached_url

        # Score each alternative
        scores = []

        for base_url in alternatives:
            # Find any endpoint stats for this base URL
            matching_endpoints = [
                ep for key, ep in self._endpoints.items()
                if key.startswith(base_url.replace("https://", ""))
            ]

            if not matching_endpoints:
                # No data, assume healthy with default score
                scores.append((base_url, 0.5))
                continue

            # Calculate aggregate score
            avg_success = statistics.mean([ep.success_rate for ep in matching_endpoints])
            avg_latency = statistics.mean([ep.avg_latency for ep in matching_endpoints]) if matching_endpoints else 100

            # Normalize latency (lower is better, cap at 1000ms)
            latency_score = 1.0 - min(avg_latency / 1000, 1.0)

            # Combined score
            score = (self.success_weight * avg_success +
                    self.latency_weight * latency_score)

            # Penalty for unhealthy endpoints
            unhealthy_count = sum(1 for ep in matching_endpoints
                                  if ep.health in [EndpointHealth.UNHEALTHY, EndpointHealth.DEAD])
            if unhealthy_count > 0:
                score *= (1 - 0.2 * unhealthy_count / len(matching_endpoints))

            scores.append((base_url, score))

        if not scores:
            return alternatives[0] if alternatives else None

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        best_url = scores[0][0]

        # Cache decision
        self._route_cache[venue] = (best_url, time.time())

        return best_url

    def get_healthy_ratio(self, venue: Optional[str] = None) -> float:
        """Get ratio of healthy endpoints"""
        endpoints = list(self._endpoints.values())

        if venue:
            endpoints = [ep for ep in endpoints if ep.venue == venue]

        if not endpoints:
            return 1.0

        healthy = sum(1 for ep in endpoints if ep.health == EndpointHealth.HEALTHY)
        return healthy / len(endpoints)

    def get_stats(self) -> Dict:
        return {
            "total_endpoints": len(self._endpoints),
            "healthy_ratio": self.get_healthy_ratio(),
            "endpoints": {
                key: ep.to_dict()
                for key, ep in self._endpoints.items()
            },
            "venues": {
                venue: {
                    "healthy_ratio": self.get_healthy_ratio(venue),
                    "endpoints": sum(1 for ep in self._endpoints.values() if ep.venue == venue)
                }
                for venue in set(ep.venue for ep in self._endpoints.values())
            }
        }


# =============================================================================
# ENHANCED PARALLEL API FETCHER
# =============================================================================

class EnhancedParallelAPIFetcher:
    """
    Enhanced parallel API fetcher with advanced features.

    Improvements over base ParallelAPIFetcher:
    1. Request Batching - Combines similar requests
    2. Predictive Pre-fetching - Anticipates data needs
    3. Failure-Aware Load Balancing - Routes around failures
    """

    def __init__(
        self,
        max_concurrent: int = 50,
        enable_batching: bool = True,
        enable_prefetching: bool = True,
        enable_load_balancing: bool = True
    ):
        self.max_concurrent = max_concurrent
        self.enable_batching = enable_batching
        self.enable_prefetching = enable_prefetching
        self.enable_load_balancing = enable_load_balancing

        # Connection pool
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Semaphore for concurrent requests
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Per-venue rate limiters (reuse from base implementation)
        self._rate_limiters: Dict[str, 'AdaptiveRateLimiter'] = {}

        # Enhanced components
        self.batcher = RequestBatcher() if enable_batching else None
        self.prefetcher = PredictivePrefetcher() if enable_prefetching else None
        self.load_balancer = FailureAwareLoadBalancer() if enable_load_balancing else None

        # Latency tracking
        self._latency_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Request builder for prefetching
        self._request_builder: Optional[Callable[[str, str, str], FetchRequest]] = None

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._connector = aiohttp.TCPConnector(
                    limit=self.max_concurrent * 2,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    timeout=timeout
                )
            return self._session

    def _get_rate_limiter(self, venue: str) -> 'AdaptiveRateLimiter':
        """Get or create rate limiter for venue"""
        if venue not in self._rate_limiters:
            rates = {
                "polymarket": 5.0,
                "kraken": 10.0,
                "coinbase": 8.0,
                "default": 5.0
            }
            rate = rates.get(venue, rates["default"])
            self._rate_limiters[venue] = AdaptiveRateLimiter(initial_rate=rate, burst=3)
        return self._rate_limiters[venue]

    def set_request_builder(self, builder: Callable[[str, str, str], FetchRequest]):
        """Set request builder for prefetching"""
        self._request_builder = builder

    async def start(self):
        """Start background tasks"""
        if self._running:
            return

        self._running = True

        # Start cache cleanup task
        if self.prefetcher:
            self._background_tasks.append(
                asyncio.create_task(self._cache_cleanup_loop())
            )

        logger.info("Enhanced API fetcher started")

    async def stop(self):
        """Stop background tasks and cleanup"""
        self._running = False

        for task in self._background_tasks:
            task.cancel()

        self._background_tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None

        if self._connector:
            await self._connector.close()
            self._connector = None

        logger.info("Enhanced API fetcher stopped")

    async def _cache_cleanup_loop(self):
        """Periodically clean up prefetch cache"""
        while self._running:
            try:
                await asyncio.sleep(5)
                if self.prefetcher:
                    await self.prefetcher.cleanup_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def fetch(self, request: FetchRequest) -> FetchResult:
        """
        Fetch a single request with all enhancements.
        """
        # Check load balancer circuit breaker
        if self.load_balancer and self.load_balancer.should_skip_endpoint(request):
            logger.warning(f"Skipping endpoint due to circuit breaker: {request.url}")
            return FetchResult(
                request=request,
                success=False,
                error="Circuit breaker open"
            )

        # Check prefetch cache
        if self.prefetcher and not request.is_prefetch:
            cached = await self.prefetcher.get_cached(request)
            if cached:
                logger.debug(f"Prefetch cache hit for {request.url}")
                return cached

        # Record access pattern
        if self.prefetcher:
            self.prefetcher.record_access(request)

        # Execute with batching or directly
        if self.batcher and request.batch_key:
            result = await self.batcher.add_request(
                request,
                lambda reqs: self._execute_batch(reqs)
            )
        else:
            result = await self._execute_single(request)

        # Record result for load balancing
        if self.load_balancer:
            await self.load_balancer.record_result(request, result)

        # Trigger prefetching for likely next requests
        if self.prefetcher and self._request_builder and result.success:
            prefetch_requests = self.prefetcher.get_prefetch_requests(
                request,
                self._request_builder
            )
            if prefetch_requests:
                # Execute prefetches in background
                asyncio.create_task(self._execute_prefetches(prefetch_requests))

        return result

    async def _execute_single(self, request: FetchRequest) -> FetchResult:
        """Execute a single request with retries"""
        session = await self._get_session()
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
                            self._latency_stats[request.venue].append(latency_ms)

                            if response.status == 200:
                                data = await response.json()
                                rate_limiter.record_success()

                                result.success = True
                                result.data = data
                                result.latency_ms = latency_ms
                                result.retry_count = attempt

                                # Cache if prefetch
                                if request.is_prefetch and self.prefetcher:
                                    await self.prefetcher.cache_result(request, result)

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
                            self._latency_stats[request.venue].append(latency_ms)

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

            # Wait before retry
            if attempt < request.retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

        return result

    async def _execute_batch(self, requests: List[FetchRequest]) -> List[FetchResult]:
        """Execute a batch of requests in parallel"""
        tasks = [self._execute_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(FetchResult(
                    request=requests[i],
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    async def _execute_prefetches(self, requests: List[FetchRequest]):
        """Execute prefetch requests in background"""
        for request in requests:
            try:
                result = await self._execute_single(request)
                if result.success and self.prefetcher:
                    await self.prefetcher.cache_result(request, result)
            except Exception as e:
                logger.debug(f"Prefetch failed: {e}")

    async def fetch_batch(
        self,
        requests: List[FetchRequest],
        return_when: str = "ALL_COMPLETED"
    ) -> List[FetchResult]:
        """Fetch multiple requests with enhancements"""
        # Sort by priority
        sorted_requests = sorted(requests, key=lambda r: r.priority.value)

        tasks = [asyncio.create_task(self.fetch(req)) for req in sorted_requests]

        if return_when == "ALL_COMPLETED":
            results = await asyncio.gather(*tasks, return_exceptions=True)

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

            for task in pending:
                task.cancel()

            results = []
            for task in done:
                try:
                    results.append(task.result())
                except Exception:
                    pass

            return results

    async def fetch_prices_parallel(
        self,
        symbols: List[str],
        venues: List[str],
        get_url_func: Callable[[str, str], str],
        get_headers_func: Optional[Callable[[str], Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch prices with load balancing"""
        requests = []

        for symbol in symbols:
            for venue in venues:
                # Use load balancer to get best endpoint if available
                base_url = None
                if self.load_balancer:
                    base_url = await self.load_balancer.get_best_endpoint(venue)

                url = get_url_func(venue, symbol)
                if base_url:
                    # Substitute base URL
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url)
                    new_parsed = parsed._replace(netloc=urllib.parse.urlparse(base_url).netloc)
                    url = urllib.parse.urlunparse(new_parsed)

                headers = get_headers_func(venue) if get_headers_func else {}

                requests.append(FetchRequest(
                    url=url,
                    venue=venue,
                    symbol=symbol,
                    endpoint_type="ticker",
                    headers=headers,
                    priority=FetchPriority.HIGH,
                    timeout=3.0,
                    batch_key=f"{venue}:ticker"
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

    def get_latency_stats(self, venue: str) -> Dict:
        """Get latency statistics for a venue"""
        if venue not in self._latency_stats or len(self._latency_stats[venue]) < 2:
            return {"avg": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "samples": 0}

        latencies = list(self._latency_stats[venue])
        sorted_latencies = sorted(latencies)

        return {
            "avg": statistics.mean(latencies),
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            "min": min(latencies),
            "max": max(latencies),
            "samples": len(latencies)
        }

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            "latency": {
                venue: self.get_latency_stats(venue)
                for venue in self._latency_stats.keys()
            },
            "rate_limiters": {
                venue: limiter.get_stats()
                for venue, limiter in self._rate_limiters.items()
            }
        }

        if self.batcher:
            stats["batching"] = self.batcher.get_stats()

        if self.prefetcher:
            stats["prefetching"] = self.prefetcher.get_stats()

        if self.load_balancer:
            stats["load_balancing"] = self.load_balancer.get_stats()

        return stats


# =============================================================================
# ADAPTIVE RATE LIMITER (copied from original for completeness)
# =============================================================================

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on API responses"""

    def __init__(self, initial_rate: float = 10.0, burst: int = 5):
        self.base_rate = initial_rate
        self.current_rate = initial_rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()

        self.error_count = 0
        self.success_count = 0
        self.last_429_time: Optional[float] = None
        self.backoff_until: Optional[float] = None

    async def acquire(self) -> float:
        """Acquire a token, returning wait time if needed"""
        async with self._lock:
            now = time.time()

            if self.backoff_until and now < self.backoff_until:
                wait_time = self.backoff_until - now
                await asyncio.sleep(wait_time)
                return wait_time

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

        if self.success_count > 10 and self.current_rate < self.base_rate:
            self.current_rate = min(self.base_rate, self.current_rate * 1.1)

    def record_error(self, status_code: int, retry_after: Optional[float] = None):
        """Record error and adjust rate"""
        self.error_count += 1
        self.success_count = 0

        if status_code == 429:
            self.last_429_time = time.time()

            if retry_after:
                self.backoff_until = time.time() + retry_after
            else:
                backoff = min(60, 2 ** self.error_count)
                self.backoff_until = time.time() + backoff

            self.current_rate = max(0.5, self.current_rate * 0.5)
            logger.warning(f"Rate limit hit. Reducing rate to {self.current_rate:.1f}/s")
        else:
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


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_enhanced_fetcher():
    """Test the enhanced parallel fetcher"""
    print("Testing Enhanced Parallel API Fetcher...\n")

    fetcher = EnhancedParallelAPIFetcher(
        max_concurrent=10,
        enable_batching=True,
        enable_prefetching=True,
        enable_load_balancing=True
    )

    await fetcher.start()

    # Create test requests
    requests = [
        FetchRequest(
            url="https://api.kraken.com/0/public/Ticker?pair=BTCUSD",
            venue="kraken",
            symbol="BTC",
            endpoint_type="ticker",
            priority=FetchPriority.HIGH,
            batch_key="kraken:ticker"
        ),
        FetchRequest(
            url="https://api.kraken.com/0/public/Ticker?pair=ETHUSD",
            venue="kraken",
            symbol="ETH",
            endpoint_type="ticker",
            priority=FetchPriority.NORMAL,
            batch_key="kraken:ticker"
        ),
        FetchRequest(
            url="https://api.kraken.com/0/public/Depth?pair=BTCUSD",
            venue="kraken",
            symbol="BTC",
            endpoint_type="orderbook",
            priority=FetchPriority.HIGH
        ),
    ]

    print("Fetching data with enhanced features...")
    results = await fetcher.fetch_batch(requests)

    for result in results:
        print(f"{result.request.symbol} ({result.request.endpoint_type}): "
              f"success={result.success}, "
              f"latency={result.latency_ms:.1f}ms, "
              f"from_batch={result.from_batch}, "
              f"from_cache={result.from_cache}")

    print("\n--- Statistics ---")
    stats = fetcher.get_stats()

    if "batching" in stats:
        print(f"\nBatching:")
        print(f"  Batches executed: {stats['batching']['batches_executed']}")
        print(f"  Requests batched: {stats['batching']['requests_batched']}")
        print(f"  Deduplicated: {stats['batching']['requests_deduplicated']}")

    if "prefetching" in stats:
        print(f"\nPrefetching:")
        print(f"  Prefetches triggered: {stats['prefetching']['prefetches_triggered']}")
        print(f"  Hit rate: {stats['prefetching']['hit_rate']:.1%}")

    if "load_balancing" in stats:
        print(f"\nLoad Balancing:")
        print(f"  Healthy ratio: {stats['load_balancing']['healthy_ratio']:.1%}")
        print(f"  Endpoints tracked: {stats['load_balancing']['total_endpoints']}")

    await fetcher.stop()
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_fetcher())
