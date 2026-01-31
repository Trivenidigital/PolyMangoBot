"""
Utility Functions Module
Common utilities for retry logic, validation, and helpers
"""

import asyncio
import functools
import logging
import time
from typing import TypeVar, Callable, Optional, Any, List, Type
from dataclasses import dataclass

from exceptions import (
    APIError, APITimeoutError, APIRateLimitError, APIConnectionError,
    PolyMangoBotError
)

logger = logging.getLogger("PolyMangoBot.utils")

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple = (APITimeoutError, APIConnectionError, APIRateLimitError)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay with optional jitter

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential growth
        jitter: Add randomness to delay

    Returns:
        Delay in seconds
    """
    import random

    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        # Add up to 25% jitter
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying async functions with exponential backoff

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback called on each retry (exception, attempt_number)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_backoff_delay(
                            attempt, base_delay, max_delay
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        # Handle rate limit with specific delay
                        if isinstance(e, APIRateLimitError) and e.retry_after_seconds:
                            delay = max(delay, e.retry_after_seconds)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,)
):
    """Decorator for retrying synchronous functions with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_backoff_delay(
                            attempt, base_delay, max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    Prevents repeated calls to failing services.
    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Service is down, calls fail fast
    - HALF_OPEN: Testing if service recovered
    """

    class State:
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_calls: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == self.State.OPEN:
                if self._should_attempt_recovery():
                    self.state = self.State.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise APIError(
                        f"Circuit breaker is OPEN. Service unavailable. "
                        f"Recovery in {self._time_until_recovery():.1f}s"
                    )

            if self.state == self.State.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise APIError("Circuit breaker is HALF_OPEN. Waiting for test call result.")
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == self.State.HALF_OPEN:
                logger.info("Circuit breaker: Service recovered. Closing circuit.")
                self.state = self.State.CLOSED

            self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.State.HALF_OPEN:
                logger.warning("Circuit breaker: Recovery failed. Re-opening circuit.")
                self.state = self.State.OPEN

            elif self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker: Failure threshold ({self.failure_threshold}) reached. "
                    "Opening circuit."
                )
                self.state = self.State.OPEN

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _time_until_recovery(self) -> float:
        """Time until circuit breaker will attempt recovery"""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    @property
    def is_open(self) -> bool:
        return self.state == self.State.OPEN


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, name: str) -> float:
    """Validate that a value is non-negative"""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_percentage(value: float, name: str) -> float:
    """Validate that a value is a valid percentage (0-100)"""
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")
    return value


def validate_fraction(value: float, name: str) -> float:
    """Validate that a value is a valid fraction (0-1)"""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on divide by zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(value, max_val))


class RateLimiter:
    """Simple token bucket rate limiter"""

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class TimingStats:
    """
    Track timing statistics for operations.

    Uses collections.deque for O(1) bounded operations instead of list slicing.
    Caches sorted array for efficient percentile calculations.
    """

    def __init__(self, max_samples: int = 1000):
        from collections import deque
        self._samples: deque = deque(maxlen=max_samples)
        self.max_samples = max_samples
        self._sorted_cache: Optional[List[float]] = None
        self._cache_valid = False

    def record(self, duration_ms: float) -> None:
        """Record a timing sample - O(1) operation"""
        self._samples.append(duration_ms)
        self._cache_valid = False  # Invalidate cache

    def _ensure_sorted_cache(self) -> List[float]:
        """Ensure sorted cache is up to date"""
        if not self._cache_valid or self._sorted_cache is None:
            self._sorted_cache = sorted(self._samples)
            self._cache_valid = True
        return self._sorted_cache

    @property
    def count(self) -> int:
        """Number of samples recorded"""
        return len(self._samples)

    @property
    def samples(self) -> List[float]:
        """Get samples as list (for compatibility)"""
        return list(self._samples)

    @property
    def avg_ms(self) -> float:
        """Average duration in milliseconds"""
        if not self._samples:
            return 0
        return sum(self._samples) / len(self._samples)

    @property
    def min_ms(self) -> float:
        """Minimum duration in milliseconds"""
        if not self._samples:
            return 0
        return min(self._samples)

    @property
    def max_ms(self) -> float:
        """Maximum duration in milliseconds"""
        if not self._samples:
            return 0
        return max(self._samples)

    @property
    def p50_ms(self) -> float:
        """50th percentile (median) duration in milliseconds"""
        if not self._samples:
            return 0
        sorted_samples = self._ensure_sorted_cache()
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    @property
    def p95_ms(self) -> float:
        """95th percentile duration in milliseconds"""
        if not self._samples:
            return 0
        sorted_samples = self._ensure_sorted_cache()
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99_ms(self) -> float:
        """99th percentile duration in milliseconds"""
        if not self._samples:
            return 0
        sorted_samples = self._ensure_sorted_cache()
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> dict:
        """Export stats as dictionary"""
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3)
        }

    def reset(self) -> None:
        """Clear all samples"""
        self._samples.clear()
        self._cache_valid = False
        self._sorted_cache = None


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a value as currency"""
    if value >= 0:
        return f"{symbol}{value:,.2f}"
    return f"-{symbol}{abs(value):,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format a value as percentage"""
    return f"{value:.{decimal_places}f}%"
