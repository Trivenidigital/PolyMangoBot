"""
Tests for utility functions.
Tests retry logic, circuit breaker, rate limiter, and timing stats.
"""

import pytest
import asyncio
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    calculate_backoff_delay,
    retry_async,
    retry_sync,
    CircuitBreaker,
    RateLimiter,
    TimingStats,
    validate_positive,
    validate_non_negative,
    validate_percentage,
    validate_fraction,
    safe_divide,
    clamp,
    format_currency,
    format_percentage
)
from exceptions import APIError, APITimeoutError, APIConnectionError


class TestBackoffDelay:
    """Tests for exponential backoff calculation"""

    def test_initial_delay(self):
        """First attempt should use base delay"""
        delay = calculate_backoff_delay(0, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Delay should grow exponentially"""
        delay_1 = calculate_backoff_delay(1, base_delay=1.0, max_delay=30.0, jitter=False)
        delay_2 = calculate_backoff_delay(2, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay"""
        delay = calculate_backoff_delay(10, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_jitter_adds_randomness(self):
        """Jitter should add variability to delay"""
        delays = [
            calculate_backoff_delay(1, base_delay=1.0, max_delay=30.0, jitter=True)
            for _ in range(10)
        ]
        # Not all delays should be identical with jitter
        assert len(set(delays)) > 1


class TestRetryAsync:
    """Tests for async retry decorator"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Successful function should not retry"""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await success_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on retryable exception"""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, retryable_exceptions=(APITimeoutError,))
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APITimeoutError("timeout")
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Should raise after max attempts"""
        @retry_async(max_attempts=2, base_delay=0.01, retryable_exceptions=(APITimeoutError,))
        async def always_fails():
            raise APITimeoutError("always fails")

        with pytest.raises(APITimeoutError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Should not retry non-retryable exceptions"""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, retryable_exceptions=(APITimeoutError,))
        async def raises_other():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await raises_other()
        assert call_count == 1


class TestCircuitBreaker:
    """Tests for circuit breaker pattern"""

    @pytest.mark.asyncio
    async def test_closed_state_allows_calls(self):
        """Closed circuit should allow calls through"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=0.1)

        async def success_func():
            return "ok"

        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitBreaker.State.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Circuit should open after failure threshold"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=1.0)

        async def failing_func():
            raise Exception("fail")

        # First failure
        with pytest.raises(Exception):
            await cb.call(failing_func)

        # Second failure - should trigger open
        with pytest.raises(Exception):
            await cb.call(failing_func)

        assert cb.state == CircuitBreaker.State.OPEN

    @pytest.mark.asyncio
    async def test_open_state_fails_fast(self):
        """Open circuit should fail fast without calling function"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=10.0)

        async def failing_func():
            raise Exception("fail")

        # Trigger open state
        with pytest.raises(Exception):
            await cb.call(failing_func)

        # Should fail fast with APIError
        with pytest.raises(APIError) as exc_info:
            await cb.call(failing_func)
        assert "OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Circuit should transition to half-open after timeout"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.1)

        async def failing_func():
            raise Exception("fail")

        # Trigger open state
        with pytest.raises(Exception):
            await cb.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open on next call
        async def success_func():
            return "ok"

        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitBreaker.State.CLOSED


class TestRateLimiter:
    """Tests for rate limiter"""

    @pytest.mark.asyncio
    async def test_allows_burst(self):
        """Should allow burst requests"""
        limiter = RateLimiter(rate=10.0, burst=3)

        start = time.time()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.time() - start

        # Burst should complete almost immediately
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limits_after_burst(self):
        """Should rate limit after burst exhausted"""
        limiter = RateLimiter(rate=10.0, burst=1)

        await limiter.acquire()  # Use burst

        start = time.time()
        await limiter.acquire()  # Should wait
        elapsed = time.time() - start

        # Should have waited ~0.1 seconds (1/10 rate)
        assert elapsed >= 0.05


class TestTimingStats:
    """Tests for timing statistics"""

    def test_empty_stats(self):
        """Empty stats should return zeros"""
        stats = TimingStats()
        assert stats.avg_ms == 0
        assert stats.min_ms == 0
        assert stats.max_ms == 0
        assert stats.p95_ms == 0
        assert stats.count == 0

    def test_record_samples(self):
        """Should record and calculate stats correctly"""
        stats = TimingStats()
        for i in range(1, 11):
            stats.record(float(i))

        assert stats.count == 10
        assert stats.avg_ms == 5.5
        assert stats.min_ms == 1.0
        assert stats.max_ms == 10.0

    def test_bounded_storage(self):
        """Should not exceed max_samples"""
        stats = TimingStats(max_samples=5)
        for i in range(10):
            stats.record(float(i))

        assert stats.count == 5
        # Should only have last 5 samples (5-9)
        assert stats.min_ms == 5.0
        assert stats.max_ms == 9.0

    def test_percentiles(self):
        """Should calculate percentiles correctly"""
        stats = TimingStats()
        for i in range(1, 101):
            stats.record(float(i))

        assert stats.p50_ms == 50.0 or stats.p50_ms == 51.0  # Median
        assert stats.p95_ms >= 95.0
        assert stats.p99_ms >= 99.0

    def test_to_dict(self):
        """Should export stats as dictionary"""
        stats = TimingStats()
        stats.record(10.0)
        stats.record(20.0)

        result = stats.to_dict()
        assert "count" in result
        assert "avg_ms" in result
        assert "p95_ms" in result
        assert result["count"] == 2

    def test_reset(self):
        """Should clear all samples on reset"""
        stats = TimingStats()
        stats.record(10.0)
        stats.reset()
        assert stats.count == 0


class TestValidation:
    """Tests for validation functions"""

    def test_validate_positive_valid(self):
        """Should return valid positive value"""
        assert validate_positive(5.0, "test") == 5.0

    def test_validate_positive_invalid(self):
        """Should raise ValueError for non-positive"""
        with pytest.raises(ValueError):
            validate_positive(0, "test")
        with pytest.raises(ValueError):
            validate_positive(-1, "test")

    def test_validate_non_negative_valid(self):
        """Should accept zero and positive"""
        assert validate_non_negative(0, "test") == 0
        assert validate_non_negative(5.0, "test") == 5.0

    def test_validate_non_negative_invalid(self):
        """Should reject negative values"""
        with pytest.raises(ValueError):
            validate_non_negative(-1, "test")

    def test_validate_percentage_valid(self):
        """Should accept 0-100"""
        assert validate_percentage(0, "test") == 0
        assert validate_percentage(50, "test") == 50
        assert validate_percentage(100, "test") == 100

    def test_validate_percentage_invalid(self):
        """Should reject values outside 0-100"""
        with pytest.raises(ValueError):
            validate_percentage(-1, "test")
        with pytest.raises(ValueError):
            validate_percentage(101, "test")

    def test_validate_fraction_valid(self):
        """Should accept 0-1"""
        assert validate_fraction(0, "test") == 0
        assert validate_fraction(0.5, "test") == 0.5
        assert validate_fraction(1.0, "test") == 1.0

    def test_validate_fraction_invalid(self):
        """Should reject values outside 0-1"""
        with pytest.raises(ValueError):
            validate_fraction(-0.1, "test")
        with pytest.raises(ValueError):
            validate_fraction(1.1, "test")


class TestHelperFunctions:
    """Tests for helper functions"""

    def test_safe_divide_normal(self):
        """Should perform normal division"""
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_by_zero(self):
        """Should return default on divide by zero"""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1

    def test_clamp(self):
        """Should clamp value to range"""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_format_currency(self):
        """Should format currency correctly"""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(-100) == "-$100.00"
        assert format_currency(100, "€") == "€100.00"

    def test_format_percentage(self):
        """Should format percentage correctly"""
        assert format_percentage(50.5) == "50.50%"
        assert format_percentage(33.333, 1) == "33.3%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
