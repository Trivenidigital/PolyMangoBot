"""
Async Logging Module
Non-blocking logging for async hot paths.

This module provides:
- Async-safe logging that doesn't block the event loop
- Log queue with background writer
- Structured logging support
- Automatic log sanitization
"""

import asyncio
import logging
import sys
import time
import threading
import queue
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import contextmanager

from security import sanitize_for_logging


# =============================================================================
# LOG RECORD STRUCTURE
# =============================================================================

class LogLevel(Enum):
    """Log levels matching Python logging"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class AsyncLogRecord:
    """Structured log record for async logging"""
    timestamp: float
    level: LogLevel
    logger_name: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)
    exc_info: Optional[tuple] = None

    def format(self, include_timestamp: bool = True) -> str:
        """Format the log record as a string"""
        parts = []

        if include_timestamp:
            dt = datetime.fromtimestamp(self.timestamp)
            parts.append(dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

        parts.append(f"| {self.logger_name:30s}")
        parts.append(f"| {self.level.name:8s}")
        parts.append(f"| {self.message}")

        if self.extra:
            extra_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
            parts.append(f"| {extra_str}")

        return " ".join(parts)


# =============================================================================
# ASYNC LOG QUEUE
# =============================================================================

class AsyncLogQueue:
    """
    Thread-safe async log queue with background writer.

    Logs are queued and written by a background thread to avoid
    blocking the async event loop.
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        flush_interval: float = 0.1,
        handlers: Optional[list] = None
    ):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._flush_interval = flush_interval
        self._handlers = handlers or [logging.StreamHandler(sys.stdout)]
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None
        self._dropped_count = 0

    def start(self):
        """Start the background writer thread"""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="AsyncLogWriter"
        )
        self._writer_thread.start()

    def stop(self, timeout: float = 5.0):
        """Stop the background writer thread"""
        self._running = False
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)

    def enqueue(self, record: AsyncLogRecord) -> bool:
        """
        Add a log record to the queue.

        Returns True if enqueued, False if queue is full.
        """
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self._dropped_count += 1
            return False

    def _writer_loop(self):
        """Background thread that writes logs"""
        batch = []
        last_flush = time.time()

        while self._running or not self._queue.empty():
            try:
                # Get records with timeout
                try:
                    record = self._queue.get(timeout=0.05)
                    batch.append(record)
                except queue.Empty:
                    pass

                # Flush if interval elapsed or batch is large
                now = time.time()
                if batch and (now - last_flush >= self._flush_interval or len(batch) >= 100):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                # Don't let writer thread die on errors
                print(f"AsyncLogQueue writer error: {e}", file=sys.stderr)

        # Final flush
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list):
        """Write a batch of log records"""
        for record in batch:
            formatted = record.format()
            for handler in self._handlers:
                try:
                    # Create a standard logging record for handler compatibility
                    log_record = logging.LogRecord(
                        name=record.logger_name,
                        level=record.level.value,
                        pathname="",
                        lineno=0,
                        msg=formatted,
                        args=(),
                        exc_info=record.exc_info
                    )
                    handler.emit(log_record)
                except Exception:
                    pass

    @property
    def dropped_count(self) -> int:
        """Number of log records dropped due to full queue"""
        return self._dropped_count

    @property
    def queue_size(self) -> int:
        """Current queue size"""
        return self._queue.qsize()


# =============================================================================
# ASYNC LOGGER
# =============================================================================

class AsyncLogger:
    """
    Async-safe logger that doesn't block the event loop.

    Usage:
        logger = AsyncLogger("MyComponent")
        logger.info("Operation completed", extra={"duration_ms": 123})
    """

    # Shared queue for all async loggers
    _shared_queue: Optional[AsyncLogQueue] = None
    _initialized: bool = False

    @classmethod
    def initialize(
        cls,
        handlers: Optional[list] = None,
        max_queue_size: int = 10000
    ):
        """Initialize the shared async logging system"""
        if cls._initialized:
            return

        cls._shared_queue = AsyncLogQueue(
            max_queue_size=max_queue_size,
            handlers=handlers
        )
        cls._shared_queue.start()
        cls._initialized = True

    @classmethod
    def shutdown(cls, timeout: float = 5.0):
        """Shutdown the async logging system"""
        if cls._shared_queue:
            cls._shared_queue.stop(timeout=timeout)
        cls._initialized = False

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        sanitize: bool = True
    ):
        self.name = name
        self.level = level
        self.sanitize = sanitize

        # Ensure system is initialized
        if not AsyncLogger._initialized:
            AsyncLogger.initialize()

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message at this level should be logged"""
        return level.value >= self.level.value

    def _create_record(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[tuple] = None
    ) -> AsyncLogRecord:
        """Create a log record"""
        # Sanitize message and extra data if enabled
        if self.sanitize:
            message = sanitize_for_logging(message)
            if extra:
                extra = {k: sanitize_for_logging(v) for k, v in extra.items()}

        return AsyncLogRecord(
            timestamp=time.time(),
            level=level,
            logger_name=self.name,
            message=message,
            extra=extra or {},
            exc_info=exc_info
        )

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[tuple] = None
    ):
        """Internal log method"""
        if not self._should_log(level):
            return

        record = self._create_record(level, message, extra, exc_info)

        if AsyncLogger._shared_queue:
            AsyncLogger._shared_queue.enqueue(record)
        else:
            # Fallback to standard logging if queue not available
            print(record.format())

    def debug(self, message: str, **extra):
        """Log at DEBUG level"""
        self._log(LogLevel.DEBUG, message, extra)

    def info(self, message: str, **extra):
        """Log at INFO level"""
        self._log(LogLevel.INFO, message, extra)

    def warning(self, message: str, **extra):
        """Log at WARNING level"""
        self._log(LogLevel.WARNING, message, extra)

    def error(self, message: str, exc_info: bool = False, **extra):
        """Log at ERROR level"""
        exc = sys.exc_info() if exc_info else None
        self._log(LogLevel.ERROR, message, extra, exc)

    def critical(self, message: str, exc_info: bool = False, **extra):
        """Log at CRITICAL level"""
        exc = sys.exc_info() if exc_info else None
        self._log(LogLevel.CRITICAL, message, extra, exc)

    def exception(self, message: str, **extra):
        """Log an exception with traceback"""
        self._log(LogLevel.ERROR, message, extra, sys.exc_info())


# =============================================================================
# STRUCTURED LOGGING CONTEXT
# =============================================================================

class LogContext:
    """
    Context manager for adding structured context to logs.

    Usage:
        with LogContext(trade_id="abc123", venue="kraken"):
            logger.info("Processing trade")  # Includes trade_id and venue
    """

    _context: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    @contextmanager
    def add(cls, **context):
        """Add context for the duration of the with block"""
        old_context = cls._context.copy()
        with cls._lock:
            cls._context.update(context)
        try:
            yield
        finally:
            with cls._lock:
                cls._context = old_context

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current context"""
        return cls._context.copy()

    @classmethod
    def clear(cls):
        """Clear all context"""
        with cls._lock:
            cls._context.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_async_logger(name: str, level: LogLevel = LogLevel.INFO) -> AsyncLogger:
    """Get an async logger instance"""
    return AsyncLogger(name, level)


def init_async_logging(
    handlers: Optional[list] = None,
    level: LogLevel = LogLevel.INFO
):
    """Initialize the async logging system"""
    if handlers is None:
        # Create default handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(message)s'
        ))
        handlers = [handler]

    AsyncLogger.initialize(handlers=handlers)


def shutdown_async_logging(timeout: float = 5.0):
    """
    Shutdown the async logging system.

    This function can be called from both sync and async contexts.
    For async contexts, it returns a coroutine that can be awaited.
    """
    AsyncLogger.shutdown(timeout=timeout)


async def async_shutdown_logging(timeout: float = 5.0):
    """
    Async-compatible shutdown for use with await.

    This wraps the sync shutdown in an executor to avoid blocking.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: AsyncLogger.shutdown(timeout=timeout))


# =============================================================================
# LATENCY-AWARE LOGGER
# =============================================================================

class LatencyAwareLogger(AsyncLogger):
    """
    Logger that tracks and logs operation latencies.

    Usage:
        logger = LatencyAwareLogger("API")
        with logger.timed("fetch_data"):
            data = await fetch()
        # Logs: "fetch_data completed | duration_ms=123.45"
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        slow_threshold_ms: float = 100.0
    ):
        super().__init__(name, level)
        self.slow_threshold_ms = slow_threshold_ms

    @contextmanager
    def timed(self, operation: str, log_always: bool = False):
        """
        Context manager that times an operation.

        Args:
            operation: Name of the operation
            log_always: If True, always log. If False, only log slow operations.
        """
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_ms = elapsed_ns / 1_000_000

            if log_always or elapsed_ms > self.slow_threshold_ms:
                level = LogLevel.WARNING if elapsed_ms > self.slow_threshold_ms else LogLevel.DEBUG
                self._log(
                    level,
                    f"{operation} completed",
                    {"duration_ms": round(elapsed_ms, 3)}
                )
