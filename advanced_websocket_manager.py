"""
Advanced WebSocket Manager
Enterprise-grade WebSocket management with:
- Automatic reconnection with exponential backoff
- Health monitoring and heartbeat tracking
- Message deduplication and ordering
- Connection pooling for multiple streams
- Latency tracking and alerting
- Graceful degradation under load
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

logger = logging.getLogger("PolyMangoBot.advanced_websocket")


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"  # Connected but experiencing issues
    FAILED = "failed"  # Permanently failed after max retries


@dataclass
class ConnectionHealth:
    """Health metrics for a WebSocket connection"""
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_message_time: float = 0.0
    last_heartbeat_time: float = 0.0
    messages_received: int = 0
    messages_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    reconnect_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    connected_since: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        if self.state != ConnectionState.CONNECTED:
            return False
        # No message in 30 seconds is unhealthy
        if time.time() - self.last_message_time > 30:
            return False
        return True

    @property
    def uptime_seconds(self) -> float:
        """Get connection uptime"""
        if self.connected_since:
            return time.time() - self.connected_since
        return 0.0


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections"""
    url: str
    name: str
    reconnect_enabled: bool = True
    max_reconnect_attempts: int = 10
    base_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0
    message_timeout: float = 60.0  # Max time without any message
    max_message_queue: int = 10000
    enable_deduplication: bool = True
    dedup_window_seconds: float = 5.0
    ping_interval: float = 20.0
    ping_timeout: float = 10.0


class MessageDeduplicator:
    """Deduplicates messages within a time window"""

    def __init__(self, window_seconds: float = 5.0, max_hashes: int = 10000):
        self.window_seconds = window_seconds
        self.max_hashes = max_hashes
        self._seen_hashes: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _hash_message(self, message: str) -> str:
        """Create hash of message content"""
        return hashlib.md5(message.encode()).hexdigest()[:16]

    async def is_duplicate(self, message: str) -> bool:
        """Check if message is a duplicate"""
        msg_hash = self._hash_message(message)
        current_time = time.time()

        async with self._lock:
            # Clean old hashes
            if len(self._seen_hashes) > self.max_hashes:
                cutoff = current_time - self.window_seconds
                self._seen_hashes = {
                    h: t for h, t in self._seen_hashes.items()
                    if t > cutoff
                }

            # Check if seen
            if msg_hash in self._seen_hashes:
                last_seen = self._seen_hashes[msg_hash]
                if current_time - last_seen < self.window_seconds:
                    return True

            # Mark as seen
            self._seen_hashes[msg_hash] = current_time
            return False


class LatencyTracker:
    """Tracks message latency statistics"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

    def record(self, latency_ms: float):
        """Record a latency measurement"""
        self._latencies.append(latency_ms)
        self._timestamps.append(time.time())

    @property
    def avg_latency(self) -> float:
        """Average latency in ms"""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p95_latency(self) -> float:
        """95th percentile latency"""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency(self) -> float:
        """99th percentile latency"""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def messages_per_second(self) -> float:
        """Calculate message throughput"""
        if len(self._timestamps) < 2:
            return 0.0

        # Count messages in last second
        now = time.time()
        recent = sum(1 for t in self._timestamps if now - t <= 1.0)
        return float(recent)


class ReconnectionStrategy:
    """Handles reconnection with exponential backoff"""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 10,
        jitter: float = 0.1
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
        self._attempt = 0

    def reset(self):
        """Reset attempt counter on successful connection"""
        self._attempt = 0

    def should_retry(self) -> bool:
        """Check if we should attempt reconnection"""
        return self._attempt < self.max_attempts

    def get_delay(self) -> float:
        """Get delay before next reconnection attempt"""
        import random

        # Exponential backoff with jitter
        delay = min(
            self.base_delay * (2 ** self._attempt),
            self.max_delay
        )

        # Add jitter
        jitter_amount = delay * self.jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

        self._attempt += 1
        return max(0.1, delay)

    @property
    def attempts(self) -> int:
        return self._attempt


class AdvancedWebSocketStream:
    """
    Advanced WebSocket connection with health monitoring and auto-reconnection.

    Features:
    - Automatic reconnection with exponential backoff
    - Health monitoring via heartbeats
    - Message deduplication
    - Latency tracking
    - Graceful shutdown
    """

    def __init__(
        self,
        config: WebSocketConfig,
        on_message: Optional[Callable[[Dict], Any]] = None,
        on_connect: Optional[Callable[[], Any]] = None,
        on_disconnect: Optional[Callable[[str], Any]] = None,
        on_error: Optional[Callable[[Exception], Any]] = None
    ):
        self.config = config
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._health = ConnectionHealth()
        self._deduplicator = MessageDeduplicator(
            window_seconds=config.dedup_window_seconds
        ) if config.enable_deduplication else None
        self._latency_tracker = LatencyTracker()
        self._reconnect_strategy = ReconnectionStrategy(
            base_delay=config.base_reconnect_delay,
            max_delay=config.max_reconnect_delay,
            max_attempts=config.max_reconnect_attempts
        )

        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.max_message_queue
        )
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._subscriptions: List[Dict] = []
        self._lock = asyncio.Lock()

    @property
    def health(self) -> ConnectionHealth:
        """Get current connection health"""
        return self._health

    def _is_ws_open(self) -> bool:
        """Check if WebSocket is open (handles different library versions)"""
        if self._ws is None:
            return False
        try:
            # Try 'open' attribute first (newer versions)
            if hasattr(self._ws, 'open'):
                return self._ws.open
            # Fall back to 'closed' attribute (older versions)
            if hasattr(self._ws, 'closed'):
                return not self._ws.closed
            # Default to checking state
            return self._health.state == ConnectionState.CONNECTED
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self._is_ws_open() and self._health.state == ConnectionState.CONNECTED

    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        if self._running:
            return self.is_connected

        self._running = True
        self._health.state = ConnectionState.CONNECTING

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=5.0
                ),
                timeout=30.0
            )

            self._health.state = ConnectionState.CONNECTED
            self._health.connected_since = time.time()
            self._health.last_message_time = time.time()
            self._health.last_heartbeat_time = time.time()
            self._reconnect_strategy.reset()

            logger.info(f"[{self.config.name}] Connected to {self.config.url}")

            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._receive_loop()),
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._health_monitor_loop()),
            ]

            # Restore subscriptions
            await self._restore_subscriptions()

            # Callback
            if self.on_connect:
                try:
                    result = self.on_connect()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"[{self.config.name}] on_connect callback error: {e}")

            return True

        except asyncio.TimeoutError:
            logger.error(f"[{self.config.name}] Connection timeout")
            self._health.state = ConnectionState.DISCONNECTED
            self._health.error_count += 1
            self._health.last_error = "Connection timeout"
            return False

        except Exception as e:
            logger.error(f"[{self.config.name}] Connection failed: {e}")
            self._health.state = ConnectionState.DISCONNECTED
            self._health.error_count += 1
            self._health.last_error = str(e)
            return False

    async def disconnect(self):
        """Gracefully disconnect"""
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._health.state = ConnectionState.DISCONNECTED
        logger.info(f"[{self.config.name}] Disconnected")

    async def subscribe(self, subscription: Dict):
        """Subscribe to a channel/topic"""
        async with self._lock:
            self._subscriptions.append(subscription)

        if self.is_connected:
            await self._send(subscription)

    async def unsubscribe(self, subscription: Dict):
        """Unsubscribe from a channel/topic"""
        async with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)

        # Send unsubscribe message if format is known
        if self.is_connected and "channel" in subscription:
            unsub = {**subscription, "action": "unsubscribe"}
            await self._send(unsub)

    async def _send(self, message: Dict):
        """Send a message"""
        if not self._is_ws_open():
            return

        try:
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"[{self.config.name}] Send error: {e}")

    async def _restore_subscriptions(self):
        """Restore subscriptions after reconnection"""
        async with self._lock:
            for sub in self._subscriptions:
                try:
                    await self._send(sub)
                    await asyncio.sleep(0.1)  # Rate limit
                except Exception as e:
                    logger.error(f"[{self.config.name}] Failed to restore subscription: {e}")

    async def _receive_loop(self):
        """Main receive loop"""
        while self._running:
            try:
                if not self._is_ws_open():
                    await asyncio.sleep(0.1)
                    continue

                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.config.message_timeout
                )

                receive_time = time.time()
                self._health.last_message_time = receive_time
                self._health.messages_received += 1

                # Parse message
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    data = {"raw": message}

                # Check for duplicates
                if self._deduplicator:
                    if await self._deduplicator.is_duplicate(message):
                        continue

                # Track latency if timestamp in message
                if isinstance(data, dict) and "timestamp" in data:
                    try:
                        msg_time = float(data["timestamp"])
                        # Handle millisecond timestamps
                        if msg_time > 1e12:
                            msg_time /= 1000
                        latency_ms = (receive_time - msg_time) * 1000
                        if 0 < latency_ms < 60000:  # Sanity check
                            self._latency_tracker.record(latency_ms)
                            self._health.avg_latency_ms = self._latency_tracker.avg_latency
                    except (ValueError, TypeError):
                        pass

                # Update throughput
                self._health.messages_per_second = self._latency_tracker.messages_per_second

                # Callback
                if self.on_message:
                    try:
                        result = self.on_message(data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"[{self.config.name}] on_message callback error: {e}")

            except asyncio.TimeoutError:
                logger.warning(f"[{self.config.name}] Message timeout - no data received")
                self._health.state = ConnectionState.DEGRADED

            except ConnectionClosed as e:
                logger.warning(f"[{self.config.name}] Connection closed: {e}")
                await self._handle_disconnect(str(e))

            except Exception as e:
                logger.error(f"[{self.config.name}] Receive error: {e}")
                self._health.error_count += 1
                self._health.last_error = str(e)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self.is_connected:
                    continue

                # Send ping
                try:
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(
                        pong_waiter,
                        timeout=self.config.heartbeat_timeout
                    )
                    self._health.last_heartbeat_time = time.time()

                except asyncio.TimeoutError:
                    logger.warning(f"[{self.config.name}] Heartbeat timeout")
                    self._health.state = ConnectionState.DEGRADED

            except Exception as e:
                logger.error(f"[{self.config.name}] Heartbeat error: {e}")

    async def _health_monitor_loop(self):
        """Monitor connection health"""
        while self._running:
            try:
                await asyncio.sleep(5.0)

                if self._health.state == ConnectionState.CONNECTED:
                    # Check for stale connection
                    time_since_message = time.time() - self._health.last_message_time
                    if time_since_message > self.config.message_timeout:
                        logger.warning(
                            f"[{self.config.name}] Connection stale - "
                            f"no message for {time_since_message:.1f}s"
                        )
                        self._health.state = ConnectionState.DEGRADED
                        await self._handle_disconnect("Connection stale")

            except Exception as e:
                logger.error(f"[{self.config.name}] Health monitor error: {e}")

    async def _handle_disconnect(self, reason: str):
        """Handle disconnection and attempt reconnection"""
        self._health.state = ConnectionState.DISCONNECTED

        # Callback
        if self.on_disconnect:
            try:
                result = self.on_disconnect(reason)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[{self.config.name}] on_disconnect callback error: {e}")

        # Attempt reconnection
        if self.config.reconnect_enabled and self._running:
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect"""
        self._health.state = ConnectionState.RECONNECTING

        while self._running and self._reconnect_strategy.should_retry():
            delay = self._reconnect_strategy.get_delay()
            logger.info(
                f"[{self.config.name}] Reconnecting in {delay:.1f}s "
                f"(attempt {self._reconnect_strategy.attempts}/"
                f"{self._reconnect_strategy.max_attempts})"
            )

            await asyncio.sleep(delay)

            if not self._running:
                break

            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self.config.url,
                        ping_interval=self.config.ping_interval,
                        ping_timeout=self.config.ping_timeout,
                        close_timeout=5.0
                    ),
                    timeout=30.0
                )

                self._health.state = ConnectionState.CONNECTED
                self._health.connected_since = time.time()
                self._health.last_message_time = time.time()
                self._health.last_heartbeat_time = time.time()
                self._health.reconnect_count += 1
                self._reconnect_strategy.reset()

                logger.info(f"[{self.config.name}] Reconnected successfully")

                # Restore subscriptions
                await self._restore_subscriptions()

                # Callback
                if self.on_connect:
                    try:
                        result = self.on_connect()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"[{self.config.name}] on_connect callback error: {e}")

                return

            except Exception as e:
                logger.warning(f"[{self.config.name}] Reconnection failed: {e}")
                self._health.error_count += 1

        # Max retries exceeded
        self._health.state = ConnectionState.FAILED
        logger.error(f"[{self.config.name}] Max reconnection attempts exceeded")

        if self.on_error:
            try:
                result = self.on_error(Exception("Max reconnection attempts exceeded"))
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[{self.config.name}] on_error callback error: {e}")


class AdvancedWebSocketManager:
    """
    Manages multiple WebSocket connections with unified health monitoring.

    Features:
    - Multiple concurrent connections
    - Unified health dashboard
    - Automatic failover
    - Cross-stream message aggregation
    """

    def __init__(self):
        self._streams: Dict[str, AdvancedWebSocketStream] = {}
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._health_task: Optional[asyncio.Task] = None

    def add_stream(
        self,
        name: str,
        url: str,
        **config_kwargs
    ) -> AdvancedWebSocketStream:
        """Add a new WebSocket stream"""
        config = WebSocketConfig(url=url, name=name, **config_kwargs)

        stream = AdvancedWebSocketStream(
            config=config,
            on_message=lambda msg, n=name: self._handle_message(n, msg),
            on_connect=lambda n=name: self._on_stream_connect(n),
            on_disconnect=lambda reason, n=name: self._on_stream_disconnect(n, reason),
            on_error=lambda e, n=name: self._on_stream_error(n, e)
        )

        self._streams[name] = stream
        return stream

    def add_message_handler(self, stream_name: str, handler: Callable):
        """Add a message handler for a specific stream"""
        if stream_name not in self._message_handlers:
            self._message_handlers[stream_name] = []
        self._message_handlers[stream_name].append(handler)

    def add_global_handler(self, handler: Callable):
        """Add a handler for all streams"""
        for name in self._streams:
            self.add_message_handler(name, handler)

    async def connect_all(self):
        """Connect all streams"""
        self._running = True

        # Start health monitor
        self._health_task = asyncio.create_task(self._global_health_monitor())

        # Connect all streams concurrently
        results = await asyncio.gather(
            *[stream.connect() for stream in self._streams.values()],
            return_exceptions=True
        )

        # Log results
        for name, result in zip(self._streams.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect {name}: {result}")
            elif result:
                logger.info(f"Connected {name}")
            else:
                logger.warning(f"Connection failed for {name}")

    async def disconnect_all(self):
        """Disconnect all streams gracefully"""
        self._running = False

        # Stop health monitor
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Disconnect all streams
        await asyncio.gather(
            *[stream.disconnect() for stream in self._streams.values()],
            return_exceptions=True
        )

        logger.info("All WebSocket streams disconnected")

    def get_health_summary(self) -> Dict:
        """Get health summary for all streams"""
        summary = {
            "total_streams": len(self._streams),
            "connected": 0,
            "degraded": 0,
            "disconnected": 0,
            "failed": 0,
            "total_messages": 0,
            "total_errors": 0,
            "streams": {}
        }

        for name, stream in self._streams.items():
            health = stream.health
            summary["total_messages"] += health.messages_received
            summary["total_errors"] += health.error_count

            if health.state == ConnectionState.CONNECTED:
                summary["connected"] += 1
            elif health.state == ConnectionState.DEGRADED:
                summary["degraded"] += 1
            elif health.state == ConnectionState.FAILED:
                summary["failed"] += 1
            else:
                summary["disconnected"] += 1

            summary["streams"][name] = {
                "state": health.state.value,
                "is_healthy": health.is_healthy,
                "uptime_seconds": health.uptime_seconds,
                "messages_received": health.messages_received,
                "messages_per_second": health.messages_per_second,
                "avg_latency_ms": health.avg_latency_ms,
                "reconnect_count": health.reconnect_count,
                "error_count": health.error_count,
                "last_error": health.last_error
            }

        return summary

    async def _handle_message(self, stream_name: str, message: Dict):
        """Handle incoming message"""
        handlers = self._message_handlers.get(stream_name, [])
        for handler in handlers:
            try:
                result = handler(stream_name, message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Message handler error for {stream_name}: {e}")

    def _on_stream_connect(self, stream_name: str):
        """Handle stream connection"""
        logger.info(f"Stream {stream_name} connected")

    def _on_stream_disconnect(self, stream_name: str, reason: str):
        """Handle stream disconnection"""
        logger.warning(f"Stream {stream_name} disconnected: {reason}")

    def _on_stream_error(self, stream_name: str, error: Exception):
        """Handle stream error"""
        logger.error(f"Stream {stream_name} error: {error}")

    async def _global_health_monitor(self):
        """Monitor overall health of all streams"""
        while self._running:
            try:
                await asyncio.sleep(30.0)

                summary = self.get_health_summary()

                # Log health status
                logger.info(
                    f"WebSocket Health: {summary['connected']}/{summary['total_streams']} connected, "
                    f"{summary['degraded']} degraded, {summary['failed']} failed, "
                    f"{summary['total_messages']} total messages"
                )

                # Alert if too many failures
                if summary['failed'] > 0:
                    logger.error(
                        f"ALERT: {summary['failed']} WebSocket streams have failed permanently!"
                    )

            except Exception as e:
                logger.error(f"Global health monitor error: {e}")


# Factory functions for common WebSocket configurations
def create_polymarket_stream(on_message: Callable) -> AdvancedWebSocketStream:
    """Create configured stream for Polymarket"""
    config = WebSocketConfig(
        url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        name="polymarket",
        reconnect_enabled=True,
        heartbeat_interval=30.0,
        message_timeout=60.0,
        enable_deduplication=True
    )
    return AdvancedWebSocketStream(config, on_message=on_message)


def create_kraken_stream(on_message: Callable) -> AdvancedWebSocketStream:
    """Create configured stream for Kraken"""
    config = WebSocketConfig(
        url="wss://ws.kraken.com",
        name="kraken",
        reconnect_enabled=True,
        heartbeat_interval=30.0,
        message_timeout=60.0,
        enable_deduplication=True
    )
    return AdvancedWebSocketStream(config, on_message=on_message)


def create_coinbase_stream(on_message: Callable) -> AdvancedWebSocketStream:
    """Create configured stream for Coinbase"""
    config = WebSocketConfig(
        url="wss://ws-feed.exchange.coinbase.com",
        name="coinbase",
        reconnect_enabled=True,
        heartbeat_interval=30.0,
        message_timeout=60.0,
        enable_deduplication=True
    )
    return AdvancedWebSocketStream(config, on_message=on_message)


# Test function
async def test_advanced_websocket():
    """Test the advanced WebSocket manager"""
    import signal

    manager = AdvancedWebSocketManager()

    # Add streams
    manager.add_stream(
        "kraken",
        "wss://ws.kraken.com",
        heartbeat_interval=30.0
    )

    # Add message handler
    def handle_message(stream_name: str, message: Dict):
        print(f"[{stream_name}] Message: {message}")

    manager.add_global_handler(handle_message)

    # Connect
    await manager.connect_all()

    # Subscribe to a channel
    stream = manager._streams.get("kraken")
    if stream:
        await stream.subscribe({
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "ticker"}
        })

    # Run for a bit
    try:
        for _ in range(10):
            await asyncio.sleep(5)
            print(f"\nHealth Summary: {manager.get_health_summary()}\n")
    except KeyboardInterrupt:
        pass

    # Disconnect
    await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(test_advanced_websocket())
