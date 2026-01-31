"""
Health Check Module
Comprehensive health monitoring for all bot components.

This module provides:
- Component health checks
- System health status
- Readiness and liveness probes
- Health reporting for monitoring
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# psutil is optional - gracefully handle if not installed
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("PolyMangoBot.health")


# =============================================================================
# HEALTH STATUS
# =============================================================================

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "last_check_ago_seconds": round(time.time() - self.last_check, 1),
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details
        }


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    components: List[ComponentHealth]
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Ready if healthy or degraded"""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "uptime_human": str(timedelta(seconds=int(self.uptime_seconds))),
            "timestamp": self.timestamp,
            "components": [c.to_dict() for c in self.components]
        }


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """
    Manages health checks for all bot components.

    Usage:
        checker = HealthChecker()
        checker.register("api", api_health_check)
        checker.register("websocket", ws_health_check)

        health = await checker.check_all()
        print(health.to_dict())
    """

    def __init__(self, start_time: Optional[float] = None):
        self._start_time = start_time or time.time()
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, ComponentHealth] = {}
        self._check_timeout = 5.0  # seconds

    def register(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
        is_async: bool = True
    ):
        """
        Register a health check function.

        Args:
            name: Component name
            check_func: Function that returns ComponentHealth
            is_async: Whether the function is async
        """
        self._checks[name] = (check_func, is_async)

    def unregister(self, name: str):
        """Unregister a health check"""
        self._checks.pop(name, None)

    async def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component"""
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="No health check registered"
            )

        check_func, is_async = self._checks[name]
        start = time.perf_counter_ns()

        try:
            if is_async:
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=self._check_timeout
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    check_func
                )

            result.latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            self._last_results[name] = result
            return result

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self._check_timeout}s",
                latency_ms=self._check_timeout * 1000
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.perf_counter_ns() - start) / 1_000_000
            )

    async def check_all(self) -> SystemHealth:
        """Check health of all registered components"""
        tasks = [
            self.check_component(name)
            for name in self._checks
        ]

        # Add system checks
        tasks.append(self._check_system_resources())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        components = []
        for result in results:
            if isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=str(result)
                ))
            elif isinstance(result, ComponentHealth):
                components.append(result)

        # Determine overall status
        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall,
            components=components,
            uptime_seconds=time.time() - self._start_time
        )

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        # If psutil is not available, return unknown status
        if not PSUTIL_AVAILABLE:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed - resource monitoring unavailable"
            )

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }

            # Determine status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage"
            elif cpu_percent > 75 or memory.percent > 80 or disk.percent > 85:
                status = HealthStatus.DEGRADED
                message = "High resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Resources OK"

            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check resources: {e}"
            )

    def get_last_result(self, name: str) -> Optional[ComponentHealth]:
        """Get the last health check result for a component"""
        return self._last_results.get(name)

    @property
    def uptime_seconds(self) -> float:
        """Get current uptime in seconds"""
        return time.time() - self._start_time


# =============================================================================
# COMPONENT HEALTH CHECK FACTORIES
# =============================================================================

def create_api_health_check(api_manager) -> Callable:
    """Create a health check for the API manager"""

    async def check() -> ComponentHealth:
        try:
            is_connected = (
                api_manager.polymarket.is_connected and
                api_manager.exchange.is_connected
            )

            if is_connected:
                return ComponentHealth(
                    name="api_manager",
                    status=HealthStatus.HEALTHY,
                    message="All API connections active",
                    details={
                        "polymarket_connected": api_manager.polymarket.is_connected,
                        "exchange_connected": api_manager.exchange.is_connected,
                        "exchange_name": api_manager.exchange.exchange_name
                    }
                )
            else:
                return ComponentHealth(
                    name="api_manager",
                    status=HealthStatus.DEGRADED,
                    message="Some API connections inactive",
                    details={
                        "polymarket_connected": api_manager.polymarket.is_connected,
                        "exchange_connected": api_manager.exchange.is_connected
                    }
                )
        except Exception as e:
            return ComponentHealth(
                name="api_manager",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    return check


def create_websocket_health_check(ws_manager) -> Callable:
    """Create a health check for the WebSocket manager"""

    async def check() -> ComponentHealth:
        try:
            stats = ws_manager.get_statistics()

            polymarket_ok = stats.get("polymarket_connected", False)
            exchange_connections = stats.get("exchange_connections", {})
            any_exchange_ok = any(exchange_connections.values())

            if polymarket_ok and any_exchange_ok:
                status = HealthStatus.HEALTHY
                message = "WebSocket connections active"
            elif polymarket_ok or any_exchange_ok:
                status = HealthStatus.DEGRADED
                message = "Some WebSocket connections inactive"
            else:
                status = HealthStatus.UNHEALTHY
                message = "All WebSocket connections down"

            return ComponentHealth(
                name="websocket_manager",
                status=status,
                message=message,
                details={
                    "polymarket_connected": polymarket_ok,
                    "exchange_connections": exchange_connections,
                    "total_events": stats.get("total_events", 0),
                    "is_running": stats.get("is_running", False)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="websocket_manager",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    return check


def create_circuit_breaker_health_check(circuit_breakers: Dict) -> Callable:
    """Create a health check for circuit breakers"""

    async def check() -> ComponentHealth:
        try:
            states = {}
            open_count = 0

            for name, cb in circuit_breakers.items():
                states[name] = cb.state
                if cb.is_open:
                    open_count += 1

            if open_count == 0:
                status = HealthStatus.HEALTHY
                message = "All circuit breakers closed"
            elif open_count < len(circuit_breakers):
                status = HealthStatus.DEGRADED
                message = f"{open_count} circuit breaker(s) open"
            else:
                status = HealthStatus.UNHEALTHY
                message = "All circuit breakers open"

            return ComponentHealth(
                name="circuit_breakers",
                status=status,
                message=message,
                details={"states": states, "open_count": open_count}
            )
        except Exception as e:
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )

    return check


def create_execution_health_check(order_executor) -> Callable:
    """Create a health check for order execution"""

    async def check() -> ComponentHealth:
        try:
            stats = order_executor.get_execution_stats()
            active_trades = stats.get("active_trades", 0)
            total_trades = stats.get("total_trades", 0)

            # Check circuit breakers
            cb_states = stats.get("circuit_breakers", {})
            cb_open = sum(1 for s in cb_states.values() if s == "open")

            if cb_open > 0:
                status = HealthStatus.DEGRADED
                message = f"{cb_open} venue circuit breaker(s) open"
            elif active_trades > 5:
                status = HealthStatus.DEGRADED
                message = f"High active trade count: {active_trades}"
            else:
                status = HealthStatus.HEALTHY
                message = "Order execution healthy"

            return ComponentHealth(
                name="order_executor",
                status=status,
                message=message,
                details={
                    "active_trades": active_trades,
                    "total_trades": total_trades,
                    "avg_execution_ms": round(stats.get("avg_execution_ms", 0), 2),
                    "circuit_breakers": cb_states
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="order_executor",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    return check


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================

class HealthEndpoint:
    """
    Simple health endpoint for monitoring.

    Can be used with aiohttp or other async frameworks.
    """

    def __init__(self, health_checker: HealthChecker):
        self.checker = health_checker

    async def liveness(self) -> Dict[str, Any]:
        """
        Liveness probe - is the application running?

        Returns 200 if alive, used by orchestrators to detect crashes.
        """
        return {
            "status": "alive",
            "uptime_seconds": self.checker.uptime_seconds
        }

    async def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe - is the application ready to accept traffic?

        Returns 200 if ready, 503 if not ready.
        """
        health = await self.checker.check_all()
        return {
            "ready": health.is_ready,
            "status": health.status.value,
            "message": "Ready" if health.is_ready else "Not ready"
        }

    async def health(self) -> Dict[str, Any]:
        """
        Full health check with component details.
        """
        health = await self.checker.check_all()
        return health.to_dict()


# =============================================================================
# GLOBAL HEALTH CHECKER
# =============================================================================

_global_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance"""
    global _global_checker
    if _global_checker is None:
        _global_checker = HealthChecker()
    return _global_checker


def init_health_checker(start_time: Optional[float] = None) -> HealthChecker:
    """Initialize the global health checker"""
    global _global_checker
    _global_checker = HealthChecker(start_time=start_time)
    return _global_checker
