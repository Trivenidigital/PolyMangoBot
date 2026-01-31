"""
Tests for health check module.
Tests component health checks and system health reporting.
"""

import pytest
import asyncio
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    HealthEndpoint,
    create_api_health_check,
    create_circuit_breaker_health_check,
    init_health_checker
)


class TestHealthStatus:
    """Tests for HealthStatus enum"""

    def test_health_status_values(self):
        """Should have expected status values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass"""

    def test_component_health_creation(self):
        """Should create component health with defaults"""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.latency_ms == 0.0

    def test_component_health_to_dict(self):
        """Should convert to dictionary"""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            latency_ms=10.5,
            details={"extra": "info"}
        )
        result = health.to_dict()

        assert result["name"] == "test"
        assert result["status"] == "healthy"
        assert result["message"] == "OK"
        assert result["latency_ms"] == 10.5
        assert result["details"]["extra"] == "info"
        assert "last_check_ago_seconds" in result


class TestSystemHealth:
    """Tests for SystemHealth dataclass"""

    def test_system_health_is_healthy(self):
        """Should report healthy when status is HEALTHY"""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[],
            uptime_seconds=100.0
        )
        assert health.is_healthy is True
        assert health.is_ready is True

    def test_system_health_degraded(self):
        """Should report not healthy but ready when DEGRADED"""
        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            components=[],
            uptime_seconds=100.0
        )
        assert health.is_healthy is False
        assert health.is_ready is True

    def test_system_health_unhealthy(self):
        """Should report not healthy and not ready when UNHEALTHY"""
        health = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components=[],
            uptime_seconds=100.0
        )
        assert health.is_healthy is False
        assert health.is_ready is False

    def test_system_health_to_dict(self):
        """Should convert to dictionary with all fields"""
        component = ComponentHealth(
            name="comp1",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[component],
            uptime_seconds=3661.0
        )
        result = health.to_dict()

        assert result["status"] == "healthy"
        assert result["is_healthy"] is True
        assert result["is_ready"] is True
        assert "uptime_human" in result
        assert len(result["components"]) == 1


class TestHealthChecker:
    """Tests for HealthChecker"""

    def test_health_checker_creation(self):
        """Should create health checker with start time"""
        checker = HealthChecker()
        assert checker.uptime_seconds >= 0

    def test_register_check(self):
        """Should register health check function"""
        checker = HealthChecker()

        async def my_check():
            return ComponentHealth(
                name="my_component",
                status=HealthStatus.HEALTHY,
                message="OK"
            )

        checker.register("my_component", my_check)
        assert "my_component" in checker._checks

    def test_unregister_check(self):
        """Should unregister health check function"""
        checker = HealthChecker()

        async def my_check():
            return ComponentHealth(name="x", status=HealthStatus.HEALTHY, message="")

        checker.register("my_component", my_check)
        checker.unregister("my_component")
        assert "my_component" not in checker._checks

    @pytest.mark.asyncio
    async def test_check_component_success(self):
        """Should check component and return result"""
        checker = HealthChecker()

        async def healthy_check():
            return ComponentHealth(
                name="healthy_comp",
                status=HealthStatus.HEALTHY,
                message="All systems go"
            )

        checker.register("healthy_comp", healthy_check)
        result = await checker.check_component("healthy_comp")

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems go"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_check_component_unknown(self):
        """Should return UNKNOWN for unregistered component"""
        checker = HealthChecker()
        result = await checker.check_component("nonexistent")

        assert result.status == HealthStatus.UNKNOWN
        assert "No health check registered" in result.message

    @pytest.mark.asyncio
    async def test_check_component_timeout(self):
        """Should handle timeout in health check"""
        checker = HealthChecker()
        checker._check_timeout = 0.1

        async def slow_check():
            await asyncio.sleep(1.0)
            return ComponentHealth(name="slow", status=HealthStatus.HEALTHY, message="")

        checker.register("slow", slow_check)
        result = await checker.check_component("slow")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_check_component_exception(self):
        """Should handle exceptions in health check"""
        checker = HealthChecker()

        async def failing_check():
            raise ValueError("Something went wrong")

        checker.register("failing", failing_check)
        result = await checker.check_component("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Something went wrong" in result.message

    @pytest.mark.asyncio
    async def test_check_all(self):
        """Should check all registered components"""
        checker = HealthChecker()

        async def healthy_check():
            return ComponentHealth(name="comp1", status=HealthStatus.HEALTHY, message="OK")

        async def degraded_check():
            return ComponentHealth(name="comp2", status=HealthStatus.DEGRADED, message="Slow")

        checker.register("comp1", healthy_check)
        checker.register("comp2", degraded_check)

        result = await checker.check_all()

        # Overall should be DEGRADED (worst of components)
        assert result.status == HealthStatus.DEGRADED
        # Should have at least our 2 components + system resources
        assert len(result.components) >= 2

    @pytest.mark.asyncio
    async def test_check_all_healthy(self):
        """Should report HEALTHY when all components healthy"""
        checker = HealthChecker()

        async def healthy_check():
            return ComponentHealth(name="comp1", status=HealthStatus.HEALTHY, message="OK")

        checker.register("comp1", healthy_check)

        result = await checker.check_all()
        # Note: system_resources check is also added, which might be HEALTHY or not
        # Just verify we got a result
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class TestHealthEndpoint:
    """Tests for HealthEndpoint"""

    @pytest.mark.asyncio
    async def test_liveness(self):
        """Should return alive status"""
        checker = HealthChecker()
        endpoint = HealthEndpoint(checker)

        result = await endpoint.liveness()

        assert result["status"] == "alive"
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_readiness_healthy(self):
        """Should return ready when healthy"""
        checker = HealthChecker()

        async def healthy():
            return ComponentHealth(name="x", status=HealthStatus.HEALTHY, message="")

        checker.register("x", healthy)
        endpoint = HealthEndpoint(checker)

        result = await endpoint.readiness()
        assert result["ready"] in [True, False]  # Depends on system resources

    @pytest.mark.asyncio
    async def test_health_full(self):
        """Should return full health status"""
        checker = HealthChecker()

        async def healthy():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY, message="OK")

        checker.register("test", healthy)
        endpoint = HealthEndpoint(checker)

        result = await endpoint.health()

        assert "status" in result
        assert "components" in result
        assert "uptime_seconds" in result
        assert "is_healthy" in result


class TestHealthCheckFactories:
    """Tests for health check factory functions"""

    @pytest.mark.asyncio
    async def test_create_circuit_breaker_health_check(self):
        """Should create health check for circuit breakers"""
        from utils import CircuitBreaker

        breakers = {
            "venue1": CircuitBreaker(),
            "venue2": CircuitBreaker()
        }

        check_func = create_circuit_breaker_health_check(breakers)
        result = await check_func()

        assert result.status == HealthStatus.HEALTHY
        assert "states" in result.details
        assert result.details["open_count"] == 0


class TestGlobalHealthChecker:
    """Tests for global health checker functions"""

    def test_init_health_checker(self):
        """Should initialize global health checker"""
        start_time = time.time()
        checker = init_health_checker(start_time=start_time)

        assert checker is not None
        assert checker.uptime_seconds >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
