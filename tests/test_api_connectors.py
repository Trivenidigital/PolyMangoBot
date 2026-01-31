"""
Tests for API connectors module.
Tests connection pooling, API client base, and error handling.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_connectors import (
    get_connection_pool,
    close_connection_pool,
    APIManager
)
from exceptions import APIError, APITimeoutError, APIConnectionError


class TestConnectionPool:
    """Tests for connection pooling"""

    @pytest.mark.asyncio
    async def test_get_connection_pool_creates_pool(self):
        """Should create a connection pool on first call"""
        # Clean up any existing pool first
        await close_connection_pool()
        pool = get_connection_pool()
        assert pool is not None
        await close_connection_pool()

    @pytest.mark.asyncio
    async def test_get_connection_pool_reuses_pool(self):
        """Should return same pool on subsequent calls"""
        await close_connection_pool()
        pool1 = get_connection_pool()
        pool2 = get_connection_pool()
        assert pool1 is pool2
        await close_connection_pool()

    @pytest.mark.asyncio
    async def test_close_connection_pool(self):
        """Should close the connection pool"""
        pool = get_connection_pool()
        await close_connection_pool()
        # After closing, the pool should be closed
        assert pool.closed


class TestAPIManager:
    """Tests for APIManager"""

    @pytest.mark.asyncio
    async def test_initialization_default(self):
        """Should initialize with default settings"""
        manager = APIManager()
        assert manager is not None
        await close_connection_pool()

    @pytest.mark.asyncio
    async def test_initialization_with_websocket(self):
        """Should initialize with WebSocket when enabled"""
        manager = APIManager(enable_websocket=True)
        assert manager.ws_manager is not None
        await close_connection_pool()

    @pytest.mark.asyncio
    async def test_initialization_without_websocket(self):
        """Should work without WebSocket"""
        manager = APIManager(enable_websocket=False)
        assert manager.ws_manager is None
        await close_connection_pool()


class TestAPIErrorHandling:
    """Tests for API error handling"""

    def test_api_error_message(self):
        """APIError should preserve message"""
        error = APIError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_api_timeout_error(self):
        """APITimeoutError should be catchable as APIError"""
        error = APITimeoutError("Request timed out")
        assert isinstance(error, APIError)
        assert "timed out" in str(error)

    def test_api_connection_error(self):
        """APIConnectionError should be catchable as APIError"""
        error = APIConnectionError("Connection failed")
        assert isinstance(error, APIError)
        assert "Connection failed" in str(error)


class TestAPIClientIntegration:
    """Integration tests for API clients"""

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self):
        """Should cleanup resources on error"""
        manager = APIManager(enable_websocket=False)

        # The manager should handle cleanup gracefully
        await manager.disconnect_all()
        await close_connection_pool()

    @pytest.mark.asyncio
    async def test_multiple_disconnect_calls(self):
        """Should handle multiple disconnect calls gracefully"""
        manager = APIManager(enable_websocket=False)

        # Multiple disconnects should not raise
        await manager.disconnect_all()
        await manager.disconnect_all()
        await close_connection_pool()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
