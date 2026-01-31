"""
API Connector Module
Handles all communication with Polymarket and exchanges
With proper error handling, retry logic, and rate limiting
"""

import aiohttp
import asyncio
import os
import logging
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional, Callable
from websocket_manager import WebSocketManager, WebSocketConfig
from exceptions import (
    APIConnectionError, APITimeoutError, APIRateLimitError,
    APIResponseError, APIAuthenticationError
)
from utils import retry_async, RateLimiter

load_dotenv()

logger = logging.getLogger("PolyMangoBot.api")


class PolymarketAPI:
    """Polymarket API wrapper with retry logic and rate limiting"""

    def __init__(self, rate_limit: float = 10.0):  # 10 requests per second
        self.base_url = "https://clob.polymarket.com"
        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(rate=rate_limit, burst=5)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.session is not None

    async def connect(self):
        """Initialize async session with proper timeout configuration"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self._connected = True
        logger.info("Polymarket API connected")

    async def disconnect(self):
        """Close session gracefully"""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        logger.info("Polymarket API disconnected")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

    @retry_async(
        max_attempts=3,
        base_delay=1.0,
        retryable_exceptions=(APITimeoutError, APIConnectionError)
    )
    async def get_order_book(self, market_id: str) -> Dict:
        """Get order book for a specific market with retry logic"""
        if not self.is_connected:
            raise APIConnectionError("Polymarket API not connected")

        await self.rate_limiter.acquire()

        url = f"{self.base_url}/order-book/{market_id}"
        headers = self._get_headers()

        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 401:
                    raise APIAuthenticationError("Polymarket authentication failed")
                elif resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 60))
                    raise APIRateLimitError(
                        "Polymarket rate limit exceeded",
                        retry_after_seconds=retry_after
                    )
                else:
                    body = await resp.text()
                    logger.warning(f"Polymarket error {resp.status}: {body[:200]}")
                    raise APIResponseError(
                        f"Polymarket API error",
                        status_code=resp.status,
                        response_body=body[:500]
                    )
        except asyncio.TimeoutError:
            raise APITimeoutError("Polymarket order book request timed out")
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Polymarket connection error: {e}")
    
    @retry_async(max_attempts=3, base_delay=1.0, retryable_exceptions=(APITimeoutError, APIConnectionError))
    async def get_markets(self) -> List[Dict]:
        """Get all active markets with retry logic"""
        if not self.is_connected:
            raise APIConnectionError("Polymarket API not connected")

        await self.rate_limiter.acquire()

        url = f"{self.base_url}/markets"
        headers = self._get_headers()

        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    raise APIRateLimitError("Polymarket rate limit exceeded")
                else:
                    logger.warning(f"Polymarket markets error: {resp.status}")
                    return []
        except asyncio.TimeoutError:
            raise APITimeoutError("Polymarket markets request timed out")
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Polymarket connection error: {e}")

    @retry_async(max_attempts=2, base_delay=0.5, retryable_exceptions=(APITimeoutError,))
    async def place_order(self, order: Dict) -> Dict:
        """Place an order on Polymarket with minimal retry (order placement is time-sensitive)"""
        if not self.is_connected:
            raise APIConnectionError("Polymarket API not connected")

        await self.rate_limiter.acquire()

        url = f"{self.base_url}/orders"
        headers = self._get_headers()

        logger.debug(f"Placing Polymarket order: {order}")

        try:
            async with self.session.post(url, json=order, headers=headers) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    logger.info(f"Polymarket order placed: {result.get('order_id', 'unknown')}")
                    return result
                elif resp.status == 401:
                    raise APIAuthenticationError("Polymarket authentication failed for order")
                elif resp.status == 429:
                    raise APIRateLimitError("Polymarket rate limit exceeded during order placement")
                else:
                    body = await resp.text()
                    logger.error(f"Polymarket order failed {resp.status}: {body[:200]}")
                    raise APIResponseError(
                        f"Order placement failed",
                        status_code=resp.status,
                        response_body=body[:500]
                    )
        except asyncio.TimeoutError:
            raise APITimeoutError("Polymarket order placement timed out")
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Polymarket order connection error: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order - no retry to avoid race conditions"""
        if not self.is_connected:
            logger.warning("Cannot cancel order - Polymarket API not connected")
            return False

        if not order_id:
            return True

        await self.rate_limiter.acquire()

        url = f"{self.base_url}/orders/{order_id}"
        headers = self._get_headers()

        try:
            async with self.session.delete(url, headers=headers) as resp:
                success = resp.status in (200, 204, 404)  # 404 means already cancelled/filled
                if success:
                    logger.info(f"Polymarket order cancelled: {order_id}")
                else:
                    logger.warning(f"Polymarket cancel failed for {order_id}: {resp.status}")
                return success
        except asyncio.TimeoutError:
            logger.error(f"Polymarket cancel timeout for {order_id}")
            return False
        except Exception as e:
            logger.error(f"Polymarket cancel error for {order_id}: {e}")
            return False


class ExchangeAPI:
    """Generic exchange API wrapper with retry logic and rate limiting"""

    # Exchange base URLs
    EXCHANGE_URLS = {
        "kraken": "https://api.kraken.com",
        "coinbase": "https://api.coinbase.com",
    }

    def __init__(self, exchange_name: str = "kraken", rate_limit: float = 10.0):
        self.exchange_name = exchange_name.lower()
        self.api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
        self.api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
        self.base_url = self.EXCHANGE_URLS.get(self.exchange_name, "https://api.kraken.com")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(rate=rate_limit, burst=5)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.session is not None

    async def connect(self):
        """Initialize async session with proper timeout configuration"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self._connected = True
        logger.info(f"{self.exchange_name.capitalize()} API connected")

    async def disconnect(self):
        """Close session gracefully"""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        logger.info(f"{self.exchange_name.capitalize()} API disconnected")

    @retry_async(max_attempts=3, base_delay=1.0, retryable_exceptions=(APITimeoutError, APIConnectionError))
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current price for a symbol with retry logic"""
        if not self.is_connected:
            raise APIConnectionError(f"{self.exchange_name} API not connected")

        await self.rate_limiter.acquire()

        if self.exchange_name == "kraken":
            url = f"{self.base_url}/0/public/Ticker?pair={symbol}"
        else:
            url = f"{self.base_url}/v2/prices/{symbol}-USD/spot"

        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if self.exchange_name == "kraken":
                        return data.get("result", {})
                    return data
                elif resp.status == 429:
                    raise APIRateLimitError(f"{self.exchange_name} rate limit exceeded")
                else:
                    logger.warning(f"{self.exchange_name} ticker error: {resp.status}")
                    return {}
        except asyncio.TimeoutError:
            raise APITimeoutError(f"{self.exchange_name} ticker request timed out")
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"{self.exchange_name} connection error: {e}")

    async def get_balance(self) -> Dict:
        """Get account balance (requires auth)"""
        if not self.is_connected:
            return {}

        logger.debug(f"Balance check on {self.exchange_name}")
        # Real implementation would use HMAC signatures
        return {}

    @retry_async(max_attempts=2, base_delay=0.5, retryable_exceptions=(APITimeoutError,))
    async def place_order(
        self,
        order: Dict = None,
        symbol: str = None,
        side: str = None,
        volume: float = None,
        price: float = None
    ) -> Dict:
        """
        Place a limit order with retry logic

        Can be called two ways:
        1. order_dict format: place_order({"market": "BTC", "side": "buy", "quantity": 1.0, "price": 42500})
        2. params format: place_order(symbol="BTC", side="buy", volume=1.0, price=42500)
        """
        # Support both dict and parameter formats
        if order and isinstance(order, dict):
            symbol = order.get("market", symbol)
            side = order.get("side", side)
            volume = order.get("quantity", order.get("volume", volume))
            price = order.get("price", price)

        # Validate inputs
        if not all([symbol, side, volume, price]):
            raise ValueError("Missing required order parameters: symbol, side, volume, price")

        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid order side: {side}")

        if volume <= 0:
            raise ValueError(f"Invalid volume: {volume}")

        if price <= 0:
            raise ValueError(f"Invalid price: {price}")

        await self.rate_limiter.acquire()

        logger.info(f"[{self.exchange_name}] Placing {side} order: {volume} {symbol} @ ${price}")

        # Simulated response - real implementation would call exchange API
        order_id = f"order_{side}_{symbol}_{int(price)}_{int(asyncio.get_event_loop().time() * 1000) % 10000}"
        return {"order_id": order_id, "status": "pending"}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order - no retry to avoid race conditions"""
        if not order_id:
            return True

        logger.info(f"[{self.exchange_name}] Canceling order {order_id}")
        # Real implementation would call exchange API
        return True


class APIManager:
    """Manages all API connections - single point of contact"""

    def __init__(self, additional_exchanges: List[str] = None, enable_websocket: bool = True):
        self.polymarket = PolymarketAPI()
        self.exchange = ExchangeAPI("kraken")
        self.additional_exchanges = {}

        # Support multiple exchanges
        if additional_exchanges:
            for exchange_name in additional_exchanges:
                self.additional_exchanges[exchange_name] = ExchangeAPI(exchange_name)

        self.connected = False

        # WebSocket support for real-time streaming
        self.enable_websocket = enable_websocket
        self.ws_manager: Optional[WebSocketManager] = None
        if enable_websocket:
            self.ws_manager = WebSocketManager(WebSocketConfig())

    async def connect_all(self):
        """Connect to all APIs (HTTP) and WebSocket (if enabled)"""
        await self.polymarket.connect()
        await self.exchange.connect()

        # Connect additional exchanges in parallel
        if self.additional_exchanges:
            await asyncio.gather(
                *[ex.connect() for ex in self.additional_exchanges.values()],
                return_exceptions=True
            )

        # Connect WebSocket if enabled
        if self.enable_websocket and self.ws_manager:
            try:
                await self.ws_manager.connect_all()
                print("[OK] WebSocket connections established")
            except Exception as e:
                print(f"[WARN] WebSocket connection failed: {e}")

        self.connected = True
        print("[OK] All APIs connected")

    async def disconnect_all(self):
        """Disconnect from all APIs (HTTP) and WebSocket"""
        tasks = [
            self.polymarket.disconnect(),
            self.exchange.disconnect(),
        ]

        # Add additional exchanges
        tasks.extend([ex.disconnect() for ex in self.additional_exchanges.values()])

        # Disconnect WebSocket if enabled
        if self.enable_websocket and self.ws_manager:
            tasks.append(self.ws_manager.disconnect_all())

        await asyncio.gather(*tasks, return_exceptions=True)

        self.connected = False
        print("[OK] All APIs disconnected")

    async def fetch_all_prices_parallel(self, symbols: List[str]) -> Dict:
        """
        Fetch prices from all venues IN PARALLEL (not sequentially)

        This is much faster than fetching one venue at a time.

        Returns:
            {
                'BTC': {
                    'polymarket': {'price': 42500, 'bid_qty': 5, 'ask_qty': 4},
                    'kraken': {'price': 42650, 'bid_qty': 50, 'ask_qty': 45}
                }
            }
        """

        results = {}

        # Create parallel tasks for each symbol
        tasks = {}
        for symbol in symbols:
            tasks[symbol] = asyncio.gather(
                self.polymarket.get_order_book(symbol),
                self.exchange.get_ticker(symbol),
                return_exceptions=True
            )

        # Execute all in parallel
        all_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map results back
        for symbol, result in zip(tasks.keys(), all_results):
            if isinstance(result, tuple) and len(result) >= 2:
                poly_data, kraken_data = result[0], result[1]

                results[symbol] = {
                    'polymarket': poly_data if isinstance(poly_data, dict) else {},
                    'kraken': kraken_data if isinstance(kraken_data, dict) else {}
                }

        return results

    def get_websocket_manager(self) -> Optional[WebSocketManager]:
        """Get WebSocket manager for real-time streaming"""
        return self.ws_manager

    async def subscribe_realtime_prices(
        self,
        polymarket_ids: List[str] = None,
        exchange: str = "kraken",
        symbols: List[str] = None,
        callback: Callable = None
    ) -> bool:
        """
        Subscribe to real-time price updates via WebSocket

        Args:
            polymarket_ids: List of Polymarket market IDs to stream
            exchange: Exchange name (kraken, coinbase)
            symbols: List of symbols to stream
            callback: Optional callback for price events

        Returns:
            True if subscription successful
        """
        if not self.ws_manager:
            print("[WARN] WebSocket not enabled")
            return False

        success = True

        if polymarket_ids:
            result = await self.ws_manager.subscribe_polymarket(polymarket_ids)
            success = success and result

        if symbols:
            result = await self.ws_manager.subscribe_exchange(exchange, symbols)
            success = success and result

        if callback:
            self.ws_manager.register_callback(callback)

        return success

    def get_latest_prices_from_websocket(self, symbol: str) -> Dict:
        """
        Get latest prices for a symbol from WebSocket streams

        Returns dict with venue -> PriceEvent
        """
        if not self.ws_manager:
            return {}

        return self.ws_manager.get_all_prices(symbol)


# Test the module
async def test_api_connector():
    """Simple test - just verifies connections work"""
    manager = APIManager()
    await manager.connect_all()
    
    # Try to get Polymarket markets
    print("\nFetching Polymarket markets...")
    markets = await manager.polymarket.get_markets()
    print(f"Found {len(markets)} markets" if markets else "No markets (API may need credentials)")
    
    await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(test_api_connector())