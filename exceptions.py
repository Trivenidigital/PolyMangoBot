"""
Custom Exceptions Module
Defines specific exceptions for better error handling
"""

from typing import Optional, Dict, Any


class PolyMangoBotError(Exception):
    """Base exception for all bot errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# API Errors
class APIError(PolyMangoBotError):
    """Base class for API-related errors"""
    pass


class APIConnectionError(APIError):
    """Failed to connect to API"""
    pass


class APITimeoutError(APIError):
    """API request timed out"""
    pass


class APIAuthenticationError(APIError):
    """API authentication failed"""
    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded"""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message, {"retry_after_seconds": retry_after_seconds})
        self.retry_after_seconds = retry_after_seconds


class APIResponseError(APIError):
    """Invalid or unexpected API response"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message, {"status_code": status_code, "response_body": response_body})
        self.status_code = status_code
        self.response_body = response_body


# Trading Errors
class TradingError(PolyMangoBotError):
    """Base class for trading-related errors"""
    pass


class InsufficientLiquidityError(TradingError):
    """Not enough liquidity to fill order"""

    def __init__(self, message: str, available: float, required: float):
        super().__init__(message, {"available": available, "required": required})
        self.available = available
        self.required = required


class OrderExecutionError(TradingError):
    """Order execution failed"""

    def __init__(self, message: str, order_id: Optional[str] = None, venue: Optional[str] = None):
        super().__init__(message, {"order_id": order_id, "venue": venue})
        self.order_id = order_id
        self.venue = venue


class OrderCancellationError(TradingError):
    """Order cancellation failed"""
    pass


class AtomicExecutionError(TradingError):
    """Atomic trade execution failed - one leg succeeded, other failed"""

    def __init__(self, message: str, successful_order_id: Optional[str] = None,
                 failed_venue: Optional[str] = None, rollback_status: Optional[str] = None):
        super().__init__(message, {
            "successful_order_id": successful_order_id,
            "failed_venue": failed_venue,
            "rollback_status": rollback_status
        })
        self.successful_order_id = successful_order_id
        self.failed_venue = failed_venue
        self.rollback_status = rollback_status


# Risk Errors
class RiskError(PolyMangoBotError):
    """Base class for risk-related errors"""
    pass


class PositionSizeExceededError(RiskError):
    """Position size exceeds maximum allowed"""

    def __init__(self, message: str, requested: float, maximum: float):
        super().__init__(message, {"requested": requested, "maximum": maximum})
        self.requested = requested
        self.maximum = maximum


class DailyLossLimitError(RiskError):
    """Daily loss limit would be exceeded"""

    def __init__(self, message: str, current_loss: float, max_loss: float):
        super().__init__(message, {"current_loss": current_loss, "max_loss": max_loss})
        self.current_loss = current_loss
        self.max_loss = max_loss


class InsufficientProfitMarginError(RiskError):
    """Trade doesn't meet minimum profit margin"""

    def __init__(self, message: str, expected_profit: float, minimum_required: float):
        super().__init__(message, {"expected_profit": expected_profit, "minimum_required": minimum_required})
        self.expected_profit = expected_profit
        self.minimum_required = minimum_required


# WebSocket Errors
class WebSocketError(PolyMangoBotError):
    """Base class for WebSocket-related errors"""
    pass


class WebSocketConnectionError(WebSocketError):
    """WebSocket connection failed"""
    pass


class WebSocketReconnectionError(WebSocketError):
    """WebSocket reconnection attempts exhausted"""

    def __init__(self, message: str, attempts: int, max_attempts: int):
        super().__init__(message, {"attempts": attempts, "max_attempts": max_attempts})
        self.attempts = attempts
        self.max_attempts = max_attempts


class WebSocketSubscriptionError(WebSocketError):
    """WebSocket subscription failed"""
    pass


# Data Errors
class DataError(PolyMangoBotError):
    """Base class for data-related errors"""
    pass


class DataValidationError(DataError):
    """Data validation failed"""
    pass


class DataParsingError(DataError):
    """Failed to parse data"""
    pass


class StaleDataError(DataError):
    """Data is too old to be reliable"""

    def __init__(self, message: str, data_age_seconds: float, max_age_seconds: float):
        super().__init__(message, {"data_age_seconds": data_age_seconds, "max_age_seconds": max_age_seconds})
        self.data_age_seconds = data_age_seconds
        self.max_age_seconds = max_age_seconds


# Configuration Errors
class ConfigurationError(PolyMangoBotError):
    """Configuration is invalid or missing"""
    pass
