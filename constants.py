"""
Constants Module
Centralized constants to replace magic numbers throughout the codebase.
"""

from enum import Enum
from typing import Final

# =============================================================================
# TRADING CONSTANTS
# =============================================================================

# Fill rate assumptions
EXPECTED_FILL_RATE: Final[float] = 0.95  # Conservative 95% fill assumption
PARTIAL_FILL_THRESHOLD: Final[float] = 0.10  # Cancel if less than 10% filled

# Position sizing
MAX_POSITION_PCT_OF_CAPITAL: Final[float] = 0.10  # Max 10% of capital per trade
DEFAULT_POSITION_PCT: Final[float] = 0.02  # Default 2% of capital if no Kelly data
MIN_POSITION_SIZE_USD: Final[float] = 10.0  # Minimum trade size

# Kelly Criterion
MIN_TRADES_FOR_KELLY_CONFIDENCE: Final[int] = 20  # Minimum trades before trusting Kelly
KELLY_HALF_MULTIPLIER: Final[float] = 0.5  # Half-Kelly conservative mode
KELLY_QUARTER_MULTIPLIER: Final[float] = 0.25  # Quarter-Kelly very conservative

# =============================================================================
# RISK MANAGEMENT CONSTANTS
# =============================================================================

# Loss limits
DEFAULT_MAX_DAILY_LOSS_PCT: Final[float] = 0.05  # 5% of capital
DEFAULT_MAX_POSITION_LOSS_PCT: Final[float] = 0.01  # 1% max loss assumption per trade

# Spread thresholds
MIN_SPREAD_PCT_FOR_TRADE: Final[float] = 0.3  # Minimum spread to consider
MAX_SPREAD_PCT_SANITY: Final[float] = 50.0  # Reject spreads > 50% as likely errors

# Profit margins
MIN_PROFIT_MARGIN_PCT: Final[float] = 0.3  # Minimum profit after fees

# =============================================================================
# FEE CONSTANTS
# =============================================================================

# Default fees (fallback when dynamic estimation unavailable)
DEFAULT_MAKER_FEE_PCT: Final[float] = 0.10
DEFAULT_TAKER_FEE_PCT: Final[float] = 0.15
DEFAULT_SLIPPAGE_PCT: Final[float] = 0.25

# Fee tiers (volume-based)
class FeeTier(Enum):
    """Volume-based fee tiers"""
    TIER_1 = (0, 100_000, 0.20)  # 0-100K volume: 0.20%
    TIER_2 = (100_000, 1_000_000, 0.15)  # 100K-1M: 0.15%
    TIER_3 = (1_000_000, 10_000_000, 0.10)  # 1M-10M: 0.10%
    TIER_4 = (10_000_000, float('inf'), 0.05)  # 10M+: 0.05%

# =============================================================================
# LATENCY CONSTANTS
# =============================================================================

# Latency thresholds (milliseconds)
SLOW_COMPONENT_THRESHOLD_MS: Final[float] = 100.0  # Warn if component > 100ms
SLOW_ITERATION_THRESHOLD_MS: Final[float] = 1000.0  # Warn if iteration > 1s
CRITICAL_LATENCY_THRESHOLD_MS: Final[float] = 500.0  # Log info if > 500ms

# Sample collection
MAX_LATENCY_SAMPLES: Final[int] = 1000  # Max samples to keep per component
MAX_ITERATION_SAMPLES: Final[int] = 1000  # Max iteration time samples

# =============================================================================
# API CONSTANTS
# =============================================================================

# Timeouts (seconds)
DEFAULT_API_TIMEOUT: Final[float] = 30.0
DEFAULT_CONNECT_TIMEOUT: Final[float] = 10.0
DEFAULT_READ_TIMEOUT: Final[float] = 15.0
ORDER_PLACEMENT_TIMEOUT: Final[float] = 10.0
ORDER_CANCEL_TIMEOUT: Final[float] = 10.0

# Retry configuration
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_BASE_DELAY: Final[float] = 1.0
DEFAULT_RETRY_MAX_DELAY: Final[float] = 30.0
ORDER_RETRY_MAX_ATTEMPTS: Final[int] = 2  # Fewer retries for time-sensitive orders

# Rate limiting
DEFAULT_RATE_LIMIT_RPS: Final[float] = 10.0  # Requests per second
DEFAULT_BURST_SIZE: Final[int] = 5  # Burst allowance

# Connection pooling
DEFAULT_CONNECTION_LIMIT: Final[int] = 100  # Total connections
DEFAULT_CONNECTION_LIMIT_PER_HOST: Final[int] = 30  # Per-host limit

# =============================================================================
# WEBSOCKET CONSTANTS
# =============================================================================

# WebSocket timeouts
WS_CONNECT_TIMEOUT: Final[float] = 60.0
WS_HEARTBEAT_INTERVAL: Final[float] = 30.0
WS_PING_TIMEOUT: Final[float] = 30.0

# Reconnection
WS_RECONNECT_DELAY: Final[float] = 1.0
WS_MAX_RECONNECT_DELAY: Final[float] = 30.0
WS_MAX_RETRIES: Final[int] = 10

# Data management
WS_MAX_PRICE_HISTORY_PER_SYMBOL: Final[int] = 1000

# =============================================================================
# CIRCUIT BREAKER CONSTANTS
# =============================================================================

CB_FAILURE_THRESHOLD: Final[int] = 5  # Failures before opening
CB_RECOVERY_TIMEOUT: Final[float] = 30.0  # Seconds before half-open
CB_HALF_OPEN_MAX_CALLS: Final[int] = 1  # Calls allowed in half-open

# Per-venue circuit breaker (more sensitive for order execution)
CB_ORDER_FAILURE_THRESHOLD: Final[int] = 3
CB_ORDER_RECOVERY_TIMEOUT: Final[float] = 60.0

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

# Log message limits
MAX_LOG_MESSAGE_LENGTH: Final[int] = 500  # Truncate long messages
MAX_RESPONSE_BODY_LOG_LENGTH: Final[int] = 200  # Truncate response bodies

# Periodic logging intervals
STATS_LOG_INTERVAL_ITERATIONS: Final[int] = 10
LATENCY_SUMMARY_LOG_INTERVAL: Final[int] = 100

# =============================================================================
# MARKET MAKER TRACKING CONSTANTS
# =============================================================================

# Inventory estimation
MM_INVENTORY_WINDOW_SIZE: Final[int] = 100  # Samples for inventory estimation
MM_INVENTORY_ZSCORE_THRESHOLD: Final[float] = 2.0  # Z-score for extreme inventory

# Spread regime thresholds (basis points)
SPREAD_ULTRA_TIGHT_BPS: Final[float] = 1.0
SPREAD_TIGHT_BPS: Final[float] = 3.0
SPREAD_NORMAL_BPS: Final[float] = 10.0
SPREAD_WIDE_BPS: Final[float] = 25.0
SPREAD_VERY_WIDE_BPS: Final[float] = 50.0

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# API key validation
MIN_API_KEY_LENGTH: Final[int] = 16  # Minimum expected key length
MIN_API_SECRET_LENGTH: Final[int] = 16  # Minimum expected secret length

# Price sanity checks
MIN_VALID_PRICE: Final[float] = 0.0001  # Minimum valid price
MAX_PRICE_CHANGE_PCT: Final[float] = 50.0  # Max % change to consider valid
