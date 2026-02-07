"""
Configuration Management Module
Centralized configuration with validation and type safety.

Includes secure credential loading with validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import os
import logging
from dotenv import load_dotenv

from constants import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RATE_LIMIT_RPS,
    DEFAULT_CONNECTION_LIMIT,
    DEFAULT_CONNECTION_LIMIT_PER_HOST,
    WS_CONNECT_TIMEOUT,
    WS_HEARTBEAT_INTERVAL,
    WS_RECONNECT_DELAY,
    WS_MAX_RECONNECT_DELAY,
    WS_MAX_RETRIES,
    WS_MAX_PRICE_HISTORY_PER_SYMBOL,
    DEFAULT_MAKER_FEE_PCT,
    DEFAULT_TAKER_FEE_PCT,
    DEFAULT_SLIPPAGE_PCT,
    MIN_PROFIT_MARGIN_PCT,
    MAX_SPREAD_PCT_SANITY,
    MAX_POSITION_PCT_OF_CAPITAL,
    MIN_TRADES_FOR_KELLY_CONFIDENCE,
)
from exceptions import ConfigurationError

load_dotenv()


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass(frozen=True)
class APIConfig:
    """API connection configuration"""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def is_configured(self) -> bool:
        """Check if API credentials are set"""
        return bool(self.api_key and self.api_secret)


@dataclass(frozen=True)
class PolymarketConfig(APIConfig):
    """Polymarket-specific configuration"""
    base_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://clob.polymarket.com/ws"


@dataclass(frozen=True)
class ExchangeConfig(APIConfig):
    """Exchange-specific configuration"""
    exchange_name: str = "kraken"
    base_url: str = "https://api.kraken.com"
    ws_url: str = "wss://ws.kraken.com"


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 1000.0
    max_daily_loss: float = 5000.0
    min_profit_margin_percent: float = 0.3
    max_spread_percent: float = 50.0
    maker_fee_percent: float = 0.1
    taker_fee_percent: float = 0.15
    slippage_percent: float = 0.25

    def __post_init__(self):
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        if self.min_profit_margin_percent < 0:
            raise ValueError("min_profit_margin_percent cannot be negative")


@dataclass(frozen=True)
class KellyConfig:
    """Kelly Criterion position sizing configuration"""
    enabled: bool = True
    capital: float = 10000.0
    mode: str = "HALF_KELLY"  # FULL_KELLY, HALF_KELLY, QUARTER_KELLY
    min_trades_for_confidence: int = 20
    max_position_percent: float = 10.0  # Max 10% of capital per trade

    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if self.mode not in ["FULL_KELLY", "HALF_KELLY", "QUARTER_KELLY"]:
            raise ValueError(f"Invalid kelly mode: {self.mode}")


@dataclass(frozen=True)
class WebSocketConfig:
    """WebSocket connection configuration"""
    enabled: bool = True
    reconnect_delay_seconds: float = 1.0
    max_reconnect_delay_seconds: float = 30.0
    heartbeat_interval_seconds: float = 30.0
    timeout_seconds: float = 60.0
    max_retries: int = 10
    price_history_max_size: int = 1000


@dataclass(frozen=True)
class InferenceConfig:
    """Cross-market inference engine configuration"""
    enabled: bool = True

    # Family discovery
    min_family_size: int = 2
    max_family_size: int = 20
    min_token_overlap: float = 0.5

    # Detection thresholds
    min_edge_pct: float = 0.5
    min_leg_liquidity: float = 500.0
    min_realizable_edge_pct: float = 0.3

    # Position sizing
    default_position_usd: float = 100.0
    max_position_usd: float = 1000.0
    max_position_pct_of_liquidity: float = 0.05

    # Fee estimation
    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.02
    base_slippage_pct: float = 0.05

    # Monitoring
    poll_interval_seconds: float = 45.0
    fast_poll_interval_seconds: float = 10.0
    min_persistence_seconds: float = 30.0

    # LLM settings
    use_llm: bool = False
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-haiku-20240307"

    # Auto-execution
    auto_execute: bool = False
    execute_edge_threshold_pct: float = 1.0


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True


@dataclass
class BotConfig:
    """Master configuration for the entire bot"""
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Bot behavior
    scan_interval_seconds: float = 5.0
    min_spread_percent: float = 0.3
    dry_run: bool = True  # Safety: default to simulation mode

    @classmethod
    def from_env(cls, validate_credentials: bool = False) -> "BotConfig":
        """
        Load configuration from environment variables.

        Args:
            validate_credentials: If True, raise ConfigurationError if credentials invalid

        Returns:
            BotConfig instance

        Raises:
            ConfigurationError: If validate_credentials=True and credentials are invalid
        """
        errors = []

        # Parse numeric values with error handling
        def parse_float(env_var: str, default: float, name: str) -> float:
            value = os.getenv(env_var, str(default))
            try:
                parsed = float(value)
                if parsed < 0:
                    errors.append(f"{name} must be non-negative, got {parsed}")
                    return default
                return parsed
            except ValueError:
                errors.append(f"Invalid {name}: '{value}' is not a valid number")
                return default

        def parse_bool(env_var: str, default: bool) -> bool:
            value = os.getenv(env_var, str(default).lower())
            return value.lower() in ("true", "1", "yes", "on")

        # Load API credentials
        polymarket_key = os.getenv("POLYMARKET_API_KEY", "").strip()
        polymarket_secret = os.getenv("POLYMARKET_API_SECRET", "").strip()
        exchange_name = os.getenv("EXCHANGE_NAME", "kraken").strip().lower()
        exchange_key = os.getenv(f"{exchange_name.upper()}_API_KEY", "").strip()
        exchange_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET", "").strip()

        # Validate credentials if required
        if validate_credentials:
            if not polymarket_key or not polymarket_secret:
                errors.append("Polymarket API credentials are required")
            if not exchange_key or not exchange_secret:
                errors.append(f"{exchange_name.capitalize()} API credentials are required")

        # Parse kelly mode with validation
        kelly_mode = os.getenv("KELLY_MODE", "HALF_KELLY").upper()
        if kelly_mode not in ["FULL_KELLY", "HALF_KELLY", "QUARTER_KELLY"]:
            errors.append(f"Invalid KELLY_MODE: {kelly_mode}")
            kelly_mode = "HALF_KELLY"

        # Parse log level with validation
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            log_level = LogLevel[log_level_str]
        except KeyError:
            errors.append(f"Invalid LOG_LEVEL: {log_level_str}")
            log_level = LogLevel.INFO

        # Raise if there are errors and validation is required
        if errors and validate_credentials:
            raise ConfigurationError(f"Configuration errors: {'; '.join(errors)}")

        return cls(
            polymarket=PolymarketConfig(
                api_key=polymarket_key,
                api_secret=polymarket_secret,
            ),
            exchange=ExchangeConfig(
                exchange_name=exchange_name,
                api_key=exchange_key,
                api_secret=exchange_secret,
            ),
            risk=RiskConfig(
                max_position_size=parse_float("MAX_POSITION_SIZE", 1000.0, "MAX_POSITION_SIZE"),
                max_daily_loss=parse_float("MAX_DAILY_LOSS", 5000.0, "MAX_DAILY_LOSS"),
                min_profit_margin_percent=parse_float("MIN_PROFIT_MARGIN", MIN_PROFIT_MARGIN_PCT, "MIN_PROFIT_MARGIN"),
            ),
            kelly=KellyConfig(
                enabled=parse_bool("KELLY_ENABLED", True),
                capital=parse_float("TRADING_CAPITAL", 10000.0, "TRADING_CAPITAL"),
                mode=kelly_mode,
            ),
            websocket=WebSocketConfig(
                enabled=parse_bool("WEBSOCKET_ENABLED", True),
            ),
            logging=LoggingConfig(
                level=log_level,
                file_path=os.getenv("LOG_FILE"),
            ),
            inference=InferenceConfig(
                enabled=parse_bool("INFERENCE_ENABLED", True),
                min_edge_pct=parse_float("INFERENCE_MIN_EDGE_PCT", 0.5, "INFERENCE_MIN_EDGE_PCT"),
                min_leg_liquidity=parse_float("INFERENCE_MIN_LIQUIDITY", 500.0, "INFERENCE_MIN_LIQUIDITY"),
                default_position_usd=parse_float("INFERENCE_POSITION_USD", 100.0, "INFERENCE_POSITION_USD"),
                poll_interval_seconds=parse_float("INFERENCE_POLL_INTERVAL", 45.0, "INFERENCE_POLL_INTERVAL"),
                use_llm=parse_bool("INFERENCE_USE_LLM", False),
                auto_execute=parse_bool("INFERENCE_AUTO_EXECUTE", False),
            ),
            scan_interval_seconds=parse_float("SCAN_INTERVAL", 5.0, "SCAN_INTERVAL"),
            min_spread_percent=parse_float("MIN_SPREAD_PERCENT", 0.3, "MIN_SPREAD_PERCENT"),
            dry_run=parse_bool("DRY_RUN", True),
        )

    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings/errors"""
        issues = []

        if not self.polymarket.is_configured():
            issues.append("WARNING: Polymarket API credentials not configured")

        if not self.exchange.is_configured():
            issues.append("WARNING: Exchange API credentials not configured")

        if self.dry_run:
            issues.append("INFO: Running in DRY RUN mode - no real trades will be executed")

        if self.kelly.enabled and self.kelly.capital < self.risk.max_position_size:
            issues.append("WARNING: Kelly capital is less than max position size")

        return issues


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Setup logging based on configuration"""
    logger = logging.getLogger("PolyMangoBot")
    logger.setLevel(config.level.value)

    formatter = logging.Formatter(config.format)

    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.file_path:
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global config instance
_config: Optional[BotConfig] = None


def get_config() -> BotConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = BotConfig.from_env()
    return _config


def set_config(config: BotConfig) -> None:
    """Set global configuration instance"""
    global _config
    _config = config
