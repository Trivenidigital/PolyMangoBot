"""
Configuration Management Module
Centralized configuration with validation and type safety
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import os
import logging
from dotenv import load_dotenv

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

    # Bot behavior
    scan_interval_seconds: float = 5.0
    min_spread_percent: float = 0.3
    dry_run: bool = True  # Safety: default to simulation mode

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables"""
        return cls(
            polymarket=PolymarketConfig(
                api_key=os.getenv("POLYMARKET_API_KEY", ""),
                api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            ),
            exchange=ExchangeConfig(
                exchange_name=os.getenv("EXCHANGE_NAME", "kraken"),
                api_key=os.getenv("KRAKEN_API_KEY", ""),
                api_secret=os.getenv("KRAKEN_API_SECRET", ""),
            ),
            risk=RiskConfig(
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "1000")),
                max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "5000")),
                min_profit_margin_percent=float(os.getenv("MIN_PROFIT_MARGIN", "0.3")),
            ),
            kelly=KellyConfig(
                enabled=os.getenv("KELLY_ENABLED", "true").lower() == "true",
                capital=float(os.getenv("TRADING_CAPITAL", "10000")),
                mode=os.getenv("KELLY_MODE", "HALF_KELLY"),
            ),
            websocket=WebSocketConfig(
                enabled=os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true",
            ),
            logging=LoggingConfig(
                level=LogLevel[os.getenv("LOG_LEVEL", "INFO").upper()],
                file_path=os.getenv("LOG_FILE"),
            ),
            scan_interval_seconds=float(os.getenv("SCAN_INTERVAL", "5")),
            min_spread_percent=float(os.getenv("MIN_SPREAD_PERCENT", "0.3")),
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
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
