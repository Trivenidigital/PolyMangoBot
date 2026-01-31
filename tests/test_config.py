"""
Tests for configuration module.
Tests BotConfig loading, validation, and LogLevel.
"""

import pytest
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BotConfig,
    PolymarketConfig,
    ExchangeConfig,
    RiskConfig,
    KellyConfig,
    WebSocketConfig,
    LoggingConfig,
    LogLevel,
    get_config,
    set_config,
    setup_logging
)
from exceptions import ConfigurationError


class TestLogLevel:
    """Tests for LogLevel enum"""

    def test_log_levels_exist(self):
        """Should have all standard log levels"""
        assert LogLevel.DEBUG is not None
        assert LogLevel.INFO is not None
        assert LogLevel.WARNING is not None
        assert LogLevel.ERROR is not None
        assert LogLevel.CRITICAL is not None

    def test_log_level_values(self):
        """Log levels should have correct numeric values"""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value


class TestAPIConfig:
    """Tests for API configuration dataclasses"""

    def test_polymarket_config_defaults(self):
        """Should have correct Polymarket defaults"""
        config = PolymarketConfig()
        assert config.base_url == "https://clob.polymarket.com"
        assert config.ws_url == "wss://clob.polymarket.com/ws"
        assert config.api_key == ""
        assert config.api_secret == ""

    def test_polymarket_config_is_configured(self):
        """Should check if credentials are configured"""
        unconfigured = PolymarketConfig()
        assert unconfigured.is_configured() is False

        configured = PolymarketConfig(api_key="key", api_secret="secret")
        assert configured.is_configured() is True

    def test_exchange_config_defaults(self):
        """Should have correct exchange defaults"""
        config = ExchangeConfig()
        assert config.exchange_name == "kraken"
        assert config.base_url == "https://api.kraken.com"


class TestRiskConfig:
    """Tests for risk configuration"""

    def test_risk_config_defaults(self):
        """Should have sane defaults"""
        config = RiskConfig()
        assert config.max_position_size == 1000.0
        assert config.max_daily_loss == 5000.0
        assert config.min_profit_margin_percent >= 0

    def test_risk_config_validation(self):
        """Should validate constraints"""
        with pytest.raises(ValueError):
            RiskConfig(max_position_size=-100)

        with pytest.raises(ValueError):
            RiskConfig(max_daily_loss=0)

        with pytest.raises(ValueError):
            RiskConfig(min_profit_margin_percent=-1)


class TestKellyConfig:
    """Tests for Kelly Criterion configuration"""

    def test_kelly_config_defaults(self):
        """Should have sane defaults"""
        config = KellyConfig()
        assert config.enabled is True
        assert config.capital == 10000.0
        assert config.mode == "HALF_KELLY"

    def test_kelly_config_modes(self):
        """Should accept valid modes"""
        for mode in ["FULL_KELLY", "HALF_KELLY", "QUARTER_KELLY"]:
            config = KellyConfig(mode=mode)
            assert config.mode == mode

    def test_kelly_config_invalid_mode(self):
        """Should reject invalid modes"""
        with pytest.raises(ValueError):
            KellyConfig(mode="INVALID_MODE")

    def test_kelly_config_capital_validation(self):
        """Should validate capital is positive"""
        with pytest.raises(ValueError):
            KellyConfig(capital=0)

        with pytest.raises(ValueError):
            KellyConfig(capital=-1000)


class TestWebSocketConfig:
    """Tests for WebSocket configuration"""

    def test_websocket_config_defaults(self):
        """Should have sane defaults"""
        config = WebSocketConfig()
        assert config.enabled is True
        assert config.reconnect_delay_seconds > 0
        assert config.max_reconnect_delay_seconds > config.reconnect_delay_seconds
        assert config.heartbeat_interval_seconds > 0


class TestBotConfig:
    """Tests for main BotConfig"""

    def test_bot_config_defaults(self):
        """Should create config with default sub-configs"""
        config = BotConfig()
        assert config.polymarket is not None
        assert config.exchange is not None
        assert config.risk is not None
        assert config.kelly is not None
        assert config.websocket is not None
        assert config.logging is not None
        assert config.dry_run is True  # Safety default

    @patch.dict(os.environ, {
        "POLYMARKET_API_KEY": "test_poly_key",
        "POLYMARKET_API_SECRET": "test_poly_secret",
        "EXCHANGE_NAME": "kraken",
        "KRAKEN_API_KEY": "test_kraken_key",
        "KRAKEN_API_SECRET": "test_kraken_secret",
        "DRY_RUN": "true",
        "KELLY_MODE": "HALF_KELLY",
        "LOG_LEVEL": "INFO"
    }, clear=False)
    def test_from_env_basic(self):
        """Should load configuration from environment"""
        config = BotConfig.from_env(validate_credentials=False)
        assert config.polymarket.api_key == "test_poly_key"
        assert config.exchange.exchange_name == "kraken"
        assert config.dry_run is True

    @patch.dict(os.environ, {
        "DRY_RUN": "false",
        "KELLY_ENABLED": "true",
        "WEBSOCKET_ENABLED": "false"
    }, clear=False)
    def test_from_env_booleans(self):
        """Should parse boolean values correctly"""
        config = BotConfig.from_env(validate_credentials=False)
        assert config.dry_run is False
        assert config.kelly.enabled is True
        assert config.websocket.enabled is False

    @patch.dict(os.environ, {
        "MAX_POSITION_SIZE": "5000",
        "MAX_DAILY_LOSS": "10000",
        "TRADING_CAPITAL": "50000",
        "SCAN_INTERVAL": "10.0"
    }, clear=False)
    def test_from_env_numeric(self):
        """Should parse numeric values correctly"""
        config = BotConfig.from_env(validate_credentials=False)
        assert config.risk.max_position_size == 5000.0
        assert config.risk.max_daily_loss == 10000.0
        assert config.kelly.capital == 50000.0
        assert config.scan_interval_seconds == 10.0

    @patch.dict(os.environ, {
        "KELLY_MODE": "INVALID"
    }, clear=False)
    def test_from_env_invalid_kelly_mode(self):
        """Should fall back to default for invalid Kelly mode"""
        config = BotConfig.from_env(validate_credentials=False)
        assert config.kelly.mode == "HALF_KELLY"

    @patch.dict(os.environ, {
        "LOG_LEVEL": "INVALID"
    }, clear=False)
    def test_from_env_invalid_log_level(self):
        """Should fall back to INFO for invalid log level"""
        config = BotConfig.from_env(validate_credentials=False)
        assert config.logging.level == LogLevel.INFO

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_required_credentials(self):
        """Should raise ConfigurationError when credentials required but missing"""
        # Clear all credential env vars
        for var in ["POLYMARKET_API_KEY", "POLYMARKET_API_SECRET",
                    "KRAKEN_API_KEY", "KRAKEN_API_SECRET"]:
            os.environ.pop(var, None)

        with pytest.raises(ConfigurationError):
            BotConfig.from_env(validate_credentials=True)

    def test_validate_unconfigured(self):
        """Should report warnings for unconfigured credentials"""
        config = BotConfig()  # Default with empty credentials
        issues = config.validate()

        # Should have warnings about missing credentials
        assert any("Polymarket" in issue for issue in issues)
        assert any("Exchange" in issue for issue in issues)

    def test_validate_dry_run(self):
        """Should report info about dry run mode"""
        config = BotConfig(dry_run=True)
        issues = config.validate()
        assert any("DRY RUN" in issue for issue in issues)


class TestGlobalConfig:
    """Tests for global config functions"""

    def test_set_and_get_config(self):
        """Should set and retrieve global config"""
        custom_config = BotConfig(
            scan_interval_seconds=15.0,
            min_spread_percent=0.5
        )
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved.scan_interval_seconds == 15.0
        assert retrieved.min_spread_percent == 0.5


class TestLoggingSetup:
    """Tests for logging setup"""

    def test_setup_logging_console(self):
        """Should create logger with console handler"""
        config = LoggingConfig(
            level=LogLevel.INFO,
            console_output=True,
            file_path=None
        )
        logger = setup_logging(config)

        assert logger is not None
        assert logger.level == LogLevel.INFO.value
        assert len(logger.handlers) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
