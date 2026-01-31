"""
Tests for security module.
Tests credential validation, log sanitization, and HMAC signing.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Import the modules we're testing
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import (
    validate_api_key,
    validate_api_secret,
    load_and_validate_credentials,
    sanitize_dict,
    sanitize_headers,
    sanitize_url,
    sanitize_for_logging,
    RequestSigner,
    SecureConfigLoader,
    REDACTED
)
from exceptions import ConfigurationError


class TestCredentialValidation:
    """Tests for credential validation functions"""

    def test_validate_api_key_valid(self):
        """Valid API key should return no errors"""
        errors = validate_api_key("abcdefghijklmnopqrstuvwxyz", "TEST_KEY")
        assert len(errors) == 0

    def test_validate_api_key_none(self):
        """None API key should return error"""
        errors = validate_api_key(None, "TEST_KEY")
        assert len(errors) == 1
        assert "not set" in errors[0]

    def test_validate_api_key_empty(self):
        """Empty API key should return error"""
        errors = validate_api_key("", "TEST_KEY")
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_validate_api_key_too_short(self):
        """Short API key should return warning"""
        errors = validate_api_key("abc", "TEST_KEY")
        assert len(errors) == 1
        assert "too short" in errors[0]

    def test_validate_api_key_placeholder(self):
        """Placeholder API key should return error"""
        errors = validate_api_key("your_api_key_here", "TEST_KEY")
        assert len(errors) == 1
        assert "placeholder" in errors[0]

    def test_validate_api_secret_valid(self):
        """Valid API secret should return no errors"""
        errors = validate_api_secret("secret_abcdefghijklmnop", "TEST_SECRET")
        assert len(errors) == 0

    def test_validate_api_secret_none(self):
        """None API secret should return error"""
        errors = validate_api_secret(None, "TEST_SECRET")
        assert len(errors) == 1
        assert "not set" in errors[0]


class TestLoadCredentials:
    """Tests for credential loading"""

    @patch.dict(os.environ, {"TEST_API_KEY": "valid_key_1234567890", "TEST_API_SECRET": "valid_secret_1234567890"})
    def test_load_valid_credentials(self):
        """Should load valid credentials without errors"""
        creds = load_and_validate_credentials("TEST_API_KEY", "TEST_API_SECRET", required=False)
        assert creds.is_valid
        assert creds.api_key == "valid_key_1234567890"
        assert len(creds.validation_errors) == 0

    @patch.dict(os.environ, {}, clear=True)
    def test_load_missing_credentials_not_required(self):
        """Should return invalid credentials when not required"""
        # Clear the env vars we're testing
        os.environ.pop("MISSING_KEY", None)
        os.environ.pop("MISSING_SECRET", None)
        creds = load_and_validate_credentials("MISSING_KEY", "MISSING_SECRET", required=False)
        assert not creds.is_valid
        assert len(creds.validation_errors) > 0

    @patch.dict(os.environ, {}, clear=True)
    def test_load_missing_credentials_required(self):
        """Should raise ConfigurationError when required credentials missing"""
        os.environ.pop("MISSING_KEY", None)
        os.environ.pop("MISSING_SECRET", None)
        with pytest.raises(ConfigurationError):
            load_and_validate_credentials("MISSING_KEY", "MISSING_SECRET", required=True)


class TestSanitization:
    """Tests for log sanitization functions"""

    def test_sanitize_dict_basic(self):
        """Should redact sensitive keys"""
        data = {
            "api_key": "secret123",
            "username": "john",
            "password": "mypassword"
        }
        sanitized = sanitize_dict(data)
        assert sanitized["api_key"] == REDACTED
        assert sanitized["username"] == "john"
        assert sanitized["password"] == REDACTED

    def test_sanitize_dict_nested(self):
        """Should redact sensitive keys in nested dicts"""
        data = {
            "config": {
                "api_key": "secret123",
                "endpoint": "https://api.example.com"
            }
        }
        sanitized = sanitize_dict(data)
        assert sanitized["config"]["api_key"] == REDACTED
        assert sanitized["config"]["endpoint"] == "https://api.example.com"

    def test_sanitize_headers(self):
        """Should redact authorization headers"""
        headers = {
            "Authorization": "Bearer secret_token",
            "Content-Type": "application/json",
            "X-Api-Key": "secret_key"
        }
        sanitized = sanitize_headers(headers)
        assert sanitized["Authorization"] == REDACTED
        assert sanitized["Content-Type"] == "application/json"
        assert sanitized["X-Api-Key"] == REDACTED

    def test_sanitize_url_with_query(self):
        """Should redact sensitive query parameters"""
        url = "https://api.example.com/endpoint?api_key=secret&data=public"
        sanitized = sanitize_url(url)
        assert "secret" not in sanitized
        assert "data=public" in sanitized or "data=" in sanitized

    def test_sanitize_url_no_query(self):
        """Should preserve URL without query params"""
        url = "https://api.example.com/endpoint"
        sanitized = sanitize_url(url)
        assert sanitized == url

    def test_sanitize_for_logging_dict(self):
        """Should sanitize dict for logging"""
        data = {"token": "secret123", "status": "ok"}
        result = sanitize_for_logging(data)
        assert "secret123" not in result
        assert "ok" in result

    def test_sanitize_for_logging_string(self):
        """Should handle plain strings"""
        result = sanitize_for_logging("simple message")
        assert result == "simple message"

    def test_sanitize_for_logging_handles_long_strings(self):
        """Should handle long strings without error"""
        long_string = "x" * 1000
        result = sanitize_for_logging(long_string)
        # Should return the string (possibly sanitized)
        assert isinstance(result, str)
        # Value should be preserved since it contains no secrets
        assert "x" in result


class TestRequestSigner:
    """Tests for HMAC request signing"""

    def test_kraken_signer_creates_headers(self):
        """Kraken signer should create API-Key and API-Sign headers"""
        signer = RequestSigner("test_key", "dGVzdF9zZWNyZXQ=", "kraken")  # base64 encoded
        headers = signer.sign_kraken_request("/0/private/Balance", {})

        assert "API-Key" in headers
        assert "API-Sign" in headers
        assert headers["API-Key"] == "test_key"
        assert len(headers["API-Sign"]) > 0

    def test_coinbase_signer_creates_headers(self):
        """Coinbase signer should create CB-ACCESS-* headers"""
        signer = RequestSigner("test_key", "dGVzdF9zZWNyZXQ=", "coinbase")
        headers = signer.sign_coinbase_request("GET", "/accounts", "")

        assert "CB-ACCESS-KEY" in headers
        assert "CB-ACCESS-SIGN" in headers
        assert "CB-ACCESS-TIMESTAMP" in headers
        assert headers["CB-ACCESS-KEY"] == "test_key"

    def test_sign_request_dispatcher(self):
        """sign_request should dispatch to correct exchange method"""
        signer = RequestSigner("key", "c2VjcmV0", "kraken")
        headers = signer.sign_request("POST", "/test", data={"a": 1})
        assert "API-Key" in headers

    def test_sign_request_unsupported_exchange(self):
        """Should raise ValueError for unsupported exchange"""
        signer = RequestSigner("key", "secret", "unknown_exchange")
        with pytest.raises(ValueError):
            signer.sign_request("GET", "/test")


class TestSecureConfigLoader:
    """Tests for SecureConfigLoader"""

    @patch.dict(os.environ, {
        "POLYMARKET_API_KEY": "poly_key_1234567890",
        "POLYMARKET_API_SECRET": "poly_secret_1234567890"
    })
    def test_load_polymarket_credentials(self):
        """Should load Polymarket credentials"""
        loader = SecureConfigLoader()
        creds = loader.load_polymarket_credentials(required=False)
        assert creds.api_key == "poly_key_1234567890"

    @patch.dict(os.environ, {
        "KRAKEN_API_KEY": "kraken_key_1234567890",
        "KRAKEN_API_SECRET": "a3Jha2VuX3NlY3JldA=="  # base64 encoded
    })
    def test_get_request_signer_kraken(self):
        """Should create request signer for Kraken"""
        loader = SecureConfigLoader()
        signer = loader.get_request_signer("kraken")
        assert signer is not None
        assert signer.exchange == "kraken"

    def test_get_request_signer_invalid_creds(self):
        """Should return None for invalid credentials"""
        loader = SecureConfigLoader()
        # Clear any existing cached credentials
        loader._credentials_cache.clear()
        loader._signers_cache.clear()
        signer = loader.get_request_signer("nonexistent_exchange")
        assert signer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
