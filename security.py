"""
Security Module
Secure secrets management, credential validation, and log sanitization.

This module provides:
- Secure loading and validation of API credentials
- Log sanitization to prevent credential leakage
- HMAC request signing for exchange APIs
"""

import os
import re
import hmac
import hashlib
import base64
import time
import urllib.parse
from typing import Dict, Optional, Any, List, Set
from dataclasses import dataclass
from functools import lru_cache
import logging

from constants import MIN_API_KEY_LENGTH, MIN_API_SECRET_LENGTH
from exceptions import ConfigurationError

logger = logging.getLogger("PolyMangoBot.security")


# =============================================================================
# SENSITIVE DATA PATTERNS
# =============================================================================

# Patterns that indicate sensitive data in strings
SENSITIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r'api[_-]?key', re.IGNORECASE),
    re.compile(r'api[_-]?secret', re.IGNORECASE),
    re.compile(r'password', re.IGNORECASE),
    re.compile(r'token', re.IGNORECASE),
    re.compile(r'bearer\s+\S+', re.IGNORECASE),
    re.compile(r'authorization', re.IGNORECASE),
    re.compile(r'private[_-]?key', re.IGNORECASE),
    re.compile(r'secret[_-]?key', re.IGNORECASE),
]

# Keys in dictionaries that should be redacted
SENSITIVE_KEYS: Set[str] = {
    'api_key', 'apikey', 'api-key',
    'api_secret', 'apisecret', 'api-secret',
    'secret', 'password', 'passwd',
    'token', 'access_token', 'refresh_token',
    'authorization', 'auth',
    'private_key', 'privatekey',
    'secret_key', 'secretkey',
    'signature', 'sig',
    'nonce', 'otp',
}

# Redaction placeholder
REDACTED: str = "[REDACTED]"


# =============================================================================
# CREDENTIAL VALIDATION
# =============================================================================

@dataclass
class ValidatedCredentials:
    """Container for validated API credentials"""
    api_key: str
    api_secret: str
    is_valid: bool
    validation_errors: List[str]

    def __post_init__(self):
        # Never store secrets in string representation
        self._key_hash = hashlib.sha256(self.api_key.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return f"ValidatedCredentials(key_hash={self._key_hash}, valid={self.is_valid})"


def validate_api_key(key: Optional[str], name: str = "API_KEY") -> List[str]:
    """
    Validate an API key format.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    if key is None:
        errors.append(f"{name} is not set")
        return errors

    if not isinstance(key, str):
        errors.append(f"{name} must be a string")
        return errors

    key = key.strip()

    if not key:
        errors.append(f"{name} is empty")
        return errors

    if len(key) < MIN_API_KEY_LENGTH:
        errors.append(f"{name} appears too short (min {MIN_API_KEY_LENGTH} chars)")

    # Check for placeholder values
    placeholder_patterns = ['xxx', 'your_', 'insert_', 'replace_', '<', '>']
    for pattern in placeholder_patterns:
        if pattern in key.lower():
            errors.append(f"{name} appears to be a placeholder value")
            break

    return errors


def validate_api_secret(secret: Optional[str], name: str = "API_SECRET") -> List[str]:
    """
    Validate an API secret format.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    if secret is None:
        errors.append(f"{name} is not set")
        return errors

    if not isinstance(secret, str):
        errors.append(f"{name} must be a string")
        return errors

    secret = secret.strip()

    if not secret:
        errors.append(f"{name} is empty")
        return errors

    if len(secret) < MIN_API_SECRET_LENGTH:
        errors.append(f"{name} appears too short (min {MIN_API_SECRET_LENGTH} chars)")

    return errors


def load_and_validate_credentials(
    key_env_var: str,
    secret_env_var: str,
    required: bool = True
) -> ValidatedCredentials:
    """
    Load and validate API credentials from environment variables.

    Args:
        key_env_var: Environment variable name for API key
        secret_env_var: Environment variable name for API secret
        required: If True, raise ConfigurationError on validation failure

    Returns:
        ValidatedCredentials object

    Raises:
        ConfigurationError: If required=True and validation fails
    """
    api_key = os.getenv(key_env_var, "").strip()
    api_secret = os.getenv(secret_env_var, "").strip()

    errors = []
    errors.extend(validate_api_key(api_key, key_env_var))
    errors.extend(validate_api_secret(api_secret, secret_env_var))

    is_valid = len(errors) == 0

    if required and not is_valid:
        error_msg = f"Credential validation failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)

    if errors:
        for error in errors:
            logger.warning(f"Credential warning: {error}")

    return ValidatedCredentials(
        api_key=api_key,
        api_secret=api_secret,
        is_valid=is_valid,
        validation_errors=errors
    )


# =============================================================================
# LOG SANITIZATION
# =============================================================================

def sanitize_value(value: Any, max_length: int = 500) -> Any:
    """
    Sanitize a single value for logging.

    Handles strings, dicts, lists, and other types.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return sanitize_dict(value, max_length)

    if isinstance(value, (list, tuple)):
        return [sanitize_value(item, max_length) for item in value[:10]]  # Limit list size

    if isinstance(value, str):
        # Check if string looks like a secret
        if len(value) > 20 and not ' ' in value:
            # Long string without spaces might be a token/key
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(value):
                    return REDACTED

        # Truncate long strings
        if len(value) > max_length:
            return value[:max_length] + "...[truncated]"

        return value

    # For other types, convert to string and truncate
    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "...[truncated]"

    return value


def sanitize_dict(data: Dict[str, Any], max_length: int = 500) -> Dict[str, Any]:
    """
    Sanitize a dictionary for logging by redacting sensitive keys.

    Args:
        data: Dictionary to sanitize
        max_length: Maximum length for string values

    Returns:
        Sanitized copy of the dictionary
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}

    for key, value in data.items():
        key_lower = key.lower().replace('-', '_')

        # Check if key is sensitive
        if key_lower in SENSITIVE_KEYS:
            sanitized[key] = REDACTED
        elif any(pattern.search(key) for pattern in SENSITIVE_PATTERNS):
            sanitized[key] = REDACTED
        else:
            # Recursively sanitize nested structures
            sanitized[key] = sanitize_value(value, max_length)

    return sanitized


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Sanitize HTTP headers for logging.

    Always redacts Authorization and other auth-related headers.
    """
    sensitive_headers = {
        'authorization', 'x-api-key', 'api-key', 'apikey',
        'x-api-secret', 'api-secret', 'cookie', 'set-cookie',
        'x-auth-token', 'x-access-token'
    }

    sanitized = {}
    for key, value in headers.items():
        if key.lower() in sensitive_headers:
            sanitized[key] = REDACTED
        else:
            sanitized[key] = value

    return sanitized


def sanitize_url(url: str) -> str:
    """
    Sanitize a URL by redacting query parameters that might contain secrets.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.query:
            return url

        query_params = urllib.parse.parse_qs(parsed.query)
        sanitized_params = {}

        for key, values in query_params.items():
            key_lower = key.lower()
            if key_lower in SENSITIVE_KEYS or any(p.search(key) for p in SENSITIVE_PATTERNS):
                sanitized_params[key] = [REDACTED]
            else:
                sanitized_params[key] = values

        sanitized_query = urllib.parse.urlencode(sanitized_params, doseq=True)
        sanitized_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, sanitized_query, parsed.fragment
        ))

        return sanitized_url
    except Exception:
        # If parsing fails, return a safe version
        return url.split('?')[0] + "?[query_redacted]"


def sanitize_for_logging(data: Any, context: str = "") -> str:
    """
    Prepare any data for safe logging.

    This is the main entry point for log sanitization.

    Args:
        data: Any data to sanitize (dict, string, object, etc.)
        context: Optional context string for the log

    Returns:
        String safe for logging
    """
    try:
        if isinstance(data, dict):
            sanitized = sanitize_dict(data)
            return str(sanitized)
        elif isinstance(data, str):
            # Check for URL
            if data.startswith(('http://', 'https://')):
                return sanitize_url(data)
            return sanitize_value(data)
        else:
            return sanitize_value(data)
    except Exception as e:
        return f"[sanitization_error: {type(data).__name__}]"


# =============================================================================
# HMAC REQUEST SIGNING
# =============================================================================

class RequestSigner:
    """
    HMAC request signing for exchange APIs.

    Supports Kraken and Coinbase signing schemes.
    """

    def __init__(self, api_key: str, api_secret: str, exchange: str = "kraken"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = exchange.lower()

    def sign_kraken_request(
        self,
        url_path: str,
        data: Dict[str, Any],
        nonce: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Sign a request for Kraken API.

        Kraken uses: HMAC-SHA512(path + SHA256(nonce + POST data), secret)

        Returns:
            Headers dict with API-Key and API-Sign
        """
        if nonce is None:
            nonce = int(time.time() * 1000)

        data['nonce'] = nonce

        # Encode POST data
        post_data = urllib.parse.urlencode(data)

        # Create signature
        # SHA256(nonce + post_data)
        sha256_hash = hashlib.sha256()
        sha256_hash.update((str(nonce) + post_data).encode('utf-8'))

        # HMAC-SHA512(url_path + sha256_hash, base64_decoded_secret)
        secret_decoded = base64.b64decode(self.api_secret)
        hmac_digest = hmac.new(
            secret_decoded,
            url_path.encode('utf-8') + sha256_hash.digest(),
            hashlib.sha512
        )

        signature = base64.b64encode(hmac_digest.digest()).decode('utf-8')

        return {
            'API-Key': self.api_key,
            'API-Sign': signature,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

    def sign_coinbase_request(
        self,
        method: str,
        request_path: str,
        body: str = "",
        timestamp: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Sign a request for Coinbase Pro API.

        Coinbase uses: HMAC-SHA256(timestamp + method + path + body, secret)

        Returns:
            Headers dict with CB-ACCESS-* headers
        """
        if timestamp is None:
            timestamp = str(time.time())

        # Create message
        message = timestamp + method.upper() + request_path + body

        # Sign with HMAC-SHA256
        secret_decoded = base64.b64decode(self.api_secret)
        signature = hmac.new(
            secret_decoded,
            message.encode('utf-8'),
            hashlib.sha256
        )
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')

        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': os.getenv('COINBASE_PASSPHRASE', ''),
            'Content-Type': 'application/json'
        }

    def sign_request(
        self,
        method: str,
        url_path: str,
        data: Optional[Dict[str, Any]] = None,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Sign a request using the appropriate method for the exchange.

        Args:
            method: HTTP method (GET, POST, etc.)
            url_path: API endpoint path
            data: Form data (for Kraken)
            body: JSON body (for Coinbase)

        Returns:
            Headers dict with authentication
        """
        if self.exchange == "kraken":
            return self.sign_kraken_request(url_path, data or {})
        elif self.exchange == "coinbase":
            return self.sign_coinbase_request(method, url_path, body)
        else:
            raise ValueError(f"Unsupported exchange for signing: {self.exchange}")


# =============================================================================
# SECURE CONFIGURATION LOADER
# =============================================================================

class SecureConfigLoader:
    """
    Secure configuration loader with validation and error handling.
    """

    def __init__(self):
        self._credentials_cache: Dict[str, ValidatedCredentials] = {}
        self._signers_cache: Dict[str, RequestSigner] = {}

    def load_polymarket_credentials(self, required: bool = True) -> ValidatedCredentials:
        """Load and validate Polymarket API credentials."""
        cache_key = "polymarket"
        if cache_key not in self._credentials_cache:
            self._credentials_cache[cache_key] = load_and_validate_credentials(
                "POLYMARKET_API_KEY",
                "POLYMARKET_API_SECRET",
                required=required
            )
        return self._credentials_cache[cache_key]

    def load_kraken_credentials(self, required: bool = True) -> ValidatedCredentials:
        """Load and validate Kraken API credentials."""
        cache_key = "kraken"
        if cache_key not in self._credentials_cache:
            self._credentials_cache[cache_key] = load_and_validate_credentials(
                "KRAKEN_API_KEY",
                "KRAKEN_API_SECRET",
                required=required
            )
        return self._credentials_cache[cache_key]

    def load_coinbase_credentials(self, required: bool = True) -> ValidatedCredentials:
        """Load and validate Coinbase API credentials."""
        cache_key = "coinbase"
        if cache_key not in self._credentials_cache:
            self._credentials_cache[cache_key] = load_and_validate_credentials(
                "COINBASE_API_KEY",
                "COINBASE_API_SECRET",
                required=required
            )
        return self._credentials_cache[cache_key]

    def get_request_signer(self, exchange: str) -> Optional[RequestSigner]:
        """
        Get a request signer for the specified exchange.

        Returns None if credentials are not valid.
        """
        exchange = exchange.lower()

        if exchange in self._signers_cache:
            return self._signers_cache[exchange]

        try:
            if exchange == "kraken":
                creds = self.load_kraken_credentials(required=False)
            elif exchange == "coinbase":
                creds = self.load_coinbase_credentials(required=False)
            else:
                logger.warning(f"No signer available for exchange: {exchange}")
                return None

            if not creds.is_valid:
                logger.warning(f"Cannot create signer for {exchange}: invalid credentials")
                return None

            signer = RequestSigner(creds.api_key, creds.api_secret, exchange)
            self._signers_cache[exchange] = signer
            return signer

        except Exception as e:
            logger.error(f"Error creating signer for {exchange}: {e}")
            return None


# Global secure config loader instance
_secure_config: Optional[SecureConfigLoader] = None


def get_secure_config() -> SecureConfigLoader:
    """Get the global secure configuration loader."""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfigLoader()
    return _secure_config
