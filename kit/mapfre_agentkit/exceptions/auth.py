class InvalidAuthHeader(Exception):
    """Raised when the Authorization header is missing or malformed."""

    pass


class JWKSFetchError(Exception):
    """Raised when there is an error fetching the JWKS from the endpoint (e.g., network, timeout, HTTP error)."""

    pass


class JWKSParseError(Exception):
    """Raised when the JWKS response cannot be parsed as valid JSON or has an unexpected format."""

    pass


class JWKSConfigError(Exception):
    """Raised when the JWKS configuration is invalid or missing required fields."""

    pass


class ConfigurationError(Exception):
    """Raised when the configuration is invalid or missing required fields."""

    pass
