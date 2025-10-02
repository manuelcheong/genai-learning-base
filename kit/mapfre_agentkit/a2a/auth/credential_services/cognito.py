import os
import time
from typing import Dict
import json
import requests
import jwt
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request
import httpx
from a2a.types import SecurityScheme
from mapfre_agentkit.a2a.auth.credential_services.base import (
    AuthStrategy,
)
from mapfre_agentkit.exceptions.auth import (
    InvalidAuthHeader,
    JWKSFetchError,
    JWKSParseError,
    JWKSConfigError,
)
import logging

logger = logging.getLogger(__name__)


class CognitoM2MCredentialService(AuthStrategy):
    """
    CredentialService implementation for AWS Cognito User Pool with machine-to-machine (M2M) authentication.

    This service uses the OAuth2 client credentials flow to fetch and cache access tokens from Cognito.
    Tokens are cached until expiration and refreshed automatically as needed.
    """

    def __init__(
        self,
        client_id=None,
        client_secret=None,
        user_pool_domain=None,
        scope=None,
        **kwargs,
    ):
        """
        Initialize the credential service with Cognito client credentials and domain.

        Args:
            client_id (str): The Cognito App Client ID.
            client_secret (str): The Cognito App Client Secret.
            user_pool_domain (str): The Cognito User Pool domain (e.g., 'my-domain.auth.eu-west-1.amazoncognito.com').
            scope (str, optional): OAuth2 scope(s) to request. Defaults to "openid".
        """
        self.client_id = client_id or os.getenv("COGNITO_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("COGNITO_CLIENT_SECRET")
        self.user_pool_domain = user_pool_domain or os.getenv(
            "COGNITO_USER_POOL_DOMAIN"
        )
        self.scope = scope or os.getenv("COGNITO_SCOPE")
        self.token_url = f"https://{self.user_pool_domain}/oauth2/token"
        self._access_token = None
        self._expires_at = 0
        logger.info(
            "CognitoM2MCredentialService initialized with scope: %s", self.scope
        )

    async def get_credentials(self, scheme_name=None, context=None):
        """
        Retrieve a valid access token for the given security scheme.

        Args:
            scheme_name (str): The security scheme name (ignored for Cognito M2M, included for interface compatibility).
            context: Optional context for the credential request (unused).

        Returns:
            str: The valid access token as a string.
        """
        if not self._access_token or time.time() > self._expires_at:
            await self._fetch_token()
        return self._access_token

    async def _fetch_token(self):
        """
        Fetch a new access token from Cognito using the client credentials flow.
        Updates the cached token and expiration time.

        Raises:
            httpx.HTTPStatusError: If the token endpoint returns an error status.
            httpx.RequestError: For network-related errors.
        """
        async with httpx.AsyncClient() as client:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope,
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            response = await client.post(self.token_url, data=data, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._expires_at = time.time() + token_data.get("expires_in", 3600) - 60

    def get_keys(self, config: SecurityScheme) -> Dict:
        """
        Retrieve the JWKS (JSON Web Key Set) from the provided endpoint in the security scheme.

        Args:
            config (SecurityScheme): The security scheme containing the JWKS endpoint URL in its description.

        Returns:
            dict: The JWKS as a dictionary.
        """

        try:
            jwt_endpoint = config.root.description
            response = requests.get(jwt_endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            raise JWKSFetchError(f"JWKS endpoint timeout: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise JWKSFetchError(f"JWKS endpoint connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            raise JWKSFetchError(f"JWKS endpoint HTTP error: {e}") from e
        except requests.exceptions.RequestException as e:
            raise JWKSFetchError(f"JWKS endpoint request failed: {e}") from e
        except ValueError as e:
            raise JWKSParseError(
                f"Invalid JSON response from JWKS endpoint: {e}"
            ) from e
        except TypeError as e:
            raise JWKSConfigError(f"Invalid config or response type: {e}") from e

    def get_token(self, request: Request) -> str:
        """
        Extract the Bearer token from the Authorization header of the incoming request.

        Args:
            request (Request): The incoming HTTP request.

        Returns:
            str: The extracted Bearer token.

        Raises:
            InvalidAuthHeader: If the Authorization header is missing or malformed.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise InvalidAuthHeader("Missing or malformed Authorization header.")

        token = auth_header.split("Bearer ")[1]
        return token

    def validate_token(self, token, config):
        """
        Validate a JWT token against the JWKS and return its payload if valid.

        Args:
            token (str): The JWT token to validate.

        Returns:
            dict: The decoded JWT payload if valid.

        Raises:
            jwt.ExpiredSignatureError: If the token is expired.
            jwt.InvalidTokenError: If the token is invalid.
            jwt.PyJWTError: For other JWT-related errors.
        """
        keys = {
            k["kid"]: RSAAlgorithm.from_jwk(json.dumps(k)) for k in config.get("keys")
        }
        header = jwt.get_unverified_header(token)
        key_id = header.get("kid")
        algorithm = header.get("alg")
        pub_key = keys.get(key_id)
        valid_token_data = jwt.decode(
            token, pub_key, audience=None, algorithms=[algorithm]
        )
        return valid_token_data
