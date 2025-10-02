import json
import os
import logging
import jwt
from jwt.algorithms import RSAAlgorithm

from a2a.types import AgentCard

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from mapfre_agentkit.config.auth_config import AuthConfig, AUTH_STRATEGIES, AuthType
from mapfre_agentkit.exceptions.auth import (
    InvalidAuthHeader,
    JWKSFetchError,
    JWKSParseError,
    JWKSConfigError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


class Middleware(BaseHTTPMiddleware):
    """
    Starlette middleware that authenticates A2A (Agent-to-Agent) access using JWT tokens.

    This middleware inspects incoming HTTP requests, checks for authentication requirements
    specified in the AgentCard, and validates JWT tokens against the agent's security configuration.
    It supports public (unauthenticated) paths and enforces required claims as specified in the AgentCard.
    """

    def __init__(
        self,
        app: Starlette,
        agent_card: AgentCard = None,
        auth: AuthConfig = None,
        public_paths: list[str] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app (Starlette): The Starlette application instance.
            agent_card (AgentCard, optional): The agent's metadata card containing security requirements.
            public_paths (list[str], optional): List of paths that do not require authentication.
        """
        super().__init__(app)
        self.agent_card = agent_card
        self.public_paths = set(public_paths or [])

        # Process the AgentCard to identify security requirements for authentication/authorization.
        self.a2a_auth = {}
        self.credential_service = None
        self.jwks = None

        try:
            if agent_card.security:
                sec_req = agent_card.security[0]
                if isinstance(sec_req, str):
                    scheme_name = sec_req
                    required_claims = []
                elif isinstance(sec_req, dict):
                    scheme_name = list(sec_req.keys())[0]
                    required_claims = sec_req[scheme_name]
                else:
                    raise ConfigurationError("Invalid security requirement format")

                self.security_scheme = agent_card.security_schemes[scheme_name]
                self.credential_service = AUTH_STRATEGIES[auth.type](auth)
                self.a2a_auth = {"required_claims": required_claims}
            else:
                self.credential_service = AUTH_STRATEGIES[AuthType.NO_AUTH]()

        except IndexError as e:
            logger.error("Security requirements list is empty", exc_info=True)
            raise ConfigurationError("No security requirements specified") from e
        except KeyError as e:
            logger.error(f"Missing key in agent card: {e}", exc_info=True)
            raise ConfigurationError("Invalid agent card structure") from e
        except TypeError as e:
            logger.error(f"Type error in agent card: {e}", exc_info=True)
            raise ConfigurationError("Malformed agent card") from e

    async def dispatch(self, request: Request, call_next):
        """
        Process each incoming request, enforcing authentication and authorization as needed.

        Args:
            request (Request): The incoming HTTP request.
            call_next (Callable): The next middleware or endpoint handler.

        Returns:
            Response: The HTTP response, either from the next handler or an error response.
        """
        path = request.url.path

        # Allow public paths and anonymous access
        if path in self.public_paths or isinstance(
            self.credential_service, AUTH_STRATEGIES[AuthType.NO_AUTH]
        ):
            return await call_next(request)

        try:
            keys = self.credential_service.get_keys(self.security_scheme)
            token = self.credential_service.get_token(request)
            payload = self.credential_service.validate_token(token, keys)
        except JWKSFetchError as e:
            logger.error(f"JWKS fetch error: {e}", exc_info=True)
            return self._service_unavailable("Unable to fetch JWKS", request)
        except JWKSParseError as e:
            logger.error(f"JWKS parse error: {e}", exc_info=True)
            return self._unauthorized("Invalid JWKS format", request)
        except JWKSConfigError as e:
            logger.error(f"JWKS config error: {e}", exc_info=True)
            return self._unauthorized("Invalid JWKS configuration", request)
        except jwt.ExpiredSignatureError as e:
            logger.error(f"Expired JWT: {e}", exc_info=True)
            return self._unauthorized(f"Expired JWT", request)
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid JWT: {e}", exc_info=True)
            return self._unauthorized(f"Invalid JWT", request)
        except jwt.PyJWTError as e:
            logger.error(f"Invalid JWT: {e}", exc_info=True)
            return self._unauthorized("Invalid JWT", request)
        except InvalidAuthHeader as e:
            logger.error(f"Invalid auth header: {e}", exc_info=True)
            return self._unauthorized("Invalid auth header", request)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return self._unauthorized("Unexpected error", request)

        # Check for required claims
        scopes = payload.get("scope", "").split()
        missing_claims = [
            claim for claim in self.a2a_auth["required_claims"] if claim not in scopes
        ]

        if missing_claims:
            logger.error(f"Missing required claims: {missing_claims}")
            return self._forbidden(
                f"Missing required claims: {missing_claims}", request
            )

        return await call_next(request)

    def _forbidden(self, reason: str, request: Request):
        """
        Return a 403 Forbidden response with an appropriate message.

        Args:
            reason (str): The reason for the forbidden response.
            request (Request): The incoming HTTP request.

        Returns:
            Response: A Starlette JSON or plain text response.
        """
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            return PlainTextResponse(
                f"error forbidden: {reason}",
                status_code=403,
                media_type="text/event-stream",
            )
        return JSONResponse({"error": "forbidden", "reason": reason}, status_code=403)

    def _unauthorized(self, reason: str, request: Request):
        """
        Return a 401 Unauthorized response with an appropriate message.

        Args:
            reason (str): The reason for the unauthorized response.
            request (Request): The incoming HTTP request.

        Returns:
            Response: A Starlette JSON or plain text response.
        """
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            return PlainTextResponse(
                f"error unauthorized: {reason}",
                status_code=401,
                media_type="text/event-stream",
            )
        return JSONResponse(
            {"error": "unauthorized", "reason": reason}, status_code=401
        )

    def _service_unavailable(self, reason: str, request: Request):
        """
        Return a 503 Service Unavailable response with an appropriate message.

        Args:
            reason (str): The reason for the service unavailable response.
            request (Request): The incoming HTTP request.

        Returns:
            Response: A Starlette JSON or plain text response.
        """
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            return PlainTextResponse(
                f"error service unavailable: {reason}",
                status_code=503,
                media_type="text/event-stream",
            )
        return JSONResponse(
            {"error": "service_unavailable", "reason": reason}, status_code=503
        )
