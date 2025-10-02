from abc import abstractmethod
from typing import Dict, Optional
from starlette.requests import Request
from a2a.client.auth.credentials import CredentialService
from a2a.types import SecurityScheme


class AuthStrategy(CredentialService):
    """
    Base class for authentication strategies.

    This service uses the OAuth2 client credentials flow to fetch and cache access tokens from Cognito.
    Tokens are cached until expiration and refreshed automatically as needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get_keys(self, config: SecurityScheme) -> Dict:
        pass

    @abstractmethod
    async def get_token(self, request: Request) -> str:
        pass

    @abstractmethod
    def validate_token(self, token: str, config: Optional[Dict] = None):
        pass
