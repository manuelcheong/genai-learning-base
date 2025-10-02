"""
Agent configuration classes and registry.
"""

from typing import Dict, Type, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from a2a.types import SecurityScheme
from mapfre_agentkit.a2a.auth.credential_services.cognito import (
    CognitoM2MCredentialService,
)
from mapfre_agentkit.a2a.auth.credential_services.noauth import (
    NoAuthCredentialService,
)
from mapfre_agentkit.a2a.auth.credential_services.base import (
    AuthStrategy,
)


class AuthType(str, Enum):
    """Enum of available auth types."""

    NO_AUTH = "noauth"
    COGNITO = "cognito"


# Registry of available auth strategies
AUTH_STRATEGIES: Dict[str, Type[AuthStrategy]] = {
    AuthType.NO_AUTH: NoAuthCredentialService,
    AuthType.COGNITO: CognitoM2MCredentialService,
}


class AuthConfig(BaseModel):
    """Configuration for authentication."""

    securitySchemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[dict[str, List[str]]]] = Field(
        default=None,
        examples=[[{"oauth": ["read"]}, {"api-key": [], "mtls": []}]],
    )
    type: AuthType = AuthType.NO_AUTH
