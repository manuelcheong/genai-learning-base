from mapfre_agentkit.a2a.auth.credential_services.base import (
    AuthStrategy,
)


class NoAuthCredentialService(AuthStrategy):
    """
    CredentialService implementation for no authentication.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the NoAuthCredentialService.
        This implementation ignores all arguments and does not require any setup.
        """
        pass

    async def get_credentials(self, scheme_name=None, context=None):
        """
        Return None as no credentials are required for unauthenticated access.
        Args:
            scheme_name (str, optional): Ignored.
            context (any, optional): Ignored.
        Returns:
            None
        """
        return None

    def get_keys(self, config=None):
        """
        Return an empty dictionary as no keys are needed for unauthenticated access.
        Args:
            config (any, optional): Ignored.
        Returns:
            dict: An empty dictionary.
        """
        return {}

    async def get_token(self, request):
        """
        Return None as no token extraction is performed for unauthenticated access.
        Args:
            request: The incoming HTTP request (ignored).
        Returns:
            None
        """
        return None

    def validate_token(self, token, config=None):
        """
        Return an empty dictionary as no token validation is performed for unauthenticated access.
        Args:
            token (any): Ignored.
            config (any, optional): Ignored.
        Returns:
            dict: An empty dictionary.
        """
        return {}
