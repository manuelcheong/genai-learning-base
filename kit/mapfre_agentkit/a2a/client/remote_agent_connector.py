import logging
from typing import Optional, AsyncGenerator
import httpx
from a2a.client.auth.interceptor import AuthInterceptor
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.middleware import ClientCallContext
from a2a.types import AgentCard, Message, SendMessageResponse
from mapfre_agentkit.config.auth_config import AuthConfig, AuthType, AUTH_STRATEGIES
from mapfre_agentkit.a2a.interceptors.session_interceptor import SessionInterceptor


logger = logging.getLogger(__name__)


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents.

    This class manages connections to remote agents using the A2A protocol.
    It provides methods for retrieving agent information and sending messages
    to remote agents.

    Attributes:
        _httpx_client (httpx.AsyncClient): The HTTP client used for asynchronous requests.
        agent_client (A2AClient): The A2A client used for interacting with the remote agent.
        card (AgentCard): The agent card containing metadata about the remote agent.
    """

    def __init__(self, agent_card: AgentCard, auth_config: Optional[AuthConfig] = None):
        """Initialize a connection to a remote agent.

        Args:
            agent_card (AgentCard): The agent card containing metadata about the remote agent.

        Raises:
            None

        Returns:
            None
        """
        self._httpx_client = httpx.AsyncClient(timeout=60)
        self.card = agent_card
        session_interceptor = SessionInterceptor()
        interceptors = [session_interceptor]
        scopes = None
        if auth_config.type != AuthType.NO_AUTH and auth_config.security:
            first_scheme = auth_config.security[0]
            if isinstance(first_scheme, dict):
                scope_lists = list(first_scheme.values())
                if scope_lists and isinstance(scope_lists[0], list):
                    scopes = " ".join(scope_lists[0])
            self.credential_service = AUTH_STRATEGIES[auth_config.type](scope=scopes)
            auth_interceptor = AuthInterceptor(self.credential_service)
            interceptors.append(auth_interceptor)

        config = ClientConfig(httpx_client=self._httpx_client)
        factory = ClientFactory(config=config)
        self.agent_client = factory.create(agent_card, interceptors=interceptors)

    def get_agent(self) -> AgentCard:
        """Get the agent card for this remote agent connection.

        Returns:
            AgentCard: The agent card containing metadata about the remote agent.
        """
        return self.card

    async def send_message(
        self,
        message_request: Message,
        call_context: Optional[ClientCallContext] = None,
    ) -> AsyncGenerator[SendMessageResponse, None]:
        """Send a message to the remote agent.

        Args:
            message_request (Message): The message request to send to the remote agent.
            call_context (Optional[ClientCallContext]): The call context to use for the request.

        Returns:
            AsyncGenerator[SendMessageResponse, None]: The response from the remote agent.
        """
        async for response in self.agent_client.send_message(
            request=message_request, context=call_context
        ):
            yield response
