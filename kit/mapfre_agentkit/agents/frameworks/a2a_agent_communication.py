import logging
import uuid
from abc import ABC, abstractmethod
import json
from typing import Any, Optional
import httpx
from mapfre_agentkit.a2a.client.remote_agent_connector import (
    RemoteAgentConnections,
)
from mapfre_agentkit.config.auth_config import AuthConfig
from a2a.client import A2ACardResolver
from a2a.client.middleware import ClientCallContext


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2AAgentCommunicationBase(ABC):
    """Base class for Agent-to-Agent communication."""

    def __init__(self, remote_agent_configs=None):
        self.remote_agent_connections = {}
        self.cards = {}
        self.agents = ""
        self.is_initialized = False
        self.remote_agent_configs = remote_agent_configs

    async def initialize(self):
        """Initialize connections to all remote agents asynchronously.

        Tests each connection individually with detailed logging to help identify
        any connection issues. It attempts to connect to each remote agent address,
        retrieve its agent card, and store the connection for later use.

        Raises:
            No exceptions are raised, but errors are logged.
        """
        if not self.remote_agent_configs or not self.remote_agent_configs[0]:
            logger.error(
                "CRITICAL FAILURE: REMOTE_AGENT_CONFIGS environment variable is empty. "
                "Cannot proceed."
            )
            self.is_initialized = True
            return

        async with httpx.AsyncClient(timeout=60) as client:
            for i, config in enumerate(self.remote_agent_configs):
                address = config.get("url")
                auth = AuthConfig(**config.get("auth"))
                logger.info(f"--- STEP 3.{i}: Attempting connection to: {address} ---")
                try:
                    card_resolver = A2ACardResolver(client, address)
                    card = await card_resolver.get_agent_card()

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, auth_config=auth
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                    logger.info(
                        f"--- STEP 5.{i}: Successfully stored connection for {card.name} ---"
                    )

                except Exception as e:
                    logger.error(
                        f"--- CRITICAL FAILURE at STEP 4.{i} for address: {address} ---"
                    )
                    logger.error(
                        f"--- The hidden exception type is: {type(e).__name__} ---"
                    )
                    logger.error(
                        f"--- Full exception details and traceback: ---", exc_info=True
                    )

        logger.info("STEP 6: Finished attempting all connections.")
        if not self.remote_agent_connections:
            logger.error(
                "FINAL VERDICT: The loop finished, but the remote agent list is still empty."
            )
        else:
            agent_info = [
                json.dumps({"name": c.name, "description": c.description})
                for c in self.cards.values()
            ]
            self.agents = "\n".join(agent_info)
            logger.info(
                f"--- FINAL SUCCESS: Initialization complete. {len(self.remote_agent_connections)} agents loaded. ---"
            )

        self.is_initialized = True

    @abstractmethod
    async def before_agent_callback(self, callback_context: Any):
        """Callback executed before agent processing begins.

        This method should be used to add guardrails.
        It's called automatically by the agent framework before processing requests.

        Args:
            callback_context: Context object containing the agent's state and other information.
        """

    @abstractmethod
    async def send_message(
        self,
        agent_name: str,
        task: str,
        tool_context: Optional[Any] = None,
    ):
        """Delegate a task to a specified remote agent.

        This method sends a message to a remote agent, requesting it to perform a task.
        It handles the creation of the message payload and manages the communication
        with the remote agent.

        Args:
            agent_name: Name of the remote agent to send the task to.
            task: Detailed description of the task for the remote agent to perform.
            tool_context: Optional context object containing state of the tool and other information.

        Returns:
            Task object if successful, None otherwise.

        Raises:
            ValueError: If the specified agent is not found in the available connections.
        """

    @abstractmethod
    def check_active_agent(self, context: Any):
        """Check if there is an active agent in the current session.

        Args:
            context: Variable containing the current session state.

        Returns:
            dict: Dictionary with 'active_agent' key containing the name of the
                active agent or 'None' if no agent is active.
        """

    def list_remote_agents(self):
        """List all available remote agents.

        Returns:
            list: List of dictionaries containing name and description for each
                remote agent. Empty list if no agents are available.
        """
        if not self.cards:
            return []
        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    @staticmethod
    def create_send_message_payload(
        text: str,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a message payload for sending to a remote agent.

        Args:
            text: The text content of the message.
            task_id: Optional task ID to associate with the message.
            context_id: Optional context ID to associate with the message.

        Returns:
            dict: A dictionary containing the formatted message payload ready
                to be sent to a remote agent.
        """
        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}],
                "message_id": uuid.uuid4().hex,
            },
        }
        if task_id:
            payload["message"]["task_id"] = task_id
        if context_id:
            payload["message"]["context_id"] = context_id
        return payload
