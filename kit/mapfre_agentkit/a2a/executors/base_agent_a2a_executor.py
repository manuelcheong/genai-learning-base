import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater

from a2a.types import (
    AgentCard,
)
from google.genai import types


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseAgentA2AExecutor(AgentExecutor, ABC):
    """Base abstract class for Agent-to-Agent (A2A) executors.

    This class defines the common interface and shared functionality for all
    executor implementations, regardless of the underlying agent framework.

    Attributes:
        _card (AgentCard): The agent card containing metadata about the agent.
        _active_sessions (set[str]): Set of active session IDs for tracking.
    """

    def __init__(self, card: AgentCard):
        """Initialize the BaseAgentA2AExecutor.

        Args:
            card (AgentCard): The agent card containing metadata about the agent.
        """
        self._card = card
        # Track active sessions for potential cancellation
        self._active_sessions: set[str] = set()

    @abstractmethod
    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
        propagation_headers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Process an incoming request message.

        This abstract method must be implemented by concrete executor classes to handle
        the core execution of a task using the specific agent framework.

        Args:
            new_message (types.Content): The message to process.
            session_id (str): The session ID for this request.
            task_updater (TaskUpdater): The updater for reporting task progress and results.
            propagation_headers (Optional[dict[str, Any]]): The headers to be propagated to the remote agent.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Execute a task based on the provided request context.

        This is the main entry point for task execution. It sets up the task
        updater, notifies about task submission and status changes, and processes
        the request.

        Args:
            context (RequestContext): The context containing task information and message.
            event_queue (EventQueue): The queue for sending task events and updates.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the execution for the given context.

        Base implementation that logs the cancellation attempt and raises an error
        since cancellation may not be supported by all frameworks.

        Args:
            context (RequestContext): The context containing task information.
            event_queue (EventQueue): The queue for sending task events and updates.

        Returns:
            None

        Raises:
            ServerError: Raised with UnsupportedOperationError since
                cancellation may not be supported.
        """
        pass
