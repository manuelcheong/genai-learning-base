import logging
from typing import TYPE_CHECKING, Optional, Any
from mapfre_agentkit.a2a.executors.utils.utils import (
    convert_genai_part_to_a2a,
    generate_payload,
)
from mapfre_agentkit.a2a.executors.base_agent_a2a_executor import BaseAgentA2AExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils.errors import ServerError

from a2a.types import (
    AgentCard,
    TaskState,
    UnsupportedOperationError,
)
from google.adk.runners import Runner

from google.genai import types

if TYPE_CHECKING:
    from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ADKAgentA2AExecutor(BaseAgentA2AExecutor):
    """Executor for Agent-to-Agent (A2A) communication using Google ADK.

    This class handles the execution of agent tasks, manages sessions, and processes
    requests and responses between agents using the A2A protocol and Google ADK runner.

    Attributes:
        runner (Runner): The Google ADK runner instance for executing agent tasks.
        _card (AgentCard): The agent card containing metadata about the agent.
        _active_sessions (set[str]): Set of active session IDs for tracking.
    """

    def __init__(self, runner: Runner, card: AgentCard):
        """Initialize the ADKAgentA2AExecutor.

        Args:
            runner (Runner): The Google ADK runner instance for executing agent tasks.
            card (AgentCard): The agent card containing metadata about the agent.
        """
        super().__init__(card)
        self.runner = runner

    async def _process_request(
        self,
        session_id: str,
        user_id: str,
        new_message: types.Content,
        task_updater: TaskUpdater,
        propagation_headers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Process an incoming request message using the ADK runner.

        This method handles the core execution of a task, managing the session
        lifecycle, processing response events from the runner, and updating the
        task status.

        Args:
            session_id (str): The session ID for this request.
            user_id (str): The user ID for this request.
            new_message (types.Content): The content of the message to process.
            task_updater (TaskUpdater): The updater for reporting task progress and results.
            propagation_headers (Optional[dict[str, Any]]): The headers to be propagated to the remote agent.

        Returns:
            None
        """
        session_obj = await self._upsert_session(session_id=session_id, user_id=user_id)
        session_id = session_obj.id

        self._active_sessions.add(session_id)

        try:
            async for event in self.runner.run_async(
                session_id=session_id,
                user_id=user_id,
                new_message=new_message,
                state_delta=propagation_headers,
            ):
                if event.is_final_response():
                    parts = []
                    if event.content:
                        parts = [
                            convert_genai_part_to_a2a(part)
                            for part in event.content.parts
                            if (part.text or part.file_data or part.inline_data)
                        ]
                    logger.debug("Yielding final response: %s", parts)
                    await task_updater.add_artifact(parts)
                    await task_updater.update_status(TaskState.completed, final=True)
                    break
                if not event.get_function_calls():
                    logger.debug("Yielding update response")
                    message_parts = []
                    if event.content:
                        message_parts = [
                            convert_genai_part_to_a2a(part)
                            for part in event.content.parts
                            if (part.text)
                        ]
                    await task_updater.update_status(
                        TaskState.working,
                        message=task_updater.new_agent_message(message_parts),
                    )
                else:
                    logger.debug("Skipping event")
        finally:
            # Remove from active sessions when done
            self._active_sessions.discard(session_id)

    async def _upsert_session(self, session_id: str, user_id: str) -> "Session":
        """Retrieve a session if it exists, otherwise create a new one.

        Ensures that async session service methods are properly awaited.

        Args:
            session_id (str): The ID of the session to retrieve or create.

        Returns:
            Session: The retrieved or newly created session object.
        """
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        return session

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
        # Run the agent until either complete or the task is suspended.
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        # Immediately notify that the task is submitted.
        if not context.current_task:
            await updater.update_status(TaskState.submitted)
        await updater.update_status(TaskState.working)
        payload = generate_payload(context, updater)
        await self._process_request(**payload)

        logger.debug("[ADKAgentA2AExecutor] execute exiting")

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
        session_id = context.call_context.state.get("headers", {}).get(
            "x-mapfre-session-id", context.context_id
        )
        if session_id in self._active_sessions:
            logger.info(f"Cancellation requested for active session: {session_id}")
            self._active_sessions.discard(session_id)
        else:
            logger.debug(f"Cancellation requested for inactive session: {session_id}")

        raise ServerError(error=UnsupportedOperationError())
