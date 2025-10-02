import logging
from typing import Any, Optional
from mapfre_agentkit.a2a.executors.utils.utils import (
    convert_genai_part_to_a2a,
    generate_payload,
)
from mapfre_agentkit.a2a.executors.base_agent_a2a_executor import BaseAgentA2AExecutor
from google.genai import types
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils.errors import ServerError
from a2a.types import (
    AgentCard,
    UnsupportedOperationError,
    TaskState,
    InternalError,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LangChainAgentA2AExecutor(BaseAgentA2AExecutor):
    """Executor for Agent-to-Agent (A2A) communication using LangChain.

    This class handles the execution of agent tasks using LangChain's AgentExecutor.

    Attributes:
        agent_executor (LangChainAgentExecutor): The LangChain agent executor instance.
        _card (AgentCard): The agent card containing metadata about the agent.
        _active_sessions (set[str]): Set of active session IDs for tracking.
    """

    def __init__(self, agent_executor: Any, card: AgentCard):
        """Initialize the LangChainAgentA2AExecutor.

        Args:
            agent_executor: The LangChain agent executor instance.
            card (AgentCard): The agent card containing metadata about the agent.
        """
        super().__init__(card)
        self.agent_executor = agent_executor

    async def _process_request(
        self,
        session_id: str,
        user_id: str,
        new_message: types.Content,
        task_updater: TaskUpdater,
        propagation_headers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Process an incoming request message using the LangChain agent executor.

        This method handles the core execution of a task using LangChain's agent executor.

        Args:
            session_id (str): The session ID for this request.
            user_id (str): The user ID for this request.
            new_message (types.Content): The content of the message to process.
            task_updater (TaskUpdater): The updater for reporting task progress and results.
            propagation_headers (Optional[dict[str, Any]]): The headers to be propagated to the remote agent.

        Returns:
            None
        """
        try:
            # Extract query text from the message parts
            query = ""
            if new_message and hasattr(new_message, "parts") and new_message.parts:
                for part in new_message.parts:
                    if hasattr(part, "text") and part.text:
                        query += part.text + " "
            query = query.strip()
            logger.info(
                f"User ID: {user_id}, Session ID: {session_id}, Propagation Headers: {propagation_headers}"
            )

            if not query:
                logger.warning("No text content found in the message parts")
                # Create error message using the same conversion pattern
                error_genai_part = types.Part(
                    text="No text content found in the message"
                )
                error_message_parts = [convert_genai_part_to_a2a(error_genai_part)]

                await task_updater.update_status(
                    state=TaskState.failed,
                    message=task_updater.new_agent_message(error_message_parts),
                    final=True,
                )
                return

            logger.debug(f"Processing query: {query} with session_id: {session_id}")

            async for item in self.agent_executor.stream(
                query, session_id, propagation_headers
            ):
                is_task_complete = item.get("is_task_complete", False)
                require_user_input = item.get("require_user_input", False)
                content = item.get("content", "")
                logger.debug(
                    f"is_task_complete: {is_task_complete}, require_user_input: {require_user_input}, content: {content}"
                )

                if require_user_input:
                    genai_part = types.Part(text=content)
                    message_parts = [convert_genai_part_to_a2a(genai_part)]
                    logger.debug("Agent requires user input")
                    await task_updater.update_status(
                        state=TaskState.input_required,
                        message=task_updater.new_agent_message(message_parts),
                        final=True,
                    )
                elif is_task_complete:
                    genai_part = types.Part(text=content)
                    message_parts = [convert_genai_part_to_a2a(genai_part)]
                    logger.debug("Task completed successfully, sending final response")
                    await task_updater.add_artifact(
                        message_parts,
                        name="final_agent_response",
                    )
                    await task_updater.complete()
                else:
                    logger.debug("Skipping event")

        except Exception as e:
            logger.error(
                f"An error occurred while streaming the response: {e}", exc_info=True
            )
            error_genai_part = types.Part(text=f"Error processing request: {str(e)}")
            error_message_parts = [convert_genai_part_to_a2a(error_genai_part)]

            await task_updater.update_status(
                state=TaskState.failed,
                message=task_updater.new_agent_message(error_message_parts),
                final=True,
            )
            raise ServerError(error=InternalError()) from e

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
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.update_status(TaskState.submitted)
        await updater.update_status(TaskState.working)
        payload = generate_payload(context, updater)
        await self._process_request(**payload)

        logger.debug("[LangChainAgentA2AExecutor] execute exiting")

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
        raise ServerError(error=UnsupportedOperationError())
