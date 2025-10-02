from typing import Any, List, Optional, Union, Dict
import logging
import uuid
from opentelemetry.propagate import inject
from mapfre_agentkit.a2a.executors.adk.agent_a2a_executor import ADKAgentA2AExecutor
from mapfre_agentkit.agents.frameworks.base_agent_builder import AgentStrategy
from mapfre_agentkit.config.model_config import ModelConfig
from mapfre_agentkit.observability.observability import Observability
from google.adk.agents import Agent as ADKAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from a2a.types import (
    Task,
    Message,
)
from a2a.client.middleware import ClientCallContext

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


class ADKAgentStrategy(AgentStrategy):
    """Strategy for creating Google ADK agents.

    This strategy is always available as ADK is a core dependency.
    """

    def __init__(self, remote_agent_configs: list[Dict]):
        super().__init__(remote_agent_configs)

    def create_agent(
        self,
        name: str,
        model_config: ModelConfig,
        instruction: str,
        tools: List[Any],
        before_agent_callback: Optional[Any] = None,
    ) -> ADKAgent:
        """Create and return an ADK agent instance.

        Args:
            name: Name of the agent
            model_config: The LLM model configuration to use
            instruction: Instructions for the agent
            tools: List of tools available to the agent
            before_agent_callback: Optional callback to run before agent execution

        Returns:
            ADKAgent: The created ADK agent instance
        """
        return ADKAgent(
            name=name,
            model=self.get_model(model_config=model_config),
            before_agent_callback=before_agent_callback,
            instruction=instruction,
            tools=tools,
        )

    def create_executor(self, agent: Any, agent_card: Any) -> ADKAgentA2AExecutor:

        runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        return ADKAgentA2AExecutor(runner=runner, card=agent_card)

    def get_model(self, model_config: ModelConfig) -> Union[str, LiteLlm]:
        """Process the model configuration and return the appropriate model object.

        Returns:
            Either a model name string or a LiteLlm instance
        """
        try:
            logger.info(f"Parsed model config: {model_config}")

            if not model_config.provider:
                logger.info(f"Using model name: {model_config.name}")
                return model_config.name
            else:
                logger.info(
                    f"Using LiteLlm with provider: {model_config.provider.name}"
                )
                return LiteLlm(f"{model_config.provider.name}/{model_config.name}")
        except Exception as e:
            logger.error(f"Error in _get_model: {e}")
            raise e

    async def before_agent_callback(self, callback_context: CallbackContext):
        """Callback executed before agent processing begins.

        This method should be used to add guardrails.
        It's called automatically by the agent framework before processing requests.

        Args:
            callback_context: Context object containing the agent's state and other information.
        """
        logger.info("guardrails should be implemented here")

    async def send_message(
        self,
        agent_name: str,
        task: str,
        tool_context: ToolContext,
    ):
        """Delegate a task to a specified remote agent.

        This method sends a message to a remote agent, requesting it to perform a task.
        It handles the creation of the message payload and manages the communication
        with the remote agent.

        Args:
            agent_name: Name of the remote agent to send the task to.
            task: Detailed description of the task for the remote agent to perform.
            tool_context: Context object containing state of the tool and other information.

        Returns:
            Task object if successful, None otherwise.

        Raises:
            ValueError: If the specified agent is not found in the available connections.
        """
        logger.info(
            f"`send_message` triggered with agent_name: {agent_name}, task: {task}"
        )
        if agent_name not in self.remote_agent_connections:
            logger.error(
                f"LLM tried to call '{agent_name}' but it was not found. "
                f"Available agents: {list(self.remote_agent_connections.keys())}"
            )
            raise ValueError(f"Agent '{agent_name}' not found.")
        state = tool_context.state
        client = self.remote_agent_connections[agent_name]
        if "remote_agent_contexts" not in state:
            state["remote_agent_contexts"] = {}
        if agent_name not in state["remote_agent_contexts"]:
            logger.info(f"Creating new context for agent: {agent_name}")
            state["remote_agent_contexts"][agent_name] = {
                "context_id": str(uuid.uuid4())
            }
        context_id = state["remote_agent_contexts"][agent_name]["context_id"]
        task_id = state.get("task_id", None)
        message_id = state.get("input_message_metadata", {}).get(
            "message_id", str(uuid.uuid4())
        )
        payload = self.create_send_message_payload(task, task_id, context_id)
        payload["message"]["message_id"] = message_id
        logger.info("`send_message` triggered with payload: %s", payload)
        send_response = None
        async for resp in client.send_message(
            message_request=Message(**payload.get("message")),
            call_context=ClientCallContext(state=state.to_dict()),
        ):
            send_response = resp
        # Unpack tuple if needed
        if isinstance(send_response, tuple):
            send_response, _ = send_response
        # Now send_response is a Task, not SendMessageResponse
        if not isinstance(send_response, Task):
            return None
        return send_response

    def check_active_agent(self, context: ReadonlyContext):
        """Check if there is an active agent in the current session.

        Args:
            context: ReadonlyContext containing the current session state.

        Returns:
            dict: Dictionary with 'active_agent' key containing the name of the
                active agent or 'None' if no agent is active.
        """
        state = context.state
        if (
            "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f'{state["active_agent"]}'}
        return {"active_agent": "None"}
