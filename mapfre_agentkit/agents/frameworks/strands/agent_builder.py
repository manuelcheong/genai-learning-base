from typing import Any, List, Optional, Union, Dict
import logging
import uuid
from mapfre_agentkit.agents.frameworks.base_agent_builder import AgentStrategy
from mapfre_agentkit.config.model_config import ModelConfig
from strands import Agent
from strands.multiagent.a2a.executor import StrandsA2AExecutor
from strands.models.litellm import LiteLLMModel
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    Message,
)

logger = logging.getLogger(__name__)


class StrandsAgentStrategy(AgentStrategy):
    """Strategy for creating Strands agents.

    This strategy is always available as Strands is a core dependency.
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
    ) -> Agent:
        """Create and return an ADK agent instance.

        Args:
            name: Name of the agent
            model_config: The LLM model to use
            instruction: Instructions for the agent
            tools: List of tools available to the agent
            before_agent_callback: Optional callback to run before agent execution

        Returns:
            Agent: The created Strands agent instance
        """

        return Agent(
            name=name,
            model=self.get_model(model_config),
            system_prompt=instruction,
            tools=tools,
        )

    def create_executor(self, agent: Any, agent_card: Any) -> StrandsA2AExecutor:

        return StrandsA2AExecutor(agent=agent)

    def get_model(self, model_config: ModelConfig) -> Union[str, LiteLLMModel]:
        """Process the model configuration and return the appropriate model object.

        Returns:
            Either a model name string or a LiteLLMModel instance
        """
        try:
            logger.info(f"Parsed model config: {model_config}")

            if not model_config.provider:
                logger.info(f"Using model name: {model_config.name}")
                return model_config.name
            else:
                logger.info(
                    f"Using LiteLLM with provider: {model_config.provider.name}"
                )
                return LiteLLMModel(
                    model_id=f"{model_config.provider.name}/{model_config.name}"
                )
        except Exception as e:
            logger.error(f"Error in get_model: {e}")
            raise e

    async def before_agent_callback(self, callback_context: Any):
        """Callback executed before agent processing begins.

        This method should be used to add guardrails.
        It's called automatically by the agent framework before processing requests.

        Args:
            callback_context: Context object containing the agent's state and other information.
        """
        logger.info("guardrails should be implemented here")

    async def send_message(
        self, agent_name: str, task: str, tool_context: Optional[Any] = None
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
        logger.info(
            f"`send_message` triggered with agent_name: {agent_name}, task: {task}"
        )
        if agent_name not in self.remote_agent_connections:
            logger.error(
                f"LLM tried to call '{agent_name}' but it was not found. "
                f"Available agents: {list(self.remote_agent_connections.keys())}"
            )
            raise ValueError(f"Agent '{agent_name}' not found.")

        client = self.remote_agent_connections[agent_name]

        message_id = str(uuid.uuid4())
        task_id = None
        context_id = None

        payload = self.create_send_message_payload(task, task_id, context_id)
        payload["message"]["message_id"] = message_id
        logger.info("`send_message` triggered with payload: %s", payload)

        send_response = None
        async for resp in client.send_message(
            message_request=Message(**payload.get("message"))
        ):
            send_response: SendMessageResponse = resp

        if isinstance(send_response, tuple):
            send_response, _ = send_response
        if not isinstance(send_response, Task):
            return None
        return send_response

    def check_active_agent(self, context: Any):
        """Check if there is an active agent in the current session.

        Args:
            context: ReadonlyContext containing the current session state.

        Returns:
            dict: Dictionary with 'active_agent' key containing the name of the
                active agent or 'None' if no agent is active.
        """
        raise NotImplementedError
