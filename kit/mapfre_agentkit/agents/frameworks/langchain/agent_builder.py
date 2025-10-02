from typing import Any, AsyncIterable, Dict, List, Optional, Literal
import types
import logging
import uuid

# Removed unused import of ANNOTATED_FIELD_UNTOUCHED_TYPES from pydantic.v1.main
from mapfre_agentkit.agents.frameworks.base_agent_builder import AgentStrategy
from mapfre_agentkit.a2a.executors.langchain.agent_a2a_executor import (
    LangChainAgentA2AExecutor,
)
from mapfre_agentkit.config.model_config import ModelConfig
from langchain.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from a2a.types import (
    Message,
    Task,
)
from a2a.client.middleware import ClientCallContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    """Structured response format for LangChain agents."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class LangChainAgentStrategy(AgentStrategy):
    """Strategy for creating agents using LangChain.

    This strategy provides a implementation for creating LangChain
    agents with proper error handling, tool validation, and
    configuration management. It supports both regular invocation
    and streaming responses for real-time interaction.
    """

    def __init__(self, remote_agent_configs: list[Dict]):
        super().__init__(remote_agent_configs)

    @staticmethod
    def _flatten_tools(tools: List[Any]) -> List[Any]:
        """Flatten a list of tools that may contain nested lists."""
        flattened_tools = []
        for item in tools:
            if isinstance(item, list):
                flattened_tools.extend(item)
            else:
                flattened_tools.append(item)
        return flattened_tools

    def create_agent(
        self,
        name: str,
        model_config: Any,
        instruction: str,
        tools: List[Any],
        before_agent_callback: Optional[Any] = None,
    ) -> Any:
        """Create and return a LangChain agent instance.

        Args:
            name: The name of the agent
            model: The LLM to use (must be a LangChain-compatible model)
            instruction: The agent instructions/system prompt
            tools: List of tools for the agent to use
            before_agent_callback: Optional callback to execute before agent runs

        Returns:
            Agent: A configured LangChain agent with streaming capabilities

        Raises:
            ValueError: If required parameters are invalid
            RuntimeError: If agent creation fails
        """

        if not name:
            raise ValueError("Agent name is required")
        if not instruction:
            raise ValueError("Agent instruction is required")
        if not model_config:
            raise ValueError("Model configuration is required")

        try:
            llm = self._initialize_llm(self.get_model(model_config))
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {str(e)}") from e

        # Removed commented-out code related to `validated_tools`.

        flattened_tools = self._flatten_tools(tools)

        try:
            # Create a React agent with structured response format
            agent = create_react_agent(
                model=llm,
                tools=flattened_tools,
                checkpointer=MemorySaver(),
                prompt=instruction,
            )

            # Add streaming capabilities to the agent
            bound_method = types.MethodType(self._stream, agent)
            setattr(agent, "stream", bound_method)
            # setattr(agent, "get_response_from_state", cls._get_response_from_state)

            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create LangChain agent: {str(e)}") from e

    def create_executor(self, agent: Any, agent_card: Any) -> LangChainAgentA2AExecutor:

        return LangChainAgentA2AExecutor(agent_executor=agent, card=agent_card)

    def get_model(self, model_config: ModelConfig) -> str:
        """Process the model configuration and return the appropriate model object.

        Returns:
            A model name string
        """
        try:
            logger.info(f"Parsed model config: {model_config}")

            if not model_config.provider:
                logger.info(f"Using model name: {model_config.name}")
                return model_config.name
            else:
                logger.info(f"Using model with provider: {model_config.provider.name}")
                return f"{model_config.provider.name}:{model_config.name}"
        except Exception as e:
            logger.error(f"Error in _get_model: {e}")
            raise e

    async def _stream(
        self,
        agent: Any,
        query: str,
        session_id: str,
        propagation_headers: Optional[dict[str, Any]] = None,
    ) -> AsyncIterable[Dict[str, Any]]:
        """Stream method for the agent to provide real-time responses.

        Args:
            agent: The LangGraph agent instance
            query: User query to process
            session_id: Unique session identifier for state management

        Returns:
            AsyncIterable of response chunks
        """
        logger.info(
            f"Session ID: {session_id}, Propagation Headers: {propagation_headers}"
        )
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        if propagation_headers:
            config["configurable"]["propagation_headers"] = propagation_headers
        langgraph_input = {"messages": [("user", query)]}

        try:
            async for chunk in agent.astream_events(
                langgraph_input, config, version="v1"
            ):
                event_name = chunk.get("event")
                data = chunk.get("data", {})
                content_to_yield = None

                if event_name == "on_chat_model_stream":
                    message_chunk = data.get("chunk")
                    if (
                        isinstance(message_chunk, AIMessageChunk)
                        and message_chunk.content
                    ):
                        content_to_yield = message_chunk.content

                if content_to_yield:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": content_to_yield,
                    }

            # After all events, get the final structured response
            final_response = self._get_response_from_state(config, agent)
            yield final_response

        except Exception as e:
            logging.error(f"Error during agent streaming: {e}", exc_info=True)
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"An error occurred during streaming: {str(e)}",
            }

    def _get_response_from_state(
        self, config: RunnableConfig, agent: Any
    ) -> Dict[str, Any]:
        """Extract the final response from the agent's state.

        Args:
            config: The runnable configuration with thread_id
            agent: The LangGraph agent instance

        Returns:
            Structured response dictionary
        """
        try:
            if not hasattr(agent, "get_state"):
                logging.error("Agent does not have get_state method")
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Internal error: Agent state retrieval misconfigured.",
                }

            current_state = agent.get_state(config)
            state_values = getattr(current_state, "values", None)

            if not state_values:
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Error: Agent state is unavailable.",
                }

            # Try to get structured response first
            structured_response = (
                state_values.get("structured_response")
                if isinstance(state_values, dict)
                else getattr(state_values, "structured_response", None)
            )

            if structured_response and isinstance(structured_response, ResponseFormat):
                if structured_response.status == "completed":
                    return {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": structured_response.message,
                    }
                return {
                    "is_task_complete": False,
                    "require_user_input": structured_response.status
                    == "input_required",
                    "content": structured_response.message,
                }

            # Fallback to last AI message if structured response not available
            final_messages = (
                state_values.get("messages", [])
                if isinstance(state_values, dict)
                else getattr(state_values, "messages", [])
            )

            if final_messages and isinstance(final_messages[-1], AIMessage):
                ai_content = final_messages[-1].content
                if isinstance(ai_content, str) and ai_content:
                    return {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": ai_content,
                    }

            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your request at the moment.",
            }

        except Exception as e:
            logging.error(f"Error getting response from state: {e}", exc_info=True)
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error retrieving response: {str(e)}",
            }

    def _initialize_llm(self, model_config: Any) -> Any:
        """Initialize a LangChain LLM from the provided model configuration.

        Args:
            model_config: Model configuration (name or dict with parameters)

        Returns:
            A LangChain LLM instance
        """
        if isinstance(model_config, str):
            return init_chat_model(model=model_config)

        if isinstance(model_config, dict):
            model_name = model_config.get("name")
            if not model_name:
                raise ValueError("Model name is required in model configuration")

            model_params = {k: v for k, v in model_config.items() if k != "name"}
            return init_chat_model(model=model_name, **model_params)

        raise ValueError(f"Unsupported model configuration type: {type(model_config)}")

    def _validate_tools(self, tools: List[Any]) -> List[BaseTool]:
        """Validate and convert tools to LangChain BaseTool format.

        Args:
            tools: List of tools in various formats

        Returns:
            List of validated LangChain BaseTools
        """
        validated_tools = []

        for tool in tools:
            try:
                if isinstance(tool, BaseTool):
                    validated_tools.append(tool)
                    continue
                if (
                    isinstance(tool, dict)
                    and "name" in tool
                    and "description" in tool
                    and "func" in tool
                ):
                    validated_tools.append(
                        Tool(
                            name=tool["name"],
                            description=tool["description"],
                            func=tool["func"],
                        )
                    )
                    continue

                if (
                    hasattr(tool, "name")
                    and hasattr(tool, "description")
                    and hasattr(tool, "__call__")
                ):
                    validated_tools.append(
                        Tool(
                            name=tool.name,
                            description=tool.description,
                            func=tool.__call__,
                        )
                    )
                    continue

                logging.warning(f"Skipping invalid tool: {tool}")

            except Exception as e:
                logging.error(f"Error validating tool: {str(e)}")
                raise e

        return validated_tools

    async def before_agent_callback(self, callback_context: Any):
        """Callback executed before agent processing begins.

        This method should be used to add guardrails.
        It's called automatically by the agent framework before processing requests.

        Args:
            callback_context: Context object containing the agent's state and other information.
        """
        logger.info("guardrails should be implemented here")

    async def send_message(
        self, agent_name: str, task: str, tool_context: RunnableConfig
    ):
        """Delegate a task to a specified remote agent.

        This method sends a message to a remote agent, requesting it to perform a task.
        It handles the creation of the message payload and manages the communication
        with the remote agent.

        Args:
            agent_name: Name of the remote agent to send the task to.
            task: Detailed description of the task for the remote agent to perform.
            tool_context: RunnableConfig object containing the tool context.

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

        configurable_context = tool_context.get("configurable", {})
        # TODO: populate task_id as it currently raises an unexpected error
        # task_id = configurable_context.get("__pregel_task_id")
        context_id = configurable_context.get("thread_id")
        task_id = None
        message_id = str(uuid.uuid4())

        payload = self.create_send_message_payload(task, task_id, context_id)
        payload["message"]["message_id"] = message_id
        logger.info("`send_message` triggered with payload: %s", payload)

        send_response = None
        call_context = configurable_context.get("propagation_headers", {})
        async for resp in client.send_message(
            message_request=Message(**payload.get("message")),
            call_context=ClientCallContext(state=call_context),
        ):
            send_response = resp

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
