"""
Agent implementation strategies for different agent frameworks.

This module defines the strategy pattern for creating agents using different
frameworks such as Google ADK or LangChain.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from mapfre_agentkit.agents.frameworks.a2a_agent_communication import (
    A2AAgentCommunicationBase,
)

logger = logging.getLogger(__name__)


class AgentStrategy(A2AAgentCommunicationBase, ABC):
    """Abstract base class for different agent implementation strategies."""

    def __init__(self, remote_agent_configs=None):
        super().__init__(remote_agent_configs)

    @abstractmethod
    def create_agent(
        self,
        name: str,
        model: Any,
        instruction: str,
        tools: List[Any],
        before_agent_callback: Optional[Any] = None,
    ) -> Any:
        """Create and return an agent instance.

        Args:
            name: The name of the agent
            model: The model to use
            instruction: The agent instructions
            tools: List of tools for the agent to use
            before_agent_callback: Optional callback to run before agent execution

        Returns:
            An agent instance
        """
        pass

    @abstractmethod
    def create_executor(self, agent: Any, agent_card: Any) -> Any:
        """Create and return an agent executor.

        Args:
            agent: The agent instance to execute.

        Returns:
            An agent executor instance.
        """
        pass

    @abstractmethod
    def get_model(self):
        """Process the model configuration and return the appropriate model object.

        Returns:
            Either a model name string or a model config instance
        """
        pass
