"""
Agent configuration classes and registry.
"""

from typing import Dict, Type
from enum import Enum
from pydantic import BaseModel

# Import model configuration from separate file to avoid circular imports
from mapfre_agentkit.config.model_config import ModelConfig

# Import agent strategies after model types to avoid circular imports
from mapfre_agentkit.agents.frameworks.base_agent_builder import AgentStrategy
from mapfre_agentkit.agents.frameworks.adk.agent_builder import ADKAgentStrategy
from mapfre_agentkit.agents.frameworks.langchain.agent_builder import (
    LangChainAgentStrategy,
)
from mapfre_agentkit.agents.frameworks.strands.agent_builder import (
    StrandsAgentStrategy,
)


class AgentType(str, Enum):
    """Enum of available agent types."""

    ADK = "adk"
    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"
    STRANDS = "strands"


# Registry of available agent strategies
AGENT_STRATEGIES: Dict[str, Type[AgentStrategy]] = {
    AgentType.ADK: ADKAgentStrategy,
    AgentType.LANGCHAIN: LangChainAgentStrategy,
    AgentType.STRANDS: StrandsAgentStrategy,
}


class AgentConfig(BaseModel):
    """Configuration for the agent section."""

    type: AgentType
    model: ModelConfig
