"""Tool Factory for dynamically loading and managing agent tools.

This module provides a flexible way to load different types of tools based on
configuration. It uses the Strategy pattern and Pydantic for validation.
"""

from typing import Any, Dict, Type, Optional
from mapfre_agentkit.config.tool_config import (
    ToolConfig,
    ToolType,
    TOOLS_CONFIG_MODELS,
)
from mapfre_agentkit.tools.loaders.base_tool_loader import ToolLoaderStrategy
from mapfre_agentkit.tools.loaders.mcp_tool_loader import (
    MCPToolLoaderADK,
    MCPToolLoaderLangchain,
    MCPToolLoaderStrands,
)
from mapfre_agentkit.tools.loaders.function_tool_loader import FunctionToolLoader
from mapfre_agentkit.tools.loaders.method_tool_loader import ClassMethodToolLoader
from mapfre_agentkit.config.agent_config import AgentConfig
from mapfre_agentkit.config.agent_config import AgentType

from pydantic import ValidationError


class ToolFactory:
    """Factory for creating tools based on configuration."""

    def __init__(self, agent_config: AgentConfig):
        """Initialize the tool factory with default strategies."""
        self.agent_config = agent_config
        self._strategies = {
            ToolType.MCP: self._get_mcp_tool_loader(),
            ToolType.FUNCTION: FunctionToolLoader(),
            ToolType.CLASS_METHOD: ClassMethodToolLoader(),
        }

        self._config_models = TOOLS_CONFIG_MODELS

    def _get_mcp_tool_loader(self) -> ToolLoaderStrategy:
        if self.agent_config.type == AgentType.ADK:
            return MCPToolLoaderADK()
        elif self.agent_config.type == AgentType.LANGCHAIN:
            return MCPToolLoaderLangchain()
        elif self.agent_config.type == AgentType.STRANDS:
            return MCPToolLoaderStrands()
        else:
            raise ValueError(f"Unknown agent type: {self.agent_config.type}")

    def register_tool_type(
        self,
        tool_type: ToolType,
        strategy: ToolLoaderStrategy,
        config_model: Type[ToolConfig],
    ):
        """Register a new tool type with its strategy and model."""
        self._strategies[tool_type] = strategy
        self._config_models[tool_type] = config_model

    def create_tool(self, tool_config: Dict[str, Any]) -> Optional[Any]:
        """Create a tool based on its configuration.

        Args:
            tool_config: Dictionary containing tool configuration

        Returns:
            The created tool or None if creation failed

        Raises:
            ValueError: If tool type is not supported or config is invalid
        """
        tool_type_str = tool_config.get("type")

        # Convert string to ToolType enum
        try:
            tool_type = ToolType(tool_type_str)
        except ValueError:
            raise ValueError(f"Unsupported tool type: {tool_type_str}")

        if tool_type not in self._strategies:
            raise ValueError(f"Tool type not registered: {tool_type}")

        config_model = self._config_models[tool_type]
        try:
            validated_config = config_model(**tool_config)
        except ValidationError as e:
            raise ValueError(f"Invalid tool configuration: {e}")

        return self._strategies[tool_type].load_tool(validated_config)
