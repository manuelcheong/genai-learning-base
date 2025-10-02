from abc import ABC, abstractmethod
from typing import Any
from mapfre_agentkit.config.tool_config import (
    ToolConfig,
)


class ToolLoaderStrategy(ABC):
    """Interface for tool loading strategies."""

    @abstractmethod
    def load_tool(self, config: ToolConfig) -> Any:
        """Load a tool based on configuration."""
        pass
