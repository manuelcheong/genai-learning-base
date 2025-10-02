import importlib
from typing import Any
from mapfre_agentkit.config.tool_config import FunctionToolConfig
from mapfre_agentkit.tools.loaders.base_tool_loader import ToolLoaderStrategy


class FunctionToolLoader(ToolLoaderStrategy):
    """Strategy for loading function-based tools."""

    def load_tool(self, config: FunctionToolConfig) -> Any:
        """Load a function tool from configuration."""
        try:
            module = importlib.import_module(config.module_path)
            return getattr(module, config.function_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Error loading function tool: {e}")
