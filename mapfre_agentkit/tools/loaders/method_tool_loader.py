import importlib
from typing import Any
from mapfre_agentkit.config.tool_config import ClassMethodToolConfig
from mapfre_agentkit.tools.loaders.base_tool_loader import ToolLoaderStrategy


class ClassMethodToolLoader(ToolLoaderStrategy):
    """Strategy for loading class method-based tools."""

    def load_tool(self, config: ClassMethodToolConfig) -> Any:
        """Load a class method tool from configuration."""
        try:
            module = importlib.import_module(config.module_path)
            tool_class = getattr(module, config.class_name)
            class_instance = tool_class(**config.init_params)
            return getattr(class_instance, config.method_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Error loading class method tool: {e}")
