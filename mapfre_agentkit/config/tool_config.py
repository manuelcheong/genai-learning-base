from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ToolType(str, Enum):
    """Enum for tool types."""

    MCP = "mcp"
    FUNCTION = "function"
    CLASS_METHOD = "class_method"


class Header(BaseModel):
    """Model for header."""

    header_name: Optional[str] = None
    header_value: Optional[str] = None


class Auth(BaseModel):
    """Model for auth."""

    headers: Optional[List[Header]] = None


class MCPTypeStreamable(BaseModel):
    """Model for streamable tool type."""

    url: str
    port: int
    path: str = "/"
    auth: Optional[Auth] = None


class MCPTypeStdio(BaseModel):
    """Model for stdio tool type."""

    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None


class ToolConfig(BaseModel):
    """Base model for tool configuration."""

    type: ToolType
    name: str = ""
    description: str = ""


class MCPToolConfig(ToolConfig):
    """Configuration for MCP tools."""

    mcp_config: MCPTypeStreamable | MCPTypeStdio


class FunctionToolConfig(ToolConfig):
    """Configuration for function-based tools."""

    module_path: str
    function_name: str


class ClassMethodToolConfig(ToolConfig):
    """Configuration for class method-based tools."""

    module_path: str
    class_name: str
    method_name: str
    init_params: Dict[str, Any] = {}


TOOLS_CONFIG_MODELS: Dict[ToolType, Type[ToolConfig]] = {
    ToolType.MCP: MCPToolConfig,
    ToolType.FUNCTION: FunctionToolConfig,
    ToolType.CLASS_METHOD: ClassMethodToolConfig,
}
