import os
from typing import Any, Optional, Dict
import asyncio
import logging
from mapfre_agentkit.config.tool_config import (
    MCPToolConfig,
    MCPTypeStreamable,
    MCPTypeStdio,
)
from mapfre_agentkit.tools.loaders.base_tool_loader import ToolLoaderStrategy
from mapfre_agentkit.utils import expand_env_vars, get_protocol

from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
    StreamableHTTPConnectionParams,
)
from mcp.client.streamable_http import streamablehttp_client
from mcp import stdio_client
from mcp import StdioServerParameters as MCPStdioServerParameters

from strands.tools.mcp.mcp_client import MCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class MCPToolLoaderADK(ToolLoaderStrategy):
    """Strategy for loading MCP tools for ADK."""

    def load_tool(self, config: MCPToolConfig) -> Any:
        """Load an MCP tool from configuration."""

        if isinstance(config.mcp_config, MCPTypeStreamable):
            protocol = get_protocol()
            url = f"{protocol}://{config.mcp_config.url}:{config.mcp_config.port}{config.mcp_config.path}"
            logger.info(f"parsed headers: {self.get_headers(config)}")
            return MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=url, headers=self.get_headers(config)
                )
            )
        elif isinstance(config.mcp_config, MCPTypeStdio):
            command = config.mcp_config.command
            args = config.mcp_config.args
            env = expand_env_vars(config.mcp_config.env)

            logger.debug(f"Loading MCP tool with command: {command}, args: {args}")
            return MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=command, args=args, env=env
                    )
                )
            )

    def get_headers(self, config: MCPToolConfig) -> Optional[Dict[str, str]]:
        if isinstance(config.mcp_config, MCPTypeStreamable) and config.mcp_config.auth:
            logger.info(f"headers: {config.mcp_config.auth.headers}")
            return {
                header.header_name: os.getenv(header.header_value)
                for header in config.mcp_config.auth.headers
            }
        return None


class MCPToolLoaderLangchain(ToolLoaderStrategy):
    """Strategy for loading MCP tools for Langchain."""

    def load_tool(self, config: MCPToolConfig) -> Any:
        """Load an MCP tool from configuration."""

        async def load_tool_async(config: MCPToolConfig) -> Any:

            if isinstance(config.mcp_config, MCPTypeStreamable):
                protocol = get_protocol()
                url = f"{protocol}://{config.mcp_config.url}:{config.mcp_config.port}{config.mcp_config.path}"
                tool_name = config.name

                client = MultiServerMCPClient(
                    {
                        tool_name: {
                            "url": url,
                            "transport": "streamable_http",
                            "headers": self.get_headers(config),
                        }
                    }
                )
                return await client.get_tools()
            elif isinstance(config.mcp_config, MCPTypeStdio):
                tool_name = config.name
                env = expand_env_vars(config.mcp_config.env)

                client = MultiServerMCPClient(
                    {
                        tool_name: {
                            "command": config.mcp_config.command,
                            "args": config.mcp_config.args,
                            "env": env,
                            "transport": "stdio",
                        }
                    }
                )
                return await client.get_tools()

        tools = asyncio.run(load_tool_async(config))
        logger.info(f"Tools type: {type(tools)}, with value: {tools}")
        return tools

    def get_headers(self, config: MCPToolConfig) -> Optional[Dict[str, str]]:
        if isinstance(config.mcp_config, MCPTypeStreamable) and config.mcp_config.auth:
            return {
                header.header_name: os.getenv(header.header_value)
                for header in config.mcp_config.auth.headers
            }
        return None


class MCPToolLoaderStrands(ToolLoaderStrategy):
    """Strategy for loading MCP tools for Strands."""

    def load_tool(self, config: MCPToolConfig) -> Any:
        """Load an MCP tool from configuration."""
        if isinstance(config.mcp_config, MCPTypeStreamable):
            protocol = get_protocol()
            url = f"{protocol}://{config.mcp_config.url}:{config.mcp_config.port}{config.mcp_config.path}"
            streamable_http_mcp_client = MCPClient(
                lambda: streamablehttp_client(url=url)
            )
            with streamable_http_mcp_client:
                return streamable_http_mcp_client.list_tools_sync()
        elif isinstance(config.mcp_config, MCPTypeStdio):
            env = expand_env_vars(config.mcp_config.env)
            stdio_mcp_client = MCPClient(
                lambda: stdio_client(
                    MCPStdioServerParameters(
                        command=config.mcp_config.command,
                        args=config.mcp_config.args,
                        env=env,
                    )
                )
            )
            with stdio_mcp_client:
                return stdio_mcp_client.list_tools_sync()
