import os
from typing import Dict, List, Any, Optional
import logging
import yaml
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from mapfre_agentkit.utils import safe_async_run, get_protocol
from mapfre_agentkit.agents.frameworks.base_agent_builder import AgentStrategy
from mapfre_agentkit.config.agent_config import (
    AGENT_STRATEGIES,
    AgentConfig,
    AgentType,
)
from mapfre_agentkit.config.auth_config import AuthType
from mapfre_agentkit.tools.generators.tool_generator import ToolFactory

logger = logging.getLogger(__name__)


class AgentA2AFactory:
    """Factory class for creating Agent-to-Agent (A2A) communication components.

    This class handles the configuration, initialization, and creation of agents
    for A2A communication. It loads configuration from a YAML file, sets up tracing,
    and builds agent instances with the appropriate tools and capabilities.
    """

    def __init__(self, config_path: str):
        """Initialize the AgentA2AFactory with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        """Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.
        """
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return self._normalize_auth_blocks(config)

    def _normalize_auth_blocks(self, config: dict) -> dict:
        """
        Normalize and update the 'auth' blocks in the agent configuration.

        This method ensures that the main 'auth' section and each remote agent's 'auth' section
        (if present) are normalized according to the expected structure using _parse_auth.

        Args:
            config (dict): The agent configuration dictionary. It should contain an optional 'auth' section
                and an optional list of 'remote_agents_addresses', each possibly with its own 'auth'.

        Returns:
            dict: The updated configuration dictionary with normalized 'auth' sections.
        """
        config["auth"] = self._parse_auth(config.get("auth", {}))
        for agent in config.get("remote_agents_addresses", []):
            if "auth" in agent:
                agent["auth"] = self._parse_auth(agent["auth"])
        return config

    def _parse_auth(self, auth: dict) -> dict:
        """
        Normalize and validate the 'auth' section from an agent configuration.

        Converts the 'security' list to a valid OpenAPI format (a list of dicts with scopes),
        and returns a dictionary with the normalized fields: 'security', 'security_schemes', and 'type'.

        Args:
            auth (dict): The authentication configuration dictionary. May include 'security',
                'securitySchemes', and 'type'.

        Returns:
            dict: A normalized authentication configuration dictionary with keys:
                - 'security': List of security requirements in OpenAPI format (list of dicts)
                - 'security_schemes': Security schemes definition (if present)
                - 'type': Authentication type (default: AuthType.NO_AUTH)
        """
        security = auth.get("security")
        security_schemes = auth.get("securitySchemes")
        auth_type = auth.get("type", AuthType.NO_AUTH)
        if security:
            normalized_security = []
            for sec in security:
                if isinstance(sec, str):
                    normalized_security.append({sec: []})
                elif isinstance(sec, dict):
                    normalized_security.append(sec)
            security = normalized_security
        return {
            "security": security,
            "security_schemes": security_schemes,
            "type": auth_type,
        }

    def _generate_remote_agent_configs(self):
        """
        Generate configuration dictionaries for remote agents based on their addresses.

        Iterates through the 'remote_agents_addresses' section of the main config, generating a config
        dictionary for each remote agent, including its URL and normalized 'auth' configuration.

        Returns:
            list[dict] | None: A list of dictionaries, each containing 'url' and 'auth' for a remote agent,
            or None if there are no remote agents configured.
        """
        remote_agents_addresses = self.config.get("remote_agents_addresses", [])
        if remote_agents_addresses:
            configs = []
            for remote_agent in remote_agents_addresses:
                url = self._generate_url(
                    host=remote_agent.get("host"),
                    port=remote_agent.get("port"),
                    path=remote_agent.get("path"),
                )
                auth_config = remote_agent.get("auth", {})
                configs.append({"url": url, "auth": auth_config})
            return configs
        return None

    def _generate_url(
        self, host: str, port: Optional[int] = None, path: Optional[str] = None
    ) -> str:
        """Generate a URL from host, optional port, and optional path components.

        Args:
            host (str): Hostname or IP address.
            port (Optional[int], optional): Port number. If None, it's omitted.
            path (Optional[str], optional): URL path. May start with or without "/". If None or empty, it's omitted.

        Returns:
            str: Complete URL such as "https://host/path" or "https://host:8080/path".
        """
        protocol = get_protocol()
        url = f"{protocol}://{host}"

        if isinstance(port, int):
            url = f"{url}:{port}"

        if path is not None:
            path_str = str(path).strip()
            if not path_str.startswith("/"):
                url = f"{url}/{path_str}"
            else:
                url = f"{url}{path_str}"

        return url

    def _get_agent_instructions(self) -> str:
        """Get the agent instructions from the configuration.

        Returns:
            str: The agent instructions string, or an empty string if not specified.
        """
        return self.config.get("instruction", "")

    def _get_agent_skills(self) -> List[AgentSkill]:
        """Create a list of AgentSkill objects from the configuration.

        Converts the skills section of the configuration into AgentSkill objects
        with appropriate properties (id, name, description, tags, examples).

        Returns:
            List[AgentSkill]: List of agent skills defined in the configuration.
        """
        return [
            AgentSkill(
                id=skill["id"],
                name=skill["name"],
                description=skill["description"],
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
            )
            for skill in self.config.get("skills", [])
        ]

    def _get_agent_tools(self) -> List[Any]:
        """Create a list of agent tools from the configuration.

        Supports multiple tool types through the ToolFactory:
        - MCP tools: Model Control Protocol tools
        - Function tools: Standalone functions loaded dynamically
        - Class method tools: Methods bound to class instances

        Returns:
            List[Any]: List of tool objects based on configuration.
        """
        tool_factory = ToolFactory(AgentConfig(**self.config.get("agent", {})))
        tools = []

        for tool_config in self.config.get("tools", []):
            try:
                tool = tool_factory.create_tool(tool_config)
                if tool:
                    tools.append(tool)
            except ValueError as e:
                logger.error(f"Error creating tool: {e}")
                logger.error(f"Tool config: {tool_config}")

        return tools

    def _get_agent_strategy(
        self, agent_type: str | AgentType, remote_agent_configs: List[Dict] = None
    ) -> AgentStrategy:
        """Get the appropriate agent strategy instance based on the agent type.

        Args:
            agent_type: The type of agent to create (e.g., AgentType.ADK, AgentType.LANGCHAIN, or string)
            remote_agent_connections: List of remote agent connections to initialize the strategy with

        Returns:
            AgentStrategy: An initialized instance of the agent strategy

        Raises:
            ValueError: If the agent type is unknown or required dependencies
                are not available
        """
        # Convert string to enum if needed
        if isinstance(agent_type, str):
            agent_type = agent_type.lower()
            try:
                agent_type = AgentType(agent_type)
            except ValueError:
                available_types = ", ".join([t.value for t in AgentType])
                raise ValueError(
                    f"Unknown agent type: {agent_type}. Available types: {available_types}"
                )

        strategy_class = AGENT_STRATEGIES[agent_type]

        # Create and return an instance of the strategy class
        strategy_instance = strategy_class(remote_agent_configs=remote_agent_configs)

        # Initialize A2A communication if remote_agent_connections are provided
        if remote_agent_configs:
            safe_async_run(strategy_instance.initialize())

        return strategy_instance

    def build_agent(self) -> Any:
        """Build and configure an Agent instance based on the configuration.

        Creates an Agent with the appropriate name, model, instructions, and tools.
        If remote agent addresses are configured, also sets up the A2A remote factory
        and adds its callback and send_message tool to the agent.

        The agent type (ADK or LangChain) is determined by the 'agent.type' field
        in the configuration. If not specified, defaults to 'adk'.

        Returns:
            Any: Fully configured Agent instance ready for use.
        """
        before_agent_callback = None
        tools = self._get_agent_tools()
        remote_agent_configs = self._generate_remote_agent_configs()
        agent_config = AgentConfig(**self.config.get("agent", {}))

        # Get an initialized strategy instance
        strategy = self._get_agent_strategy(agent_config.type, remote_agent_configs)

        # Add the send_message method from the strategy to the tools
        if remote_agent_configs:
            before_agent_callback = strategy.before_agent_callback
            tools.append(strategy.send_message)
        return strategy.create_agent(
            name=self.config["name"],
            model_config=agent_config.model,
            instruction=self._get_agent_instructions(),
            tools=tools,
            before_agent_callback=before_agent_callback,
        )

    def build_executor(self, agent, agent_card) -> Any:
        agent_config = AgentConfig(**self.config.get("agent", {}))

        strategy = self._get_agent_strategy(agent_config.type)
        return strategy.create_executor(agent=agent, agent_card=agent_card)

    def build_agent_card(self) -> AgentCard:
        """Build an AgentCard instance based on the configuration.

        Creates an AgentCard with metadata about the agent including name,
        description, URL, capabilities, and skills. This card is used for
        agent discovery and communication in the A2A protocol.

        Returns:
            AgentCard: Agent card with metadata from the configuration.
        """
        capabilities = AgentCapabilities(**self.config.get("capabilities"))
        auth = self.config.get("auth", {})
        return AgentCard(
            name=self.config.get("name"),
            description=self.config.get("description"),
            url=self.config.get("url"),
            version=self.config.get("version", "1.0.0"),
            default_input_modes=self.config.get(
                "defaultInputModes", ["text", "text/plain"]
            ),
            default_output_modes=self.config.get(
                "defaultOutputModes", ["text", "text/plain"]
            ),
            capabilities=capabilities,
            skills=self._get_agent_skills(),
            security=auth.get("security"),
            security_schemes=auth.get("security_schemes"),
        )
