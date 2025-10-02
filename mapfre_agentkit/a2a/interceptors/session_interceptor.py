import logging
from typing import Any

from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
from a2a.types import AgentCard

logger = logging.getLogger(__name__)


class SessionInterceptor(ClientCallInterceptor):
    """An interceptor that automatically adds session details to requests.

    Based on the agent's security schemes.
    """

    async def intercept(
        self,
        method_name: str,
        request_payload: dict[str, Any],
        http_kwargs: dict[str, Any],
        agent_card: AgentCard | None,
        context: ClientCallContext | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        if not context or not context.state:
            return request_payload, http_kwargs

        headers = http_kwargs.setdefault("headers", {})

        for key, value in context.state.items():

            if key == "propagation_headers" and isinstance(value, dict):
                for header_key, header_value in value.items():
                    if isinstance(header_value, (str, bytes)):
                        headers[header_key] = header_value

            elif isinstance(value, (str, bytes)):
                headers[key] = value

        return request_payload, http_kwargs
