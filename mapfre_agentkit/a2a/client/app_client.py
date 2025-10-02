from __future__ import annotations

from typing import Any, AsyncGenerator, Optional, List
from uuid import uuid4
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import Request
from a2a.client import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.middleware import ClientCallContext
from a2a.types import AgentCard, JSONRPCErrorResponse, Message, Role, TextPart


def get_card_resolver(
    client: httpx.AsyncClient, agent_card_url: str
) -> A2ACardResolver:
    """Build an A2ACardResolver from a full card URL, mirroring a2a-inspector."""
    parsed_url = urlparse(agent_card_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    path_with_query = urlunparse(("", "", parsed_url.path, "", parsed_url.query, ""))
    card_path = path_with_query.lstrip("/")
    if card_path:
        return A2ACardResolver(client, base_url, agent_card_path=card_path)
    return A2ACardResolver(client, base_url)


class A2AGatewayClient:
    """
    High-level client that wraps the low-level A2A client to send messages to agents,
    supporting both non-streaming (message/send) and streaming (message/stream).

    Usage:
      gateway = await A2AGatewayClient.from_card_url("https://agent_host/.well-known/agent")
      result = await gateway.send_message("hola")
      async for chunk in gateway.send_message_streaming("hola"):
          ...
    """

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        agent_card: AgentCard,
        interceptors: Optional[List[Any]] = None,
    ) -> None:
        self._httpx = httpx_client
        self._card = agent_card
        self._interceptors = interceptors or []
        self._agent_client = self.build_client()
        self._owns_client = False

    def build_client(self):
        """
        Creates and configures a low-level A2A client instance.

        This uses the ClientFactory with the instance's httpx configuration and
        interceptors to build the client that will communicate with the agent.

        Returns:
            A configured A2A client instance.
        """
        config = ClientConfig(httpx_client=self._httpx)
        factory = ClientFactory(config=config)
        return factory.create(self._card, interceptors=self._interceptors)

    @classmethod
    async def from_card_url(
        cls,
        agent_card_url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 60.0,
        interceptors: Optional[List[Any]] = None,
    ) -> "A2AGatewayClient":
        """
        Asynchronously create an instance from an Agent Card URL.

        This factory method manages the creation and lifecycle of an internal
        `httpx.AsyncClient` to resolve the Agent Card and configure the client.

        Args:
            agent_card_url (str): The full URL of the agent's Agent Card.
            headers (Optional[dict[str, str]]): Optional HTTP headers for the client.
            timeout (float): Timeout in seconds for HTTP requests.
            interceptors (Optional[List[Any]]): Optional list of interceptors for the A2A client.

        Returns:
            A2AGatewayClient: A new instance of A2AGatewayClient.
        """
        client = httpx.AsyncClient(timeout=timeout, headers=headers)
        try:
            resolver = get_card_resolver(client, agent_card_url)
            card = await resolver.get_agent_card()
            inst = cls(client, card, interceptors=interceptors)
            inst._owns_client = True
            return inst
        except Exception:
            await client.aclose()
            raise

    async def aclose(self) -> None:
        """
        Closes the underlying httpx.AsyncClient if it was created by this instance.

        It's important to call this method to release resources when the client
        is no longer needed, especially if instantiated with `from_card_url`.
        """
        if self._owns_client:
            await self._httpx.aclose()

    def supports_streaming(self) -> bool:
        """
        Checks if the target agent supports the streaming capability.

        Returns:
            bool: `True` if the Agent Card indicates support for streaming, `False` otherwise.
        """
        return bool(
            hasattr(self._card, "capabilities")
            and getattr(self._card.capabilities, "streaming", False) is True
        )

    async def send_message(
        self,
        text: str,
        *,
        message_id: Optional[str] = None,
        context_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
        context: Optional[ClientCallContext] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Sends a message to an agent and yields normalized events for streaming and non-streaming.

        This async generator produces events in a standardized dictionary format,
        `{"event": <str>, "data": <dict>, "id": <str>}`, regardless of whether
        the agent supports streaming. The final message from the agent is yielded
        with `event="message"`.

        Args:
            text (str): The text content of the message to send.
            message_id (Optional[str]): Optional ID for the message. A new one is generated if not provided.
            context_id (Optional[str]): Optional ID for the conversation or context.
            metadata (Optional[dict[str, Any]]): Optional metadata to attach to the message.
            request_id (Optional[str]): Optional ID for the request, used in the yielded events.
            context (Optional[ClientCallContext]): Optional call context for the A2A client.

        Yields:
            AsyncGenerator[dict[str, Any], None]: A generator that yields dictionaries
            representing each event from the agent interaction.
        """
        mid = message_id or uuid4().hex
        rid = request_id or mid
        message = Message(
            role=Role.user,
            parts=[TextPart(text=str(text))],
            message_id=mid,
            context_id=context_id,
            metadata=metadata or {},
        )
        async for item in self._agent_client.send_message(
            request=message, context=context
        ):
            if hasattr(item, "root"):
                if isinstance(item.root, JSONRPCErrorResponse):
                    err = item.root.error.model_dump(exclude_none=True)
                    yield {"event": "error", "data": err, "id": rid}
                    return
                event_payload = item.root.result
                data = (
                    event_payload.model_dump(exclude_none=True)
                    if hasattr(event_payload, "model_dump")
                    else dict(event_payload)
                )
                data["id"] = getattr(event_payload, "id", rid)
                yield {"event": "message", "data": data, "id": rid}
                continue

            if isinstance(item, tuple):
                item, _ = item

            if hasattr(item, "artifacts") and getattr(item, "artifacts"):
                data = (
                    item.model_dump(exclude_none=True)
                    if hasattr(item, "model_dump")
                    else dict(item)
                )
                data["id"] = getattr(item, "id", rid) if hasattr(item, "id") else rid
                yield {"event": "message", "data": data, "id": rid}
            elif hasattr(item, "parts"):
                data = (
                    item.model_dump(exclude_none=True)
                    if hasattr(item, "model_dump")
                    else dict(item)
                )
                data["id"] = getattr(item, "id", rid) if hasattr(item, "id") else rid
                yield {"event": "message", "data": data, "id": rid}
            else:
                event_name = (
                    getattr(item, "event", None)
                    or getattr(item, "type", None)
                    or "update"
                )

                payload = {}
                if hasattr(item, "model_dump"):
                    payload = item.model_dump(exclude_none=True)
                elif hasattr(item, "_asdict"):
                    payload = item._asdict()
                elif isinstance(item, dict):
                    payload = item
                else:
                    payload = {"value": str(item)}

                yield {"event": str(event_name), "data": payload, "id": rid}

    def build_propagation_context(self, request: Request) -> ClientCallContext:
        """
        Creates a ClientCallContext to propagate headers from an incoming request.

        This is useful for forwarding tracing headers, authentication tokens, etc.,
        from the request the gateway receives to the request it sends to the agent.

        Args:
            request (Request): The incoming FastAPI request.

        Returns:
            ClientCallContext: A call context containing the request headers.
        """
        incoming = dict(request.headers)
        return ClientCallContext(state={"propagation_headers": incoming})
