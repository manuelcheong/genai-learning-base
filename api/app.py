import os
import uvicorn
from fastapi import FastAPI, APIRouter, Body, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from mapfre_agentkit.a2a.client.app_client import A2AGatewayClient
from mapfre_agentkit.a2a.interceptors.headers_propagation_interceptor import (
    HeadersPropagationInterceptor,
)

from a2a.types import TextPart, FilePart, FileWithUri, FileWithBytes

# --- Configuración ---
# URL donde está corriendo tu agente 'custom_client'
AGENT_CARD_URL = os.environ.get("AGENT_CARD_URL", "http://agentkit-orchestrator-agent:8080")

# Variable para almacenar el cliente del gateway
gateway_client_instance = None

# ----------------------------------------------------------------------------
# FastAPI router factory to expose the gateway as API endpoints
# ----------------------------------------------------------------------------


def build_router(gateway: A2AGatewayClient) -> APIRouter:
    """
    Builds a FastAPI APIRouter to expose the A2AGatewayClient via an HTTP API.

    The router exposes the following endpoints:
      - `GET /card`: Returns the agent's Agent Card.
      - `POST /messages`: Sends a message to the agent, with optional support for
        streaming via Server-Sent Events (SSE).

    Args:
        gateway (A2AGatewayClient): The gateway client instance to expose.

    Returns:
        APIRouter: A configured FastAPI router with the endpoints.
    """
    router = APIRouter()

    @router.get("/card")
    async def get_card() -> dict[str, Any]:
        """
        Endpoint to get the configured agent's Agent Card.

        Returns:
            dict[str, Any]: A dictionary representation of the Agent Card.
        """
        return gateway._card.model_dump(exclude_none=True)

# GENERIC AGENT
    @router.post("/messages")
    async def post_message(
        body: dict = Body(
            ...,
            description="{ message: str, contextId?, metadata?}",
        ),
        stream: bool = Query(False, description="Enable streaming if supported"),
        request: Request = None,
    ):
        """
        Endpoint to send a message to an agent and receive a response.

        Supports two modes:
        1.  **Non-streaming** (default): Returns a single JSON response with the final message.
        2.  **Streaming** (if `stream=true` and supported by the agent): Returns a
            `text/event-stream` response (Server-Sent Events) with intermediate
            events and the final message.

        Args:
            body (dict): The request body, containing the message to send.
            stream (bool): A query parameter to enable streaming mode.
            request (Request): The FastAPI request object.

        Returns:
            Union[JSONResponse, StreamingResponse]: A JSON response or a streaming SSE response.
        """
        text = str(body.get("message", ""))
        kind = str(body.get("kind", "text"))
        messageContent = str(body.get("messageContent", ""))
        context_id = body.get("contextId")
        metadata = body.get("metadata") or {}

        # CHANGE AGENT CARD URL

        call_context = gateway.build_propagation_context(request)

        if stream and gateway.supports_streaming():

            async def event_gen() -> AsyncGenerator[bytes, None]:
                async for evt in gateway.send_message(
                    text,
                    context_id=context_id,
                    metadata=metadata,
                    context=call_context,
                ):
                    # SSE framing: event + data
                    event_name = evt.get("event", "message")
                    payload = evt.get("data", {})
                    yield f"event: {event_name}\n".encode()
                    yield f"data: {JSONResponse(content=payload).body.decode()}\n\n".encode()

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        # Non-streaming path
        last_payload: Optional[dict[str, Any]] = None
        async for evt in gateway.send_message(
            text,
            context_id=context_id,
            metadata=metadata,
            context=call_context,
        ):
            if evt.get("event") == "message":
                last_payload = evt.get("data")
        return JSONResponse(content=last_payload or {})

    return router

# --- Lógica de la aplicación ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida del A2AGatewayClient.
    Lo crea al iniciar la app, monta el router y lo cierra al terminar.
    """
    global gateway_client_instance
    print(f"Connecting to agent at: {AGENT_CARD_URL}")

    interceptors = [HeadersPropagationInterceptor()]

    gateway_client_instance = await A2AGatewayClient.from_card_url(
        AGENT_CARD_URL, interceptors=interceptors
    )

    print("Gateway client created successfully.")

    # Construye el router y lo añade a la aplicación
    api_router = build_router(gateway_client_instance)
    app.include_router(api_router, prefix="/agent")
    print("Agent API router included.")

    yield  # La aplicación se ejecuta aquí

    print("Closing gateway client...")
    if gateway_client_instance:
        await gateway_client_instance.aclose()
    print("Gateway client closed.")


# --- Creación de la App FastAPI ---

app = FastAPI(
    title="Agent API Gateway",
    description="Exposes a MAPFRE AgentKit agent via a RESTful API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root endpoint for health checks")
async def read_root():
    """Endpoint raíz para verificar que el servicio está activo."""
    return {"status": "ok", "message": "Agent API Gateway is running"}


# --- Punto de entrada para ejecutar el servidor ---

if __name__ == "__main__":
    # Este servidor correría en otro puerto, por ejemplo, 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
