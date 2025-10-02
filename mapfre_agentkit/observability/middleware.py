import logging
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from opentelemetry import trace, context, baggage
from opentelemetry.propagate import extract
from mapfre_agentkit.observability.observability import Observability
from mapfre_agentkit.observability.custom_processor import CUSTOM_HEADERS_KEY


logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that authenticates A2A (Agent-to-Agent) access using JWT tokens.

    This middleware inspects incoming HTTP requests, checks for authentication requirements
    specified in the AgentCard, and validates JWT tokens against the agent's security configuration.
    It supports public (unauthenticated) paths and enforces required claims as specified in the AgentCard.
    """

    def __init__(
        self,
        app: Starlette,
        public_paths: list[str] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app (Starlette): The Starlette application instance.
            agent_card (AgentCard, optional): The agent's metadata card containing security requirements.
            public_paths (list[str], optional): List of paths that do not require authentication.
        """
        super().__init__(app)
        providers = Observability().get_tracers_provider()
        for provider in providers:
            trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)

    async def dispatch(self, request: Request, call_next):
        """
        Procesa cada solicitud, extrayendo el contexto de la traza de las cabeceras
        y adjuntando atributos personalizados.
        """

        carrier = dict(request.headers.items())
        parent_context = extract(carrier=carrier)

        default_headers = {
            "x-custom-id": "app-001",
            "x-app-name": "observability-demo",
            "x-agent-version": "1.0.0",
        }

        new_context = context.set_value(
            CUSTOM_HEADERS_KEY, default_headers, parent_context
        )

        with self.tracer.start_as_current_span("dispatch", context=new_context) as span:
            response = await call_next(request)
            return response
