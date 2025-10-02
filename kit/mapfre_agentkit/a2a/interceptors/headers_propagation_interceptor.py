import logging
from typing import Any, Iterable

from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
from a2a.types import AgentCard

logger = logging.getLogger(__name__)


class HeadersPropagationInterceptor(ClientCallInterceptor):
    """Propagates selected headers from the call context to the outbound HTTP request.

    This interceptor expects the per-call headers to be provided in the
    ClientCallContext.state under the key 'propagation_headers'.

    It applies simple allowlist/prefix rules to avoid leaking hop-by-hop or
    sensitive headers. Configure via constructor.
    """

    def __init__(
        self,
        *,
        allowed_prefixes: Iterable[str] | None = None,
        allowed_names: Iterable[str] | None = None,
        allow_authorization: bool = False,
    ) -> None:
        self.allowed_prefixes = tuple(
            (p.lower() for p in (allowed_prefixes or ("x-mapfre-",)))
        )
        self.allowed_names = {
            n.lower()
            for n in (
                allowed_names or ("x-request-id", "x-correlation-id", "accept-language")
            )
        }
        self.allow_authorization = allow_authorization

        # Hop-by-hop and sensitive headers that must never be forwarded
        self._blocked = {
            "connection",
            "keep-alive",
            "transfer-encoding",
            "te",
            "trailer",
            "upgrade",
            "host",
            "content-length",
        }
        if not self.allow_authorization:
            self._blocked.add("authorization")
            self._blocked.add("cookie")

    def _is_allowed(self, name: str) -> bool:
        lower = name.lower()
        if lower in self._blocked:
            return False
        if lower in self.allowed_names:
            return True
        return any(lower.startswith(p) for p in self.allowed_prefixes)

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

        candidate_headers: dict[str, Any] | None = None
        # Prefer explicit propagation set by the caller
        if isinstance(context.state.get("propagation_headers"), dict):
            candidate_headers = context.state.get("propagation_headers")  # type: ignore[assignment]

        if not candidate_headers:
            return request_payload, http_kwargs

        headers = http_kwargs.setdefault("headers", {})
        added = {}
        for k, v in candidate_headers.items():
            if isinstance(v, (str, bytes)) and self._is_allowed(k):
                headers[k] = v
                added[k] = v

        if added:
            logger.debug(
                "HeadersPropagationInterceptor added headers: %s", list(added.keys())
            )
        return request_payload, http_kwargs
