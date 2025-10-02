from typing import Optional
import logging
from opentelemetry import context, baggage
from opentelemetry.trace.span import Span
from phoenix.otel import SimpleSpanProcessor

logger = logging.getLogger(__name__)


CUSTOM_HEADERS_KEY = context.create_key("custom.request.headers")


class ForceCustomAttributes(SimpleSpanProcessor):
    """
    Añade atributos al span a partir de un diccionario de cabeceras
    almacenadas en el contexto de la traza por el middleware.
    """

    def on_start(
        self, span: Span, parent_context: Optional[context.Context] = None
    ) -> None:
        if parent_context:
            custom_headers = context.get_value(
                CUSTOM_HEADERS_KEY, context=parent_context
            )
            if custom_headers and isinstance(custom_headers, dict):
                logger.info(f"Headers found in parent context: {custom_headers}")
                for key, value in custom_headers.items():
                    span.set_attribute(f"request.header.{key}", value)
                    logger.info("Attribute added: request.header.%s = %s", key, value)
            else:
                logger.warning(
                    "Parent context exists, but custom headers were not found or are not a dictionary."
                )
