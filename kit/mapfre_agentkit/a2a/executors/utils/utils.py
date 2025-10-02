import logging
from a2a.types import (
    FilePart,
    FileWithBytes,
    Part,
    TextPart,
)
from a2a.server.agent_execution.context import RequestContext
from a2a.server.tasks import TaskUpdater

from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "self"


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type.

    Args:
        part (Part): The A2A Part to convert.

    Returns:
        types.Part: The equivalent Google Gen AI Part.

    Raises:
        ValueError: If the part type is not supported.
    """
    part = part.root
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    raise ValueError(f"Unsupported part type: {type(part)}")


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type.

    Args:
        part (types.Part): The Google Gen AI Part to convert.

    Returns:
        Part: The equivalent A2A Part.

    Raises:
        ValueError: If the part type is not supported.
    """
    if part.text:
        return TextPart(text=part.text)
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f"Unsupported part type: {part}")


def generate_payload(request: RequestContext, updater: TaskUpdater):
    """Generate a payload dictionary for the Langchain runner execution.
    Extracts user and session information from the request context headers
    and prepares the request payload data structure needed by the Langchain runner. Falls
    back to default values if headers are not available.
    Args:
        request (RequestContext): The incoming request context containing
            message data and call context with headers.
        updater (TaskUpdater): The task updater for reporting progress
            and results during execution.
    Returns:
        dict: A dictionary containing user_id, session_id, new_message,
            and task_updater for the Langchain runner.
    Raises:
        ValueError: If the request message is None.
    """
    if not request.message:
        raise ValueError("Request message cannot be None")
    return {
        "user_id": request.call_context.state.get("headers", {}).get(
            "x-mapfre-user-id", DEFAULT_USER_ID
        ),
        "session_id": request.call_context.state.get("headers", {}).get(
            "x-mapfre-session-id", request.context_id
        ),
        "new_message": types.UserContent(
            parts=[convert_a2a_part_to_genai(part) for part in request.message.parts]
        ),
        "task_updater": updater,
        "propagation_headers": get_propagated_headers(request),
    }


def get_propagated_headers(request: RequestContext):
    """Build and return curated headers to propagate to downstream A2A calls.
    Extracts inbound headers from the request call context and returns a payload
    dict under the 'propagation_headers' key containing only safe, serializable
    items. Currently this method filters headers to those that start with
    'x-mapfre-' and whose values are strings or bytes.
    Args:
        request (RequestContext): The incoming request context containing the
            call context with inbound HTTP headers.
    Returns:
        dict[str, dict[str, str]]: A dictionary with a single 'propagation_headers'
            key mapping to the filtered headers that should be forwarded to
            remote agents.
    """
    headers = request.call_context.state.get("headers", {})
    propagated_headers = {}
    propagated_headers = propagated_headers.setdefault("propagation_headers", {})
    for key, value in headers.items():
        if isinstance(value, (str, bytes)) and key.startswith("x-mapfre-"):
            propagated_headers[key] = value
    return propagated_headers
