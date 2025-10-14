"""
Model configuration classes for agent configuration.
"""
from fastapi import Request
from pydantic import BaseModel, Base64Bytes, Base64Str


from typing import Optional

class KindType(str, Enum):
    """Enum of available kind types."""

    TEXT = "text"
    FILE = "file"

class ApiBaseConfig(BaseModel):
    """Configuration for a model provider."""

    body: BodyConfig
    stream: bool
    request: Request

class BodyConfig(BaseModel):
    """Configuration for a model."""

    kind: KindType
    messages: str
    attachment: Optional[Base64Bytes|Base64Str|str] = None




    