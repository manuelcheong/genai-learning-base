"""
Model configuration classes for agent configuration.
"""

from typing import Optional
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a model provider."""

    name: str
    endpoint: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for a model."""

    name: str
    provider: Optional[ProviderConfig] = None
