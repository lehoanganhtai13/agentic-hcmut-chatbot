from enum import Enum


class EmbedderBackend(Enum):
    """Enum for different embedder backends."""
    OPENAI = "openai"
    LOCAL = "local"


class EmbedderConfig:
    """Base configuration for embedder."""
    def __init__(self, backend: EmbedderBackend, **kwargs):
        self.backend = backend
        self.config = kwargs
