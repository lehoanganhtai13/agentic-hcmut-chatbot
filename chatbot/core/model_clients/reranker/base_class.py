from enum import Enum
from abc import ABC


class RerankerBackend(Enum):
    """Enumeration of supported reranker backends."""
    VLLM = "vllm"
    COHERE = "cohere"


class RerankerConfig(ABC):
    """
    Base configuration class for all reranker implementations.
    
    This abstract class provides the foundation for reranker-specific
    configurations, ensuring consistent interface across different backends.
    """

    def __init__(self, backend: RerankerBackend, **kwargs):
        """
        Initialize the base reranker configuration.
        
        Args:
            backend (RerankerBackend): The reranker backend type.
            **kwargs: Additional configuration parameters.
        """
        self.backend = backend
