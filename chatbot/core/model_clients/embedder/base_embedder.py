from abc import ABC, abstractmethod
from typing import Any, List

from chatbot.core.model_clients.embedder.base_class import EmbedderConfig
from chatbot.utils.embeddings import DenseEmbedding


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: EmbedderConfig, **kwargs):
        self.config = config
        self._initialize_embedder(**kwargs)
    
    @abstractmethod
    def _initialize_embedder(self, **kwargs) -> None:
        """Initialize the embedder."""
        pass
        
    @abstractmethod
    def get_query_embedding(self, query: str, **kwargs: Any) -> DenseEmbedding:
        """
        Get the embedding for the query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            DenseEmbedding: The embedding for the query.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def get_text_embedding(self, text: str, **kwargs: Any) -> DenseEmbedding:
        """
        Get the embedding for the text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            DenseEmbedding: The embedding for the text.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def get_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[DenseEmbedding]:
        """
        Get the embeddings for the list of texts.

        Args:
            texts (List[str]): The list of texts to get embeddings for.

        Returns:
            List[DenseEmbedding]: The list of embeddings for the texts.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    async def aget_query_embedding(self, query: str, **kwargs: Any) -> DenseEmbedding:
        """
        Asynchronously get the embedding for the query.

        Args:
            query (str): The query to get the embedding for.

        Returns:
            DenseEmbedding: The embedding for the query.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    async def aget_text_embedding(self, text: str, **kwargs: Any) -> DenseEmbedding:
        """
        Asynchronously get the embedding for the text.

        Args:
            text (str): The text to get the embedding for.

        Returns:
            DenseEmbedding: The embedding for the text.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    @abstractmethod
    async def aget_text_embeddings(self, texts: List[str], **kwargs: Any) -> List[DenseEmbedding]:
        """
        Asynchronously get the embeddings for the list of texts.

        Args:
            texts (List[str]): The list of texts to get embeddings for.

        Returns:
            List[DenseEmbedding]: The list of embeddings for the texts.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
    