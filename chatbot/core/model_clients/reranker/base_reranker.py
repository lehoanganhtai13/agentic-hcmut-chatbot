from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from chatbot.core.model_clients.reranker.base_class import RerankerConfig


class BaseReranker(ABC):
    """
    Abstract base class for all reranker implementations.
    
    This class defines the interface that all reranker clients must implement,
    providing both synchronous and asynchronous methods for reranking documents
    based on query relevance.
    """

    def __init__(self, config: RerankerConfig, **kwargs):
        """
        Initialize the base reranker with configuration.
        
        Args:
            config (RerankerConfig): Configuration object for the reranker.
            **kwargs: Additional keyword arguments for specific implementations.
        """
        self.config = config
        self._initialize_reranker(**kwargs)

    @abstractmethod
    def _initialize_reranker(self, **kwargs) -> None:
        """
        Initialize reranker-specific components.
        
        This method should be implemented by each reranker to set up
        their specific clients, connections, or other resources.
        
        Args:
            **kwargs: Additional keyword arguments for initialization.
        """
        pass

    # --- Synchronous Methods ---

    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            **kwargs: Additional keyword arguments for specific implementations.
            
        Returns:
            List[Tuple[int, float]]: List of tuples containing (original_index, relevance_score)
                sorted by relevance score in descending order.
        """
        pass

    # --- Asynchronous Methods ---

    @abstractmethod
    async def arerank(
        self, 
        query: str, 
        documents: List[str], 
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """
        Asynchronously rerank documents based on query relevance.
        
        Args:
            query (str): The search query.
            documents (List[str]): List of documents to rerank.
            **kwargs: Additional keyword arguments for specific implementations.
            
        Returns:
            List[Tuple[int, float]]: List of tuples containing (original_index, relevance_score)
                sorted by relevance score in descending order.
        """
        pass