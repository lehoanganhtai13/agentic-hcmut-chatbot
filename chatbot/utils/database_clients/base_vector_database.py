from abc import ABC, abstractmethod
from typing import Any, List, Dict

from chatbot.utils.database_clients.base_class import (
    EmbeddingData,
    VectorDBConfig
)


class BaseVectorDatabase(ABC):
    """Abstract base class for vector database operations."""
    
    def __init__(self, config: VectorDBConfig, **kwargs):
        self.config = config
        self._initialize_client(**kwargs)
    
    @abstractmethod
    def _initialize_client(self, **kwargs) -> None:
        """Initialize the database client."""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        collection_structure: List[Any],
        **kwargs,
    ) -> None:
        """Create a new collection in the vector database."""
        pass

    @abstractmethod
    def load_collection(self, collection_name: str, **kwargs) -> bool:
        """Load the collection into memory for faster search operations."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str, **kwargs) -> None:
        """Delete a collection from the database."""
        pass
    
    @abstractmethod
    def list_collections(self, **kwargs) -> List[str]:
        """List all collections in the database."""
        pass
    
    @abstractmethod
    def has_collection(self, collection_name: str, **kwargs) -> bool:
        """Check if a collection exists."""
        pass

    @abstractmethod
    def insert_vectors(self, data: List[Dict], **kwargs) -> None:
        """Insert vectors into the collection."""
        pass

    @abstractmethod
    def get_items(self, ids: List[str], **kwargs) -> List[Dict]:
        """Get items from the collection by their IDs."""
        pass
    
    @abstractmethod
    def search_dense_vectors(
        self,
        query_embeddings: List[List[float]],
        field_name: str,
        output_fields: List[str],
        filtering_expr: str,
        top_k: int,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Search for dense vectors in the collection.
        The output is a nested list of dictionaries containing the search results.
        Each dictionary corresponds to a search result and contains the fields specified in `output_fields` and `_score` field.
        """
        pass

    @abstractmethod
    def hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int,
        **kwargs,
    ) -> List[Dict]:
        """
        Perform hybrid search for vectors in the collection.
        The output is a list of dictionaries containing the search results.
        Each dictionary corresponds to a search result and contains the fields specified in `output_fields` and `_score` field.
        """
        pass

    @abstractmethod
    def async_create_collection(
        self,
        collection_name: str,
        collection_structure: List[Dict],
        **kwargs,
    ) -> None:
        """Asynchronously create a new collection in the vector database."""
        pass

    @abstractmethod
    async def async_load_collection(self, collection_name: str, **kwargs) -> bool:
        """Asynchronously load the collection into memory for faster search operations."""
        pass

    @abstractmethod
    async def async_delete_collection(self, collection_name: str, **kwargs) -> None:
        """Asynchronously delete a collection from the database."""
        pass

    @abstractmethod
    async def async_list_collections(self, **kwargs) -> List[str]:
        """Asynchronously list all collections in the database."""
        pass

    @abstractmethod
    async def async_has_collection(self, collection_name: str, **kwargs) -> bool:
        """Asynchronously check if a collection exists."""
        pass

    @abstractmethod
    async def async_insert_vectors(self, data: List[Dict], **kwargs) -> None:
        """Asynchronously insert vectors into the collection."""
        pass

    @abstractmethod
    async def async_get_items(self, ids: List[str], **kwargs) -> List[Dict]:
        """Asynchronously get items from the collection by their IDs."""
        pass

    @abstractmethod
    async def async_search_dense_vectors(
        self,
        query_embeddings: List[List[float]],
        field_name: str,
        output_fields: List[str],
        filtering_expr: str,
        top_k: int,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Asynchronously search for dense vectors in the collection.
        The output is a nested list of dictionaries containing the search results.
        Each dictionary corresponds to a search result and contains the fields specified in `output_fields` and `_score` field.
        """
        pass

    @abstractmethod
    async def async_hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int,
        **kwargs,
    ) -> List[Dict]:
        """
        Asynchronously hybrid search for vectors in the collection.
        The output is a list of dictionaries containing the search results.
        Each dictionary corresponds to a search result and contains the fields specified in `output_fields` and `_score` field.
        """
        pass
    