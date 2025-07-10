from typing import Optional

from enum import Enum
from pydantic import BaseModel, ConfigDict

from chatbot.utils.embeddings import (
    DenseEmbedding,
    SparseEmbedding,
    BinaryEmbedding
)


class VectorDBBackend(Enum):
    """Enum for different vector database backends."""
    MILVUS = "milvus"
    LANCEDB = "lancedb"


class VectorDBConfig:
    """Base configuration for vector database."""
    def __init__(self, backend: VectorDBBackend, **kwargs):
        self.backend = backend
        self.config = kwargs


class IndexValueType(Enum):
    """
    Enum for different types of indexed values.
    """
    STRING = "varchar"
    INT = "double"
    BOOL = "bool"


class IndexParam(BaseModel):
    """
    Parameters for indexing JSON fields in a vector database.
    
    Args:
        indexed_key (str): Path of the indexed key in the JSON object.
            You can target nested keys, array positions, or both 
            (e.g., `metadata["product_info"]["category"]` or `metadata["tags"][0]`)
        index_name (str): Name of the index in the vector database.
        value_type (IndexValueType): Type of the value to be indexed.
    """
    indexed_key: str
    index_name: str
    value_type: IndexValueType

    class Config:
        arbitrary_types_allowed = True


class EmbeddingType(Enum):
    """Enum for different types of embeddings."""
    DENSE = "dense"
    SPARSE = "sparse"
    BINARY = "binary"


class EmbeddingData(BaseModel):
    """
    Data structure for embedding data used for searching in a vector database.
    
    Args:
        field_name (str): Name of the field in the JSON object.
        embeddings (Optional[DenseEmbedding | BinaryEmbedding | SparseEmbedding]): embedding (dense, sparse, or binary).
        query (Optional[str]): Query string for full-text search.
        filtering_expr (Optional[str]): Filtering expression for the embeddings.
        embedding_type (Optional[EmbeddingType]): Type of the embedding (dense, sparse, or binary).
    """
    field_name: str
    embeddings: Optional[DenseEmbedding | BinaryEmbedding | SparseEmbedding] = None
    query: Optional[str] = None
    filtering_expr: Optional[str] = None
    embedding_type: Optional[EmbeddingType] = EmbeddingType.DENSE

    model_config = ConfigDict(arbitrary_types_allowed=True)
