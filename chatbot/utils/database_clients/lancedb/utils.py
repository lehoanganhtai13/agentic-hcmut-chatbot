from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ElementType(Enum):
    """
    Enum representing the types of elements of an array in LanceDB.
    """
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    VECTOR = "vector"
    BOOL = "bool"


class DataType(Enum):
    """
    Enum representing the data types supported by LanceDB.
    """
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    VECTOR = "vector"
    BOOL = "bool"
    ARRAY = "array"
    BINARY = "binary"


class IndexConfig(BaseModel):
    """
    Class representing the configuration for an index in LanceDB.

    Attributes:
        index (bool): Whether to create an index for the field.
        full_text_search (bool): Whether to enable full-text search for the field.
    """
    index: bool = False
    full_text_search: bool = False


class SchemaField(BaseModel):
    """
    Class representing a field in a LanceDB schema.

    Attributes:
        field_name (str): The name of the field.
        field_type (DataType): The data type of the field.
        element_type (Optional[ElementType]): The type of elements in the field if it is an array.
        dimension (Optional[int]): The dimension of the vector if the field type is VECTOR.
        index_config (Optional[IndexConfig]): The configuration for indexing the field.
    """
    field_name: str
    field_type: DataType
    element_type: Optional[ElementType] = None
    dimension: Optional[int] = None
    index_config: Optional[IndexConfig] = None


class MetricType(Enum):
    """
    Enum for different types of metrics used in LanceDB.
    """
    L2 = "L2"
    IP = "dot"
    COSINE = "cosine"
    HAMMING = "hamming"


class VectorIndexType(Enum):
    """
    Enum for different types of indexes for vector used in LanceDB.
    """
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    IVF_HNSW_SQ = "IVF_HNSW_SQ"
    IVF_HNSW_PQ = "IVF_HNSW_PQ"


class ScalarIndexType(Enum):
    """
    Enum for different types of indexes for scalar used in LanceDB.
    """
    BTREE = "BTREE"
    BITMAP = "BITMAP"
    LABEL_LIST = "LABEL_LIST"


class QueryType(Enum):
    """
    Enum for different types of queries supported by LanceDB.
    """
    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"


class CreateMode(Enum):
    """
    Enum for different modes of creating a collection in LanceDB.
    """
    CREATE = "create"
    OVERWRITE = "overwrite"
