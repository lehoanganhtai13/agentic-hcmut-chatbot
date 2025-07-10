from enum import Enum
from pydantic import BaseModel
from typing import Optional


class IndexValueType(Enum):
    """
    Enum for different types of indexed values.
    """
    STRING = "varchar"
    INT = "double"
    BOOL = "bool"


class IndexParam(BaseModel):
    """
    Parameters for indexing JSON fields in Milvus database.
    
    Args:
        indexed_key (str): Path of the indexed key in the JSON object.
            You can target nested keys, array positions, or both 
            (e.g., `metadata["product_info"]["category"]` or `metadata["tags"][0]`)
        index_name (str): Name of the index in Milvus database.
        value_type (IndexValueType): Type of the value to be indexed.
    """
    indexed_key: str
    index_name: str
    value_type: IndexValueType

    class Config:
        arbitrary_types_allowed = True


class MetricType(Enum):
    """
    Enum for different types of metrics used in Milvus databases.
    """
    L2 = "L2"
    IP = "IP"
    COSINE = "COSINE"
    HAMMING = "HAMMING"
    

class IndexType(Enum):
    """
    Enum for different types of indexes used in Milvus databases.
    """
    AUTOINDEX = "AUTOINDEX"
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    BIN_FLAT = "BIN_FLAT"
    STL_SORT = "STL_SORT"
    HNSW = "HNSW"
    INVERTED = "INVERTED"
    SPARSE_INVERTED_INDEX = "SPARSE_INVERTED_INDEX"


class ElementType(Enum):
    """
    Enum representing the types of elements of an array in Milvus.
    """
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"


class DataType(Enum):
    """
    Enum representing the data types supported by Milvus.
    """
    INT = "int"
    STRING = "string"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    BOOL = "bool"
    ARRAY = "array"
    BINARY = "binary"
    JSON = "json"


class IndexConfig(BaseModel):
    """
    Class representing the configuration for an index in Milvus.
    """
    index: bool = False
    hnsw_m: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None
    index_type: Optional[IndexType] = None
    metric_type: Optional[MetricType] = None


class SchemaField(BaseModel):
    """
    Class representing a field in a Milvus schema.
    """
    field_name: str
    field_type: DataType
    is_primary: bool = False
    field_description: Optional[str] = None
    element_type: Optional[ElementType] = None
    dimension: Optional[int] = None
    max_capacity: Optional[int] = None
    index_config: Optional[IndexConfig] = None
