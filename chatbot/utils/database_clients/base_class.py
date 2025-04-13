from pydantic import BaseModel
from enum import Enum


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
