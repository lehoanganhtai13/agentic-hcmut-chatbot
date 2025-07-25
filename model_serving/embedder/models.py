from pydantic import BaseModel
from typing import List


class EmbeddingRequest(BaseModel):
    input: List[str]

class EmbeddingItem(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingItem]