from pydantic import BaseModel
from typing import List


class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankItem(BaseModel):
    index: int
    score: float
    object: str = "score"

class RerankResponse(BaseModel):
    data: List[RerankItem]