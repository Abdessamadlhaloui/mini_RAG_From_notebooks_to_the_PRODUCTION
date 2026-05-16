from pydantic import BaseModel, Field
from typing import List, Dict, Any
class EmbeddingModel(BaseModel):
    text: str = Field(..., description='The original text chunk')
    embedding: List[float] = Field(..., description='The vector representation of the text')
    metadata: Dict[str, Any] = Field(default_factory=dict, description='Associated metadata (source filename, page number, etc.)')