"""
Embedding data model.
Provides a typed structure for a text chunk paired with its vector representation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class EmbeddingModel(BaseModel):
    """Represents a text chunk and its corresponding vector embedding."""

    text: str = Field(..., description="The original text chunk")
    embedding: List[float] = Field(..., description="The vector representation of the text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (source filename, page number, etc.)",
    )
