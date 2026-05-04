"""
Pydantic schemas for the RAG query endpoint.
These schemas enforce strict request/response validation and power the Swagger docs.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class QueryRequest(BaseModel):
    """Incoming RAG query payload."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural-language question",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of context documents to retrieve (1-20)",
    )


class SourceDocument(BaseModel):
    """A single retrieved source chunk returned alongside the answer."""

    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response payload for a successful RAG query."""

    answer: str
    sources: List[SourceDocument]
    cached: bool = Field(default=False, description="Whether this response was served from cache")


class ErrorResponse(BaseModel):
    """Standardized error payload returned by all error handlers."""

    detail: str
    request_id: Optional[str] = None
