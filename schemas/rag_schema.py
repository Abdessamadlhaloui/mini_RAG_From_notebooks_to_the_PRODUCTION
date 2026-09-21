from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's natural-language question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description='Number of context documents to retrieve (1-20)')
    conversation_id: Optional[str] = Field(default=None, description='Conversation ID for multi-turn queries (optional)')
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description='Optional metadata filter for retrieval (document_id, source_filename, etc.)')


class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class Citation(BaseModel):
    citation_id: str
    source_index: int
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source_filename: Optional[str] = None
    excerpt: str
    score: Optional[float] = None


class VerificationResult(BaseModel):
    grounded: bool
    grounding_score: float
    hallucination_risk: float
    source_coverage: float
    unsupported_claims: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    conversation_id: str = Field(default='', description='Conversation ID for multi-turn tracking')
    tokens_used: Optional[int] = Field(default=None, description='Total tokens used (if available)')
    cached: bool = Field(default=False, description='Whether this response was served from cache')
    citations: List[Citation] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    confidence_score: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    evaluation_id: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
