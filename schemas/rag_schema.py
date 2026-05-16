from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's natural-language question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description='Number of context documents to retrieve (1-20)')
    conversation_id: Optional[str] = Field(default=None, description='Conversation ID for multi-turn queries (optional)')
class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    conversation_id: str = Field(default='', description='Conversation ID for multi-turn tracking')
    tokens_used: Optional[int] = Field(default=None, description='Total tokens used (if available)')
    cached: bool = Field(default=False, description='Whether this response was served from cache')
class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None