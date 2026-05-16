from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, field_validator
from config.settings import get_settings
class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    @field_validator('title')
    @classmethod
    def trim_title(cls, v: Optional[str]) -> Optional[str]:
        settings = get_settings()
        if v:
            return v.strip()[:settings.CONVERSATION_TITLE_MAX_CHARS]
        return v
class UpdateConversationRequest(BaseModel):
    title: str
    @field_validator('title')
    @classmethod
    def trim_title(cls, v: str) -> str:
        settings = get_settings()
        return v.strip()[:settings.CONVERSATION_TITLE_MAX_CHARS]
class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    sources: list
    tokens_used: int
    created_at: datetime
class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]
class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummaryResponse]
    total: int
    page: int
    limit: int