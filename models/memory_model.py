from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

class MemoryEntry(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    memory_type: str  # short_term, long_term, semantic, profile, conversation, summarized
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0

    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('memory_id')
        return d

    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'MemoryEntry':
        data = dict(d)
        data['memory_id'] = data.pop('_id')
        return cls(**data)

class ConversationSummary(BaseModel):
    conversation_id: str
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('conversation_id')
        return d

    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'ConversationSummary':
        data = dict(d)
        data['conversation_id'] = data.pop('_id')
        return cls(**data)
