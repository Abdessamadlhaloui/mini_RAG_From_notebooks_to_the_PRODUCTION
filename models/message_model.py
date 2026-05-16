from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid
class SourceChunk(BaseModel):
    content: str
    score: float
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
class MessageModel(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: Literal['user', 'assistant']
    content: str
    sources: List[SourceChunk] = Field(default_factory=list)
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('message_id')
        return d
    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'MessageModel':
        data = dict(d)
        data['message_id'] = data.pop('_id')
        return cls(**data)