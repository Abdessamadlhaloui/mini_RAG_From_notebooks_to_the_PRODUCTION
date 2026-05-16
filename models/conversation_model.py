from datetime import datetime
from pydantic import BaseModel, Field
import uuid
class ConversationModel(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = 'New conversation'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    metadata: dict = Field(default_factory=dict)
    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('conversation_id')
        return d
    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'ConversationModel':
        data = dict(d)
        data['conversation_id'] = data.pop('_id')
        return cls(**data)