from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class EvaluationResult(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    
    faithfulness: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    answer_correctness: Optional[float] = None
    hallucination_rate: Optional[float] = None
    
    latency_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('evaluation_id')
        return d

    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'EvaluationResult':
        data = dict(d)
        data['evaluation_id'] = data.pop('_id')
        return cls(**data)
