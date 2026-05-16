from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
class DocumentModel(BaseModel):
    filename: str = Field(..., description='Original name of the uploaded file')
    chunks_processed: int = Field(default=0, description='Number of text chunks created')
    status: str = Field(..., description="Ingestion status: 'success' or 'failed'")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description='UTC timestamp of ingestion')
    error: Optional[str] = Field(default=None, description='Error message if ingestion failed')
    def to_mongo_dict(self) -> dict:
        data = self.model_dump()
        data['timestamp'] = self.timestamp.isoformat()
        return data
    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}