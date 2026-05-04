"""
MongoDB document model for ingestion log records.
Used by the IngestionController to persist structured records instead of raw dicts.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DocumentModel(BaseModel):
    """Represents an ingested document record stored in MongoDB."""

    filename: str = Field(..., description="Original name of the uploaded file")
    chunks_processed: int = Field(default=0, description="Number of text chunks created")
    status: str = Field(..., description="Ingestion status: 'success' or 'failed'")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of ingestion")
    error: Optional[str] = Field(default=None, description="Error message if ingestion failed")

    def to_mongo_dict(self) -> dict:
        """Serializes the model to a dict suitable for MongoDB insertion."""
        data = self.model_dump()
        # Convert datetime to ISO string for consistent storage
        data["timestamp"] = self.timestamp.isoformat()
        return data

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}
