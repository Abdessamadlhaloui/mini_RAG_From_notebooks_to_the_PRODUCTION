from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class DocumentModel(BaseModel):
    document_id: Optional[str] = Field(default=None, description="Unique UUID for document")
    filename: str = Field(..., description="Original name of the uploaded file")
    file_type: Optional[str] = Field(default=None, description="Type of the file (pdf, txt, png, etc.)")
    chunks_processed: int = Field(default=0, description="Number of text chunks created")
    status: str = Field(..., description="Ingestion status: 'success' or 'failed'")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of ingestion")
    error: Optional[str] = Field(default=None, description="Error message if ingestion failed")
    
    page_count: Optional[int] = None
    author: Optional[str] = None
    language: Optional[str] = None
    creation_date: Optional[str] = None
    document_title: Optional[str] = None
    
    has_ocr: bool = False
    ocr_confidence: Optional[float] = None
    
    tables_count: int = 0
    images_count: int = 0
    formulas_count: int = 0
    
    structure_id: Optional[str] = None
    embedding_version: Optional[str] = None

    def to_mongo_dict(self) -> dict:
        data = self.model_dump()
        data['timestamp'] = self.timestamp.isoformat()
        return data

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}
