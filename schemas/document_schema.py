from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    chunks_processed: int = 0
    embedding_version: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_processed: int
    error: Optional[str] = None
    page_count: Optional[int] = None
    has_ocr: bool = False
    tables_count: int = 0
    images_count: int = 0
    timestamp: datetime
