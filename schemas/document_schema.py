"""
Pydantic schemas for the document ingestion endpoint.
"""
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Response payload for a successful document ingestion."""

    status: str = Field(..., description="Ingestion result: 'success' or 'error'")
    message: str = Field(..., description="Human-readable status message")
    filename: str = Field(..., description="Name of the ingested file")
    chunks_processed: int = Field(..., description="Number of text chunks indexed")
