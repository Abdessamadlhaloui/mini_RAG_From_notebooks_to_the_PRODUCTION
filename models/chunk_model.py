from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChunkModel(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    document_id: str
    section_id: Optional[str] = None
    
    document_title: Optional[str] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    page: Optional[int] = None
    author: Optional[str] = None
    language: Optional[str] = None
    date: Optional[str] = None
    source_filename: str
    
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    
    chunk_type: str = "text" # text, table, image, formula
    embedding_version: Optional[str] = None
    
    content: str
    
    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('chunk_id')
        return d

    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'ChunkModel':
        data = dict(d)
        data['chunk_id'] = data.pop('_id')
        return cls(**data)
