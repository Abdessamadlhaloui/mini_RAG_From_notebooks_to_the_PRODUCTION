from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SectionNode(BaseModel):
    section_id: str
    parent_section_id: Optional[str] = None
    level: str  # title, heading, subheading, paragraph
    title: Optional[str] = None
    content: Optional[str] = None
    page: Optional[int] = None
    children: List['SectionNode'] = Field(default_factory=list)

class TableData(BaseModel):
    table_id: str
    page: int
    section_id: Optional[str] = None
    title: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    raw_json: str

class ImageData(BaseModel):
    image_id: str
    page: int
    section_id: Optional[str] = None
    caption: Optional[str] = None
    position: Optional[List[float]] = None
    image_path: str

class FormulaData(BaseModel):
    formula_id: str
    page: int
    section_id: Optional[str] = None
    latex: str
    rendered_text: Optional[str] = None

class DocumentStructure(BaseModel):
    document_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    sections: List[SectionNode] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    images: List[ImageData] = Field(default_factory=list)
    formulas: List[FormulaData] = Field(default_factory=list)

    def to_mongo_dict(self) -> dict:
        d = self.model_dump()
        d['_id'] = d.pop('document_id')
        return d

    @classmethod
    def from_mongo_dict(cls, d: dict) -> 'DocumentStructure':
        data = dict(d)
        data['document_id'] = data.pop('_id')
        return cls(**data)
