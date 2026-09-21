import logging
from typing import Dict, Any, List
from models.document_structure_model import DocumentStructure, SectionNode, ImageData

logger = logging.getLogger(__name__)

class LayoutService:
    def __init__(self):
        try:
            import fitz
            self.enabled = True
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed. Layout extraction disabled.")
            self.enabled = False

    def extract_structure(self, pdf_path: str, doc_id: str) -> DocumentStructure:
        if not self.enabled:
            return DocumentStructure(document_id=doc_id)
            
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            structure = DocumentStructure(
                document_id=doc_id,
                page_count=len(doc),
                metadata=doc.metadata if doc.metadata else {}
            )
            
            sections = []
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for b in blocks:
                    if b['type'] == 0:
                        text = "".join([l["text"] for l in b["lines"] for s in l["spans"]])
                        if text.strip():
                            sections.append(SectionNode(
                                section_id=f"sec_{page_num}_{len(sections)}",
                                level="paragraph",
                                content=text.strip(),
                                page=page_num + 1
                            ))
                    elif b['type'] == 1:
                        image_id = f"img_{page_num}_{len(images)}"
                        images.append(ImageData(
                            image_id=image_id,
                            page=page_num + 1,
                            position=b.get("bbox", []),
                            image_path=f"extracted_{image_id}.png"
                        ))
                        
            structure.sections = sections
            structure.images = images
            doc.close()
            return structure
        except Exception as e:
            logger.error(f"Layout extraction failed for {pdf_path}: {e}")
            return DocumentStructure(document_id=doc_id)
