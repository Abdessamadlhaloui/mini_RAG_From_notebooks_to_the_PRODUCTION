import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from config.settings import get_settings
from services.parent_document_service import ParentDocumentService

logger = logging.getLogger("api_logger")


class ParentContextService:
    def __init__(self, parent_document_service: ParentDocumentService | None = None) -> None:
        self._parent_service = parent_document_service or ParentDocumentService()
        self._settings = get_settings()

    async def expand(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        if not self._settings.RETRIEVAL_ENABLE_PARENT_CONTEXT:
            return results
        expanded: List[Tuple[Document, float]] = []
        for doc, score in results:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id:
                expanded.append((doc, score))
                continue
            parent = await self._parent_service.get_parent_document(str(parent_id))
            if not parent or not parent.content.strip():
                expanded.append((doc, score))
                continue
            metadata = dict(doc.metadata)
            metadata["child_excerpt"] = doc.page_content
            metadata["parent_context"] = True
            parent_doc = Document(page_content=parent.content, metadata=metadata)
            expanded.append((parent_doc, score))
        return expanded
