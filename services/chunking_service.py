import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from config.settings import get_settings
from services.embedding_service import EmbeddingService
from services.contextual_enrichment_service import ContextualEnrichmentService
from services.parent_document_service import ParentDocumentService

logger = logging.getLogger(__name__)

class SemanticSplitter:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 1500,
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    async def split_text(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
            
        embeddings = await self.embedding_service.embed_documents(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sim = self.cosine_similarity(embeddings[i-1], embeddings[i])
            
            if sim >= self.similarity_threshold and (current_length + len(sentence)) <= self.max_chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class ChunkingService:
    def __init__(self, embedding_service: EmbeddingService | None = None, parent_document_service: ParentDocumentService | None = None):
        settings = get_settings()
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService()
        self.parent_document_service = parent_document_service or ParentDocumentService()
        self.contextual_enrichment_service = ContextualEnrichmentService()
        self.splitter = SemanticSplitter(
            self.embedding_service,
            similarity_threshold=settings.chunking_similarity_threshold,
            max_chunk_size=settings.chunking_max_chunk_size,
        )
        
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            parent_id = await self.parent_document_service.store_parent_document(
                document_id=str(doc.metadata.get("document_id") or doc.metadata.get("source_filename") or "unknown"),
                content=text,
                metadata=doc.metadata,
            )
            chunks = await self.splitter.split_text(text)
            if not chunks:
                chunks = [text]
            chunk_window = self.settings.chunking_context_window
            for i, chunk_text in enumerate(chunks):
                metadata = dict(doc.metadata or {})
                metadata["parent_id"] = parent_id
                metadata["parent_chunk_id"] = parent_id
                metadata["chunk_index"] = i
                metadata["chunk_count"] = len(chunks)
                metadata["chunk_type"] = metadata.get("chunk_type") or "text"
                metadata["hierarchy_level"] = "chunk"
                metadata["document_hierarchy"] = self._document_hierarchy(metadata, i, len(chunks))
                metadata["chunk_context"] = self._context_window(chunks, i, chunk_window)
                metadata["semantic_chunk"] = True
                chunk_doc = Document(page_content=chunk_text, metadata=metadata)
                if self.settings.document_enable_contextual_metadata:
                    chunk_doc = await self.contextual_enrichment_service.enrich_chunk(
                        chunk_doc,
                        parent_context=text,
                    )
                    chunk_doc.metadata = dict(chunk_doc.metadata or {})
                    chunk_doc.metadata["contextual_chunk"] = True
                chunked_docs.append(chunk_doc)
        return chunked_docs

    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return await self.split_documents(documents)

    @staticmethod
    def _document_hierarchy(metadata: Dict[str, Any], index: int, total: int) -> List[str]:
        hierarchy = [
            str(metadata.get("document_id") or metadata.get("source_filename") or "document"),
            str(metadata.get("section_id") or metadata.get("page") or "page"),
            f"chunk_{index + 1}_of_{total}",
        ]
        return hierarchy

    @staticmethod
    def _context_window(chunks: List[str], index: int, window: int) -> str:
        start = max(0, index - window)
        end = min(len(chunks), index + window + 1)
        context_chunks = [chunk.strip() for chunk in chunks[start:end] if chunk.strip()]
        return "\n\n".join(context_chunks)
