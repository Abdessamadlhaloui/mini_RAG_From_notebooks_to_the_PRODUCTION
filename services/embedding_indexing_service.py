import logging
import uuid
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from config.settings import get_settings
from database.mongo_db import MongoDBClient
from models.chunk_model import ChunkModel
from services.embedding_service import EmbeddingService
from services.multi_vector_service import MultiVectorService
from services.query_enhancement_service import GraphRetrievalService
from services.vector_db_service import VectorDBService

logger = logging.getLogger("api_logger")


@dataclass(frozen=True)
class EmbeddingIndexResult:
    document_id: str
    embedding_version: str
    chunks_indexed: int
    chunk_ids: List[str]


class EmbeddingIndexingService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_db_service: VectorDBService | None = None,
        graph_service: GraphRetrievalService | None = None,
        multi_vector_service: MultiVectorService | None = None,
    ) -> None:
        self._settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()
        self.graph_service = graph_service or GraphRetrievalService()
        self.multi_vector_service = multi_vector_service or MultiVectorService()

    async def index_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: List[Document],
    ) -> EmbeddingIndexResult:
        if not document_id:
            raise ValueError("document_id must be provided.")
        if not filename:
            raise ValueError("filename must be provided.")
        if not chunks:
            return EmbeddingIndexResult(
                document_id=document_id,
                embedding_version=self.embedding_service.version,
                chunks_indexed=0,
                chunk_ids=[],
            )

        version = self.embedding_service.version
        prepared = self._prepare_chunks(document_id=document_id, filename=filename, chunks=chunks, version=version)
        embeddings = await self.embedding_service.embed_documents([chunk.page_content for chunk in prepared])
        if len(embeddings) != len(prepared):
            raise RuntimeError(f"Embedding count mismatch: expected {len(prepared)} got {len(embeddings)}.")

        stored = await self.vector_db.store_chunks(prepared, embeddings)
        if not stored:
            raise RuntimeError("Vector database rejected chunk upsert.")

        await self._persist_chunks(prepared, version)
        await self._index_graph_chunks(prepared)
        if self._settings.document_enable_multi_vector_indexing:
            await self._index_multi_vector_chunks(prepared)
        chunk_ids = [str(chunk.metadata["chunk_id"]) for chunk in prepared]
        logger.info(
            "Embedding index complete document_id=%s chunks=%s version=%s",
            document_id,
            len(prepared),
            version,
        )
        return EmbeddingIndexResult(
            document_id=document_id,
            embedding_version=version,
            chunks_indexed=len(prepared),
            chunk_ids=chunk_ids,
        )

    @staticmethod
    def _prepare_chunks(
        document_id: str,
        filename: str,
        chunks: List[Document],
        version: str,
    ) -> List[Document]:
        prepared: List[Document] = []
        for index, chunk in enumerate(chunks):
            content = (chunk.page_content or "").strip()
            if not content:
                continue
            metadata = dict(chunk.metadata or {})
            metadata["chunk_id"] = str(metadata.get("chunk_id") or uuid.uuid4())
            metadata["document_id"] = document_id
            metadata["source_filename"] = metadata.get("source_filename") or filename
            metadata["filename"] = metadata.get("filename") or filename
            metadata["chunk_index"] = metadata.get("chunk_index", index)
            metadata["embedding_version"] = version
            prepared.append(Document(page_content=content, metadata=metadata))
        return prepared

    @staticmethod
    async def _persist_chunks(chunks: List[Document], version: str) -> None:
        db = MongoDBClient.get_db()
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            chunk_id = str(metadata["chunk_id"])
            model = ChunkModel(
                chunk_id=chunk_id,
                parent_id=metadata.get("parent_id"),
                document_id=str(metadata["document_id"]),
                section_id=metadata.get("section_id"),
                document_title=metadata.get("document_title"),
                section=metadata.get("section"),
                heading=metadata.get("heading"),
                page=metadata.get("page"),
                author=metadata.get("author"),
                language=metadata.get("language"),
                date=metadata.get("date"),
                source_filename=str(metadata["source_filename"]),
                keywords=list(metadata.get("keywords") or []),
                entities=list(metadata.get("entities") or []),
                summary=metadata.get("summary"),
                chunk_type=str(metadata.get("chunk_type") or "text"),
                embedding_version=version,
                content=chunk.page_content,
            )
            await db["chunks"].update_one(
                {"_id": chunk_id},
                {"$set": model.to_mongo_dict()},
                upsert=True,
            )

    async def _index_graph_chunks(self, chunks: List[Document]) -> None:
        if not self._settings.document_enable_graph_indexing:
            return
        for chunk in chunks:
            try:
                await self.graph_service.index_chunk(
                    chunk_id=str(chunk.metadata["chunk_id"]),
                    document_id=str(chunk.metadata["document_id"]),
                    content=chunk.page_content,
                    metadata=dict(chunk.metadata),
                )
            except Exception as exc:
                logger.warning("Graph indexing failed chunk_id=%s error=%s", chunk.metadata.get("chunk_id"), exc)

    async def _index_multi_vector_chunks(self, chunks: List[Document]) -> None:
        for chunk in chunks:
            try:
                await self.multi_vector_service.index_chunk_multi_vector(chunk)
            except Exception as exc:
                logger.warning("Multi-vector indexing failed chunk_id=%s error=%s", chunk.metadata.get("chunk_id"), exc)
