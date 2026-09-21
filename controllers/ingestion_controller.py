import logging
import uuid

from fastapi import HTTPException, UploadFile

from database.mongo_db import MongoDBClient
from helpers.file_helper import save_upload_file_tmp
from models.document_model import DocumentModel
from schemas.document_schema import IngestResponse
from services.document_processing_service import DocumentProcessingService
from services.chunking_service import ChunkingService
from services.embedding_indexing_service import EmbeddingIndexingService
from services.embedding_service import EmbeddingService
from services.multi_vector_service import MultiVectorService
from services.vector_db_service import VectorDBService

logger = logging.getLogger("api_logger")


class IngestionController:
    def __init__(
        self,
        chunking_service: ChunkingService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_db_service: VectorDBService | None = None,
    ) -> None:
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()
        self.document_processing_service = DocumentProcessingService()
        self.embedding_indexer = EmbeddingIndexingService(self.embedding_service, self.vector_db)

    async def ingest_file(self, file: UploadFile) -> tuple[IngestResponse, str]:
        filename = file.filename or "unknown"
        if not filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(status_code=400, detail="Unsupported file format. Upload a PDF or TXT file.")

        document_id = str(uuid.uuid4())
        tmp_path = ""
        try:
            tmp_path = await save_upload_file_tmp(file)
            processing_result = await self.document_processing_service.process_file(tmp_path, filename, document_id=document_id)
            documents = processing_result.documents
            if not documents:
                raise HTTPException(status_code=400, detail="Could not extract any text from the uploaded file.")

            for document in documents:
                document.metadata["document_id"] = document_id

            chunks = await self.chunking_service.chunk_documents(documents)
            result = await self.embedding_indexer.index_chunks(document_id=document_id, filename=filename, chunks=chunks)
            await self._log_to_mongo(
                document_id=document_id,
                filename=filename,
                chunks_processed=result.chunks_indexed,
                status="success",
                embedding_version=result.embedding_version,
                document_model=processing_result.document,
            )
            logger.info("Ingested document_id=%s filename=%s chunks=%d", document_id, filename, result.chunks_indexed)
            return (
                IngestResponse(
                    document_id=document_id,
                    status="success",
                    message="File successfully processed and ingested.",
                    filename=filename,
                    chunks_processed=result.chunks_indexed,
                    embedding_version=result.embedding_version,
                ),
                tmp_path,
            )
        except HTTPException as exc:
            await self._log_to_mongo(
                filename=filename,
                chunks_processed=0,
                status="failed",
                error=str(exc.detail),
                document_id=document_id,
                document_model=None,
            )
            raise
        except Exception as exc:
            await self._log_to_mongo(
                filename=filename,
                chunks_processed=0,
                status="failed",
                error=str(exc),
                document_id=document_id,
                document_model=None,
            )
            logger.error("Ingestion failed document_id=%s filename=%s error=%s", document_id, filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    @staticmethod
    async def _log_to_mongo(
        filename: str,
        chunks_processed: int,
        status: str,
        error: str | None = None,
        document_id: str | None = None,
        embedding_version: str | None = None,
        document_model: DocumentModel | None = None,
    ) -> None:
        try:
            db = MongoDBClient.get_db()
            if db is None:
                return
            base_payload = document_model.model_dump() if document_model else {}
            record = DocumentModel(
                document_id=document_id or base_payload.get("document_id"),
                filename=filename,
                chunks_processed=chunks_processed,
                status=status,
                error=error,
                embedding_version=embedding_version or base_payload.get("embedding_version"),
                file_type=base_payload.get("file_type"),
                page_count=base_payload.get("page_count"),
                author=base_payload.get("author"),
                language=base_payload.get("language"),
                creation_date=base_payload.get("creation_date"),
                document_title=base_payload.get("document_title"),
                has_ocr=bool(base_payload.get("has_ocr", False)),
                ocr_confidence=base_payload.get("ocr_confidence"),
                tables_count=int(base_payload.get("tables_count") or 0),
                images_count=int(base_payload.get("images_count") or 0),
                formulas_count=int(base_payload.get("formulas_count") or 0),
                structure_id=base_payload.get("structure_id"),
            )
            payload = record.to_mongo_dict()
            if document_id:
                payload["_id"] = document_id
                update_payload = dict(payload)
                update_payload.pop("_id", None)
                await db["documents"].update_one({"_id": document_id}, {"$set": update_payload}, upsert=True)
            else:
                await db["documents"].insert_one(payload)
        except Exception as mongo_exc:
            logger.warning("Failed to log ingestion to MongoDB: %s", mongo_exc)
