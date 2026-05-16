from fastapi import UploadFile, HTTPException
from helpers.file_helper import save_upload_file_tmp, load_document
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from database.mongo_db import MongoDBClient
from schemas.document_schema import IngestResponse
from models.document_model import DocumentModel
import logging
logger = logging.getLogger('api_logger')
class IngestionController:
    def __init__(self, chunking_service: ChunkingService | None=None, embedding_service: EmbeddingService | None=None, vector_db_service: VectorDBService | None=None) -> None:
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db = vector_db_service or VectorDBService()
    async def ingest_file(self, file: UploadFile) -> tuple[IngestResponse, str]:
                   
        filename = file.filename or 'unknown'
        if not filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail='Unsupported file format. Upload a PDF or TXT file.')
        try:
            tmp_path = await save_upload_file_tmp(file)
            documents = await load_document(tmp_path, filename)
            if not documents:
                raise HTTPException(status_code=400, detail='Could not extract any text from the uploaded file.')
            chunks = await self.chunking_service.chunk_documents(documents)
            texts_to_embed = [chunk.page_content for chunk in chunks]
            embeddings = await self.embedding_service.get_embeddings(texts_to_embed)
            success = await self.vector_db.store_chunks(chunks, embeddings)
            if not success:
                raise HTTPException(status_code=500, detail='Failed to store vectors in the database.')
            await self._log_to_mongo(filename=filename, chunks_processed=len(chunks), status='success')
            logger.info("Ingested '%s' — %d chunks stored.", filename, len(chunks))
            response = IngestResponse(status='success', message='File successfully processed and ingested.', filename=filename, chunks_processed=len(chunks))
            return (response, tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            await self._log_to_mongo(filename=filename, chunks_processed=0, status='failed', error=str(exc))
            logger.error("Ingestion failed for '%s': %s", filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f'Ingestion failed: {exc}')
    @staticmethod
    async def _log_to_mongo(filename: str, chunks_processed: int, status: str, error: str | None=None) -> None:
                                                                              
        try:
            db = MongoDBClient.get_db()
            if db is not None:
                record = DocumentModel(filename=filename, chunks_processed=chunks_processed, status=status, error=error)
                await db.ingestion_logs.insert_one(record.to_mongo_dict())
        except Exception as mongo_exc:
            logger.warning('Failed to log ingestion to MongoDB: %s', mongo_exc)