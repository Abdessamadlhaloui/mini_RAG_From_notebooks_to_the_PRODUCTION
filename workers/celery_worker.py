import asyncio
import logging
from celery import Celery
from config.settings import get_settings
logger = logging.getLogger(__name__)
settings = get_settings()
celery_app = Celery('rag_worker', broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND)
celery_app.conf.update(task_serializer='json', result_serializer='json', accept_content=['json'], timezone='UTC', enable_utc=True, task_track_started=True, task_acks_late=True, worker_prefetch_multiplier=1)
@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_document_task(self, file_path: str, original_filename: str) -> dict:
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_run_ingestion(file_path, original_filename))
        loop.close()
        return result
    except Exception as exc:
        logger.error(f'Ingestion task failed for {original_filename}: {exc}')
        raise self.retry(exc=exc)
async def _run_ingestion(file_path: str, original_filename: str) -> dict:
           
    from controllers.ingestion_controller import IngestionController
    from database.mongo_db import MongoDBClient
    from helpers.file_helper import load_document
    MongoDBClient.connect()
    try:
        controller = IngestionController()
        documents = await load_document(file_path, original_filename)
        if not documents:
            return {'status': 'failed', 'chunks': 0, 'detail': 'No text extracted.'}
        chunks = await controller.chunking_service.chunk_documents(documents)
        texts_to_embed = [c.page_content for c in chunks]
        embeddings = await controller.embedding_service.get_embeddings(texts_to_embed)
        success = await controller.vector_db.store_chunks(chunks, embeddings)
        if not success:
            return {'status': 'failed', 'chunks': 0, 'detail': 'Vector store failed.'}
        await controller._log_to_mongo(filename=original_filename, chunks_processed=len(chunks), status='success')
        return {'status': 'completed', 'chunks': len(chunks)}
    finally:
        MongoDBClient.close()