import os
import asyncio
from celery import Celery
from config.settings import get_settings
from database.mongo_db import MongoDBClient
from database.vector_store import VectorDatabase
from database.qdrant_db import QdrantProvider
from models.document_model import DocumentModel

from services.ocr_service import OCRService
from services.layout_service import LayoutService
from services.table_extraction_service import TableExtractionService
from services.image_captioning_service import ImageCaptioningService
from services.formula_service import FormulaService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService

from services.parent_document_service import ParentDocumentService
from services.contextual_enrichment_service import ContextualEnrichmentService
from services.multi_vector_service import MultiVectorService
from services.query_enhancement_service import GraphRetrievalService
from models.chunk_model import ChunkModel
import uuid

from langchain_core.documents import Document

import logging
logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery(
    'rag_worker',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@celery_app.task(name='process_document_task')
def process_document_task(file_path: str, document_id: str):
    asyncio.run(_async_process_document(file_path, document_id))

async def _async_process_document(file_path: str, document_id: str):
    MongoDBClient.connect()
    if settings.VECTOR_DB_PROVIDER.lower() == 'qdrant':
        VectorDatabase.initialize(QdrantProvider())
        
    db = MongoDBClient.get_db()
    
    try:
        ocr = OCRService()
        layout = LayoutService()
        tables = TableExtractionService()
        captioning = ImageCaptioningService()
        formula = FormulaService()
        
        ext = os.path.splitext(file_path)[1].lower()
        has_ocr = False
        tables_count = 0
        images_count = 0
        
        if ext == '.pdf':
            structure = layout.extract_structure(file_path, document_id)
            extracted_tables = tables.extract_tables(file_path)
            structure.tables = extracted_tables
            
            tables_count = len(structure.tables)
            images_count = len(structure.images)
            
            await db['document_structures'].insert_one(structure.to_mongo_dict())
            texts = [s.content for s in structure.sections if s.content]
            
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.heic']:
            ocr_res = ocr.extract_text(file_path)
            texts = [ocr_res["text"]] if ocr_res["text"] else []
            has_ocr = True if texts else False
            
            cap = captioning.generate_caption(file_path)
            form = formula.extract_formula(file_path)
            
            from models.document_structure_model import DocumentStructure
            struct = DocumentStructure(document_id=document_id)
            if cap: struct.metadata["caption"] = cap
            if form: struct.metadata["formula"] = form
            await db['document_structures'].insert_one(struct.to_mongo_dict())
            images_count = 1
            
        else:
            texts = []
            
        if texts:
            chunker = ChunkingService()
            parent_service = ParentDocumentService()
            context_service = ContextualEnrichmentService()
            multi_vector = MultiVectorService()
            graph_service = GraphRetrievalService()
            
            full_context = "\n\n".join(texts)
            filename = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]
            
            parent_id = await parent_service.store_parent_document(
                document_id=document_id,
                content=full_context,
                metadata={"filename": filename}
            )
            
            docs = [Document(page_content=t, metadata={"document_id": document_id, "parent_id": parent_id, "filename": filename}) for t in texts]
            
            chunks = await chunker.split_documents(docs)
            enriched_chunks = await context_service.enrich_chunks(chunks, full_context)
            
            embed_service = EmbeddingService()
            current_version = embed_service.version

            for chunk in enriched_chunks:
                chunk_uuid = str(uuid.uuid4())
                chunk.metadata["chunk_id"] = chunk_uuid
                chunk.metadata["embedding_version"] = current_version

                chunk_model = ChunkModel(
                    chunk_id=chunk_uuid,
                    parent_id=parent_id,
                    document_id=document_id,
                    source_filename=filename,
                    content=chunk.page_content,
                    chunk_type="text",
                    embedding_version=current_version,
                )
                await db['chunks'].insert_one(chunk_model.to_mongo_dict())

                await multi_vector.index_chunk_multi_vector(chunk)
                await graph_service.index_chunk(
                    chunk_id=chunk_uuid,
                    document_id=document_id,
                    content=chunk.page_content,
                    metadata={
                        "source_filename": filename,
                        "parent_id": parent_id,
                    },
                )
                
            chunks_processed = len(enriched_chunks)
        else:
            chunks_processed = 0
            
        await db['documents'].update_one(
            {'_id': document_id},
            {'$set': {
                'status': 'success', 
                'chunks_processed': chunks_processed,
                'has_ocr': has_ocr,
                'tables_count': tables_count,
                'images_count': images_count
            }}
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Processing failed: {traceback.format_exc()}")
        await db['documents'].update_one(
            {'_id': document_id},
            {'$set': {'status': 'failed', 'error': str(e)}}
        )
    finally:
        MongoDBClient.close()
        if VectorDatabase._provider:
            VectorDatabase._provider.close()


@celery_app.task(name="reindex_documents_task")
def reindex_documents_task() -> None:
    asyncio.run(_async_reindex_documents())


async def _async_reindex_documents() -> None:
    MongoDBClient.connect()
    if settings.VECTOR_DB_PROVIDER.lower() == "qdrant":
        VectorDatabase.initialize(QdrantProvider())

    db = MongoDBClient.get_db()
    embed_service = EmbeddingService()
    vector_db = VectorDBService()
    current_version = embed_service.version
    batch_size = settings.EMBEDDING_REINDEX_BATCH_SIZE
    pending: list[tuple[str, Document]] = []
    processed = 0
    failed = 0
    run_id = str(uuid.uuid4())

    try:
        await db["reindex_runs"].insert_one(
            {
                "_id": run_id,
                "embedding_version": current_version,
                "status": "running",
                "processed": 0,
                "failed": 0,
            }
        )
        cursor = db["chunks"].find(
            {
                "$or": [
                    {"embedding_version": {"$exists": False}},
                    {"embedding_version": None},
                    {"embedding_version": {"$ne": current_version}},
                ]
            }
        )
        async for chunk_data in cursor:
            chunk_id = chunk_data["_id"]
            doc = Document(
                page_content=chunk_data["content"],
                metadata={
                    "chunk_id": str(chunk_id),
                    "document_id": chunk_data.get("document_id"),
                    "parent_id": chunk_data.get("parent_id"),
                    "source_filename": chunk_data.get("source_filename"),
                    "embedding_version": current_version,
                },
            )
            pending.append((str(chunk_id), doc))
            if len(pending) >= batch_size:
                try:
                    processed += await _process_reindex_batch(pending, embed_service, vector_db, db, current_version)
                except Exception:
                    failed += len(pending)
                    raise
                pending = []

        if pending:
            try:
                processed += await _process_reindex_batch(pending, embed_service, vector_db, db, current_version)
            except Exception:
                failed += len(pending)
                raise

        await db["documents"].update_many(
            {"embedding_version": {"$ne": current_version}, "status": "success"},
            {"$set": {"embedding_version": current_version}},
        )
        await db["reindex_runs"].update_one(
            {"_id": run_id},
            {"$set": {"status": "success", "processed": processed, "failed": failed}},
        )
        logger.info("Re-index completed for embedding version %s", current_version)
    except Exception:
        import traceback

        try:
            await db["reindex_runs"].update_one(
                {"_id": run_id},
                {"$set": {"status": "failed", "processed": processed, "failed": failed}},
            )
        except Exception:
            logger.warning("Failed to persist re-index failure status for run_id=%s", run_id)
        logger.error("Re-indexing failed: %s", traceback.format_exc())
        raise
    finally:
        MongoDBClient.close()
        if VectorDatabase._provider:
            VectorDatabase._provider.close()


async def _process_reindex_batch(
    batch: list[tuple[str, Document]],
    embed_service: EmbeddingService,
    vector_db: VectorDBService,
    db,
    version: str,
) -> int:
    ids, docs = zip(*batch)
    embeddings = await embed_service.embed_documents([d.page_content for d in docs])
    stored = await vector_db.store_chunks(list(docs), embeddings)
    if not stored:
        raise RuntimeError("Vector database rejected re-index batch.")
    for chunk_id in ids:
        await db["chunks"].update_one({"_id": chunk_id}, {"$set": {"embedding_version": version}})
    return len(ids)
