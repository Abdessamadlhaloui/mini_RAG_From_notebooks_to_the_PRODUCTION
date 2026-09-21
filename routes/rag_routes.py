from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from config.settings import get_settings
from database.mongo_db import MongoDBClient
from schemas.rag_schema import QueryRequest, QueryResponse, ErrorResponse
from schemas.document_schema import IngestResponse
from controllers.rag_controller import RagController
from controllers.ingestion_controller import IngestionController
from middlewares.auth_middleware import verify_api_key
from helpers.file_helper import cleanup_temp_file
from services.evaluation_service import EvaluationService
from services.observability_service import MetricsService
from services.rag_service import RagService

router = APIRouter(tags=['RAG Pipeline'], dependencies=[Depends(verify_api_key)])


@router.post('/query', response_model=QueryResponse, summary='Query the RAG system', description='Submit a natural-language question. The system retrieves relevant context from ingested documents and generates an answer using an LLM.', responses={200: {'description': 'Successful answer generation'}, 401: {'model': ErrorResponse, 'description': 'Missing or invalid auth token'}, 500: {'model': ErrorResponse, 'description': 'Internal server error'}})
async def query_rag(request: QueryRequest):
    controller = RagController()
    return await controller.handle_query(request)


@router.post('/query/stream', summary='Stream a RAG answer', description='Streams newline-delimited JSON events for metadata, tokens, and completion.', responses={200: {'description': 'Streaming answer generation'}, 401: {'model': ErrorResponse, 'description': 'Missing or invalid auth token'}, 500: {'model': ErrorResponse, 'description': 'Internal server error'}})
async def stream_query_rag(request: QueryRequest):
    service = RagService()
    return StreamingResponse(service.stream_pipeline(request), media_type='application/x-ndjson')


@router.get('/evaluations', summary='List recent RAG evaluations')
async def list_evaluations(limit: int = Query(default=50, ge=1, le=500)):
    service = EvaluationService()
    return {'evaluations': await service.list_recent(limit)}


@router.get('/metrics/snapshot', summary='Return in-process metrics snapshot')
async def metrics_snapshot():
    return {'prometheus': MetricsService.render_prometheus()}


@router.get('/overview', summary='Return workspace overview')
async def workspace_overview():
    settings = get_settings()
    db = MongoDBClient.get_db()

    async def _count(collection: str) -> int:
        try:
            return int(await db[collection].count_documents({}))
        except Exception:
            return 0

    async def _recent(collection: str, sort_field: str, limit: int = 8) -> list[dict]:
        try:
            cursor = db[collection].find({}).sort(sort_field, -1).limit(limit)
            rows = await cursor.to_list(length=limit)
            output: list[dict] = []
            for row in rows:
                payload = dict(row)
                if '_id' in payload and 'document_id' not in payload:
                    payload['document_id'] = payload['_id']
                output.append(payload)
            return output
        except Exception:
            return []

    return {
        'version': settings.environment,
        'counts': {
            'documents': await _count('documents'),
            'conversations': await _count('conversations'),
            'messages': await _count('messages'),
            'memories': await _count('memories'),
            'evaluations': await _count('evaluations'),
        },
        'feature_flags': {
            'pdf_processing': settings.document_enable_pdf_processing,
            'ocr': settings.document_enable_ocr,
            'layout_parsing': settings.document_enable_layout_parsing,
            'tables': settings.document_enable_table_extraction,
            'formulas': settings.document_enable_formula_recognition,
            'image_captioning': settings.document_enable_image_captioning,
            'contextual_metadata': settings.document_enable_contextual_metadata,
            'multi_vector_indexing': settings.document_enable_multi_vector_indexing,
            'graph_indexing': settings.document_enable_graph_indexing,
            'streaming': settings.GENERATION_ENABLE_STREAMING,
            'verification': settings.GENERATION_ENABLE_VERIFICATION,
            'citations': settings.GENERATION_ENABLE_CITATIONS,
        },
        'recent_documents': await _recent('documents', 'timestamp'),
        'recent_conversations': await _recent('conversations', 'updated_at'),
        'recent_evaluations': await _recent('evaluations', 'created_at', limit=6),
        'metrics': MetricsService.render_prometheus(),
    }


@router.post('/ingest', response_model=IngestResponse, summary='Ingest a document', description='Upload a PDF or TXT file. The file is parsed, chunked, embedded, and stored in the vector database for future retrieval.', responses={200: {'description': 'Document successfully ingested'}, 400: {'model': ErrorResponse, 'description': 'Invalid file format or empty file'}, 401: {'model': ErrorResponse, 'description': 'Missing or invalid auth token'}, 500: {'model': ErrorResponse, 'description': 'Ingestion pipeline error'}})
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile=File(...)):
           
    controller = IngestionController()
    response, tmp_path = await controller.ingest_file(file)
    background_tasks.add_task(cleanup_temp_file, tmp_path)
    return response
