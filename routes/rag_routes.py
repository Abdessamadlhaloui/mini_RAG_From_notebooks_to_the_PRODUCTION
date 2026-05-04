"""
RAG pipeline routes — query and document ingestion.

All endpoints are protected by the ``verify_api_key`` dependency.
File cleanup after ingestion is handled via FastAPI BackgroundTasks.
"""
from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File
from schemas.rag_schema import QueryRequest, QueryResponse, ErrorResponse
from schemas.document_schema import IngestResponse
from controllers.rag_controller import RagController
from controllers.ingestion_controller import IngestionController
from middlewares.auth_middleware import verify_api_key
from helpers.file_helper import cleanup_temp_file

router = APIRouter(
    tags=["RAG Pipeline"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Submit a natural-language question. The system retrieves "
    "relevant context from ingested documents and generates an answer "
    "using an LLM.",
    responses={
        200: {"description": "Successful answer generation"},
        401: {"model": ErrorResponse, "description": "Missing or invalid auth token"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_rag(request: QueryRequest):
    """Delegates to the RagController for end-to-end RAG processing."""
    controller = RagController()
    return await controller.handle_query(request)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document",
    description="Upload a PDF or TXT file. The file is parsed, chunked, "
    "embedded, and stored in the vector database for future retrieval.",
    responses={
        200: {"description": "Document successfully ingested"},
        400: {"model": ErrorResponse, "description": "Invalid file format or empty file"},
        401: {"model": ErrorResponse, "description": "Missing or invalid auth token"},
        500: {"model": ErrorResponse, "description": "Ingestion pipeline error"},
    },
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Uploads and ingests a document.
    Temporary files are cleaned up in a background task after the response
    is sent, so the client is never blocked by filesystem I/O.
    """
    controller = IngestionController()
    response, tmp_path = await controller.ingest_file(file)

    # Schedule safe temp-file deletion AFTER the response is sent
    background_tasks.add_task(cleanup_temp_file, tmp_path)

    return response
