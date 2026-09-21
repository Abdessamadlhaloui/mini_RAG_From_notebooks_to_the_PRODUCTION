from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging

from routes import health_routes, rag_routes
from routes.conversation_routes import router as conversation_router
from middlewares.logging_middleware import StructuredLoggingMiddleware, get_request_id
from middlewares.rate_limit_middleware import RateLimitMiddleware

from database.mongo_db import MongoDBClient
from database.redis_db import RedisClient
from database.indexes import create_indexes
from database.vector_store import VectorDatabase
from database.qdrant_db import QdrantProvider
from database.neo4j_db import Neo4jClient
from config.settings import get_settings
from services.observability_service import configure_optional_tracing

logger = logging.getLogger('api_logger')

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / 'frontend'
FRONTEND_ASSETS_DIR = FRONTEND_DIR / 'assets'
FRONTEND_INDEX = FRONTEND_DIR / 'index.html'

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_optional_tracing()
    
    MongoDBClient.connect()
    await create_indexes()
    await RedisClient.connect()
    
    if settings.VECTOR_DB_PROVIDER.lower() == 'qdrant':
        VectorDatabase.initialize(QdrantProvider())
    
    await Neo4jClient.connect()
    
    logger.info('Application startup complete.')
    yield
    
    MongoDBClient.close()
    await RedisClient.close()
    if VectorDatabase._provider:
        VectorDatabase._provider.close()
    await Neo4jClient.close()
    
    logger.info('Application shutdown complete.')

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title='Enterprise RAG Platform', description='A robust, asynchronous Retrieval-Augmented Generation system.', version='2.0.0', lifespan=lifespan)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    if FRONTEND_ASSETS_DIR.exists():
        app.mount('/assets', StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name='assets')
    app.include_router(health_routes.router)
    app.include_router(conversation_router)
    app.include_router(rag_routes.router, prefix='/api/v1')

    @app.get('/')
    async def frontend_index():
        if FRONTEND_INDEX.exists():
            return FileResponse(FRONTEND_INDEX)
        return JSONResponse(status_code=404, content={'detail': 'Frontend assets are not available.'})
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail, 'request_id': get_request_id()})
        
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={'detail': 'Request validation failed. Check your input.', 'errors': exc.errors(), 'request_id': get_request_id()})
        
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error('Unhandled exception on %s %s: %s', request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(status_code=500, content={'detail': 'An unexpected internal error occurred.', 'request_id': get_request_id()})
        
    return app

app = create_app()
