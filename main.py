from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging
from routes import health_routes, rag_routes
from routes.conversation_routes import router as conversation_router
from middlewares.logging_middleware import StructuredLoggingMiddleware, get_request_id
from middlewares.rate_limit_middleware import RateLimitMiddleware
from database.vector_db import ChromaDBClient
from database.mongo_db import MongoDBClient
from database.redis_db import RedisClient
from database.indexes import create_indexes
from config.settings import get_settings
logger = logging.getLogger('api_logger')
@asynccontextmanager
async def lifespan(app: FastAPI):
                                                                      
    ChromaDBClient.get_client()
    MongoDBClient.connect()
    await create_indexes()
    await RedisClient.connect()
    logger.info('Application startup complete.')
    yield
    MongoDBClient.close()
    await RedisClient.close()
    logger.info('Application shutdown complete.')
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title='Production RAG API', description='A robust, asynchronous Retrieval-Augmented Generation system. Ingest documents and query them using natural language.', version='1.0.0', lifespan=lifespan)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    app.include_router(health_routes.router)
    app.include_router(conversation_router)
    app.include_router(rag_routes.router, prefix='/api/v1')
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