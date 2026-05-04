"""
FastAPI application entry point.

Configures:
- Lifespan events (startup / shutdown) for DB connections.
- CORS with restricted origins from .env.
- Structured logging middleware with X-Request-ID.
- Global exception handlers that return sanitized JSON responses.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging

from routes import health_routes, rag_routes
from middlewares.logging_middleware import StructuredLoggingMiddleware, get_request_id
from database.vector_db import ChromaDBClient
from database.mongo_db import MongoDBClient
from config.settings import get_settings

logger = logging.getLogger("api_logger")


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup") / ("shutdown")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup; tear them down on shutdown."""
    # Startup
    ChromaDBClient.get_client()
    MongoDBClient.connect()
    logger.info("Application startup complete.")
    yield
    # Shutdown
    MongoDBClient.close()
    logger.info("Application shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Builds and returns the fully configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Production RAG API",
        description=(
            "A robust, asynchronous Retrieval-Augmented Generation system. "
            "Ingest documents and query them using natural language."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # --- Middlewares (order matters: last added = first executed) ---
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(health_routes.router)
    app.include_router(rag_routes.router, prefix="/api/v1")

    # --- Global Exception Handlers ---

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Returns a sanitized JSON payload for all HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Returns a structured 422 response without leaking internal paths."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed. Check your input.",
                "errors": exc.errors(),
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Catches any unhandled exception and returns a generic 500 response.
        Internal stack traces are NEVER leaked to the client.
        """
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected internal error occurred.",
                "request_id": get_request_id(),
            },
        )

    return app


app = create_app()
