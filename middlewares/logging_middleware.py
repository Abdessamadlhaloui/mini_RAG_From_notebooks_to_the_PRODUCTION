"""
Structured logging middleware with X-Request-ID correlation.

Every request is assigned a unique request ID (or one is read from the
incoming X-Request-ID header). The ID is propagated via contextvars so
that any logger in the application can include it automatically.
"""
import time
import json
import uuid
import logging
import contextvars
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api_logger")

# Context variable to hold the current request ID across async boundaries
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_request_id() -> str:
    """Returns the current request's correlation ID from context."""
    return request_id_ctx.get()


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Assigns / propagates an X-Request-ID.
    2. Logs every request and response as structured JSON.
    3. Measures and reports latency in milliseconds.
    """

    async def dispatch(self, request: Request, call_next):
        # Read or generate correlation ID
        req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request_id_ctx.set(req_id)

        start_time = time.time()

        try:
            response = await call_next(request)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            log_dict = {
                "request_id": req_id,
                "method": request.method,
                "url": str(request.url.path),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "client": request.client.host if request.client else "unknown",
            }
            logger.info(json.dumps(log_dict))

            # Echo the request ID back to the caller
            response.headers["X-Request-ID"] = req_id
            return response

        except Exception as exc:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            log_dict = {
                "request_id": req_id,
                "method": request.method,
                "url": str(request.url.path),
                "error": str(exc),
                "latency_ms": latency_ms,
            }
            logger.error(json.dumps(log_dict))
            raise
