import time
import json
import uuid
import logging
import contextvars
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('api_logger')
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')
def get_request_id() -> str:
    return request_id_ctx.get()
class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get('X-Request-ID', uuid.uuid4().hex)
        request_id_ctx.set(req_id)
        start_time = time.time()
        try:
            response = await call_next(request)
            latency_ms = round((time.time() - start_time) * 1000, 2)
            log_dict = {'request_id': req_id, 'method': request.method, 'url': str(request.url.path), 'status_code': response.status_code, 'latency_ms': latency_ms, 'client': request.client.host if request.client else 'unknown'}
            logger.info(json.dumps(log_dict))
            response.headers['X-Request-ID'] = req_id
            return response
        except Exception as exc:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            log_dict = {'request_id': req_id, 'method': request.method, 'url': str(request.url.path), 'error': str(exc), 'latency_ms': latency_ms}
            logger.error(json.dumps(log_dict))
            raise