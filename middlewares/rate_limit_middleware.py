import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from database.redis_db import RedisClient
logger = logging.getLogger(__name__)
EXCLUDED_PATHS = {'/health', '/docs', '/openapi.json', '/redoc'}
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)
        identifier = request.headers.get('X-Forwarded-For', request.client.host)
        allowed = await RedisClient.check_rate_limit(identifier)
        if not allowed:
            logger.warning(f'Rate limit exceeded for {identifier}')
            return JSONResponse(status_code=429, content={'detail': 'Too many requests. Please slow down.'}, headers={'Retry-After': '60'})
        return await call_next(request)