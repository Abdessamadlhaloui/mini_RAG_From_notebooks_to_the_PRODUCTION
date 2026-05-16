import logging
from typing import Optional
import redis.asyncio as aioredis
from config.settings import get_settings
logger = logging.getLogger(__name__)
class RedisClient:
    _client: Optional[aioredis.Redis] = None
    @classmethod
    async def connect(cls) -> None:
                                                
        settings = get_settings()
        try:
            cls._client = aioredis.from_url(settings.REDIS_URL, encoding='utf-8', decode_responses=True)
            await cls._client.ping()
            logger.info('Redis connected successfully.')
        except Exception as e:
            logger.warning(f'Redis unavailable — caching disabled. Reason: {e}')
            cls._client = None
    @classmethod
    async def close(cls) -> None:
        if cls._client:
            await cls._client.close()
    @classmethod
    def get_client(cls) -> Optional[aioredis.Redis]:
        return cls._client
    @classmethod
    async def get_cache(cls, key: str) -> Optional[str]:
        if not cls._client:
            return None
        try:
            return await cls._client.get(key)
        except Exception:
            return None
    @classmethod
    async def set_cache(cls, key: str, value: str, ttl: Optional[int]=None) -> None:
        if not cls._client:
            return
        settings = get_settings()
        expire = ttl or settings.REDIS_CACHE_TTL
        try:
            await cls._client.setex(key, expire, value)
        except Exception as e:
            logger.warning(f'Redis set_cache failed: {e}')
    @classmethod
    async def delete_cache(cls, key: str) -> None:
        if not cls._client:
            return
        try:
            await cls._client.delete(key)
        except Exception as e:
            logger.warning(f'Redis delete_cache failed: {e}')
    @classmethod
    async def check_rate_limit(cls, identifier: str) -> bool:
                   
        if not cls._client:
            return True
        import time
        settings = get_settings()
        now = time.time()
        window_start = now - settings.REDIS_RATE_LIMIT_WINDOW
        key = f'rate_limit:{identifier}'
        try:
            pipe = cls._client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, settings.REDIS_RATE_LIMIT_WINDOW * 2)
            results = await pipe.execute()
            current_count = results[1]
            return current_count < settings.REDIS_RATE_LIMIT_MAX
        except Exception as e:
            logger.warning(f'Rate limit check failed: {e}')
            return True