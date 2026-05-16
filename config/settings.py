from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List
class Settings(BaseSettings):
    environment: str = 'dev'
    api_key: str = 'super-secret-key'
    openai_api_key: str = ''
    chroma_persist_directory: str = './chroma_db'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    mongo_uri: str = 'mongodb://localhost:27017/'
    mongo_db_name: str = 'mini-rag'
    allowed_origins: str = 'http://localhost:3000,http://localhost:8080'
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_CACHE_TTL: int = 3600
    REDIS_RATE_LIMIT_WINDOW: int = 60
    REDIS_RATE_LIMIT_MAX: int = 100
    CELERY_BROKER_URL: str = 'redis://localhost:6379/1'
    CELERY_RESULT_BACKEND: str = 'redis://localhost:6379/2'
    JWT_SECRET_KEY: str = 'change-me-in-production'
    JWT_ALGORITHM: str = 'HS256'
    JWT_EXPIRE_MINUTES: int = 60
    MAX_HISTORY_TURNS: int = 10
    MAX_HISTORY_TOKENS: int = 2000
    CONVERSATION_TITLE_MAX_CHARS: int = 80
    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(',') if origin.strip()]
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
@lru_cache()
def get_settings() -> Settings:
    return Settings()