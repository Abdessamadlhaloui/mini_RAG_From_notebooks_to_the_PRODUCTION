from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    environment: str = 'dev'
    api_key: str = 'super-secret-key'
    openai_api_key: str = ''
    
    VECTOR_DB_PROVIDER: str = 'qdrant'
    chroma_persist_directory: str = './chroma_db'
    QDRANT_URL: str = 'http://localhost:6333'
    QDRANT_API_KEY: str = ''
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_similarity_threshold: float = 0.75
    chunking_max_chunk_size: int = 1500
    chunking_context_window: int = 1
    document_enable_pdf_processing: bool = True
    document_enable_ocr: bool = True
    document_enable_layout_parsing: bool = True
    document_enable_table_extraction: bool = True
    document_enable_formula_recognition: bool = True
    document_enable_image_extraction: bool = True
    document_enable_image_captioning: bool = True
    document_enable_contextual_metadata: bool = True
    document_enable_multi_vector_indexing: bool = True
    document_enable_graph_indexing: bool = True
    top_k: int = 4
    RETRIEVAL_INITIAL_K: int = 50
    RETRIEVAL_FINAL_K: int = 5
    RETRIEVAL_RRF_K: int = 60
    RETRIEVAL_BM25_CACHE_TTL_SECONDS: int = 300
    RETRIEVAL_ENABLE_BM25: bool = True
    RETRIEVAL_ENABLE_GRAPH: bool = True
    RETRIEVAL_ENABLE_MULTI_QUERY: bool = True
    RETRIEVAL_ENABLE_QUERY_EXPANSION: bool = True
    RETRIEVAL_ENABLE_HYDE: bool = True
    RETRIEVAL_ENABLE_SELF_QUERY: bool = True
    RETRIEVAL_ENABLE_RERANK: bool = True
    RETRIEVAL_ENABLE_PARENT_CONTEXT: bool = True
    RETRIEVAL_MULTI_QUERY_COUNT: int = 3
    CROSS_ENCODER_MODEL: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    CROSS_ENCODER_DEVICE: str = 'auto'
    CROSS_ENCODER_BATCH_SIZE: int = 16
    GENERATION_MODEL: str = 'gpt-4o-mini'
    GENERATION_TIMEOUT_SECONDS: float = 30.0
    GENERATION_MAX_RETRIES: int = 3
    GENERATION_ENABLE_STREAMING: bool = True
    GENERATION_ENABLE_VERIFICATION: bool = True
    GENERATION_ENABLE_CITATIONS: bool = True
    GENERATION_MIN_GROUNDING_SCORE: float = 0.25
    MEMORY_ENABLE_SEMANTIC: bool = True
    MEMORY_MAX_RELEVANT_ITEMS: int = 5
    MEMORY_COLLECTION_NAME: str = 'rag_memories'
    EVALUATION_ENABLE_BACKGROUND: bool = True
    EVALUATION_ENABLE_RAGAS: bool = True
    METRICS_ENABLE_PROMETHEUS: bool = True
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = 'enterprise-rag-platform'
    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = 'enterprise-rag-platform'
    RAGAS_ENABLED: bool = True
    RAGAS_SAMPLE_SIZE: int = 20
    RAGAS_MIN_CONTEXTS: int = 1
    COST_INPUT_TOKEN_USD_PER_1K: float = 0.00015
    COST_OUTPUT_TOKEN_USD_PER_1K: float = 0.0006
    
    EMBEDDING_PROVIDER: str = 'openai'
    EMBEDDING_MODEL: str = 'text-embedding-3-large'
    EMBEDDING_DIMENSION: int = 3072
    EMBEDDING_BATCH_SIZE: int = 100
    EMBEDDING_DEVICE: str = 'auto'
    BGE_M3_MODEL: str = 'BAAI/bge-m3'
    BGE_M3_DIMENSION: int = 1024
    BGE_M3_INFERENCE_BATCH_SIZE: int = 12
    BGE_M3_MAX_LENGTH: int = 8192
    EMBEDDING_REINDEX_BATCH_SIZE: int = 100
    
    NEO4J_URI: str = 'bolt://localhost:7687'
    NEO4J_USER: str = 'neo4j'
    NEO4J_PASSWORD: str = 'password'
    NEO4J_DATABASE: str = 'neo4j'
    
    mongo_uri: str = 'mongodb://localhost:27017/'
    mongo_db_name: str = 'mini-rag'
    
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_CACHE_TTL: int = 3600
    REDIS_RATE_LIMIT_WINDOW: int = 60
    REDIS_RATE_LIMIT_MAX: int = 100
    CELERY_BROKER_URL: str = 'redis://localhost:6379/1'
    CELERY_RESULT_BACKEND: str = 'redis://localhost:6379/2'
    
    allowed_origins: str = 'http://localhost:3000,http://localhost:8080'
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
