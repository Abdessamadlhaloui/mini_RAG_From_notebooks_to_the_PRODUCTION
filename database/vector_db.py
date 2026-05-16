import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import get_settings
import logging
logger = logging.getLogger('api_logger')
class ChromaDBClient:
    _instance = None
    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        if cls._instance is None:
            settings = get_settings()
            cls._instance = chromadb.PersistentClient(path=settings.chroma_persist_directory, settings=ChromaSettings(anonymized_telemetry=False))
            logger.info("ChromaDB PersistentClient initialized at '%s'.", settings.chroma_persist_directory)
        return cls._instance
    @classmethod
    def reset(cls) -> None:
        cls._instance = None