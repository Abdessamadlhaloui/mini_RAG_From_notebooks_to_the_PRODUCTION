"""
Asynchronous MongoDB client singleton using Motor.
Manages the connection lifecycle (connect / close) and exposes the database handle.
"""
from motor.motor_asyncio import AsyncIOMotorClient
from config.settings import get_settings
import logging

logger = logging.getLogger("api_logger")


class MongoDBClient:
    """Async singleton for the MongoDB connection via Motor."""

    _client: AsyncIOMotorClient | None = None
    _db = None

    @classmethod
    def connect(cls) -> None:
        """Establishes the MongoDB connection if not already connected."""
        if cls._client is None:
            settings = get_settings()
            try:
                cls._client = AsyncIOMotorClient(settings.mongo_uri)
                cls._db = cls._client[settings.mongo_db_name]
                logger.info(
                    "MongoDB connected: uri=%s db=%s",
                    settings.mongo_uri,
                    settings.mongo_db_name,
                )
            except Exception as exc:
                logger.error("Failed to connect to MongoDB: %s", exc)
                raise

    @classmethod
    def close(cls) -> None:
        """Gracefully closes the MongoDB connection."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB connection closed.")

    @classmethod
    def get_db(cls):
        """Returns the database handle, lazily connecting if necessary."""
        if cls._db is None:
            cls.connect()
        return cls._db
