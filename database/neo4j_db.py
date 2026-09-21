import logging
from typing import Optional, Dict, Any, List
from neo4j import AsyncGraphDatabase, AsyncDriver
from config.settings import get_settings

logger = logging.getLogger(__name__)

class Neo4jClient:
    _driver: Optional[AsyncDriver] = None

    @classmethod
    async def connect(cls) -> None:
        if cls._driver is None:
            settings = get_settings()
            try:
                cls._driver = AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                await cls._driver.verify_connectivity()
                logger.info(f"Neo4j connected successfully at {settings.NEO4J_URI}")
            except Exception as exc:
                logger.error(f"Failed to connect to Neo4j: {exc}")
                cls._driver = None

    @classmethod
    async def close(cls) -> None:
        if cls._driver is not None:
            await cls._driver.close()
            cls._driver = None
            logger.info("Neo4j connection closed.")

    @classmethod
    def get_driver(cls) -> Optional[AsyncDriver]:
        return cls._driver

    @classmethod
    async def execute_query(cls, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if cls._driver is None:
            raise RuntimeError("Neo4j driver is not initialized.")
            
        parameters = parameters or {}
        async with cls._driver.session() as session:
            result = await session.run(query, parameters)
            records = await result.data()
            return records
