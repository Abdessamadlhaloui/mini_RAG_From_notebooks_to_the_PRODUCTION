import logging
from database.mongo_db import MongoDBClient
logger = logging.getLogger(__name__)
async def create_indexes() -> None:
    db = MongoDBClient.get_db()
    try:
        await db['conversations'].create_index([('updated_at', -1)])
        await db['messages'].create_index([('conversation_id', 1), ('created_at', 1)])
        await db['documents'].create_index([('filename', 1)])
        logger.info('MongoDB indexes created/verified.')
    except Exception as e:
        logger.warning(f'Index creation failed (non-fatal): {e}')