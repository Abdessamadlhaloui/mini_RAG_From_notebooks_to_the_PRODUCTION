import logging
from database.mongo_db import MongoDBClient

logger = logging.getLogger(__name__)

async def create_indexes() -> None:
    db = MongoDBClient.get_db()
    try:
        await db['conversations'].create_index([('updated_at', -1)])
        await db['messages'].create_index([('conversation_id', 1), ('created_at', 1)])
        await db['documents'].create_index([('filename', 1)])
        await db['documents'].create_index([('embedding_version', 1), ('status', 1)])
        
        await db['document_structures'].create_index([('_id', 1)])
        await db['chunks'].create_index([('document_id', 1), ('section_id', 1), ('parent_id', 1)])
        await db['chunks'].create_index([('embedding_version', 1)])
        await db['chunks'].create_index([('source_filename', 1)])
        await db['tables'].create_index([('document_id', 1), ('page', 1)])
        await db['images'].create_index([('document_id', 1), ('page', 1)])
        await db['formulas'].create_index([('document_id', 1), ('page', 1)])
        await db['memories'].create_index([('user_id', 1), ('memory_type', 1), ('created_at', 1)])
        await db['memories'].create_index([('metadata.conversation_id', 1), ('access_count', -1)])
        await db['evaluations'].create_index([('created_at', -1)])
        await db['evaluations'].create_index([('query', 1), ('created_at', -1)])
        await db['reindex_runs'].create_index([('embedding_version', 1), ('status', 1)])
        
        logger.info('MongoDB indexes created/verified.')
    except Exception as e:
        logger.warning(f'Index creation failed (non-fatal): {e}')
