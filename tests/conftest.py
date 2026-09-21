import pytest


@pytest.fixture(autouse=True)
def _patch_external_dependencies(monkeypatch):
    from database.mongo_db import MongoDBClient
    from database.redis_db import RedisClient
    from database.vector_store import VectorDatabase
    from database.neo4j_db import Neo4jClient
    
    monkeypatch.setattr(MongoDBClient, 'connect', classmethod(lambda cls: None))
    monkeypatch.setattr(MongoDBClient, 'close', classmethod(lambda cls: None))
    class _FakeProvider:
        def connect(self): pass
        def close(self): pass
        async def store_chunks(self, chunks, embeddings, collection_name="rag_documents"): return True
        async def search(self, query_embedding, top_k=4, collection_name="rag_documents", filter_criteria=None): return []
        async def delete_by_document_id(self, document_id, collection_name="rag_documents"): return True
        
    monkeypatch.setattr(VectorDatabase, 'initialize', classmethod(lambda cls, provider: setattr(cls, '_provider', _FakeProvider())))
    monkeypatch.setattr(VectorDatabase, 'get_provider', classmethod(lambda cls: _FakeProvider()))
    
    async def _noop_async(): return None
    monkeypatch.setattr(Neo4jClient, 'connect', classmethod(lambda cls: _noop_async()))
    monkeypatch.setattr(Neo4jClient, 'close', classmethod(lambda cls: _noop_async()))

    monkeypatch.setattr(RedisClient, 'connect', classmethod(lambda cls: _noop_async()))
    monkeypatch.setattr(RedisClient, 'close', classmethod(lambda cls: _noop_async()))
    async def _allow_rate_limit(identifier: str) -> bool:
        return True
    monkeypatch.setattr(RedisClient, 'check_rate_limit', classmethod(lambda cls, identifier: _allow_rate_limit(identifier)))
    import main as main_module
    async def _noop_indexes():
        return None
    monkeypatch.setattr(main_module, 'create_indexes', _noop_indexes)
