import pytest
@pytest.fixture(autouse=True)
def _patch_external_dependencies(monkeypatch):
    from database.mongo_db import MongoDBClient
    from database.redis_db import RedisClient
    from database.vector_db import ChromaDBClient
    monkeypatch.setattr(MongoDBClient, 'connect', classmethod(lambda cls: None))
    monkeypatch.setattr(MongoDBClient, 'close', classmethod(lambda cls: None))
    class _FakeChromaCollection:
        def __init__(self):
            self._docs = []
            self._metadatas = []
            self._ids = []
        def add(self, embeddings, documents, metadatas, ids):
            self._docs.extend(list(documents or []))
            self._metadatas.extend(list(metadatas or []))
            self._ids.extend(list(ids or []))
        def query(self, query_embeddings, n_results, include):
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        def get(self, include=None):
            return {'documents': list(self._docs)}
    class _FakeChromaClient:
        def __init__(self):
            self._collections = {}
        def get_or_create_collection(self, name: str):
            if name not in self._collections:
                self._collections[name] = _FakeChromaCollection()
            return self._collections[name]
    monkeypatch.setattr(ChromaDBClient, 'get_client', classmethod(lambda cls: _FakeChromaClient()))
    async def _noop_async():
        return None
    monkeypatch.setattr(RedisClient, 'connect', classmethod(lambda cls: _noop_async()))
    monkeypatch.setattr(RedisClient, 'close', classmethod(lambda cls: _noop_async()))
    async def _allow_rate_limit(identifier: str) -> bool:
        return True
    monkeypatch.setattr(RedisClient, 'check_rate_limit', classmethod(lambda cls, identifier: _allow_rate_limit(identifier)))
    import main as main_module
    async def _noop_indexes():
        return None
    monkeypatch.setattr(main_module, 'create_indexes', _noop_indexes)