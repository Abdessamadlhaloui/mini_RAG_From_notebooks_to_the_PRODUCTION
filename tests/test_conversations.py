from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient
from config.settings import get_settings
from database.mongo_db import MongoDBClient
from database.redis_db import RedisClient
from main import app
from schemas.rag_schema import QueryResponse
from services.conversation_service import ConversationService
from services.history_service import HistoryService
from services.rag_service import RagService
from database.vector_db import ChromaDBClient
class _InsertOneResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id
class _DeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count
class _UpdateResult:
    def __init__(self, matched_count: int=1):
        self.matched_count = matched_count
class _FakeCursor:
    def __init__(self, docs: List[dict]):
        self._docs = docs
        self._sort: Optional[tuple[str, int]] = None
        self._skip = 0
        self._limit: Optional[int] = None
    def sort(self, field: str, direction: int):
        self._sort = (field, direction)
        return self
    def skip(self, n: int):
        self._skip = n
        return self
    def limit(self, n: int):
        self._limit = n
        return self
    async def to_list(self, length: int=100):
        docs = list(self._docs)
        if self._sort:
            field, direction = self._sort
            reverse = direction == -1
            docs.sort(key=lambda d: d.get(field), reverse=reverse)
        docs = docs[self._skip:]
        if self._limit is not None:
            docs = docs[:self._limit]
        return docs[:length]
class _FakeCollection:
    def __init__(self):
        self._docs: Dict[str, dict] = {}
        self._indexes: List[Any] = []
    async def create_index(self, spec):
        self._indexes.append(spec)
        return 'idx'
    async def insert_one(self, doc: dict):
        _id = doc.get('_id')
        if not _id:
            raise ValueError('Missing _id')
        self._docs[_id] = dict(doc)
        return _InsertOneResult(_id)
    async def find_one(self, query: dict):
        _id = query.get('_id')
        if _id is not None:
            return self._docs.get(_id)
        for d in self._docs.values():
            ok = True
            for k, v in query.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                return d
        return None
    def find(self, query: dict, projection: Optional[dict]=None):
        out = []
        for d in self._docs.values():
            ok = True
            for k, v in query.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                if projection:
                    filtered = {}
                    for key, include in projection.items():
                        if include and key in d:
                            filtered[key] = d[key]
                    out.append(filtered)
                else:
                    out.append(dict(d))
        return _FakeCursor(out)
    async def count_documents(self, query: dict):
        cursor = self.find(query)
        docs = await cursor.to_list(length=100000)
        return len(docs)
    async def delete_one(self, query: dict):
        _id = query.get('_id')
        if _id in self._docs:
            del self._docs[_id]
            return _DeleteResult(1)
        return _DeleteResult(0)
    async def delete_many(self, query: dict):
        to_delete = []
        for _id, d in self._docs.items():
            ok = True
            for k, v in query.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                to_delete.append(_id)
        for _id in to_delete:
            del self._docs[_id]
        return _DeleteResult(len(to_delete))
    async def update_one(self, query: dict, update: dict):
        _id = query.get('_id')
        if _id not in self._docs:
            return _UpdateResult(matched_count=0)
        d = self._docs[_id]
        inc = update.get('$inc', {})
        for k, v in inc.items():
            d[k] = int(d.get(k, 0)) + int(v)
        setv = update.get('$set', {})
        for k, v in setv.items():
            d[k] = v
        self._docs[_id] = d
        return _UpdateResult(matched_count=1)
    async def find_one_and_update(self, query: dict, update: dict, return_document=None):
        _id = query.get('_id')
        if _id not in self._docs:
            return None
        await self.update_one(query, update)
        return self._docs[_id]
class _FakeDB:
    def __init__(self):
        self._collections: Dict[str, _FakeCollection] = {}
    def __getitem__(self, name: str) -> _FakeCollection:
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]
    @property
    def conversations(self) -> _FakeCollection:
        return self['conversations']
    @property
    def messages(self) -> _FakeCollection:
        return self['messages']
    @property
    def documents(self) -> _FakeCollection:
        return self['documents']
@pytest.fixture
def client(monkeypatch):
    fake_db = _FakeDB()
    monkeypatch.setattr(MongoDBClient, 'get_db', classmethod(lambda cls: fake_db))
    monkeypatch.setattr(MongoDBClient, 'connect', classmethod(lambda cls: None))
    monkeypatch.setattr(MongoDBClient, 'close', classmethod(lambda cls: None))
    async def _noop_async():
        return None
    monkeypatch.setattr(RedisClient, 'connect', classmethod(lambda cls: _noop_async()))
    monkeypatch.setattr(RedisClient, 'close', classmethod(lambda cls: _noop_async()))
    async def _allow_rate_limit(identifier: str) -> bool:
        return True
    monkeypatch.setattr(RedisClient, 'check_rate_limit', classmethod(lambda cls, identifier: _allow_rate_limit(identifier)))
    async def _fake_run_pipeline(self: RagService, request) -> QueryResponse:
        conv_service = ConversationService()
        hist_service = HistoryService()
        conversation = await conv_service.get_or_create(getattr(request, 'conversation_id', None))
        conv_id = conversation.conversation_id
        await hist_service.add_message(conv_id, 'user', request.query)
        await hist_service.add_message(conv_id, 'assistant', f'Echo: {request.query}', sources=[], tokens_used=0)
        await conv_service.increment_message_count(conv_id)
        await conv_service.increment_message_count(conv_id)
        return QueryResponse(answer=f'Echo: {request.query}', sources=[], conversation_id=conv_id, tokens_used=0, cached=False)
    monkeypatch.setattr(RagService, 'run_pipeline', _fake_run_pipeline)
    with TestClient(app) as c:
        yield c
@pytest.fixture
def auth_headers():
    settings = get_settings()
    return {'Authorization': f'Bearer {settings.api_key}'}
class TestConversationCreate:
    def test_create_default_title(self, client: TestClient, auth_headers: dict):
        r = client.post('/api/v1/conversations', json={}, headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert 'conversation_id' in data
        assert data['title'] == 'New conversation'
    def test_create_custom_title_trimmed(self, client: TestClient, auth_headers: dict):
        long_title = 'x' * 500
        r = client.post('/api/v1/conversations', json={'title': long_title}, headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert len(data['title']) <= get_settings().CONVERSATION_TITLE_MAX_CHARS
class TestConversationList:
    def test_list_paginated(self, client: TestClient, auth_headers: dict):
        client.post('/api/v1/conversations', json={'title': 'A'}, headers=auth_headers)
        client.post('/api/v1/conversations', json={'title': 'B'}, headers=auth_headers)
        r = client.get('/api/v1/conversations?page=1&limit=20', headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert 'conversations' in data
        assert data['total'] >= 2
class TestConversationDetail:
    def test_detail_returns_messages_in_order(self, client: TestClient, auth_headers: dict):
        create = client.post('/api/v1/conversations', json={}, headers=auth_headers).json()
        conv_id = create['conversation_id']
        client.post('/api/v1/query', json={'query': 'What is RAG?', 'conversation_id': conv_id}, headers=auth_headers)
        r = client.get(f'/api/v1/conversations/{conv_id}', headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data['conversation_id'] == conv_id
        assert isinstance(data['messages'], list)
        assert len(data['messages']) >= 2
        assert data['messages'][0]['role'] == 'user'
        assert data['messages'][1]['role'] == 'assistant'
class TestConversationUpdate:
    def test_update_title(self, client: TestClient, auth_headers: dict):
        create = client.post('/api/v1/conversations', json={}, headers=auth_headers).json()
        conv_id = create['conversation_id']
        r = client.patch(f'/api/v1/conversations/{conv_id}', json={'title': 'RAG deep dive'}, headers=auth_headers)
        assert r.status_code == 200
        assert r.json()['title'] == 'RAG deep dive'
class TestConversationDelete:
    def test_delete_removes_conversation(self, client: TestClient, auth_headers: dict):
        create = client.post('/api/v1/conversations', json={}, headers=auth_headers).json()
        conv_id = create['conversation_id']
        r = client.delete(f'/api/v1/conversations/{conv_id}', headers=auth_headers)
        assert r.status_code == 200
        assert r.json()['deleted'] is True
class TestMultiTurnQuery:
    def test_two_turns_persist_messages(self, client: TestClient, auth_headers: dict):
        create = client.post('/api/v1/conversations', json={}, headers=auth_headers).json()
        conv_id = create['conversation_id']
        r1 = client.post('/api/v1/query', json={'query': 'Explain RAG in one sentence.', 'conversation_id': conv_id}, headers=auth_headers)
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1['conversation_id'] == conv_id
        r2 = client.post('/api/v1/query', json={'query': 'Give an example.', 'conversation_id': conv_id}, headers=auth_headers)
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2['conversation_id'] == conv_id
        detail = client.get(f'/api/v1/conversations/{conv_id}', headers=auth_headers).json()
        assert len(detail['messages']) == 4
class TestRateLimiting:
    def test_rate_limited_returns_429(self, client: TestClient, auth_headers: dict, monkeypatch):
        async def _deny(identifier: str) -> bool:
            return False
        monkeypatch.setattr(RedisClient, 'check_rate_limit', classmethod(lambda cls, identifier: _deny(identifier)))
        r = client.post('/api/v1/query', json={'query': 'test'}, headers=auth_headers)
        assert r.status_code == 429
        assert r.headers.get('Retry-After') == '60'