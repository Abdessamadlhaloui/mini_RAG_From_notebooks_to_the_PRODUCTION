# 🧠 Mini RAG API — The Legendary Production Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-00ADEE?style=for-the-badge&logo=google-cloud)](https://www.trychroma.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb)](https://www.mongodb.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis)](https://redis.io/)

A state-of-the-art, asynchronous **Retrieval-Augmented Generation (RAG)** system built for scale. This project transforms raw documents into a conversational intelligence layer with multi-turn memory, hybrid search, and production-grade observability.

---

## ✨ Features

*   **⚡ High Performance**: Built on **FastAPI** with fully asynchronous I/O and non-blocking background tasks.
*   **🗣️ Multi-Turn Conversations**: Advanced conversation management with persistent history and context-aware generation.
*   **🔍 Hybrid Retrieval**: Combines semantic vector search (ChromaDB) with metadata filtering for precision.
*   **🧠 Intelligent Memory**:
    *   **L1 Cache**: In-memory LRU cache for lightning-fast repeated queries.
    *   **L2 Cache**: Redis-based distributed caching for cross-worker consistency.
*   **🛡️ Production Hardened**:
    *   **HMAC Auth**: Secure API key validation using hashed comparison.
    *   **Rate Limiting**: Integrated Redis-based rate limiting to prevent abuse.
    *   **Structured Logging**: X-Request-ID tracking across the entire stack.
*   **🚀 Scalable Infrastructure**:
    *   **Celery Workers**: Offload heavy document ingestion to background workers.
    *   **Dockerized**: Multi-stage Docker builds for lean, secure production images.

---

## 🏗️ Architecture

```text
mini_RAG/
├── config/           # Pydantic-settings centralized configuration
├── controllers/      # Orchestration layer (Ingestion, RAG, Conversations)
├── database/         # Singleton clients (ChromaDB, MongoDB, Redis)
├── docker/           # Production-ready Dockerfile & configurations
├── helpers/          # Secure file handling, prompt templates, context utils
├── middlewares/      # Security, Logging, and Rate Limiting
├── models/           # ODM (MongoDB) and Database schemas
├── routes/           # RESTful API endpoints (FastAPI)
├── schemas/          # Pydantic request/reponse validation
├── services/         # Core Brain (Chunking, Embedding, Generation, Retrieval)
├── workers/          # Celery worker definitions for background tasks
├── main.py           # Application factory & lifespan management
└── docker-compose.yml # Full-stack orchestration
```

---

## 🚀 Quick Start

### Docker Deployment (Recommended)

```bash
# 1. Clone and configure
cp .env.example .env  # Add your OPENAI_API_KEY

# 2. Spin up the entire stack
docker-compose up --build
```

The stack includes:
- **API Server** (`:8000`)
- **MongoDB** (Metadata Store)
- **ChromaDB** (Vector Store)
- **Redis** (Cache & Broker)
- **Celery Worker** (Ingestion Pipeline)

---

## 🛠️ API Reference

### 🔐 Authentication
All `/api/v1/*` endpoints require a Bearer token in the header:
`Authorization: Bearer <your-secret-key>`

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness and Readiness probe |
| `POST` | `/api/v1/query` | Conversation-aware RAG query |
| `POST` | `/api/v1/ingest` | Multi-stage document ingestion |
| `GET` | `/api/v1/conversations` | List user conversation history |

### Example: Conversational Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the summary of the ingested PDF?",
    "conversation_id": "optional-uuid-here",
    "top_k": 4
  }'
```

---

## 🧪 Testing

```bash
# Run the full test suite with Pytest
pytest tests/ -v --asyncio-mode=auto
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
