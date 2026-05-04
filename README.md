# 🧠 Mini RAG API — Production-Ready

A robust, asynchronous **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangChain**, **ChromaDB**, **MongoDB**, and **OpenAI**.

## Architecture

```
project/
├── config/           # Pydantic-settings configuration
├── controllers/      # Thin controller layer (request → service → response)
├── database/         # Singleton DB clients (ChromaDB, MongoDB)
├── docker/           # Dockerfile (multi-stage, gunicorn)
├── helpers/          # File upload, text sanitization, prompt utilities
├── middlewares/      # Auth (hashed API key), logging (X-Request-ID)
├── models/           # Pydantic data models for MongoDB records
├── routes/           # FastAPI route definitions with Swagger docs
├── schemas/          # Request/response validation schemas
├── services/         # Core business logic (chunking, embedding, RAG)
├── tests/            # Pytest integration tests
├── .github/workflows # CI pipeline (flake8 + pytest)
├── main.py           # App factory with lifespan events & exception handlers
├── docker-compose.yml
├── requirements.txt
└── .env / .env.example
```

## Quick Start

### Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Start MongoDB (if not using Docker)
# Ensure MongoDB is running on localhost:27017

# 5. Run the server
uvicorn main:app --reload
```

### Docker (Recommended)

```bash
docker-compose up --build
```

This starts:
- **API** on `http://localhost:8000`
- **MongoDB** on `localhost:27017`

## API Endpoints

| Method | Path             | Description                     | Auth Required |
|--------|------------------|---------------------------------|---------------|
| GET    | `/health`        | Health check                    | No            |
| POST   | `/api/v1/query`  | Query the RAG system            | Yes           |
| POST   | `/api/v1/ingest` | Upload & ingest a document      | Yes           |

### Authentication

All `/api/v1/*` endpoints require a Bearer token:

```
Authorization: Bearer <your-api-key>
```

### Example: Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer super-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval augmented generation?", "top_k": 4}'
```

### Example: Ingest

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer super-secret-key" \
  -F "file=@document.pdf"
```

## Running Tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable                  | Description                          | Default                        |
|---------------------------|--------------------------------------|--------------------------------|
| `ENVIRONMENT`             | Runtime environment (dev/prod)       | `dev`                          |
| `API_KEY`                 | Bearer token for API auth            | `super-secret-key`             |
| `OPENAI_API_KEY`          | OpenAI API key                       | —                              |
| `CHROMA_PERSIST_DIRECTORY`| Path to ChromaDB storage             | `./chroma_db`                  |
| `CHUNK_SIZE`              | Text chunk size in characters        | `1000`                         |
| `CHUNK_OVERLAP`           | Overlap between consecutive chunks   | `200`                          |
| `TOP_K`                   | Default number of documents to retrieve | `4`                         |
| `MONGO_URI`               | MongoDB connection string            | `mongodb://localhost:27017/`   |
| `MONGO_DB_NAME`           | MongoDB database name                | `mini-rag`                     |
| `ALLOWED_ORIGINS`         | Comma-separated CORS origins         | `http://localhost:3000,...`     |

## Production Deployment

The Dockerfile uses **gunicorn** with **UvicornWorker** for production:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4
```

## License

See [LICENSE](LICENSE) for details.
