# 🧠 Autonomous Research Agent API

A production-ready REST API that accepts a research query, autonomously browses the web, synthesizes findings across multiple sources, and returns a structured report with citations.

Built with **FastAPI**, **LangGraph**, **Celery**, and **PostgreSQL**.

---

## Architecture

```
Client → FastAPI → Task Queue (Celery + Redis)
                        ↓
                  LangGraph Agent
                  ├── Plan: generate targeted search queries
                  ├── Search: DuckDuckGo web search
                  ├── Scrape: extract content from top URLs
                  └── Synthesize: structured report with citations
                        ↓
                  PostgreSQL (job state + results)
                  Redis (cache + queue)
```

## Features

- **Async job processing** — submit a query, poll or stream results
- **LangGraph agent** — multi-step: plan → search → scrape → synthesize
- **SSE streaming** — real-time agent step updates
- **Result caching** — Redis cache deduplicates identical queries
- **Configurable depth** — `quick` / `standard` / `deep` modes
- **Rate limiting** — per-IP with `slowapi`
- **Structured logging** — JSON logs via `structlog`
- **Full observability** — LangSmith tracing integration
- **Docker Compose** — one-command deployment

---

## Quick Start

### 1. Prerequisites
- Docker + Docker Compose
- OpenAI API key

### 2. Setup

```bash
git clone <your-repo>
cd research-agent
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and set SECRET_KEY
```

### 3. Start all services

```bash
docker compose up --build
```

This starts: **API** (port 8000), **Celery worker**, **PostgreSQL**, **Redis**, **Flower** (queue dashboard, port 5555).

### 4. Run DB migrations

```bash
docker compose exec api alembic upgrade head
```

### 5. Try it

```bash
# Submit a research job
curl -X POST http://localhost:8000/api/v1/research \
  -H "X-API-Key: dev-key-replace-in-production" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest advancements in quantum computing?", "depth": "standard"}'

# Poll the result (replace JOB_ID)
curl http://localhost:8000/api/v1/research/<JOB_ID> \
  -H "X-API-Key: dev-key-replace-in-production"

# Stream agent steps
curl -N http://localhost:8000/api/v1/research/<JOB_ID>/stream \
  -H "X-API-Key: dev-key-replace-in-production"
```

Or use the test client:
```bash
python scripts/test_client.py "What is the future of AI in healthcare?"
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (DB + Redis) |
| `POST` | `/api/v1/research` | Submit research query → `job_id` |
| `GET` | `/api/v1/research/{job_id}` | Poll job status + result |
| `GET` | `/api/v1/research/{job_id}/stream` | SSE stream of agent steps |

Full OpenAPI docs: `http://localhost:8000/docs`

### Request body (`POST /api/v1/research`)

```json
{
  "query": "What are the latest advancements in quantum computing?",
  "depth": "standard"
}
```

- `depth`: `"quick"` (3 sources, ~30s) | `"standard"` (5 sources, ~60s) | `"deep"` (10 sources, ~120s)

### Response (`GET /api/v1/research/{job_id}`)

```json
{
  "job_id": "uuid",
  "status": "completed",
  "query": "...",
  "depth": "standard",
  "result": {
    "summary": "2-3 paragraph synthesis...",
    "key_findings": ["Finding 1", "Finding 2"],
    "citations": [{"url": "...", "title": "...", "snippet": "..."}],
    "confidence_score": 0.87,
    "sources_scraped": 5
  }
}
```

---

## Local Development (without Docker)

```bash
# Install dependencies
poetry install

# Start Postgres and Redis (Docker just for infra)
docker compose up db redis -d

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Start Celery worker (separate terminal)
celery -A app.worker worker --loglevel=info
```

---

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=app --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Configuration

All settings are in `app/core/config.py` and loaded from `.env`.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `DATABASE_URL` | Postgres connection string | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis URL | `redis://localhost:6379/0` |
| `RATE_LIMIT_PER_MINUTE` | Requests per IP per minute | `20` |
| `CACHE_TTL` | Result cache TTL in seconds | `3600` |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `false` |

---

## Production Checklist

- [ ] Replace DuckDuckGo search with [Tavily](https://tavily.com) or SerpAPI
- [ ] Replace in-memory API key check with DB-backed key management
- [ ] Add HTTPS (nginx or Caddy in front of the API)
- [ ] Set `ENVIRONMENT=production` and `DEBUG=false`
- [ ] Restrict `CORS` origins
- [ ] Add Prometheus metrics (`/metrics` endpoint)
- [ ] Configure LangSmith for production tracing

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Agent | LangGraph + LangChain |
| LLM | GPT-4o (OpenAI) |
| Queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy (async) |
| Logging | structlog (JSON) |
| Observability | LangSmith |
| Rate Limiting | slowapi |
| Containerization | Docker Compose |
| Testing | pytest + httpx |
