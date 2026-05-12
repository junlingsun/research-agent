# Autonomous Research Agent API
![CI](https://github.com/junlingsun/research-agent/actions/workflows/ci.yml/badge.svg)

A production-ready multi-agent research platform built with FastAPI and LangGraph. Submit a research query and receive a structured, cited report synthesized from live web search and your private knowledge base.

---

## Architecture

```
Client
  ↓
FastAPI  →  Redis Queue  →  Celery Worker
                                  ↓
                         Orchestrator Graph (LangGraph)
                         │
                         ├── PlanAgent (subgraph)
                         │   ├── generate_queries
                         │   ├── critique_queries     ← internal loop
                         │   └── finalize_queries
                         │
                         ├── search_node             ← parallel Tavily search
                         ├── scrape_node             ← parallel URL scraping
                         ├── retrieve_docs_node      ← Qdrant private KB
                         │
                         ├── SynthesizeAgent (subgraph)
                         │   ├── write_draft
                         │   ├── critique_draft      ← internal loop
                         │   └── finalize_draft
                         │
                         └── evaluate_node           ← outer quality loop
                               ↓ rejected (with feedback)
                         loops back to PlanAgent
                               ↓ approved
                              END
                         │
                         ↓
               PostgreSQL + Qdrant + Redis
```

### Multi-agent design

Three specialized agents coordinate through an orchestrator graph:

**PlanAgent** — autonomously generates and critiques search queries before passing them to search. Internal generate → critique → refine loop ensures queries are specific and distinct.

**SynthesizeAgent** — autonomously drafts and self-critiques research reports. Internal draft → critique → revise loop improves report quality before the external evaluator sees it.

**EvaluateNode** — scores the final report (0-1) and either approves it or loops back to PlanAgent with specific gap feedback. Circuit breaker prevents infinite loops.

---

## Features

- **Multi-agent orchestration** — three specialized LangGraph subgraphs coordinating via shared state
- **Autonomous quality loop** — evaluator critiques reports and retries with feedback until score ≥ 0.7
- **Private knowledge base** — ingest PDFs, URLs, and text; automatically searched at synthesis time
- **Parallel execution** — search queries and URL scraping run concurrently via asyncio.gather
- **Depth control** — quick / standard / deep modes tune queries, sources, and iteration limits
- **Async job processing** — Celery + Redis queue; submit and poll or stream via SSE
- **Result caching** — Redis deduplicates identical queries, skips the agent entirely
- **Full observability** — LangSmith traces every agent decision, node input/output, and latency
- **Structured outputs** — Pydantic schemas enforce LLM response shape at every node
- **Structured logging** — JSON logs via structlog for production log aggregators
- **Rate limiting** — per-IP request limiting via slowapi
- **Docker Compose** — one-command deployment of all services

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Agent orchestration | LangGraph |
| LLM | Groq (Llama 3.3) or OpenAI (GPT-4o) |
| Web search | Tavily |
| Vector DB | Qdrant |
| Queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy (async) |
| Observability | LangSmith + structlog |
| Rate limiting | slowapi |
| Infra | Docker Compose |
| Testing | pytest + httpx |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- API keys: Groq or OpenAI, Tavily, LangSmith (optional)

### 1. Clone and configure

```bash
git clone <your-repo>
cd research-agent
cp .env.example .env
# Fill in your API keys
```

Minimum required in .env:

```bash
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=tvly-...
SECRET_KEY=any-random-32-char-string
```

### 2. Start all services

```bash
docker compose up --build -d
```

Starts: API (8000), Celery worker, Celery beat, PostgreSQL, Redis, Qdrant (6333), Flower (5555).

### 3. Run database migrations

```bash
docker compose exec api alembic upgrade head
```

### 4. Open the API docs

```
http://localhost:8000/docs
```

---

## API Reference

### Research

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/research | Submit research query → job_id |
| GET | /api/v1/research/{job_id} | Poll status and result |
| GET | /api/v1/research/{job_id}/stream | SSE stream of agent steps |

**Request:**
```json
{
  "query": "What are the latest advancements in quantum computing in 2025?",
  "depth": "standard"
}
```

**Depth levels:**

| Depth | Queries | Sources | Max iterations | Max revisions |
|---|---|---|---|---|
| quick | 2 | 3 | 1 | 1 |
| standard | 3 | 5 | 3 | 2 |
| deep | 5 | 10 | 5 | 3 |

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "summary": "Comprehensive 2-3 paragraph answer...",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "citations": [{"url": "...", "title": "...", "snippet": "..."}],
    "confidence_score": 0.87,
    "sources_scraped": 5,
    "iterations": 2,
    "evaluation_history": [
      {"iteration": 1, "score": 0.6, "approved": false, "gaps": ["missing X"]},
      {"iteration": 2, "score": 0.87, "approved": true, "gaps": []}
    ]
  }
}
```

### Documents (Private Knowledge Base)

| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/documents/url | Ingest a URL |
| POST | /api/v1/documents/pdf | Upload a PDF |
| POST | /api/v1/documents/text | Add plain text |
| GET | /api/v1/documents | List all documents |
| GET | /api/v1/documents/{id} | Get document metadata |
| DELETE | /api/v1/documents/{id} | Delete document + chunks |
| GET | /api/v1/documents/search/query?q=... | Semantic search |

Ingested documents are automatically searched at synthesis time — the agent combines web results with your private knowledge base via semantic similarity in Qdrant.

### Other

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /docs | Interactive OpenAPI docs |

---

## Observability

### LangSmith

Add to .env:
```bash
LANGSMITH_API_KEY=ls__...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=research-agent
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=research-agent
```

Every research job produces a nested trace showing each agent node, inputs/outputs, latency, and token usage. PlanAgent and SynthesizeAgent subgraph traces appear as nested spans.

### Flower (Celery dashboard)

```
http://localhost:5555
```

### LangGraph Studio

```bash
pip install langgraph-cli
langgraph dev
```

---

## Configuration

| Variable | Description | Default |
|---|---|---|
| LLM_PROVIDER | groq or openai | openai |
| LLM_MODEL | Model name | gpt-4o |
| GROQ_API_KEY | Groq API key | — |
| OPENAI_API_KEY | OpenAI API key | — |
| TAVILY_API_KEY | Tavily search key | — |
| DATABASE_URL | Postgres connection | postgresql+asyncpg://... |
| REDIS_URL | Redis URL | redis://redis:6379/0 |
| QDRANT_URL | Qdrant URL | http://qdrant:6333 |
| QDRANT_DOCUMENT_COLLECTION | Collection name | document_chunks |
| EMBEDDING_MODEL | OpenAI embedding model | text-embedding-3-small |
| LANGSMITH_TRACING | Enable LangSmith | false |
| RATE_LIMIT_PER_MINUTE | Requests per IP/min | 20 |
| CACHE_TTL | Redis cache TTL (seconds) | 3600 |
| SECRET_KEY | App secret | required |

---

## Project Structure

```
app/
├── agents/
│   ├── constants.py          ← shared depth config
│   ├── research_agent.py     ← orchestrator graph
│   ├── plan_agent.py         ← PlanAgent subgraph
│   └── synthesize_agent.py   ← SynthesizeAgent subgraph
├── api/routes/
│   ├── research.py           ← research endpoints
│   ├── documents.py          ← document ingestion endpoints
│   └── health.py             ← health check
├── core/
│   ├── config.py             ← all settings
│   ├── logging.py            ← structlog setup
│   └── security.py           ← API key auth
├── db/
│   └── session.py            ← async SQLAlchemy session
├── models/
│   ├── job.py                ← SQLAlchemy + Pydantic schemas
│   ├── research.py           ← research schemas
│   └── document.py           ← document schemas
├── services/
│   ├── job_service.py        ← research job DB operations
│   ├── cache_service.py      ← Redis cache
│   ├── document_service.py   ← document DB operations
│   └── ingestion_service.py  ← chunk, embed, Qdrant storage
├── tools/
│   ├── search.py             ← Tavily web search
│   └── scraper.py            ← URL scraper
├── main.py                   ← FastAPI app
└── worker.py                 ← Celery task

migrations/versions/
├── 001_initial_tables.py
└── 002_add_documents.py
```

---

## Design Decisions

**Why LangGraph over plain LangChain?** LangGraph models the agent as an explicit state machine with typed state, conditional edges, and subgraphs. This makes the agent flow inspectable, testable, and easier to extend than a chain-based approach.

**Why separate PlanAgent and SynthesizeAgent subgraphs?** Each agent has its own internal quality loop. Separating them keeps concerns isolated and lets each evolve independently. The orchestrator only sees clean inputs and outputs at the boundary.

**Why Qdrant over pgvector?** Qdrant is a dedicated vector database purpose-built for semantic search. Keeping vector search separate from operational data means each can scale independently and the knowledge base can grow to include any document type without touching the operational schema.

**Why Groq?** Groq's LPU hardware runs inference significantly faster than GPU-based providers, which matters in a multi-agent system making 6-10 sequential LLM calls per job.

**Why Celery over FastAPI background tasks?** Celery tasks survive API restarts, support retry logic with backoff, and are observable via Flower. FastAPI background tasks are in-process and lost on restart.

**Why depth-dependent config in one place?** All tunable parameters (max_queries, max_sources, max_iterations, max_revisions) live in constants.py. Changing behavior for a depth level requires editing one dict, not hunting through multiple files.

---

## Production Checklist

- [ ] Replace hardcoded API key with DB-backed key management
- [ ] Add HTTPS (nginx or Caddy in front of API)
- [ ] Set ENVIRONMENT=production and DEBUG=false
- [ ] Restrict CORS origins
- [ ] Add Prometheus metrics endpoint
- [ ] Set up log aggregation (Datadog, CloudWatch)
- [ ] Configure Celery worker autoscaling
- [ ] Add database connection pooling (PgBouncer)
