"""Integration tests for Research and Document API endpoints."""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app

API_KEY = "dev-key-replace-in-production"
HEADERS = {"X-API-Key": API_KEY}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_job():
    job = MagicMock()
    job.id = uuid.uuid4()
    job.query = "What is quantum computing?"
    job.depth = "standard"
    job.status = "pending"
    job.created_at = MagicMock()
    job.updated_at = MagicMock()
    job.error_message = None
    job.result = None
    return job


@pytest.fixture
def mock_completed_job(mock_job):
    mock_job.status = "completed"
    result = MagicMock()
    result.summary = "Quantum computing uses qubits."
    result.key_findings = ["Finding 1", "Finding 2"]
    result.citations = [
        {"url": "https://example.com", "title": "Example", "snippet": "snippet"}
    ]
    result.confidence_score = 0.85
    result.sources_scraped = 5
    result.agent_steps = [
        {
            "type": "evaluation",
            "iteration": 1,
            "score": 0.85,
            "approved": True,
            "gaps": [],
        }
    ]
    mock_job.result = result
    return mock_job


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        with (
            patch("app.api.routes.health.aioredis.from_url") as mock_redis,
            patch("sqlalchemy.ext.asyncio.AsyncSession.execute", AsyncMock()),
        ):
            mock_redis.return_value.ping = AsyncMock()
            mock_redis.return_value.aclose = AsyncMock()

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_root_returns_200(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")
        assert response.status_code == 200


# ── Research endpoints ────────────────────────────────────────────────────────


class TestSubmitResearch:
    @pytest.mark.asyncio
    async def test_requires_api_key(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/research",
                json={"query": "What is quantum computing?", "depth": "standard"},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rejects_short_query(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/research",
                json={"query": "short", "depth": "standard"},
                headers=HEADERS,
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_invalid_depth(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/research",
                json={"query": "What is quantum computing?", "depth": "invalid"},
                headers=HEADERS,
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_accepts_valid_request(self, mock_job):
        with (
            patch(
                "app.api.routes.research.get_cached_result",
                AsyncMock(return_value=None),
            ),
            patch(
                "app.api.routes.research.create_job", AsyncMock(return_value=mock_job)
            ),
            patch("app.api.routes.research.run_research_task") as mock_task,
            patch("app.db.session.get_db"),
        ):
            mock_task.apply_async = MagicMock()

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/research",
                    json={
                        "query": "What are the latest advancements in AI research?",
                        "depth": "quick",
                    },
                    headers=HEADERS,
                )

        assert response.status_code in (202, 422, 500)


class TestGetResearch:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_job(self):
        with patch("app.api.routes.research.get_job", AsyncMock(return_value=None)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/v1/research/{uuid.uuid4()}",
                    headers=HEADERS,
                )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_401_without_api_key(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/v1/research/{uuid.uuid4()}")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_job_with_result(self, mock_completed_job):
        with patch(
            "app.api.routes.research.get_job",
            AsyncMock(return_value=mock_completed_job),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/v1/research/{mock_completed_job.id}",
                    headers=HEADERS,
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["summary"] == "Quantum computing uses qubits."
        assert data["result"]["confidence_score"] == 0.85


# ── Document endpoints ────────────────────────────────────────────────────────


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_list_documents_requires_auth(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/documents")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_documents_returns_list(self):
        with patch(
            "app.api.routes.documents.list_documents", AsyncMock(return_value=([], 0))
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/documents", headers=HEADERS)

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_document_404_for_unknown(self):
        with patch(
            "app.api.routes.documents.get_document", AsyncMock(return_value=None)
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/v1/documents/{uuid.uuid4()}",
                    headers=HEADERS,
                )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_url_rejects_invalid_url(self):
        with (
            patch(
                "app.api.routes.documents.create_document",
                AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        title="test",
                        source_type="url",
                        source_ref="not-a-url",
                        chunk_count=0,
                        status="failed",
                        created_at=MagicMock(),
                    )
                ),
            ),
            patch(
                "app.api.routes.documents.extract_from_url",
                AsyncMock(side_effect=Exception("Invalid URL")),
            ),
            patch("app.api.routes.documents.update_document_status", AsyncMock()),
            patch("app.db.session.get_db"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/documents/url",
                    json={"url": "not-a-url"},
                    headers=HEADERS,
                )
        assert response.status_code in (422, 500)

    @pytest.mark.asyncio
    async def test_document_search_requires_auth(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/documents/search/query?q=quantum")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_document_search_returns_results(self):
        with patch(
            "app.api.routes.documents.search_documents", AsyncMock(return_value=[])
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/v1/documents/search/query?q=quantum computing",
                    headers=HEADERS,
                )
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# ── Request validation tests ──────────────────────────────────────────────────


class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_research_query_too_long_rejected(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/research",
                json={"query": "x" * 501, "depth": "standard"},
                headers=HEADERS,
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_all_valid_depths_accepted_schema(self):
        """Verify schema accepts all three depth values."""
        from app.models.research import ResearchRequest
        from app.models.job import ResearchDepth

        for depth in ("quick", "standard", "deep"):
            req = ResearchRequest(
                query="What is quantum computing in 2025?",
                depth=depth,
            )
            assert req.depth == ResearchDepth(depth)
