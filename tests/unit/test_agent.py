"""Unit tests for agent routing functions and constants."""
import pytest
from unittest.mock import MagicMock


# ── Constants tests ───────────────────────────────────────────────────────────

class TestDepthConfig:
    def test_all_depths_present(self):
        from app.agents.constants import DEPTH_CONFIG
        assert "quick" in DEPTH_CONFIG
        assert "standard" in DEPTH_CONFIG
        assert "deep" in DEPTH_CONFIG

    def test_all_required_keys_present(self):
        from app.agents.constants import DEPTH_CONFIG
        required = {"max_queries", "max_sources", "max_iterations", "max_revisions"}
        for depth, config in DEPTH_CONFIG.items():
            assert required == set(config.keys()), f"Missing keys in {depth}"

    def test_deep_more_than_quick(self):
        from app.agents.constants import DEPTH_CONFIG
        for key in ("max_queries", "max_sources", "max_iterations", "max_revisions"):
            assert DEPTH_CONFIG["deep"][key] > DEPTH_CONFIG["quick"][key]

    def test_get_depth_config_returns_correct(self):
        from app.agents.constants import get_depth_config, DEPTH_CONFIG
        assert get_depth_config("quick") == DEPTH_CONFIG["quick"]
        assert get_depth_config("standard") == DEPTH_CONFIG["standard"]
        assert get_depth_config("deep") == DEPTH_CONFIG["deep"]

    def test_get_depth_config_unknown_falls_back_to_standard(self):
        from app.agents.constants import get_depth_config, DEPTH_CONFIG
        result = get_depth_config("unknown_depth")
        assert result == DEPTH_CONFIG["standard"]


# ── Routing function tests ────────────────────────────────────────────────────

class TestRouteAfterEvaluation:
    def _state(self, score=0.8, approved=True, iteration_count=1,
                depth="standard", search_results=None):
        return {
            "evaluation": {"approved": approved, "score": score, "gaps": []},
            "iteration_count": iteration_count,
            "depth": depth,
            "search_results": search_results if search_results is not None
                              else [{"url": "https://example.com"}],
        }

    def test_approved_when_score_above_threshold(self):
        from app.agents.research_agent import route_after_evaluation
        assert route_after_evaluation(self._state(score=0.8)) == "approved"

    def test_approved_when_score_exactly_threshold(self):
        from app.agents.research_agent import route_after_evaluation
        assert route_after_evaluation(self._state(score=0.7, approved=False)) == "approved"

    def test_retry_when_score_below_threshold(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.5, approved=False, iteration_count=1)
        assert route_after_evaluation(state) == "retry"

    def test_approved_at_circuit_breaker(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.5, approved=False, iteration_count=3, depth="standard")
        assert route_after_evaluation(state) == "approved"

    def test_score_checked_before_circuit_breaker(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.8, approved=True, iteration_count=3, depth="standard")
        assert route_after_evaluation(state) == "approved"

    def test_approved_when_no_new_sources(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.4, approved=False, search_results=[])
        assert route_after_evaluation(state) == "approved"

    def test_circuit_breaker_respects_quick_depth(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.4, approved=False, iteration_count=1, depth="quick")
        assert route_after_evaluation(state) == "approved"

    def test_circuit_breaker_respects_deep_depth(self):
        from app.agents.research_agent import route_after_evaluation
        state = self._state(score=0.4, approved=False, iteration_count=3, depth="deep")
        assert route_after_evaluation(state) == "retry"


class TestPlanAgentRouting:
    def _state(self, is_good_enough=True, revision_count=0, depth="standard"):
        critique = MagicMock()
        critique.is_good_enough = is_good_enough
        return {"critique": critique, "revision_count": revision_count, "depth": depth}

    def test_finalize_when_approved(self):
        from app.agents.plan_agent import route_after_critique
        assert route_after_critique(self._state(is_good_enough=True)) == "finalize"

    def test_refine_when_not_approved(self):
        from app.agents.plan_agent import route_after_critique
        assert route_after_critique(self._state(is_good_enough=False)) == "refine"

    def test_finalize_at_max_revisions(self):
        from app.agents.plan_agent import route_after_critique
        state = self._state(is_good_enough=False, revision_count=2, depth="standard")
        assert route_after_critique(state) == "finalize"

    def test_no_critique_falls_through_to_refine(self):
        from app.agents.plan_agent import route_after_critique
        state = {"critique": None, "revision_count": 0, "depth": "standard"}
        assert route_after_critique(state) == "refine"


class TestSynthesizeAgentRouting:
    def _state(self, is_good_enough=True, revision_count=0, depth="standard"):
        critique = MagicMock()
        critique.is_good_enough = is_good_enough
        return {"critique": critique, "revision_count": revision_count, "depth": depth}

    def test_finalize_when_approved(self):
        from app.agents.synthesize_agent import route_after_critique
        assert route_after_critique(self._state(is_good_enough=True)) == "finalize"

    def test_revise_when_not_approved(self):
        from app.agents.synthesize_agent import route_after_critique
        assert route_after_critique(self._state(is_good_enough=False)) == "revise"

    def test_finalize_at_max_revisions(self):
        from app.agents.synthesize_agent import route_after_critique
        state = self._state(is_good_enough=False, revision_count=2, depth="standard")
        assert route_after_critique(state) == "finalize"


# ── Chunking tests ────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        from app.services.ingestion_service import chunk_text
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        from app.services.ingestion_service import chunk_text
        text = "A word here. " * 200
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_all_chunks_meet_minimum_length(self):
        from app.services.ingestion_service import chunk_text
        text = "Word " * 300
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert len(chunk) >= 50

    def test_empty_like_text_filtered(self):
        from app.services.ingestion_service import chunk_text
        text = "\n\n\n\n" + "Real content here. " * 100
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip() != ""


# ── Point ID tests ────────────────────────────────────────────────────────────

class TestChunkPointId:
    def test_same_inputs_same_id(self):
        import uuid
        from app.services.ingestion_service import _chunk_point_id
        doc_id = uuid.uuid4()
        assert _chunk_point_id(doc_id, 0) == _chunk_point_id(doc_id, 0)

    def test_different_chunk_index_different_id(self):
        import uuid
        from app.services.ingestion_service import _chunk_point_id
        doc_id = uuid.uuid4()
        assert _chunk_point_id(doc_id, 0) != _chunk_point_id(doc_id, 1)

    def test_different_document_different_id(self):
        import uuid
        from app.services.ingestion_service import _chunk_point_id
        assert _chunk_point_id(uuid.uuid4(), 0) != _chunk_point_id(uuid.uuid4(), 0)

    def test_id_is_32_chars(self):
        import uuid
        from app.services.ingestion_service import _chunk_point_id
        assert len(_chunk_point_id(uuid.uuid4(), 0)) == 32


# ── Document context formatting tests ────────────────────────────────────────

class TestFormatDocumentContext:
    def test_empty_returns_empty_string(self):
        from app.services.ingestion_service import format_document_context
        assert format_document_context([]) == ""

    def test_includes_title_and_content(self):
        from app.services.ingestion_service import format_document_context
        chunks = [{"document_id": "abc", "title": "Test Doc",
                   "chunk": "Some content.", "source_ref": None, "score": 0.9}]
        result = format_document_context(chunks)
        assert "Test Doc" in result
        assert "Some content." in result
        assert "PRIVATE KNOWLEDGE BASE CONTEXT" in result

    def test_same_document_header_appears_once(self):
        from app.services.ingestion_service import format_document_context
        chunks = [
            {"document_id": "abc", "title": "Doc A", "chunk": "chunk 1",
             "source_ref": None, "score": 0.9},
            {"document_id": "abc", "title": "Doc A", "chunk": "chunk 2",
             "source_ref": None, "score": 0.8},
        ]
        result = format_document_context(chunks)
        assert result.count("Doc A") == 1

    def test_multiple_documents_both_headers_appear(self):
        from app.services.ingestion_service import format_document_context
        chunks = [
            {"document_id": "abc", "title": "Doc A", "chunk": "content A",
             "source_ref": None, "score": 0.9},
            {"document_id": "xyz", "title": "Doc B", "chunk": "content B",
             "source_ref": None, "score": 0.8},
        ]
        result = format_document_context(chunks)
        assert "Doc A" in result
        assert "Doc B" in result