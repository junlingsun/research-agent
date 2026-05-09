"""Unit tests for the research agent."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_run_research_agent_returns_report_and_steps(self):
        """Agent should return a report dict and list of steps."""
        mock_state = {
            "final_report": {
                "summary": "Test summary about quantum computing.",
                "key_findings": ["Finding 1", "Finding 2"],
                "citations": [{"url": "https://example.com", "title": "Example", "snippet": "snippet"}],
                "confidence_score": 0.85,
                "sources_scraped": 3,
            },
            "steps": [
                {"type": "planning", "content": "Generated 3 queries"},
                {"type": "search", "content": "Found 5 sources"},
                {"type": "scraping", "content": "Scraped 3 sources"},
                {"type": "synthesis", "content": "Report generated"},
            ],
            "scraped_content": [{"url": "https://a.com"}, {"url": "https://b.com"}, {"url": "https://c.com"}],
        }

        with patch("app.agents.research_agent.research_graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=mock_state)

            from app.agents.research_agent import run_research_agent
            report, steps = await run_research_agent("What is quantum computing?", "standard")

        assert report["summary"] == "Test summary about quantum computing."
        assert len(report["key_findings"]) == 2
        assert report["confidence_score"] == 0.85
        assert report["sources_scraped"] == 3
        assert len(steps) == 4

    @pytest.mark.asyncio
    async def test_run_research_agent_handles_missing_report(self):
        """Agent should handle state with no final_report gracefully."""
        mock_state = {
            "final_report": None,
            "steps": [],
            "scraped_content": [],
        }

        with patch("app.agents.research_agent.research_graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=mock_state)

            from app.agents.research_agent import run_research_agent
            report, steps = await run_research_agent("test query", "quick")

        assert isinstance(report, dict)
        assert isinstance(steps, list)


class TestDepthConfig:
    def test_depth_config_has_all_levels(self):
        from app.agents.research_agent import DEPTH_CONFIG
        assert "quick" in DEPTH_CONFIG
        assert "standard" in DEPTH_CONFIG
        assert "deep" in DEPTH_CONFIG

    def test_deep_has_more_sources_than_quick(self):
        from app.agents.research_agent import DEPTH_CONFIG
        assert DEPTH_CONFIG["deep"]["max_sources"] > DEPTH_CONFIG["quick"]["max_sources"]
        assert DEPTH_CONFIG["deep"]["max_queries"] > DEPTH_CONFIG["quick"]["max_queries"]
