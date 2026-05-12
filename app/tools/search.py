"""
Web search tool using Tavily — purpose-built for AI agents.
Falls back to DuckDuckGo if no Tavily API key is configured.
"""

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.tools import tool
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Core search function — returns list of dicts directly.
    Called by both the tool and the search_node.
    """
    if settings.tavily_api_key:
        return await _tavily_search(query, max_results)
    else:
        return await _duckduckgo_search(query, max_results)


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _tavily_search(query: str, max_results: int) -> list[dict]:
    """Search using Tavily API."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    logger.info("tavily_search_complete", query=query, count=len(results))
    return results


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _duckduckgo_search(query: str, max_results: int) -> list[dict]:
    """Fallback search using DuckDuckGo."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for topic in data.get("RelatedTopics", [])[:max_results]:
        if "Text" in topic and "FirstURL" in topic:
            results.append(
                {
                    "title": topic.get("Text", "")[:100],
                    "url": topic["FirstURL"],
                    "snippet": topic.get("Text", ""),
                }
            )
    return results


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information on a query.
    Returns formatted string of results.
    """
    try:
        results = await search_web(query, max_results)
        if not results:
            return f"No results found for query: {query}"
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r['title']}\n" f"    URL: {r['url']}\n" f"    {r['snippet']}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        logger.error("search_failed", query=query, error=str(e))
        return f"Search failed: {str(e)}"
