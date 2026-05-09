"""
LangGraph-powered research agent.

Graph flow:
  START → plan → search_loop → scrape → synthesize → END

The agent:
  1. Plans its search queries from the user question.
  2. Runs web searches (parallel-safe).
  3. Scrapes the top URLs for full content.
  4. Synthesizes findings into a structured report.
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict
import operator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.config import get_settings
from app.core.logging import get_logger
from app.tools.search import web_search, search_web
from app.tools.scraper import scrape_url

settings = get_settings()
logger = get_logger(__name__)

# ── State ────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    depth: str
    messages: Annotated[list[BaseMessage], operator.add]
    search_queries: list[str]
    search_results: list[dict]
    scraped_content: list[dict]
    steps: list[dict]               # audit trail of agent actions
    final_report: dict | None       # populated at the end


# ── Depth configuration ───────────────────────────────────────────────────────

DEPTH_CONFIG = {
    "quick":    {"max_queries": 2, "max_sources": 3},
    "standard": {"max_queries": 3, "max_sources": 5},
    "deep":     {"max_queries": 5, "max_sources": 10},
}

# ── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm(tools: list | None = None) -> ChatGroq:
    llm = ChatGroq(
        model="Llama-3.3-70B-Versatile",
        api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=4096,
    )
    if tools:
        return llm.bind_tools(tools)
    return llm


# ── Node functions ────────────────────────────────────────────────────────────

async def plan_node(state: ResearchState) -> dict:
    """Generate targeted search queries from the user question."""
    cfg = DEPTH_CONFIG.get(state["depth"], DEPTH_CONFIG["standard"])
    n = cfg["max_queries"]

    llm = _get_llm()
    prompt = (
        f"You are a research strategist. Generate exactly {n} distinct, specific "
        f"web search queries to comprehensively answer:\n\n\"{state['query']}\"\n\n"
        f"Return ONLY a JSON array of strings. No markdown, no explanation."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        queries: list[str] = json.loads(response.content)
        if not isinstance(queries, list):
            raise ValueError("Expected list")
    except Exception:
        # Fallback: use the original query
        queries = [state["query"]]

    step = {"type": "planning", "content": f"Generated {len(queries)} search queries", "queries": queries}
    logger.info("plan_complete", queries=queries)
    print(f"[PLAN NODE] Generated queries: {queries}", flush=True)
    return {"search_queries": queries, "steps": [step]}


async def search_node(state: ResearchState) -> dict:
    """Run all planned search queries."""
    print(f"[SEARCH NODE] Starting with queries: {state['search_queries']}", flush=True)
    all_results: list[dict] = []

    for q in state["search_queries"]:
        print(f"[SEARCH NODE] Searching: {q}", flush=True)
        results = await search_web(q, max_results=5)
        print(f"[SEARCH NODE] Got {len(results)} results", flush=True)
        all_results.extend(results)

    # Deduplicate by URL
    seen: set[str] = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique_results.append(r)

    print(f"[SEARCH NODE] Total unique results: {len(unique_results)}", flush=True)
    step = {"type": "search", "content": f"Found {len(unique_results)} unique sources"}
    logger.info("search_complete", source_count=len(unique_results))
    return {"search_results": unique_results, "steps": [step]}


def _parse_search_output(text: str, query: str) -> list[dict]:
    """Convert formatted search string to list of dicts."""
    results = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        url = ""
        title = ""
        snippet = ""
        for line in lines:
            if "URL:" in line:
                url = line.split("URL:", 1)[-1].strip()
            elif line.startswith("["):
                title = line.split("]", 1)[-1].strip()
            else:
                snippet = line.strip()
        if url:
            results.append({"url": url, "title": title, "snippet": snippet, "query": query})
    return results


async def scrape_node(state: ResearchState) -> dict:
    """Scrape top N URLs for full content."""
    cfg = DEPTH_CONFIG.get(state["depth"], DEPTH_CONFIG["standard"])
    sources_to_scrape = state["search_results"][:cfg["max_sources"]]

    scraped: list[dict] = []
    for source in sources_to_scrape:
        content = await scrape_url.ainvoke({"url": source["url"]})
        scraped.append({
            "url": source["url"],
            "title": source["title"],
            "snippet": source["snippet"],
            "content": content,
        })

    step = {"type": "scraping", "content": f"Scraped {len(scraped)} sources"}
    logger.info("scrape_complete", scraped_count=len(scraped))
    return {"scraped_content": scraped, "steps": [step]}


async def synthesize_node(state: ResearchState) -> dict:
    """Synthesize all scraped content into a structured research report."""
    llm = _get_llm()

    # Build context block
    context_parts = []
    for i, src in enumerate(state["scraped_content"], 1):
        context_parts.append(
            f"--- SOURCE {i}: {src['title']} ---\n"
            f"URL: {src['url']}\n"
            f"{src['content'][:2000]}\n"
        )
    context = "\n\n".join(context_parts)

    system = SystemMessage(content=(
        "You are an expert research synthesizer. You produce accurate, well-structured "
        "research reports grounded only in the provided sources. Never hallucinate facts."
    ))
    user = HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Sources:\n{context}\n\n"
        "Respond ONLY with a JSON object with these exact keys:\n"
        "{\n"
        '  "summary": "<2-3 paragraph comprehensive answer>",\n'
        '  "key_findings": ["<finding 1>", "<finding 2>", ...],\n'
        '  "citations": [{"url": "...", "title": "...", "snippet": "..."}, ...],\n'
        '  "confidence_score": <0.0-1.0 based on source quality and coverage>\n'
        "}\n"
        "No markdown fences. Pure JSON only."
    ))

    response = await llm.ainvoke([system, user])

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        report = json.loads(clean)
    except Exception:
        report = {
            "summary": response.content,
            "key_findings": [],
            "citations": [],
            "confidence_score": 0.5,
        }

    step = {
        "type": "synthesis",
        "content": "Research report generated",
        "confidence": report.get("confidence_score", 0),
    }
    logger.info("synthesis_complete", confidence=report.get("confidence_score"))
    return {"final_report": report, "steps": [step]}


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "scrape")
    graph.add_edge("scrape", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Singleton compiled graph
research_graph = build_research_graph()


async def run_research_agent(
    query: str,
    depth: str = "standard",
) -> tuple[dict, list[dict]]:
    """
    Run the research agent and return (report, steps).

    Returns:
        report: dict with summary, key_findings, citations, confidence_score
        steps:  list of agent step dicts for audit trail / streaming
    """
    initial_state: ResearchState = {
        "query": query,
        "depth": depth,
        "messages": [],
        "search_queries": [],
        "search_results": [],
        "scraped_content": [],
        "steps": [],
        "final_report": None,
    }

    final_state = await research_graph.ainvoke(initial_state)

    report = final_state.get("final_report") or {}
    steps = final_state.get("steps", [])
    sources_count = len(final_state.get("scraped_content", []))

    report["sources_scraped"] = sources_count
    return report, steps