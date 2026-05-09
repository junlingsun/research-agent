"""
Orchestrator graph — coordinates subagents and plain nodes.

Flow:
  START → PlanAgent → search → scrape → SynthesizeAgent → evaluate
                                                               ↓ approved
                                                              END
                                                               ↓ rejected
                                                    loop back to PlanAgent (with feedback)

PlanAgent:      generate → critique → refine (internal loop)
SynthesizeAgent: draft → self_critique → revise (internal loop)
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.research import EvaluationResult
from app.tools.search import search_web
from app.tools.scraper import scrape_url

settings = get_settings()
logger = get_logger(__name__)

MAX_ITERATIONS = 3

from app.agents.constants import DEPTH_CONFIG


# ── Orchestrator state ────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    depth: str
    messages: Annotated[list[BaseMessage], operator.add]
    search_queries: list[str]
    search_results: list[dict]
    seen_urls: set[str]
    scraped_content: Annotated[list[dict], operator.add]
    steps: Annotated[list[dict], operator.add]
    final_report: dict | None
    evaluation: dict | None
    iteration_count: int


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm():
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_tokens=4096,
    )


# ── Orchestrator nodes ────────────────────────────────────────────────────────

async def plan_node(state: ResearchState) -> dict:
    """Delegate query planning to PlanAgent subgraph."""
    from app.agents.plan_agent import run_plan_agent

    # Extract evaluator feedback for the planner
    evaluation_feedback = ""
    if state.get("evaluation") and state.get("iteration_count", 0) > 0:
        gaps = state["evaluation"].get("gaps", [])
        feedback = state["evaluation"].get("feedback", "")
        evaluation_feedback = f"Gaps: {gaps}\nInstructions: {feedback}"

    queries, internal_steps = await run_plan_agent(
        query=state["query"],
        depth=state["depth"],
        evaluation_feedback=evaluation_feedback,
    )

    logger.info("plan_complete", queries=queries, iteration=state.get("iteration_count", 0))
    step = {
        "type": "planning",
        "iteration": state.get("iteration_count", 0) + 1,
        "content": f"PlanAgent generated {len(queries)} queries with {len(internal_steps)} internal steps",
        "queries": queries,
        "internal_steps": internal_steps,
    }
    return {"search_queries": queries, "steps": [step]}


async def search_node(state: ResearchState) -> dict:
    """Run all planned queries, skipping already-seen URLs."""
    import asyncio
    seen_urls = state.get("seen_urls", set())
    all_results: list[dict] = []
    start_time = asyncio.get_event_loop().time()

    # Run all queries simultaneously
    query_results = await asyncio.gather(
        *[search_web(q, max_results=5) for q in state["search_queries"]],
        return_exceptions=True,
    )

    for q, result in zip(state["search_queries"], query_results):
        if isinstance(result, Exception):
            logger.warning("search_query_failed", query=q, error=str(result))
        else:
            all_results.extend(result)

    # Deduplicate — skip URLs seen in previous iterations
    newly_seen: set[str] = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls and r["url"] not in newly_seen:
            unique_results.append(r)
            newly_seen.add(r["url"])

    elapsed = asyncio.get_event_loop().time() - start_time
    skipped = len(all_results) - len(unique_results)
    logger.info(
        "search_complete", 
        new_sources=len(unique_results), 
        skipped=skipped,
        elapsed_seconds=round(elapsed, 2),
        queries=len(state["search_queries"])
    )

    step = {
        "type": "search",
        "iteration": state.get("iteration_count", 0) + 1,
        "content": f"Found {len(unique_results)} new sources (skipped {skipped} already seen)",
    }
    return {
        "search_results": unique_results,
        "seen_urls": seen_urls | newly_seen,
        "steps": [step],
    }


async def scrape_node(state: ResearchState) -> dict:
    """Scrape top N new URLs."""
    import asyncio
    cfg = DEPTH_CONFIG.get(state["depth"], DEPTH_CONFIG["standard"])
    sources_to_scrape = state["search_results"][:cfg["max_sources"]]
    start_time = asyncio.get_event_loop().time()

    scraped: list[dict] = []
    async def scrape_one(source: dict) -> dict:
        content = await scrape_url.ainvoke({"url": source["url"]})
        return {
            "url": source["url"],
            "title": source["title"],
            "snippet": source["snippet"],
            "content": content,
        }
    
    # Scrape all URLs simultaneously
    results = await asyncio.gather(
        *[scrape_one(source) for source in sources_to_scrape],
        return_exceptions=True,
    )

    for source, result in zip(sources_to_scrape, results):
        if isinstance(result, Exception):
            logger.warning("scrape_failed", url=source["url"], error=str(result))
        else:
            scraped.append(result)
 
    elapsed = asyncio.get_event_loop().time() - start_time

    logger.info(
        "scrape_complete", 
        scraped_count=len(scraped),
        failed=len(sources_to_scrape) - len(scraped),
        elapsed_seconds=round(elapsed, 2),
    )
    
    step = {
        "type": "scraping",
        "iteration": state.get("iteration_count", 0) + 1,
        "content": f"Scraped {len(scraped)} new sources",
    }
    return {"scraped_content": scraped, "steps": [step]}


async def synthesize_node(state: ResearchState) -> dict:
    """Delegate synthesis to SynthesizeAgent subgraph."""
    from app.agents.synthesize_agent import run_synthesize_agent

    # Extract evaluator feedback for the synthesizer
    evaluation_feedback = ""
    if state.get("evaluation") and state.get("iteration_count", 0) > 0:
        gaps = state["evaluation"].get("gaps", [])
        evaluation_feedback = f"Address these gaps: {gaps}"

    report, internal_steps = await run_synthesize_agent(
        query=state["query"],
        scraped_content=state["scraped_content"],
        evaluation_feedback=evaluation_feedback,
    )

    # Enrich report with metadata
    report["sources_scraped"] = len(state["scraped_content"])
    report["iterations"] = state.get("iteration_count", 0) + 1

    logger.info(
        "synthesis_complete",
        confidence=report.get("confidence_score"),
        iteration=state.get("iteration_count", 0),
        internal_revisions=len([s for s in internal_steps if s["type"] == "synthesize_revise"]),
    )
    step = {
        "type": "synthesis",
        "iteration": state.get("iteration_count", 0) + 1,
        "content": f"SynthesizeAgent produced report with {len(internal_steps)} internal steps",
        "confidence": report.get("confidence_score"),
        "internal_steps": internal_steps,
    }
    return {"final_report": report, "steps": [step]}


async def evaluate_node(state: ResearchState) -> dict:
    """Evaluate report quality and decide whether to approve or retry."""
    from langchain_core.messages import HumanMessage, SystemMessage

    report = state.get("final_report", {})
    iteration = state.get("iteration_count", 0)

    system = SystemMessage(content="""
You are a strict research quality editor. Find weaknesses, not praise.

Evaluation criteria:
  1. Does the summary DIRECTLY answer the research question?
  2. Are key findings specific and supported by sources?
  3. Are there obvious gaps or unanswered aspects?
  4. Are citations present and relevant?

Scoring:
  0.0-0.5: Poor — vague, off-topic, missing key aspects
  0.5-0.7: Partial — answers some aspects but misses important ones
  0.7-0.85: Good — answers well with minor gaps
  0.85-1.0: Excellent — comprehensive, specific, well-cited

approved = true only if score >= 0.7
If approved, gaps and feedback must be empty strings/lists.
Be specific in gaps — not "needs more detail" but "missing IBM quantum roadmap post-2024"
""")

    user = HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Report (iteration {iteration + 1}):\n"
        f"Summary: {report.get('summary', '')}\n\n"
        f"Key findings: {report.get('key_findings', [])}\n\n"
        f"Citations: {[c.get('url') for c in report.get('citations', [])]}\n"
        f"Confidence claimed: {report.get('confidence_score', 0)}"
    ))

    structured_llm = _get_llm().with_structured_output(EvaluationResult)
    evaluation: EvaluationResult = await structured_llm.ainvoke([system, user])

    logger.info(
        "evaluation_complete",
        approved=evaluation.approved,
        score=evaluation.score,
        iteration=iteration,
        gaps=evaluation.gaps,
    )
    step = {
        "type": "evaluation",
        "iteration": iteration + 1,
        "content": f"Score: {evaluation.score:.0%} — {'Approved ✓' if evaluation.approved else 'Rejected, looping back'}",
        "approved": evaluation.approved,
        "score": evaluation.score,
        "gaps": evaluation.gaps,
        "feedback": evaluation.feedback,
    }
    return {
        "evaluation": evaluation.model_dump(),
        "iteration_count": iteration + 1,
        "steps": [step],
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_evaluation(state: ResearchState) -> str:
    evaluation = state.get("evaluation", {})
    iteration = state.get("iteration_count", 0)

    if iteration >= MAX_ITERATIONS:
        logger.info("outer_circuit_breaker", iteration=iteration)
        return "approved"

    if not state.get("search_results"):
        logger.info("no_new_sources", iteration=iteration)
        return "approved"

    if evaluation.get("approved") or evaluation.get("score", 0) >= 0.7:
        return "approved"

    return "retry"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("evaluate", evaluate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "scrape")
    graph.add_edge("scrape", "synthesize")
    graph.add_edge("synthesize", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "approved": END,
            "retry": "plan",
        }
    )

    return graph.compile()


# Singleton
research_graph = build_research_graph()


# ── Public interface ──────────────────────────────────────────────────────────

async def run_research_agent(
    query: str,
    depth: str = "standard",
) -> tuple[dict, list[dict]]:
    """Run the full multi-agent research pipeline. Returns (report, steps)."""
    initial_state: ResearchState = {
        "query": query,
        "depth": depth,
        "messages": [],
        "search_queries": [],
        "search_results": [],
        "seen_urls": set(),
        "scraped_content": [],
        "steps": [],
        "final_report": None,
        "evaluation": None,
        "iteration_count": 0,
    }

    final_state = await research_graph.ainvoke(initial_state)

    report = final_state.get("final_report") or {}
    steps = final_state.get("steps", [])
    return report, steps