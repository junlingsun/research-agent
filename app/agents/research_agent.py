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

from app.models.research import SearchPlan, EvaluationResult

settings = get_settings()
logger = get_logger(__name__)

MAX_ITERATIONS = 3

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
    evaluation: dict | None         # evaluator's last verdict
    iteration_count: int            # circuit breaker


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


# ── Nodes ─────────────────────────────────────────────────────────────────────
 
async def plan_node(state: ResearchState) -> dict:
    """Generate targeted search queries, incorporating evaluator feedback on retry."""
    cfg = DEPTH_CONFIG.get(state["depth"], DEPTH_CONFIG["standard"])
    n = cfg["max_queries"]
    iteration = state.get("iteration_count", 0)
 
    # Build feedback section if this is a retry
    feedback_section = ""
    if iteration > 0 and state.get("evaluation"):
        gaps = state["evaluation"].get("gaps", [])
        feedback = state["evaluation"].get("feedback", "")
        feedback_section = (
            f"\n\nPREVIOUS ATTEMPT FEEDBACK (iteration {iteration}):\n"
            f"Gaps identified: {gaps}\n"
            f"Instructions: {feedback}\n"
            f"Your new queries MUST target these specific gaps."
        )
 
    system = SystemMessage(content=f"""
You are an expert research strategist at a professional research firm.
 
Your output must be a JSON object with exactly these fields:
  queries: array of exactly {n} strings
    - Each string is a precise web search query
    - Each query targets a DIFFERENT aspect of the research question
    - Use specific technical terms, not vague phrases
    - Queries must be meaningfully distinct from each other
  reasoning: string
    - One sentence explaining how these queries cover the question
 
Rules:
  - Never generate duplicate or near-duplicate queries
  - If given feedback about gaps, your queries must address those gaps specifically
  - Output only valid JSON, no markdown, no explanation
""")
 
    user = HumanMessage(content=(
        f"Research question: {state['query']}\n"
        f"Depth: {state['depth']} (generate exactly {n} queries)"
        f"{feedback_section}"
    ))
 
    structured_llm = _get_llm().with_structured_output(SearchPlan)
    result: SearchPlan = await structured_llm.ainvoke([system, user])
 
    logger.info("plan_complete", queries=result.queries, iteration=iteration)
    step = {
        "type": "planning",
        "iteration": iteration + 1,
        "content": f"Generated {len(result.queries)} queries",
        "queries": result.queries,
        "reasoning": result.reasoning,
    }
    return {"search_queries": result.queries, "steps": [step]}
 
 
async def search_node(state: ResearchState) -> dict:
    """Run all planned queries, skipping URLs already seen in previous iterations."""
    seen_urls = state.get("seen_urls", set())
    all_results: list[dict] = []
 
    for q in state["search_queries"]:
        logger.info("searching", query=q)
        results = await search_web(q, max_results=5)
        all_results.extend(results)
 
    # Deduplicate — skip URLs seen in previous iterations
    newly_seen: set[str] = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls and r["url"] not in newly_seen:
            unique_results.append(r)
            newly_seen.add(r["url"])
 
    skipped = len(all_results) - len(unique_results)
    logger.info("search_complete", new_sources=len(unique_results), skipped=skipped)
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
    """Scrape top N new URLs — scraped_content accumulates across iterations."""
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
 
    logger.info("scrape_complete", scraped_count=len(scraped))
    step = {
        "type": "scraping",
        "iteration": state.get("iteration_count", 0) + 1,
        "content": f"Scraped {len(scraped)} new sources",
    }
    # scraped_content uses operator.add so this appends to previous iterations
    return {"scraped_content": scraped, "steps": [step]}
 
 
async def synthesize_node(state: ResearchState) -> dict:
    """Synthesize ALL accumulated content into a structured report."""
    all_content = state["scraped_content"]  # full history across all iterations
    iteration = state.get("iteration_count", 0)
 
    # Build context from all sources across all iterations
    context_parts = []
    for i, src in enumerate(all_content, 1):
        context_parts.append(
            f"--- SOURCE {i}: {src['title']} ---\n"
            f"URL: {src['url']}\n"
            f"{src['content'][:2000]}\n"
        )
    context = "\n\n".join(context_parts)
 
    # Include evaluator's feedback if this is a retry
    gap_instruction = ""
    if iteration > 0 and state.get("evaluation"):
        gaps = state["evaluation"].get("gaps", [])
        gap_instruction = (
            f"\n\nIMPORTANT: Previous synthesis was rejected. "
            f"You now have additional sources. Make sure to address these gaps: {gaps}"
        )
 
    system = SystemMessage(content="""
You are an expert research synthesizer at a professional research firm.
 
Rules:
  - Stay strictly grounded in the provided sources
  - Never include claims not supported by the sources
  - If sources conflict, acknowledge the disagreement
  - summary must be 2-3 paragraphs directly answering the question
  - key_findings must have at least 3 specific, distinct items
  - citations must only include sources you actually reference
  - confidence_score must honestly reflect source quality and coverage:
      0.0-0.4: sources are weak, off-topic, or sparse
      0.4-0.7: sources partially answer the question
      0.7-1.0: sources comprehensively answer the question
  - Output only valid JSON matching the exact schema, no markdown
""")
 
    user = HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Sources ({len(all_content)} total across {iteration + 1} iteration(s)):\n"
        f"{context}"
        f"{gap_instruction}\n\n"
        "Respond with JSON containing exactly: summary, key_findings, citations (array of "
        "{url, title, snippet}), confidence_score (float 0-1)"
    ))
 
    llm = _get_llm()
    response = await llm.ainvoke([system, user])
 
    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        report = json.loads(clean)
    except Exception:
        report = {
            "summary": response.content,
            "key_findings": [],
            "citations": [],
            "confidence_score": 0.3,
        }
 
    report["sources_scraped"] = len(all_content)
    report["iterations"] = iteration + 1
 
    logger.info("synthesis_complete", confidence=report.get("confidence_score"), iteration=iteration)
    step = {
        "type": "synthesis",
        "iteration": iteration + 1,
        "content": f"Synthesized report from {len(all_content)} sources",
        "confidence": report.get("confidence_score", 0),
    }
    return {"final_report": report, "steps": [step]}
 
 
async def evaluate_node(state: ResearchState) -> dict:
    """Evaluate the report quality and decide whether to approve or request another iteration."""
    report = state.get("final_report", {})
    iteration = state.get("iteration_count", 0)
 
    system = SystemMessage(content="""
You are a strict research quality editor. Your job is to find weaknesses, not to praise.
 
Evaluation criteria:
  1. Does the summary DIRECTLY answer the research question?
  2. Are the key findings specific and supported by sources?
  3. Are there obvious gaps or unanswered aspects of the question?
  4. Are citations present and relevant?
 
Scoring guide:
  0.0-0.5: Poor — vague, off-topic, or missing key aspects
  0.5-0.7: Partial — answers some aspects but misses important ones
  0.7-0.85: Good — answers the question well with minor gaps
  0.85-1.0: Excellent — comprehensive, specific, well-cited
 
Rules:
  - Be specific in gaps — "missing recent data" is not acceptable,
    "missing information about post-2024 qubit error rates" is acceptable
  - approved must be true only if score >= 0.7
  - If approved is true, gaps and feedback must be empty
  - Output only valid JSON, no markdown
""")
 
    user = HumanMessage(content=(
        f"Research question: {state['query']}\n\n"
        f"Report to evaluate:\n"
        f"Summary: {report.get('summary', '')}\n\n"
        f"Key findings: {report.get('key_findings', [])}\n\n"
        f"Citations: {[c.get('url') for c in report.get('citations', [])]}\n\n"
        f"Confidence score claimed: {report.get('confidence_score', 0)}\n\n"
        "Respond with JSON containing exactly: approved (bool), score (float 0-1), "
        "gaps (array of strings), feedback (string)"
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
 
 
# ── Routing function ──────────────────────────────────────────────────────────
 
def route_after_evaluation(state: ResearchState) -> str:
    """Decide whether to approve or loop back for another iteration."""
    evaluation = state.get("evaluation", {})
    iteration = state.get("iteration_count", 0)
 
    # Circuit breaker — never exceed MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS:
        logger.info("circuit_breaker_triggered", iteration=iteration)
        return "approved"
 
    # No new sources were found — can't improve, accept best effort
    if not state.get("search_results"):
        logger.info("no_new_sources_found", iteration=iteration)
        return "approved"
 
    if evaluation.get("approved") or evaluation.get("score", 0) >= 0.7:
        return "approved"
 
    return "retry"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
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
            "retry": "plan",   # loop back with evaluator feedback
        }
    )
 
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
        "evaluation": None,
        "iteration_count": 0,
    }

    final_state = await research_graph.ainvoke(initial_state)

    report = final_state.get("final_report") or {}
    steps = final_state.get("steps", [])

    return report, steps