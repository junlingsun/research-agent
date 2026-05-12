"""
PlanAgent — autonomous query planning subgraph.

Internal loop:
  generate → critique → refine (until queries are sharp or max revisions hit)

The agent critiques its own queries before passing them to search,
ensuring they are specific, distinct, and cover different angles.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

MAX_REVISIONS = 2  # internal revision limit


# ── Schemas ───────────────────────────────────────────────────────────────────


class QueryCritique(BaseModel):
    """Structured output for the critique node."""

    is_good_enough: bool = Field(
        description="True if queries are specific, distinct, and comprehensive"
    )
    issues: list[str] = Field(
        description="Specific issues found. Empty if is_good_enough is True."
    )
    suggestions: list[str] = Field(
        description="Specific improvement suggestions. Empty if is_good_enough is True."
    )


class QueryPlan(BaseModel):
    """Structured output for generate and refine nodes."""

    queries: list[str] = Field(
        description="Distinct search queries each targeting a different aspect"
    )
    reasoning: str = Field(
        description="One sentence explaining how these queries comprehensively cover the question"
    )


# ── State ─────────────────────────────────────────────────────────────────────


class PlanState(TypedDict):
    # Inputs from orchestrator
    query: str
    depth: str
    evaluation_feedback: str  # gaps from outer evaluator on retry

    # Internal state
    candidate_queries: list[str]
    critique: QueryCritique | None
    revision_count: int
    internal_steps: Annotated[list[dict], operator.add]

    # Output to orchestrator
    final_queries: list[str] | None
    reasoning: str


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
        max_tokens=2048,
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def generate_node(state: PlanState) -> dict:
    """Generate initial search queries from the research question."""
    from app.agents.constants import get_depth_config

    cfg = get_depth_config(state["depth"])
    n = cfg["max_queries"]

    # Include outer evaluator feedback if this is a retry iteration
    feedback_section = ""
    if state.get("evaluation_feedback"):
        feedback_section = (
            f"\n\nOUTER EVALUATOR FEEDBACK:\n"
            f"{state['evaluation_feedback']}\n"
            f"Your queries must specifically address these gaps."
        )

    system = SystemMessage(
        content=f"""
You are an expert research strategist. Generate precise search queries.

Query quality rules:
  - Each query must target a DIFFERENT aspect of the question
  - Use specific technical terms, not vague phrases
  - Queries must be meaningfully distinct — no near-duplicates
  - Prefer queries that retrieve primary sources over opinion pieces
  - Good: "IBM quantum computing roadmap 2025 qubit count"
  - Bad: "quantum computing latest news"

Output exactly {n} queries.
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n"
            f"Generate exactly {n} search queries."
            f"{feedback_section}"
        )
    )

    structured_llm = _get_llm().with_structured_output(QueryPlan)
    result: QueryPlan = await structured_llm.ainvoke([system, user])

    logger.info("plan_generate", query_count=len(result.queries))
    step = {
        "type": "plan_generate",
        "revision": 0,
        "content": f"Generated {len(result.queries)} initial queries",
        "queries": result.queries,
    }
    return {
        "candidate_queries": result.queries,
        "reasoning": result.reasoning,
        "internal_steps": [step],
    }


async def critique_node(state: PlanState) -> dict:
    """Critique the candidate queries for quality and specificity."""
    queries = state["candidate_queries"]

    system = SystemMessage(
        content="""
You are a search strategy critic. Your job is to find weaknesses in search queries.

Evaluate queries on:
  1. Specificity — are they precise enough to return useful results?
  2. Distinctness — do they cover genuinely different aspects?
  3. Coverage — together, do they comprehensively cover the question?
  4. Redundancy — are any too similar to each other?

Be strict. Vague queries waste search budget.
Mark as good_enough only if ALL queries are specific and distinct.
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n\n"
            f"Queries to evaluate:\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(queries))
        )
    )

    structured_llm = _get_llm().with_structured_output(QueryCritique)
    critique: QueryCritique = await structured_llm.ainvoke([system, user])

    logger.info(
        "plan_critique",
        is_good_enough=critique.is_good_enough,
        issues=critique.issues,
        revision=state["revision_count"],
    )
    step = {
        "type": "plan_critique",
        "revision": state["revision_count"],
        "content": f"{'Approved' if critique.is_good_enough else 'Issues found: ' + str(critique.issues)}",
        "is_good_enough": critique.is_good_enough,
        "issues": critique.issues,
    }
    return {"critique": critique, "internal_steps": [step]}


async def refine_node(state: PlanState) -> dict:
    """Refine queries based on critique feedback."""
    from app.agents.constants import get_depth_config

    cfg = get_depth_config(state["depth"])
    n = cfg["max_queries"]
    critique = state["critique"]

    system = SystemMessage(
        content=f"""
You are an expert research strategist refining search queries based on critique feedback.

Apply the critique suggestions precisely.
Generate exactly {n} improved queries that address all identified issues.
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n\n"
            f"Current queries:\n"
            + "\n".join(
                f"  {i+1}. {q}" for i, q in enumerate(state["candidate_queries"])
            )
            + "\n\nCritique issues:\n"
            + "\n".join(f"  - {issue}" for issue in critique.issues)
            + "\n\nSuggestions:\n"
            + "\n".join(f"  - {s}" for s in critique.suggestions)
            + f"\n\nGenerate {n} improved queries addressing all issues."
        )
    )

    structured_llm = _get_llm().with_structured_output(QueryPlan)
    result: QueryPlan = await structured_llm.ainvoke([system, user])

    revision = state["revision_count"] + 1
    logger.info("plan_refine", revision=revision, queries=result.queries)
    step = {
        "type": "plan_refine",
        "revision": revision,
        "content": f"Refined queries (revision {revision})",
        "queries": result.queries,
    }
    return {
        "candidate_queries": result.queries,
        "reasoning": result.reasoning,
        "revision_count": revision,
        "internal_steps": [step],
    }


async def finalize_node(state: PlanState) -> dict:
    """Accept the current queries as final output."""
    logger.info(
        "plan_finalized",
        queries=state["candidate_queries"],
        revisions=state["revision_count"],
    )
    step = {
        "type": "plan_finalized",
        "content": f"Queries finalized after {state['revision_count']} revision(s)",
        "final_queries": state["candidate_queries"],
    }
    return {
        "final_queries": state["candidate_queries"],
        "internal_steps": [step],
    }


# ── Routing ───────────────────────────────────────────────────────────────────


def route_after_critique(state: PlanState) -> str:
    """Approve queries or refine them — with revision cap."""
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)

    # Circuit breaker
    if revision_count >= MAX_REVISIONS:
        logger.info("plan_circuit_breaker", revision_count=revision_count)
        return "finalize"

    if critique and critique.is_good_enough:
        return "finalize"

    return "refine"


# ── Graph ─────────────────────────────────────────────────────────────────────


def build_plan_agent() -> StateGraph:
    graph = StateGraph(PlanState)

    graph.add_node("generate_queries", generate_node)
    graph.add_node("critique_queries", critique_node)
    graph.add_node("refine_queries", refine_node)
    graph.add_node("finalize_queries", finalize_node)

    graph.set_entry_point("generate_queries")
    graph.add_edge("generate_queries", "critique_queries")
    graph.add_conditional_edges(
        "critique_queries",
        route_after_critique,
        {
            "finalize": "finalize_queries",
            "refine": "refine_queries",
        },
    )
    graph.add_edge("refine_queries", "critique_queries")  # internal loop
    graph.add_edge("finalize_queries", END)

    return graph.compile()


# Singleton
plan_agent = build_plan_agent()


# ── Public interface ──────────────────────────────────────────────────────────


async def run_plan_agent(
    query: str,
    depth: str,
    evaluation_feedback: str = "",
) -> tuple[list[str], list[dict]]:
    """
    Run the PlanAgent subgraph.
    Returns (final_queries, internal_steps).
    """
    initial_state: PlanState = {
        "query": query,
        "depth": depth,
        "evaluation_feedback": evaluation_feedback,
        "candidate_queries": [],
        "critique": None,
        "revision_count": 0,
        "internal_steps": [],
        "final_queries": None,
        "reasoning": "",
    }

    result = await plan_agent.ainvoke(initial_state)
    return result["final_queries"] or [], result["internal_steps"]
