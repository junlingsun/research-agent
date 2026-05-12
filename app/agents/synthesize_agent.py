"""
SynthesizeAgent — autonomous synthesis subgraph.

Internal loop:
  draft → self_critique → revise (until quality threshold or max revisions)

The agent critiques its own draft before returning to the orchestrator,
reducing how often the outer evaluator loop needs to trigger.
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

MAX_REVISIONS = 2


# ── Schemas ───────────────────────────────────────────────────────────────────


class DraftCritique(BaseModel):
    """Structured output for the self-critique node."""

    is_good_enough: bool = Field(
        description="True if the draft comprehensively answers the question with citations"
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Quality score: 0.7+ means good enough"
    )
    issues: list[str] = Field(
        description="Specific issues with the current draft. Empty if is_good_enough."
    )
    missing_aspects: list[str] = Field(
        description="Aspects of the question not covered. Empty if is_good_enough."
    )


class ResearchDraft(BaseModel):
    """Structured output for draft and revise nodes."""

    summary: str = Field(
        description="2-3 paragraph comprehensive answer grounded in sources"
    )
    key_findings: list[str] = Field(
        description="At least 3 specific discrete findings supported by sources"
    )
    citations: list[dict] = Field(
        description="Sources referenced, each with url, title, snippet"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Honest assessment of source quality and coverage"
    )


# ── State ─────────────────────────────────────────────────────────────────────


class SynthesizeState(TypedDict):
    # Inputs from orchestrator
    query: str
    scraped_content: list[dict]
    evaluation_feedback: str  # gaps from outer evaluator on retry

    # Internal state
    current_draft: dict | None
    critique: DraftCritique | None
    revision_count: int
    internal_steps: Annotated[list[dict], operator.add]

    # Output to orchestrator
    final_report: dict | None


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


def _build_context(scraped_content: list[dict]) -> str:
    """Build source context string from scraped content."""
    parts = []
    for i, src in enumerate(scraped_content, 1):
        parts.append(
            f"--- SOURCE {i}: {src['title']} ---\n"
            f"URL: {src['url']}\n"
            f"{src['content'][:2000]}\n"
        )
    return "\n\n".join(parts)


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def draft_node(state: SynthesizeState) -> dict:
    """Write an initial research report draft from all scraped sources."""
    context = _build_context(state["scraped_content"])

    # Include outer evaluator feedback if this is a retry
    feedback_section = ""
    if state.get("evaluation_feedback"):
        feedback_section = (
            f"\n\nOUTER EVALUATOR FEEDBACK:\n"
            f"{state['evaluation_feedback']}\n"
            f"Make sure your draft specifically addresses these gaps."
        )

    system = SystemMessage(
        content="""
You are an expert research synthesizer at a professional research firm.

Synthesis rules:
  - Stay strictly grounded in the provided sources
  - Never include claims not supported by the sources
  - summary must be 2-3 paragraphs DIRECTLY answering the research question
  - key_findings must have at least 3 specific, verifiable items
  - Each citation must be a source you actually referenced
  - confidence_score must honestly reflect source quality:
      0.0-0.4: sources are weak, off-topic, or sparse
      0.4-0.7: sources partially answer the question
      0.7-1.0: sources comprehensively answer the question
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n\n"
            f"Sources ({len(state['scraped_content'])} total):\n"
            f"{context}"
            f"{feedback_section}"
        )
    )

    structured_llm = _get_llm().with_structured_output(ResearchDraft)
    draft: ResearchDraft = await structured_llm.ainvoke([system, user])

    logger.info("synthesize_draft", confidence=draft.confidence_score)
    step = {
        "type": "synthesize_draft",
        "revision": 0,
        "content": f"Initial draft with confidence {draft.confidence_score:.0%}",
        "confidence": draft.confidence_score,
    }
    return {
        "current_draft": draft.model_dump(),
        "internal_steps": [step],
    }


async def self_critique_node(state: SynthesizeState) -> dict:
    """Critically evaluate the current draft against the research question."""
    draft = state["current_draft"]

    system = SystemMessage(
        content="""
You are a strict research editor. Evaluate this draft critically.

Check:
  1. Does the summary DIRECTLY and completely answer the question?
  2. Are key findings specific and verifiable (not vague)?
  3. Are there obvious aspects of the question left unaddressed?
  4. Are citations relevant to the claims made?
  5. Is the confidence score honest given the sources?

Be strict — approve only if the draft is genuinely comprehensive.
Score >= 0.75 and no critical missing aspects = is_good_enough: true
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n\n"
            f"Draft to evaluate:\n"
            f"Summary: {draft.get('summary', '')}\n\n"
            f"Key findings: {draft.get('key_findings', [])}\n\n"
            f"Citations: {[c.get('url') for c in draft.get('citations', [])]}\n"
            f"Confidence claimed: {draft.get('confidence_score', 0)}"
        )
    )

    structured_llm = _get_llm().with_structured_output(DraftCritique)
    critique: DraftCritique = await structured_llm.ainvoke([system, user])

    logger.info(
        "synthesize_critique",
        is_good_enough=critique.is_good_enough,
        score=critique.score,
        revision=state["revision_count"],
    )
    step = {
        "type": "synthesize_critique",
        "revision": state["revision_count"],
        "content": f"Score: {critique.score:.0%} — {'Approved' if critique.is_good_enough else 'Needs revision'}",
        "is_good_enough": critique.is_good_enough,
        "score": critique.score,
        "issues": critique.issues,
    }
    return {"critique": critique, "internal_steps": [step]}


async def revise_node(state: SynthesizeState) -> dict:
    """Revise the draft based on self-critique feedback."""
    context = _build_context(state["scraped_content"])
    critique = state["critique"]
    current_draft = state["current_draft"]

    system = SystemMessage(
        content="""
You are an expert research synthesizer revising a draft based on editorial feedback.
Apply the critique precisely. Address every identified issue and missing aspect.
Stay grounded in the provided sources — do not hallucinate new information.
"""
    )

    user = HumanMessage(
        content=(
            f"Research question: {state['query']}\n\n"
            f"Current draft summary:\n{current_draft.get('summary', '')}\n\n"
            f"Critique issues:\n"
            + "\n".join(f"  - {issue}" for issue in critique.issues)
            + "\n\nMissing aspects to address:\n"
            + "\n".join(f"  - {aspect}" for aspect in critique.missing_aspects)
            + f"\n\nSources to draw from:\n{context}\n\n"
            "Write an improved version addressing all issues."
        )
    )

    structured_llm = _get_llm().with_structured_output(ResearchDraft)
    revised: ResearchDraft = await structured_llm.ainvoke([system, user])

    revision = state["revision_count"] + 1
    logger.info(
        "synthesize_revise", revision=revision, confidence=revised.confidence_score
    )
    step = {
        "type": "synthesize_revise",
        "revision": revision,
        "content": f"Revised draft (revision {revision}), confidence {revised.confidence_score:.0%}",
        "confidence": revised.confidence_score,
    }
    return {
        "current_draft": revised.model_dump(),
        "revision_count": revision,
        "internal_steps": [step],
    }


async def finalize_node(state: SynthesizeState) -> dict:
    """Accept the current draft as the final report."""
    draft = state["current_draft"] or {}
    logger.info(
        "synthesize_finalized",
        revisions=state["revision_count"],
        confidence=draft.get("confidence_score"),
    )
    step = {
        "type": "synthesize_finalized",
        "content": f"Report finalized after {state['revision_count']} revision(s)",
        "confidence": draft.get("confidence_score"),
    }
    return {
        "final_report": draft,
        "internal_steps": [step],
    }


# ── Routing ───────────────────────────────────────────────────────────────────


def route_after_critique(state: SynthesizeState) -> str:
    """Approve draft or send for revision — with revision cap."""
    from app.agents.constants import get_depth_config

    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)

    max_revisions = get_depth_config(state.get("depth", "standard")).get(
        "max_revisions", MAX_REVISIONS
    )

    # Always check quality first
    if critique and critique.is_good_enough:
        return "finalize"

    # Circuit breaker
    if revision_count >= max_revisions:
        logger.info("synthesize_circuit_breaker", revision_count=revision_count)
        return "finalize"

    return "revise"


# ── Graph ─────────────────────────────────────────────────────────────────────


def build_synthesize_agent() -> StateGraph:
    graph = StateGraph(SynthesizeState)

    graph.add_node("write_draft", draft_node)
    graph.add_node("critique_draft", self_critique_node)
    graph.add_node("revise_draft", revise_node)
    graph.add_node("finalize_draft", finalize_node)

    graph.set_entry_point("write_draft")
    graph.add_edge("write_draft", "critique_draft")
    graph.add_conditional_edges(
        "critique_draft",
        route_after_critique,
        {
            "finalize": "finalize_draft",
            "revise": "revise_draft",
        },
    )
    graph.add_edge("revise_draft", "critique_draft")  # internal loop
    graph.add_edge("finalize_draft", END)

    return graph.compile()


# Singleton
synthesize_agent = build_synthesize_agent()


# ── Public interface ──────────────────────────────────────────────────────────


async def run_synthesize_agent(
    query: str,
    scraped_content: list[dict],
    evaluation_feedback: str = "",
) -> tuple[dict, list[dict]]:
    """
    Run the SynthesizeAgent subgraph.
    Returns (final_report, internal_steps).
    """
    initial_state: SynthesizeState = {
        "query": query,
        "scraped_content": scraped_content,
        "evaluation_feedback": evaluation_feedback,
        "current_draft": None,
        "critique": None,
        "revision_count": 0,
        "internal_steps": [],
        "final_report": None,
    }

    result = await synthesize_agent.ainvoke(initial_state)
    return result["final_report"] or {}, result["internal_steps"]
