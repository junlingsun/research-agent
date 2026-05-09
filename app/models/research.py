import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.job import JobStatus, ResearchDepth


# ── Request schemas ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The research question to investigate.",
        examples=["What are the latest advancements in quantum computing in 2025?"],
    )
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="Controls how many sources are scraped and how long the agent runs.",
    )

# ── Agent internal schemas (structured outputs) ───────────────────────────────
 
class SearchPlan(BaseModel):
    """Output schema for plan_node."""
    queries: list[str] = Field(
        description="Distinct search queries each targeting a different aspect of the question"
    )
    reasoning: str = Field(
        description="One sentence explaining how these queries cover the question"
    )
 
 
class EvaluationResult(BaseModel):
    """Output schema for evaluate_node."""
    approved: bool = Field(
        description="Whether the report sufficiently answers the research question"
    )
    score: float = Field(
        ge=0.0, le=1.0,
        description="Quality score: 0.0-0.4 poor, 0.4-0.7 partial, 0.7-1.0 comprehensive"
    )
    gaps: list[str] = Field(
        description="Specific aspects of the question not covered. Empty list if approved."
    )
    feedback: str = Field(
        description="Specific instructions for improving the next research iteration. Empty string if approved."
    )

    
# ── Response schemas ─────────────────────────────────────────────────────────

class Citation(BaseModel):
    url: str
    title: str
    snippet: str


class ResearchResultSchema(BaseModel):
    summary: str
    key_findings: list[str]
    citations: list[Citation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources_scraped: int


class JobResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    query: str
    depth: ResearchDepth
    created_at: datetime
    updated_at: datetime
    result: ResearchResultSchema | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class JobCreatedResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    message: str


class AgentStepEvent(BaseModel):
    """SSE event payload emitted during streaming."""
    job_id: uuid.UUID
    step: int
    type: str  # "tool_call" | "tool_result" | "reasoning" | "complete" | "error"
    content: str
    metadata: dict = Field(default_factory=dict)
