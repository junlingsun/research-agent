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
