import uuid
from datetime import datetime
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel, Field
from app.db.base import Base


# ── SQLAlchemy model ──────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "pdf" | "url" | "text"
    source_ref: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # original URL or filename
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(50), default="processing"
    )  # "processing" | "ready" | "failed"
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: uuid.UUID
    title: str
    source_type: str
    source_ref: str | None
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and ingest")
    title: str | None = Field(None, description="Optional title override")


class TextIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=50, description="Plain text content")


class DocumentSearchResult(BaseModel):
    document_id: uuid.UUID
    title: str
    chunk: str
    score: float
    source_ref: str | None