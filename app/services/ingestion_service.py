"""
Document ingestion service.

Handles three source types:
  - PDF:  extract text → chunk → embed → store
  - URL:  scrape → chunk → embed → store
  - Text: chunk → embed → store

Chunks stored in Qdrant collection "document_chunks".
Document metadata stored in Postgres.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

DOCUMENT_COLLECTION = "document_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
VECTOR_SIZE = 1536


# ── Qdrant client ─────────────────────────────────────────────────────────────

_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def ensure_collection() -> None:
    client = get_qdrant_client()
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]
    if DOCUMENT_COLLECTION not in existing:
        await client.create_collection(
            collection_name=DOCUMENT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("document_collection_created")


# ── Embedding ─────────────────────────────────────────────────────────────────


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return await embeddings.aembed_documents(texts)


async def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    vectors = await embed_texts([text])
    return vectors[0]


# ── Text chunking ─────────────────────────────────────────────────────────────


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    text = re.sub(r" {2,}", " ", text)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        paragraph_break = text.rfind("\n\n", start, end)
        if paragraph_break > start + chunk_size // 2:
            end = paragraph_break
        else:
            sentence_break = text.rfind(". ", start, end)
            if sentence_break > start + chunk_size // 2:
                end = sentence_break + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 50]


# ── Source extractors ─────────────────────────────────────────────────────────


async def extract_from_url(url: str) -> str:
    """Scrape and extract clean text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:50000]


async def extract_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import pypdf
        import io

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ValueError("pypdf not installed. Add pypdf to pyproject.toml.")


# ── Core ingestion ────────────────────────────────────────────────────────────


def _chunk_point_id(document_id: uuid.UUID, chunk_index: int) -> str:
    """Deterministic point ID for a chunk."""
    raw = f"{document_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def ingest_chunks(
    document_id: uuid.UUID,
    title: str,
    source_ref: str | None,
    chunks: list[str],
) -> int:
    """Embed chunks and store in Qdrant. Returns number of chunks stored."""
    await ensure_collection()
    client = get_qdrant_client()

    vectors = await embed_texts(chunks)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = _chunk_point_id(document_id, i)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "document_id": str(document_id),
                    "title": title,
                    "source_ref": source_ref,
                    "chunk_index": i,
                    "chunk": chunk,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        await client.upsert(
            collection_name=DOCUMENT_COLLECTION,
            points=points[i : i + batch_size],
        )

    logger.info("chunks_stored", document_id=str(document_id), chunk_count=len(points))
    return len(points)


async def delete_document_chunks(document_id: uuid.UUID) -> None:
    """Delete all chunks for a document from Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant_client()
    await client.delete(
        collection_name=DOCUMENT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=str(document_id))
                )
            ]
        ),
    )
    logger.info("chunks_deleted", document_id=str(document_id))


# ── Document search ───────────────────────────────────────────────────────────


async def search_documents(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.7,
) -> list[dict]:
    """Semantic search over ingested document chunks."""
    try:
        await ensure_collection()
        client = get_qdrant_client()

        vector = await embed_text(query)

        results = await client.search(
            collection_name=DOCUMENT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        chunks = []
        for r in results:
            payload = r.payload or {}
            chunks.append(
                {
                    "document_id": payload.get("document_id"),
                    "title": payload.get("title", ""),
                    "chunk": payload.get("chunk", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "source_ref": payload.get("source_ref"),
                    "score": r.score,
                }
            )

        logger.info("document_search", query=query[:50], results=len(chunks))
        return chunks

    except Exception as e:
        logger.error("document_search_failed", error=str(e))
        return []


def format_document_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as context for the synthesizer."""
    if not chunks:
        return ""

    parts = ["PRIVATE KNOWLEDGE BASE CONTEXT:"]
    parts.append("Note: The following is from your uploaded documents.\n")

    seen_docs: set[str] = set()
    for chunk in chunks:
        doc_id = chunk["document_id"]
        title = chunk["title"]
        if doc_id not in seen_docs:
            parts.append(f"--- Document: {title} ---")
            seen_docs.add(doc_id)
        parts.append(chunk["chunk"])
        parts.append("")

    return "\n".join(parts)
