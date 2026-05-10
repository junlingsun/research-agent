import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_document(
    db: AsyncSession,
    title: str,
    source_type: str,
    source_ref: str | None = None,
) -> Document:
    doc = Document(
        title=title,
        source_type=source_type,
        source_ref=source_ref,
        status="processing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    logger.info("document_created", document_id=str(doc.id), title=title)
    return doc


async def update_document_status(
    db: AsyncSession,
    document_id: uuid.UUID,
    status: str,
    chunk_count: int = 0,
    error_message: str | None = None,
) -> Document | None:
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return None
    doc.status = status
    doc.chunk_count = chunk_count
    if error_message:
        doc.error_message = error_message
    await db.flush()
    return doc


async def get_document(
    db: AsyncSession,
    document_id: uuid.UUID,
) -> Document | None:
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def list_documents(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Document], int]:
    total_result = await db.execute(select(func.count(Document.id)))
    total = total_result.scalar() or 0

    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    docs = list(result.scalars().all())
    return docs, total


async def delete_document(
    db: AsyncSession,
    document_id: uuid.UUID,
) -> bool:
    doc = await get_document(db, document_id)
    if not doc:
        return False
    await db.delete(doc)
    await db.flush()
    logger.info("document_deleted", document_id=str(document_id))
    return True