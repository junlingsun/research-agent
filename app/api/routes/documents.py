import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_api_key
from app.db.session import get_db
from app.models.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentSearchResult,
    TextIngestRequest,
    URLIngestRequest,
)
from app.services.document_service import (
    create_document,
    delete_document,
    get_document,
    list_documents,
    update_document_status,
)
from app.services.ingestion_service import (
    chunk_text,
    delete_document_chunks,
    extract_from_pdf,
    extract_from_url,
    ingest_chunks,
    search_documents,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/url",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a URL into the knowledge base",
)
async def ingest_url(
    request: URLIngestRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> DocumentResponse:
    """Scrape a URL and add its content to your private knowledge base."""
    # Create DB record
    doc = await create_document(
        db,
        title=request.title or request.url,
        source_type="url",
        source_ref=request.url,
    )
    await db.commit()

    try:
        # Extract text from URL
        text = await extract_from_url(request.url)

        # Chunk and embed
        chunks = chunk_text(text)
        chunk_count = await ingest_chunks(doc.id, doc.title, request.url, chunks)

        # Update status
        await update_document_status(db, doc.id, "ready", chunk_count=chunk_count)
        await db.commit()
        await db.refresh(doc)

        logger.info("url_ingested", document_id=str(doc.id), chunks=chunk_count)
        return DocumentResponse.model_validate(doc)

    except Exception as e:
        await update_document_status(db, doc.id, "failed", error_message=str(e))
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post(
    "/pdf",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF into the knowledge base",
)
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> DocumentResponse:
    """Upload a PDF and add its content to your private knowledge base."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    doc = await create_document(
        db,
        title=title,
        source_type="pdf",
        source_ref=file.filename,
    )
    await db.commit()

    try:
        pdf_bytes = await file.read()
        text = await extract_from_pdf(pdf_bytes)
        chunks = chunk_text(text)
        chunk_count = await ingest_chunks(doc.id, doc.title, file.filename, chunks)

        await update_document_status(db, doc.id, "ready", chunk_count=chunk_count)
        await db.commit()
        await db.refresh(doc)

        logger.info("pdf_ingested", document_id=str(doc.id), chunks=chunk_count)
        return DocumentResponse.model_validate(doc)

    except Exception as e:
        await update_document_status(db, doc.id, "failed", error_message=str(e))
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post(
    "/text",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest plain text into the knowledge base",
)
async def ingest_text(
    request: TextIngestRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> DocumentResponse:
    """Add plain text content directly to your private knowledge base."""
    doc = await create_document(
        db,
        title=request.title,
        source_type="text",
        source_ref=None,
    )
    await db.commit()

    try:
        chunks = chunk_text(request.content)
        chunk_count = await ingest_chunks(doc.id, doc.title, None, chunks)

        await update_document_status(db, doc.id, "ready", chunk_count=chunk_count)
        await db.commit()
        await db.refresh(doc)

        return DocumentResponse.model_validate(doc)

    except Exception as e:
        await update_document_status(db, doc.id, "failed", error_message=str(e))
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_docs(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> DocumentListResponse:
    docs, total = await list_documents(db, limit=limit, offset=offset)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document metadata",
)
async def get_doc(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> DocumentResponse:
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_doc(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> None:
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete chunks from Qdrant
    await delete_document_chunks(document_id)

    # Delete metadata from Postgres
    await delete_document(db, document_id)
    await db.commit()


@router.get(
    "/search/query",
    response_model=list[DocumentSearchResult],
    summary="Search the private knowledge base",
)
async def search_docs(
    q: str,
    top_k: int = 5,
    _: str = Depends(require_api_key),
) -> list[DocumentSearchResult]:
    """Semantic search over all ingested documents."""
    chunks = await search_documents(q, top_k=top_k)
    return [
        DocumentSearchResult(
            document_id=uuid.UUID(c["document_id"]),
            title=c["title"],
            chunk=c["chunk"],
            score=c["score"],
            source_ref=c["source_ref"],
        )
        for c in chunks
    ]
