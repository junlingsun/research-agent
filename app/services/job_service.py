import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.job import JobStatus, ResearchDepth, ResearchJob, ResearchResult
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_job(
    db: AsyncSession,
    query: str,
    depth: ResearchDepth,
) -> ResearchJob:
    job = ResearchJob(query=query, depth=depth, status=JobStatus.PENDING)
    db.add(job)
    await db.flush()
    await db.refresh(job)
    logger.info("job_created", job_id=str(job.id), depth=depth)
    return job


async def get_job(db: AsyncSession, job_id: uuid.UUID) -> ResearchJob | None:
    result = await db.execute(
        select(ResearchJob)
        .options(selectinload(ResearchJob.result))
        .where(ResearchJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def update_job_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    status: JobStatus,
    celery_task_id: str | None = None,
    error_message: str | None = None,
) -> ResearchJob | None:
    job = await get_job(db, job_id)
    if not job:
        return None
    job.status = status
    if celery_task_id:
        job.celery_task_id = celery_task_id
    if error_message:
        job.error_message = error_message
    await db.flush()
    return job


async def save_result(
    db: AsyncSession,
    job_id: uuid.UUID,
    report: dict,
    steps: list[dict],
) -> ResearchResult:
    result = ResearchResult(
        job_id=job_id,
        summary=report.get("summary", ""),
        key_findings=report.get("key_findings", []),
        citations=report.get("citations", []),
        confidence_score=float(report.get("confidence_score", 0.0)),
        sources_scraped=int(report.get("sources_scraped", 0)),
        agent_steps=steps,
    )
    db.add(result)
    await db.flush()
    logger.info("result_saved", job_id=str(job_id))
    return result
