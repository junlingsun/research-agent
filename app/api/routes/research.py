import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.core.security import require_api_key
from app.db.session import get_db
from app.models.job import JobStatus, ResearchDepth
from app.models.research import (
    AgentStepEvent,
    JobCreatedResponse,
    JobResponse,
    ResearchRequest,
    ResearchResultSchema,
    Citation,
)
from app.services.cache_service import get_cached_result
from app.services.job_service import create_job, get_job
from app.worker import run_research_task
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research query",
)
async def submit_research(
    request: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> JobCreatedResponse:
    """
    Submit a research query. Returns a job_id for polling or streaming.
    Checks cache first — cached results are returned synchronously as a completed job.
    """
    # Cache check
    cached = await get_cached_result(request.query, request.depth.value)
    if cached:
        # Create a pre-completed job for the cached result
        job = await create_job(db, request.query, request.depth)
        from app.services.job_service import update_job_status, save_result
        await update_job_status(db, job.id, JobStatus.COMPLETED)
        await save_result(db, job.id, cached, [{"type": "cache_hit", "content": "Result from cache"}])
        await db.commit()
        logger.info("cache_hit_job", job_id=str(job.id))
        return JobCreatedResponse(
            job_id=job.id,
            status=JobStatus.COMPLETED,
            message="Result retrieved from cache.",
        )

    # Create job and dispatch to Celery
    job = await create_job(db, request.query, request.depth)
    await db.commit()

    run_research_task.apply_async(
        args=[str(job.id), request.query, request.depth.value],
        task_id=str(job.id),  # use job_id as celery task_id for easy lookup
    )

    logger.info("job_dispatched", job_id=str(job.id), depth=request.depth)
    return JobCreatedResponse(
        job_id=job.id,
        status=JobStatus.PENDING,
        message="Research job accepted. Poll /research/{job_id} for status.",
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status and result",
)
async def get_research_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> JobResponse:
    """Poll a job by ID. When status is 'completed', the result field is populated."""
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    result = None
    if job.result:
        result = ResearchResultSchema(
            summary=job.result.summary,
            key_findings=job.result.key_findings,
            citations=[Citation(**c) for c in job.result.citations],
            confidence_score=job.result.confidence_score,
            sources_scraped=job.result.sources_scraped,
        )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        query=job.query,
        depth=job.depth,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=result,
        error_message=job.error_message,
    )


@router.get(
    "/{job_id}/stream",
    summary="Stream agent reasoning steps via SSE",
)
async def stream_research(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> StreamingResponse:
    """
    Stream Server-Sent Events (SSE) with real-time agent step updates.
    Connect early and receive events as the agent works.
    """
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        poll_interval = 1.5  # seconds between DB polls
        timeout = 200        # max seconds to wait
        elapsed = 0
        last_step_count = 0

        while elapsed < timeout:
            # Re-fetch job to get latest state
            async with db.begin_nested():
                current_job = await get_job(db, job_id)

            if not current_job:
                break

            if current_job.result:
                steps = current_job.result.agent_steps or []

                # Emit any new steps since last poll
                for step in steps[last_step_count:]:
                    event = AgentStepEvent(
                        job_id=job_id,
                        step=steps.index(step) + 1,
                        type=step.get("type", "reasoning"),
                        content=step.get("content", ""),
                        metadata={k: v for k, v in step.items() if k not in ("type", "content")},
                    )
                    yield f"data: {event.model_dump_json()}\n\n"
                last_step_count = len(steps)

            if current_job.status == JobStatus.COMPLETED:
                # Final complete event
                complete_event = AgentStepEvent(
                    job_id=job_id,
                    step=last_step_count + 1,
                    type="complete",
                    content="Research complete. Fetch result at GET /research/{job_id}",
                )
                yield f"data: {complete_event.model_dump_json()}\n\n"
                break

            if current_job.status == JobStatus.FAILED:
                error_event = AgentStepEvent(
                    job_id=job_id,
                    step=last_step_count + 1,
                    type="error",
                    content=current_job.error_message or "Job failed.",
                )
                yield f"data: {error_event.model_dump_json()}\n\n"
                break

            # Heartbeat to keep connection alive
            yield f": heartbeat\n\n"
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if elapsed >= timeout:
            yield f"data: {json.dumps({'type': 'timeout', 'content': 'Stream timed out.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
