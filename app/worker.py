"""
Celery worker configuration and research task.
Run with: celery -A app.worker worker --loglevel=info
"""

import asyncio
import uuid

from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "research_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # re-queue if worker dies mid-task
    worker_prefetch_multiplier=1,  # one task at a time per worker (LLM calls are heavy)
    task_soft_time_limit=180,  # 3 min soft limit
    task_time_limit=240,  # 4 min hard limit
    result_expires=3600,  # keep results for 1 hour
)


@celery_app.task(
    name="research_agent.run_research",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def run_research_task(self, job_id: str, query: str, depth: str) -> dict:
    """
    Celery task that runs the LangGraph research agent.
    Handles DB updates and cache population.
    """
    from app.agents.research_agent import run_research_agent
    from app.db.session import AsyncSessionLocal
    from app.models.job import JobStatus
    from app.services.job_service import save_result, update_job_status
    from app.services.cache_service import set_cached_result

    async def _run() -> dict:
        async with AsyncSessionLocal() as db:
            try:
                # Mark as running
                await update_job_status(
                    db,
                    uuid.UUID(job_id),
                    JobStatus.RUNNING,
                    celery_task_id=self.request.id,
                )
                await db.commit()

                # Run the agent
                report, steps = await run_research_agent(query, depth)

                # Persist result
                await save_result(db, uuid.UUID(job_id), report, steps)
                await update_job_status(db, uuid.UUID(job_id), JobStatus.COMPLETED)
                await db.commit()

                # Populate cache
                await set_cached_result(query, depth, report)

                return report

            except Exception as exc:
                await db.rollback()
                await update_job_status(
                    db,
                    uuid.UUID(job_id),
                    JobStatus.FAILED,
                    error_message=str(exc),
                )
                await db.commit()
                raise self.retry(exc=exc)

    return asyncio.get_event_loop().run_until_complete(_run())
