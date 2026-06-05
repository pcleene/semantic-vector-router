"""Job scheduler engine for SVR operational tasks.

Manages recurring jobs that execute within maintenance windows.
Runs as a background asyncio task inside SVRClient.
"""

import asyncio
import os
import socket
import time
from collections.abc import Coroutine
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from semantic_vector_router.scheduler.interval import parse_interval
from semantic_vector_router.scheduler.models import (
    JobConfig,
    JobRun,
    JobState,
    JobStatus,
    JobType,
    SchedulerStatus,
)
from semantic_vector_router.scheduler.window import is_within_window
from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)


class JobScheduler:
    """Manages recurring operational jobs with maintenance window enforcement.

    Runs as a background asyncio task. Each tick:
    1. Check which jobs are due (based on interval)
    2. Check if current time is within the job's maintenance window
    3. If both: acquire distributed lock, execute, release lock
    4. If due but outside window: mark as "queued" (will run when window opens)
    5. Emit events for job start/complete/fail/skip
    """

    def __init__(
        self,
        metadata: Any,
        config: Any,
        event_bus: Optional[Any] = None,
    ) -> None:
        self._jobs: dict[str, JobConfig] = {}
        self._metadata = metadata
        self._config = config
        self._event_bus = event_bus
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._last_tick: Optional[datetime] = None

        # Job execution handlers
        self._job_handlers: dict[JobType, Callable[..., Coroutine[Any, Any, dict[str, Any]]]] = {}

        # Custom job callbacks
        self._custom_handlers: dict[str, Callable[..., Coroutine[Any, Any, dict[str, Any]]]] = {}

        # Worker ID
        sc = config.scheduler
        self._worker_id = sc.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self._tick_interval = sc.tick_interval_seconds

    @property
    def worker_id(self) -> str:
        """Get this scheduler's worker ID."""
        return self._worker_id

    @property
    def running(self) -> bool:
        """Whether the scheduler is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the scheduler tick loop as a background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"Scheduler started (worker={self._worker_id}, "
            f"tick={self._tick_interval}s, jobs={len(self._jobs)})"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("Scheduler stopped")

    def register_job(self, job_id: str, config: JobConfig) -> None:
        """Register a job for scheduling."""
        self._jobs[job_id] = config
        logger.info(f"Registered job '{job_id}' (type={config.job_type.value})")

    def unregister_job(self, job_id: str) -> None:
        """Unregister a job."""
        self._jobs.pop(job_id, None)
        logger.info(f"Unregistered job '{job_id}'")

    def set_job_handler(
        self,
        job_type: JobType,
        handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
    ) -> None:
        """Set the execution handler for a job type."""
        self._job_handlers[job_type] = handler

    def set_custom_handler(
        self,
        job_id: str,
        handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
    ) -> None:
        """Set a custom handler for a specific job ID."""
        self._custom_handlers[job_id] = handler

    async def run_now(self, job_id: str) -> JobRun:
        """Force-execute a job immediately, bypassing maintenance window."""
        config = self._jobs.get(job_id)
        if config is None:
            raise ValueError(f"Job '{job_id}' not registered")

        return await self._execute_job(job_id, config, force=True)

    async def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        jobs_info = []
        for job_id, config in self._jobs.items():
            state = await self._get_job_state(job_id)
            jobs_info.append({
                "job_id": job_id,
                "job_type": config.job_type.value,
                "enabled": config.enabled,
                "interval": config.interval,
                "last_run_at": state.last_run_at.isoformat() if state.last_run_at else None,
                "last_status": state.last_status.value if state.last_status else None,
                "next_due_at": state.next_due_at.isoformat() if state.next_due_at else None,
                "consecutive_failures": state.consecutive_failures,
            })

        return SchedulerStatus(
            running=self._running,
            worker_id=self._worker_id,
            jobs_registered=len(self._jobs),
            tick_interval_seconds=self._tick_interval,
            last_tick_at=self._last_tick,
            jobs=jobs_info,
        )

    async def get_job_history(
        self, job_id: str, limit: int = 20
    ) -> list[JobRun]:
        """Get recent run history for a job."""
        try:
            cursor = self._metadata._coll.find(
                {"type": "job_run", "job_id": job_id}
            ).sort("scheduled_at", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [
                JobRun(
                    job_id=doc["job_id"],
                    job_type=doc.get("job_type", ""),
                    status=JobStatus(doc.get("status", "completed")),
                    scheduled_at=doc.get("scheduled_at"),
                    started_at=doc.get("started_at"),
                    completed_at=doc.get("completed_at"),
                    duration_ms=doc.get("duration_ms"),
                    result=doc.get("result", {}),
                    error=doc.get("error"),
                    skipped_reason=doc.get("skipped_reason"),
                    worker_id=doc.get("worker_id"),
                )
                for doc in docs
            ]
        except Exception as e:
            logger.warning(f"Failed to get job history: {e}")
            return []

    # --- Internal ---

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._last_tick = datetime.utcnow()

                for job_id, config in list(self._jobs.items()):
                    if not config.enabled:
                        continue

                    state = await self._get_job_state(job_id)

                    # Is job due?
                    if not self._is_due(state, config):
                        continue

                    # Is maintenance window open?
                    window = config.maintenance_window
                    if window is None:
                        # Check global maintenance window
                        window = self._config.scheduler.maintenance_window

                    if window is not None and not is_within_window(window):
                        await self._emit_event("job.skipped", job_id, reason="outside_maintenance_window")
                        await self._record_skip(job_id, config, "outside_maintenance_window")
                        continue

                    # Try to acquire distributed lock
                    if config.lock_id:
                        acquired = await self._metadata.acquire_lock(
                            config.lock_id, self._worker_id, config.lock_ttl_seconds
                        )
                        if not acquired:
                            await self._record_skip(job_id, config, "lock_held")
                            continue

                    try:
                        await self._execute_job(job_id, config)
                    finally:
                        if config.lock_id:
                            await self._metadata.release_lock(
                                config.lock_id, self._worker_id
                            )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scheduler tick error: {e}", exc_info=True)

            await asyncio.sleep(self._tick_interval)

    def _is_due(self, state: JobState, config: JobConfig) -> bool:
        """Check if a job is due for execution."""
        if state.next_due_at is not None:
            return datetime.utcnow() >= state.next_due_at

        if state.last_run_at is None:
            return True

        try:
            interval_seconds = parse_interval(config.interval)
        except ValueError:
            return False

        elapsed = (datetime.utcnow() - state.last_run_at).total_seconds()
        return elapsed >= interval_seconds

    async def _execute_job(
        self,
        job_id: str,
        config: JobConfig,
        force: bool = False,
    ) -> JobRun:
        """Execute a job and record the result."""
        now = datetime.utcnow()

        await self._emit_event("job.started", job_id)

        start_time = time.perf_counter()
        result: dict[str, Any] = {}
        error: Optional[str] = None
        status = JobStatus.COMPLETED

        try:
            # Get handler
            handler = self._custom_handlers.get(job_id)
            if handler is None:
                handler = self._job_handlers.get(config.job_type)

            if handler is None:
                raise RuntimeError(
                    f"No handler registered for job '{job_id}' "
                    f"(type={config.job_type.value})"
                )

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(), timeout=config.timeout_seconds
            )

        except asyncio.TimeoutError:
            status = JobStatus.FAILED
            error = f"Job timed out after {config.timeout_seconds}s"
            logger.error(f"Job '{job_id}' timed out")

        except Exception as e:
            status = JobStatus.FAILED
            error = str(e)
            logger.error(f"Job '{job_id}' failed: {e}", exc_info=True)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Build run record
        run = JobRun(
            job_id=job_id,
            job_type=config.job_type.value,
            status=status,
            scheduled_at=now,
            started_at=now,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            result=result or {},
            error=error,
            worker_id=self._worker_id,
        )

        # Persist run record
        await self._record_run(run)

        # Update job state
        try:
            interval_seconds = parse_interval(config.interval)
        except ValueError:
            interval_seconds = 3600.0

        next_due = datetime.utcnow() + timedelta(seconds=interval_seconds)

        state = JobState(
            job_id=job_id,
            last_run_at=datetime.utcnow(),
            last_status=status,
            next_due_at=next_due,
            consecutive_failures=(
                0
                if status == JobStatus.COMPLETED
                else (await self._get_job_state(job_id)).consecutive_failures + 1
            ),
        )
        await self._save_job_state(state)

        # Emit completion/failure event
        if status == JobStatus.COMPLETED:
            await self._emit_event(
                "job.completed", job_id,
                duration_ms=duration_ms, result=result,
            )
        else:
            await self._emit_event(
                "job.failed", job_id,
                error=error, duration_ms=duration_ms,
            )

        return run

    async def _get_job_state(self, job_id: str) -> JobState:
        """Load job state from metadata."""
        try:
            doc = await self._metadata._coll.find_one(
                {"_id": f"job_state:{job_id}", "type": "job_state"}
            )
            if doc:
                return JobState(
                    job_id=doc.get("job_id", job_id),
                    last_run_at=doc.get("last_run_at"),
                    last_status=JobStatus(doc["last_status"]) if doc.get("last_status") else None,
                    next_due_at=doc.get("next_due_at"),
                    consecutive_failures=doc.get("consecutive_failures", 0),
                )
        except Exception as e:
            logger.warning(f"Failed to load job state for '{job_id}': {e}")

        return JobState(job_id=job_id)

    async def _save_job_state(self, state: JobState) -> None:
        """Persist job state to metadata."""
        try:
            doc = {
                "_id": f"job_state:{state.job_id}",
                "type": "job_state",
                "job_id": state.job_id,
                "last_run_at": state.last_run_at,
                "last_status": state.last_status.value if state.last_status else None,
                "next_due_at": state.next_due_at,
                "consecutive_failures": state.consecutive_failures,
            }
            await self._metadata._coll.replace_one(
                {"_id": f"job_state:{state.job_id}"},
                doc,
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"Failed to save job state for '{state.job_id}': {e}")

    async def _record_run(self, run: JobRun) -> None:
        """Persist a job run record to metadata."""
        try:
            ts = run.scheduled_at or datetime.utcnow()
            doc = {
                "_id": f"job_run:{run.job_id}-{ts.isoformat()}",
                "type": "job_run",
                "job_id": run.job_id,
                "job_type": run.job_type,
                "status": run.status.value,
                "scheduled_at": run.scheduled_at,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "duration_ms": run.duration_ms,
                "result": run.result,
                "error": run.error,
                "skipped_reason": run.skipped_reason,
                "worker_id": run.worker_id,
            }
            await self._metadata._coll.insert_one(doc)
        except Exception as e:
            logger.warning(f"Failed to record job run: {e}")

    async def _record_skip(
        self, job_id: str, config: JobConfig, reason: str
    ) -> None:
        """Record a skipped job run."""
        run = JobRun(
            job_id=job_id,
            job_type=config.job_type.value,
            status=JobStatus.SKIPPED,
            scheduled_at=datetime.utcnow(),
            skipped_reason=reason,
            worker_id=self._worker_id,
        )
        await self._record_run(run)

    async def _emit_event(
        self, event_type_str: str, job_id: str, **details: Any
    ) -> None:
        """Emit a scheduler event via the event bus."""
        if self._event_bus is None:
            return

        try:
            from semantic_vector_router.events.models import SVREvent, SVREventType

            event_map = {
                "job.started": SVREventType.JOB_STARTED,
                "job.completed": SVREventType.JOB_COMPLETED,
                "job.failed": SVREventType.JOB_FAILED,
                "job.skipped": SVREventType.JOB_SKIPPED,
            }

            svr_event_type = event_map.get(event_type_str)
            if svr_event_type is None:
                return

            event = SVREvent(
                event_type=svr_event_type,
                job_id=job_id,
                details=details,
                severity="warning" if "failed" in event_type_str else "info",
                worker_id=self._worker_id,
            )
            await self._event_bus.emit(event)
        except Exception as e:
            logger.warning(f"Failed to emit scheduler event: {e}")
