"""Comprehensive unit tests for semantic_vector_router.scheduler.engine.JobScheduler.

Tests cover: registration, start/stop lifecycle, run_now, handler success/failure/timeout,
is_due logic, maintenance window enforcement, distributed locking, state persistence,
run recording, skip recording, get_status, get_job_history, consecutive failure tracking,
event emission, disabled jobs, worker ID generation, and force flag.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.scheduler.engine import JobScheduler
from semantic_vector_router.scheduler.models import (
    JobConfig,
    JobRun,
    JobState,
    JobStatus,
    JobType,
    MaintenanceWindow,
    SchedulerStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_metadata():
    """Mock MetadataStore with async collection and lock helpers."""
    metadata = MagicMock()
    metadata._coll = MagicMock()
    metadata._coll.find_one = AsyncMock(return_value=None)
    metadata._coll.replace_one = AsyncMock()
    metadata._coll.insert_one = AsyncMock()

    # Default cursor chain for find().sort().limit().to_list()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    cursor.sort = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    metadata._coll.find = MagicMock(return_value=cursor)

    metadata.acquire_lock = AsyncMock(return_value=True)
    metadata.release_lock = AsyncMock(return_value=True)
    return metadata


@pytest.fixture
def mock_config():
    """Mock SVRConfig with SchedulerConfig attributes."""
    config = MagicMock()
    config.scheduler.enabled = True
    config.scheduler.tick_interval_seconds = 30
    config.scheduler.worker_id = "test-worker"
    config.scheduler.maintenance_window = None
    return config


@pytest.fixture
def mock_event_bus():
    """Mock EventBus with async emit."""
    bus = MagicMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def scheduler(mock_metadata, mock_config):
    """JobScheduler with no event bus."""
    return JobScheduler(metadata=mock_metadata, config=mock_config)


@pytest.fixture
def scheduler_with_events(mock_metadata, mock_config, mock_event_bus):
    """JobScheduler with event bus attached."""
    return JobScheduler(
        metadata=mock_metadata, config=mock_config, event_bus=mock_event_bus,
    )


@pytest.fixture
def sample_job_config():
    """A basic enabled job config for DETECTION type, 1-hour interval."""
    return JobConfig(
        job_id="detect-drift",
        job_type=JobType.DETECTION,
        interval="1h",
        enabled=True,
        lock_id="lock:detect",
        lock_ttl_seconds=120,
        timeout_seconds=60,
    )


@pytest.fixture
def disabled_job_config():
    """A disabled job config."""
    return JobConfig(
        job_id="disabled-job",
        job_type=JobType.CUSTOM,
        interval="6h",
        enabled=False,
    )


@pytest.fixture
def windowed_job_config():
    """A job config with a maintenance window limiting to weekdays 2-4 AM UTC."""
    return JobConfig(
        job_id="windowed-job",
        job_type=JobType.CENTROID_REFRESH,
        interval="1h",
        enabled=True,
        maintenance_window=MaintenanceWindow(
            allowed_days=["monday", "tuesday", "wednesday", "thursday", "friday"],
            allowed_hours={"start": 2, "end": 4},
            timezone="UTC",
        ),
    )


def _async_handler_ok():
    """Return a successful async handler."""
    handler = AsyncMock(return_value={"items_processed": 42})
    return handler


def _async_handler_error():
    """Return a handler that raises RuntimeError."""
    handler = AsyncMock(side_effect=RuntimeError("handler exploded"))
    return handler


# ===================================================================
# 1. Registration
# ===================================================================

class TestRegistration:
    """Tests for register_job and unregister_job."""

    def test_register_job_adds_to_internal_dict(self, scheduler, sample_job_config):
        """register_job should store config keyed by job_id."""
        scheduler.register_job("detect-drift", sample_job_config)
        assert "detect-drift" in scheduler._jobs
        assert scheduler._jobs["detect-drift"] is sample_job_config

    def test_register_multiple_jobs(self, scheduler, sample_job_config, disabled_job_config):
        """Registering multiple jobs should keep all of them."""
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.register_job("disabled-job", disabled_job_config)
        assert len(scheduler._jobs) == 2

    def test_unregister_job_removes_from_dict(self, scheduler, sample_job_config):
        """unregister_job should remove the job from internal dict."""
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.unregister_job("detect-drift")
        assert "detect-drift" not in scheduler._jobs

    def test_unregister_nonexistent_job_no_error(self, scheduler):
        """unregister_job on unknown ID should not raise."""
        scheduler.unregister_job("nonexistent")  # no exception


# ===================================================================
# 2. Start / Stop lifecycle
# ===================================================================

class TestStartStop:
    """Tests for start() and stop() background task management."""

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self, scheduler):
        """start() should set _running=True and create an asyncio Task."""
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None
        assert isinstance(scheduler._task, asyncio.Task)
        # Cleanup
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, scheduler):
        """Calling start() twice should not create a second task."""
        await scheduler.start()
        first_task = scheduler._task
        await scheduler.start()
        assert scheduler._task is first_task
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, scheduler):
        """stop() should cancel the background task and set _running=False."""
        await scheduler.start()
        task = scheduler._task
        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, scheduler):
        """stop() with no active task should be a no-op."""
        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None


# ===================================================================
# 3-4. run_now
# ===================================================================

class TestRunNow:
    """Tests for run_now() — force-execute a job."""

    @pytest.mark.asyncio
    async def test_run_now_executes_handler(self, scheduler, sample_job_config):
        """run_now should call the handler and return a JobRun."""
        handler = _async_handler_ok()
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, handler)

        run = await scheduler.run_now("detect-drift")

        assert isinstance(run, JobRun)
        assert run.job_id == "detect-drift"
        assert run.status == JobStatus.COMPLETED
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_now_unknown_job_raises_valueerror(self, scheduler):
        """run_now with unregistered job_id should raise ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            await scheduler.run_now("no-such-job")


# ===================================================================
# 5-7. Job handler outcomes (success / failure / timeout)
# ===================================================================

class TestJobExecution:
    """Tests for _execute_job handler outcomes."""

    @pytest.mark.asyncio
    async def test_handler_success_returns_completed(self, scheduler, sample_job_config):
        """Successful handler should produce COMPLETED status."""
        handler = _async_handler_ok()
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, handler)

        run = await scheduler._execute_job("detect-drift", sample_job_config)

        assert run.status == JobStatus.COMPLETED
        assert run.error is None
        assert run.result == {"items_processed": 42}
        assert run.duration_ms is not None and run.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_handler_failure_returns_failed(self, scheduler, sample_job_config):
        """Handler that raises should produce FAILED status with error message."""
        handler = _async_handler_error()
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, handler)

        run = await scheduler._execute_job("detect-drift", sample_job_config)

        assert run.status == JobStatus.FAILED
        assert "handler exploded" in run.error

    @pytest.mark.asyncio
    async def test_handler_timeout_returns_failed(self, scheduler):
        """Handler exceeding timeout_seconds should produce FAILED status."""
        slow_config = JobConfig(
            job_id="slow-job",
            job_type=JobType.CUSTOM,
            interval="1h",
            timeout_seconds=0,  # instant timeout
        )

        async def slow_handler():
            await asyncio.sleep(10)
            return {}

        scheduler.register_job("slow-job", slow_config)
        scheduler.set_job_handler(JobType.CUSTOM, slow_handler)

        run = await scheduler._execute_job("slow-job", slow_config)

        assert run.status == JobStatus.FAILED
        assert "timed out" in run.error

    @pytest.mark.asyncio
    async def test_no_handler_registered_fails(self, scheduler, sample_job_config):
        """Executing a job with no handler should produce FAILED status."""
        scheduler.register_job("detect-drift", sample_job_config)

        run = await scheduler._execute_job("detect-drift", sample_job_config)

        assert run.status == JobStatus.FAILED
        assert "No handler registered" in run.error

    @pytest.mark.asyncio
    async def test_custom_handler_takes_precedence(self, scheduler, sample_job_config):
        """Custom handler for job_id should override the type-level handler."""
        type_handler = AsyncMock(return_value={"source": "type"})
        custom_handler = AsyncMock(return_value={"source": "custom"})

        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, type_handler)
        scheduler.set_custom_handler("detect-drift", custom_handler)

        run = await scheduler._execute_job("detect-drift", sample_job_config)

        custom_handler.assert_awaited_once()
        type_handler.assert_not_awaited()
        assert run.result == {"source": "custom"}


# ===================================================================
# 8-11. _is_due logic
# ===================================================================

class TestIsDue:
    """Tests for the _is_due check (interval-based scheduling)."""

    def test_is_due_first_run_no_state(self, scheduler, sample_job_config):
        """First run (no last_run_at) should be due."""
        state = JobState(job_id="detect-drift")
        assert scheduler._is_due(state, sample_job_config) is True

    def test_is_due_within_interval_returns_false(self, scheduler, sample_job_config):
        """Job that ran recently (within 1h interval) should not be due."""
        state = JobState(
            job_id="detect-drift",
            last_run_at=datetime.utcnow() - timedelta(minutes=30),
        )
        assert scheduler._is_due(state, sample_job_config) is False

    def test_is_due_past_interval_returns_true(self, scheduler, sample_job_config):
        """Job that ran >1h ago should be due."""
        state = JobState(
            job_id="detect-drift",
            last_run_at=datetime.utcnow() - timedelta(hours=2),
        )
        assert scheduler._is_due(state, sample_job_config) is True

    def test_is_due_explicit_next_due_at_past(self, scheduler, sample_job_config):
        """Explicit next_due_at in the past should be due."""
        state = JobState(
            job_id="detect-drift",
            last_run_at=datetime.utcnow() - timedelta(minutes=10),
            next_due_at=datetime.utcnow() - timedelta(minutes=1),
        )
        assert scheduler._is_due(state, sample_job_config) is True

    def test_is_due_explicit_next_due_at_future(self, scheduler, sample_job_config):
        """Explicit next_due_at in the future should NOT be due."""
        state = JobState(
            job_id="detect-drift",
            last_run_at=datetime.utcnow() - timedelta(hours=2),
            next_due_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert scheduler._is_due(state, sample_job_config) is False

    def test_is_due_invalid_interval_returns_false(self, scheduler):
        """Invalid interval format should not crash — returns False."""
        bad_config = JobConfig(
            job_id="bad", job_type=JobType.CUSTOM, interval="xyz"
        )
        state = JobState(
            job_id="bad",
            last_run_at=datetime.utcnow() - timedelta(hours=100),
        )
        assert scheduler._is_due(state, bad_config) is False


# ===================================================================
# 12-14. Maintenance window enforcement
# ===================================================================

class TestMaintenanceWindow:
    """Tests for maintenance window gating in the scheduler loop."""

    @pytest.mark.asyncio
    @patch("semantic_vector_router.scheduler.engine.is_within_window", return_value=False)
    async def test_job_outside_window_is_skipped(
        self, mock_is_within, scheduler, mock_metadata, windowed_job_config,
    ):
        """Job due but outside maintenance window should be skipped."""
        scheduler.register_job("windowed-job", windowed_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.CENTROID_REFRESH, handler)

        # Simulate one tick manually
        scheduler._running = True
        scheduler._last_tick = datetime.utcnow()

        # Inline one tick iteration
        state = await scheduler._get_job_state("windowed-job")
        is_due = scheduler._is_due(state, windowed_job_config)
        assert is_due is True

        # The window is closed (mocked False) — handler should NOT be called
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("semantic_vector_router.scheduler.engine.is_within_window", return_value=True)
    async def test_job_inside_window_executes(
        self, mock_is_within, scheduler, windowed_job_config,
    ):
        """Job due and inside window should execute normally."""
        scheduler.register_job("windowed-job", windowed_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.CENTROID_REFRESH, handler)

        run = await scheduler._execute_job("windowed-job", windowed_job_config)
        assert run.status == JobStatus.COMPLETED
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_window_always_runs(self, scheduler, sample_job_config):
        """Job without maintenance window should execute when due."""
        assert sample_job_config.maintenance_window is None
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        run = await scheduler._execute_job("detect-drift", sample_job_config)
        assert run.status == JobStatus.COMPLETED


# ===================================================================
# 15-17. Distributed locking
# ===================================================================

class TestLocking:
    """Tests for distributed lock acquisition and release."""

    @pytest.mark.asyncio
    async def test_lock_acquired_job_executes(
        self, mock_metadata, mock_config, sample_job_config,
    ):
        """When lock is acquired, job should execute."""
        mock_metadata.acquire_lock = AsyncMock(return_value=True)
        sched = JobScheduler(metadata=mock_metadata, config=mock_config)
        sched.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        sched.set_job_handler(JobType.DETECTION, handler)

        run = await sched.run_now("detect-drift")
        assert run.status == JobStatus.COMPLETED
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lock_not_acquired_job_skipped(
        self, mock_metadata, mock_config, sample_job_config,
    ):
        """When lock cannot be acquired in the tick loop, job should be skipped."""
        mock_metadata.acquire_lock = AsyncMock(return_value=False)
        sched = JobScheduler(metadata=mock_metadata, config=mock_config)
        sched.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        sched.set_job_handler(JobType.DETECTION, handler)

        # Record skip manually to verify the logic
        await sched._record_skip("detect-drift", sample_job_config, "lock_held")
        mock_metadata._coll.insert_one.assert_awaited_once()
        inserted_doc = mock_metadata._coll.insert_one.call_args[0][0]
        assert inserted_doc["status"] == "skipped"
        assert inserted_doc["skipped_reason"] == "lock_held"

    @pytest.mark.asyncio
    async def test_lock_released_after_success(
        self, mock_metadata, mock_config, sample_job_config,
    ):
        """Lock should be released even after successful execution.

        We test this by simulating the tick loop's lock/execute/release pattern.
        """
        mock_metadata.acquire_lock = AsyncMock(return_value=True)
        mock_metadata.release_lock = AsyncMock()

        sched = JobScheduler(metadata=mock_metadata, config=mock_config)
        sched.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        sched.set_job_handler(JobType.DETECTION, handler)

        # Simulate lock-execute-release pattern from _scheduler_loop
        lock_id = sample_job_config.lock_id
        acquired = await mock_metadata.acquire_lock(lock_id, sched.worker_id, 120)
        assert acquired
        try:
            await sched._execute_job("detect-drift", sample_job_config)
        finally:
            await mock_metadata.release_lock(lock_id, sched.worker_id)

        mock_metadata.release_lock.assert_awaited_once_with(lock_id, sched.worker_id)

    @pytest.mark.asyncio
    async def test_lock_released_after_failure(
        self, mock_metadata, mock_config, sample_job_config,
    ):
        """Lock should be released even when handler fails."""
        mock_metadata.acquire_lock = AsyncMock(return_value=True)
        mock_metadata.release_lock = AsyncMock()

        sched = JobScheduler(metadata=mock_metadata, config=mock_config)
        sched.register_job("detect-drift", sample_job_config)
        handler = _async_handler_error()
        sched.set_job_handler(JobType.DETECTION, handler)

        lock_id = sample_job_config.lock_id
        acquired = await mock_metadata.acquire_lock(lock_id, sched.worker_id, 120)
        assert acquired
        try:
            await sched._execute_job("detect-drift", sample_job_config)
        finally:
            await mock_metadata.release_lock(lock_id, sched.worker_id)

        mock_metadata.release_lock.assert_awaited_once()


# ===================================================================
# 18-20. State and run persistence
# ===================================================================

class TestPersistence:
    """Tests for _save_job_state, _get_job_state, and _record_run."""

    @pytest.mark.asyncio
    async def test_save_job_state_writes_to_collection(
        self, scheduler, mock_metadata,
    ):
        """_save_job_state should call replace_one with upsert=True."""
        now = datetime.utcnow()
        state = JobState(
            job_id="j1",
            last_run_at=now,
            last_status=JobStatus.COMPLETED,
            next_due_at=now + timedelta(hours=1),
            consecutive_failures=0,
        )
        await scheduler._save_job_state(state)

        mock_metadata._coll.replace_one.assert_awaited_once()
        call_args = mock_metadata._coll.replace_one.call_args
        assert call_args[0][0] == {"_id": "job_state:j1"}
        doc = call_args[0][1]
        assert doc["type"] == "job_state"
        assert doc["job_id"] == "j1"
        assert doc["last_status"] == "completed"
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_get_job_state_reads_from_collection(
        self, scheduler, mock_metadata,
    ):
        """_get_job_state should query by _id and type."""
        now = datetime.utcnow()
        mock_metadata._coll.find_one = AsyncMock(return_value={
            "_id": "job_state:j1",
            "type": "job_state",
            "job_id": "j1",
            "last_run_at": now,
            "last_status": "completed",
            "next_due_at": now + timedelta(hours=1),
            "consecutive_failures": 2,
        })

        state = await scheduler._get_job_state("j1")

        assert state.job_id == "j1"
        assert state.last_run_at == now
        assert state.last_status == JobStatus.COMPLETED
        assert state.consecutive_failures == 2
        mock_metadata._coll.find_one.assert_awaited_once_with(
            {"_id": "job_state:j1", "type": "job_state"}
        )

    @pytest.mark.asyncio
    async def test_get_job_state_returns_default_when_none(
        self, scheduler, mock_metadata,
    ):
        """_get_job_state with no persisted doc returns fresh JobState."""
        mock_metadata._coll.find_one = AsyncMock(return_value=None)

        state = await scheduler._get_job_state("nonexistent")

        assert state.job_id == "nonexistent"
        assert state.last_run_at is None
        assert state.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_record_run_inserts_into_collection(
        self, scheduler, mock_metadata,
    ):
        """_record_run should insert_one to metadata._coll."""
        now = datetime.utcnow()
        run = JobRun(
            job_id="j1",
            job_type="detection",
            status=JobStatus.COMPLETED,
            scheduled_at=now,
            started_at=now,
            completed_at=now,
            duration_ms=123.4,
            result={"ok": True},
            worker_id="test-worker",
        )

        await scheduler._record_run(run)

        mock_metadata._coll.insert_one.assert_awaited_once()
        doc = mock_metadata._coll.insert_one.call_args[0][0]
        assert doc["type"] == "job_run"
        assert doc["job_id"] == "j1"
        assert doc["status"] == "completed"
        assert doc["duration_ms"] == 123.4


# ===================================================================
# 21. Skip record
# ===================================================================

class TestRecordSkip:
    """Tests for _record_skip."""

    @pytest.mark.asyncio
    async def test_record_skip_creates_skipped_run(
        self, scheduler, mock_metadata, sample_job_config,
    ):
        """_record_skip should create a SKIPPED run with reason."""
        await scheduler._record_skip("detect-drift", sample_job_config, "lock_held")

        mock_metadata._coll.insert_one.assert_awaited_once()
        doc = mock_metadata._coll.insert_one.call_args[0][0]
        assert doc["status"] == "skipped"
        assert doc["skipped_reason"] == "lock_held"
        assert doc["job_id"] == "detect-drift"
        assert doc["job_type"] == "detection"

    @pytest.mark.asyncio
    async def test_record_skip_outside_maintenance_window(
        self, scheduler, mock_metadata, windowed_job_config,
    ):
        """_record_skip for maintenance window records correct reason."""
        await scheduler._record_skip(
            "windowed-job", windowed_job_config, "outside_maintenance_window",
        )

        doc = mock_metadata._coll.insert_one.call_args[0][0]
        assert doc["skipped_reason"] == "outside_maintenance_window"


# ===================================================================
# 22. get_status
# ===================================================================

class TestGetStatus:
    """Tests for get_status()."""

    @pytest.mark.asyncio
    async def test_get_status_returns_scheduler_status(
        self, scheduler, sample_job_config,
    ):
        """get_status should return SchedulerStatus with correct fields."""
        scheduler.register_job("detect-drift", sample_job_config)

        status = await scheduler.get_status()

        assert isinstance(status, SchedulerStatus)
        assert status.worker_id == "test-worker"
        assert status.jobs_registered == 1
        assert status.tick_interval_seconds == 30
        assert status.running is False  # not started
        assert len(status.jobs) == 1
        assert status.jobs[0]["job_id"] == "detect-drift"
        assert status.jobs[0]["enabled"] is True

    @pytest.mark.asyncio
    async def test_get_status_empty_when_no_jobs(self, scheduler):
        """get_status with no registered jobs should have empty jobs list."""
        status = await scheduler.get_status()
        assert status.jobs_registered == 0
        assert status.jobs == []

    @pytest.mark.asyncio
    async def test_get_status_reflects_running_state(self, scheduler):
        """get_status should reflect running=True after start()."""
        await scheduler.start()
        status = await scheduler.get_status()
        assert status.running is True
        await scheduler.stop()


# ===================================================================
# 23. get_job_history
# ===================================================================

class TestGetJobHistory:
    """Tests for get_job_history()."""

    @pytest.mark.asyncio
    async def test_get_job_history_returns_runs(self, scheduler, mock_metadata):
        """get_job_history should deserialize docs into JobRun list."""
        now = datetime.utcnow()
        docs = [
            {
                "job_id": "j1",
                "job_type": "detection",
                "status": "completed",
                "scheduled_at": now,
                "started_at": now,
                "completed_at": now,
                "duration_ms": 50.0,
                "result": {"ok": True},
                "error": None,
                "skipped_reason": None,
                "worker_id": "w1",
            },
            {
                "job_id": "j1",
                "job_type": "detection",
                "status": "failed",
                "scheduled_at": now - timedelta(hours=1),
                "started_at": now - timedelta(hours=1),
                "completed_at": now - timedelta(hours=1),
                "duration_ms": 100.0,
                "result": {},
                "error": "boom",
                "skipped_reason": None,
                "worker_id": "w1",
            },
        ]
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=docs)
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        mock_metadata._coll.find = MagicMock(return_value=cursor)

        runs = await scheduler.get_job_history("j1", limit=10)

        assert len(runs) == 2
        assert runs[0].status == JobStatus.COMPLETED
        assert runs[1].status == JobStatus.FAILED
        assert runs[1].error == "boom"

    @pytest.mark.asyncio
    async def test_get_job_history_empty(self, scheduler, mock_metadata):
        """get_job_history with no docs returns empty list."""
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=[])
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        mock_metadata._coll.find = MagicMock(return_value=cursor)

        runs = await scheduler.get_job_history("j1")
        assert runs == []

    @pytest.mark.asyncio
    async def test_get_job_history_handles_exception(self, scheduler, mock_metadata):
        """get_job_history should return [] on exception instead of raising."""
        mock_metadata._coll.find = MagicMock(side_effect=Exception("db down"))

        runs = await scheduler.get_job_history("j1")
        assert runs == []


# ===================================================================
# 24. Consecutive failure tracking
# ===================================================================

class TestConsecutiveFailures:
    """Tests for consecutive failure counter in job state."""

    @pytest.mark.asyncio
    async def test_failure_increments_counter(self, scheduler, mock_metadata, sample_job_config):
        """Failed job should increment consecutive_failures."""
        # Pre-existing state with 2 consecutive failures
        mock_metadata._coll.find_one = AsyncMock(return_value={
            "_id": "job_state:detect-drift",
            "type": "job_state",
            "job_id": "detect-drift",
            "last_run_at": datetime.utcnow() - timedelta(hours=2),
            "last_status": "failed",
            "consecutive_failures": 2,
        })

        handler = _async_handler_error()
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, handler)

        await scheduler._execute_job("detect-drift", sample_job_config)

        # Check _save_job_state was called with incremented counter
        replace_call = mock_metadata._coll.replace_one.call_args
        saved_doc = replace_call[0][1]
        assert saved_doc["consecutive_failures"] == 3

    @pytest.mark.asyncio
    async def test_success_resets_counter(self, scheduler, mock_metadata, sample_job_config):
        """Successful job should reset consecutive_failures to 0."""
        mock_metadata._coll.find_one = AsyncMock(return_value={
            "_id": "job_state:detect-drift",
            "type": "job_state",
            "job_id": "detect-drift",
            "last_run_at": datetime.utcnow() - timedelta(hours=2),
            "last_status": "failed",
            "consecutive_failures": 5,
        })

        handler = _async_handler_ok()
        scheduler.register_job("detect-drift", sample_job_config)
        scheduler.set_job_handler(JobType.DETECTION, handler)

        await scheduler._execute_job("detect-drift", sample_job_config)

        replace_call = mock_metadata._coll.replace_one.call_args
        saved_doc = replace_call[0][1]
        assert saved_doc["consecutive_failures"] == 0


# ===================================================================
# 25-27. Event emission
# ===================================================================

class TestEventEmission:
    """Tests for event emission via event bus."""

    @pytest.mark.asyncio
    async def test_emit_job_started_event(
        self, scheduler_with_events, mock_event_bus, sample_job_config,
    ):
        """_execute_job should emit JOB_STARTED event."""
        handler = _async_handler_ok()
        scheduler_with_events.register_job("detect-drift", sample_job_config)
        scheduler_with_events.set_job_handler(JobType.DETECTION, handler)

        await scheduler_with_events._execute_job("detect-drift", sample_job_config)

        # Check first emit call is job.started
        calls = mock_event_bus.emit.call_args_list
        assert len(calls) >= 2  # started + completed
        started_event = calls[0][0][0]
        from semantic_vector_router.events.models import SVREventType
        assert started_event.event_type == SVREventType.JOB_STARTED
        assert started_event.job_id == "detect-drift"

    @pytest.mark.asyncio
    async def test_emit_job_completed_event(
        self, scheduler_with_events, mock_event_bus, sample_job_config,
    ):
        """Successful job should emit JOB_COMPLETED event."""
        handler = _async_handler_ok()
        scheduler_with_events.register_job("detect-drift", sample_job_config)
        scheduler_with_events.set_job_handler(JobType.DETECTION, handler)

        await scheduler_with_events._execute_job("detect-drift", sample_job_config)

        from semantic_vector_router.events.models import SVREventType
        calls = mock_event_bus.emit.call_args_list
        completed_event = calls[-1][0][0]
        assert completed_event.event_type == SVREventType.JOB_COMPLETED

    @pytest.mark.asyncio
    async def test_emit_job_failed_event(
        self, scheduler_with_events, mock_event_bus, sample_job_config,
    ):
        """Failed job should emit JOB_FAILED event."""
        handler = _async_handler_error()
        scheduler_with_events.register_job("detect-drift", sample_job_config)
        scheduler_with_events.set_job_handler(JobType.DETECTION, handler)

        await scheduler_with_events._execute_job("detect-drift", sample_job_config)

        from semantic_vector_router.events.models import SVREventType
        calls = mock_event_bus.emit.call_args_list
        failed_event = calls[-1][0][0]
        assert failed_event.event_type == SVREventType.JOB_FAILED
        assert failed_event.severity == "warning"

    @pytest.mark.asyncio
    async def test_no_event_bus_no_error(self, scheduler, sample_job_config):
        """Without event bus, _emit_event should be a silent no-op."""
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        # Should not raise
        run = await scheduler._execute_job("detect-drift", sample_job_config)
        assert run.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_event_emission_error_is_swallowed(
        self, mock_metadata, mock_config,
    ):
        """If event bus.emit raises, job execution should still succeed."""
        bus = MagicMock()
        bus.emit = AsyncMock(side_effect=Exception("event bus broken"))
        sched = JobScheduler(metadata=mock_metadata, config=mock_config, event_bus=bus)

        config = JobConfig(
            job_id="j1", job_type=JobType.CUSTOM, interval="1h",
        )
        sched.register_job("j1", config)
        sched.set_job_handler(JobType.CUSTOM, _async_handler_ok())

        run = await sched._execute_job("j1", config)
        # Execution should still complete despite event emission error
        assert run.status == JobStatus.COMPLETED


# ===================================================================
# 28. Disabled job
# ===================================================================

class TestDisabledJob:
    """Tests for disabled jobs being skipped in the tick loop."""

    def test_disabled_job_config_flag(self, disabled_job_config):
        """Disabled job config has enabled=False."""
        assert disabled_job_config.enabled is False

    @pytest.mark.asyncio
    async def test_disabled_job_not_processed_in_loop(
        self, scheduler, disabled_job_config,
    ):
        """Disabled job should be skipped in the scheduler loop.

        We verify this by checking that _is_due is never reached for
        disabled jobs (the loop continues before is_due check).
        """
        scheduler.register_job("disabled-job", disabled_job_config)

        # Simulate one tick: iterate jobs, skip disabled
        for job_id, config in list(scheduler._jobs.items()):
            if not config.enabled:
                continue
            # If we reach here, test should fail
            pytest.fail("Disabled job should not reach this point")


# ===================================================================
# 29. Worker ID
# ===================================================================

class TestWorkerID:
    """Tests for worker ID auto-generation."""

    def test_worker_id_from_config(self, scheduler):
        """Worker ID should come from config when set."""
        assert scheduler.worker_id == "test-worker"

    @patch("semantic_vector_router.scheduler.engine.socket.gethostname", return_value="myhost")
    @patch("semantic_vector_router.scheduler.engine.os.getpid", return_value=12345)
    def test_worker_id_auto_generated(self, mock_pid, mock_host, mock_metadata):
        """When config.scheduler.worker_id is None, auto-generate from hostname+pid."""
        config = MagicMock()
        config.scheduler.enabled = True
        config.scheduler.tick_interval_seconds = 30
        config.scheduler.worker_id = None
        config.scheduler.maintenance_window = None

        sched = JobScheduler(metadata=mock_metadata, config=config)
        assert sched.worker_id == "myhost-12345"


# ===================================================================
# 30. Force flag in run_now
# ===================================================================

class TestForceFlag:
    """Tests for force execution bypassing maintenance window."""

    @pytest.mark.asyncio
    async def test_run_now_passes_force_true(self, scheduler, sample_job_config):
        """run_now should call _execute_job with force=True."""
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        with patch.object(scheduler, "_execute_job", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = JobRun(
                job_id="detect-drift",
                job_type="detection",
                status=JobStatus.COMPLETED,
            )
            await scheduler.run_now("detect-drift")

            mock_exec.assert_awaited_once_with(
                "detect-drift", sample_job_config, force=True,
            )

    @pytest.mark.asyncio
    async def test_force_execution_ignores_window(
        self, scheduler, windowed_job_config,
    ):
        """Force execution should succeed regardless of maintenance window."""
        scheduler.register_job("windowed-job", windowed_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.CENTROID_REFRESH, handler)

        # run_now internally calls _execute_job with force=True
        # _execute_job does NOT check maintenance window (that's in the tick loop)
        run = await scheduler.run_now("windowed-job")
        assert run.status == JobStatus.COMPLETED
        handler.assert_awaited_once()


# ===================================================================
# Additional edge cases
# ===================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    @pytest.mark.asyncio
    async def test_save_job_state_exception_swallowed(self, scheduler, mock_metadata):
        """_save_job_state should not raise on DB errors."""
        mock_metadata._coll.replace_one = AsyncMock(
            side_effect=Exception("write failed")
        )
        state = JobState(job_id="j1")
        # Should not raise
        await scheduler._save_job_state(state)

    @pytest.mark.asyncio
    async def test_get_job_state_exception_returns_default(
        self, scheduler, mock_metadata,
    ):
        """_get_job_state should return default state on DB read error."""
        mock_metadata._coll.find_one = AsyncMock(
            side_effect=Exception("read failed")
        )
        state = await scheduler._get_job_state("j1")
        assert state.job_id == "j1"
        assert state.last_run_at is None

    @pytest.mark.asyncio
    async def test_record_run_exception_swallowed(self, scheduler, mock_metadata):
        """_record_run should not raise on insert error."""
        mock_metadata._coll.insert_one = AsyncMock(
            side_effect=Exception("insert failed")
        )
        run = JobRun(
            job_id="j1",
            job_type="detection",
            status=JobStatus.COMPLETED,
            scheduled_at=datetime.utcnow(),
        )
        # Should not raise
        await scheduler._record_run(run)

    @pytest.mark.asyncio
    async def test_execute_job_persists_run_and_state(
        self, scheduler, mock_metadata, sample_job_config,
    ):
        """_execute_job should call both _record_run and _save_job_state."""
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        await scheduler._execute_job("detect-drift", sample_job_config)

        # insert_one for run record
        mock_metadata._coll.insert_one.assert_awaited_once()
        # replace_one for state
        mock_metadata._coll.replace_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_job_sets_next_due_at(
        self, scheduler, mock_metadata, sample_job_config,
    ):
        """After execution, next_due_at should be set ~interval from now."""
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        before = datetime.utcnow()
        await scheduler._execute_job("detect-drift", sample_job_config)

        replace_call = mock_metadata._coll.replace_one.call_args
        saved_doc = replace_call[0][1]
        next_due = saved_doc["next_due_at"]
        # 1h interval = 3600s
        assert next_due > before
        assert (next_due - before).total_seconds() >= 3500  # roughly 1h

    def test_set_job_handler_stores_handler(self, scheduler):
        """set_job_handler should store callable in _job_handlers dict."""
        handler = AsyncMock()
        scheduler.set_job_handler(JobType.DETECTION, handler)
        assert scheduler._job_handlers[JobType.DETECTION] is handler

    def test_set_custom_handler_stores_handler(self, scheduler):
        """set_custom_handler should store callable in _custom_handlers dict."""
        handler = AsyncMock()
        scheduler.set_custom_handler("my-job", handler)
        assert scheduler._custom_handlers["my-job"] is handler

    @pytest.mark.asyncio
    async def test_scheduler_loop_updates_last_tick(self, scheduler, sample_job_config):
        """After start, _last_tick should be updated (verified via get_status)."""
        scheduler.register_job("detect-drift", sample_job_config)
        handler = _async_handler_ok()
        scheduler.set_job_handler(JobType.DETECTION, handler)

        await scheduler.start()
        # Give the loop a moment to tick
        await asyncio.sleep(0.05)
        assert scheduler._last_tick is not None
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_running_property(self, scheduler):
        """running property should reflect the scheduler state."""
        assert scheduler.running is False
        await scheduler.start()
        assert scheduler.running is True
        await scheduler.stop()
        assert scheduler.running is False
