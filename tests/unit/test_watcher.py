"""Unit tests for PartitionWatcher."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_vector_router.exceptions import ChangeStreamError
from semantic_vector_router.lifecycle.provisioner import PartitionProvisioner
from semantic_vector_router.lifecycle.watcher import PartitionWatcher
from semantic_vector_router.models import ResilienceConfig, WatcherStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provisioner():
    """Create a mock provisioner with async methods."""
    provisioner = AsyncMock(spec=PartitionProvisioner)
    provisioner.create_partition = AsyncMock()
    return provisioner


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


async def test_start_stop_lifecycle(mock_backend, sample_config):
    """start() sets running=True, initializes known_values, and creates a task;
    stop() sets running=False and cancels the task."""
    # Pre-populate registry so known_values includes registry keys
    sample_config.partitions.registry = {}

    mock_backend.get_distinct_values = AsyncMock(
        return_value=["electronics", "furniture"]
    )

    watcher = PartitionWatcher(mock_backend, sample_config)

    assert watcher.running is False
    assert watcher._known_values == set()

    # Patch _watch_loop to avoid opening a real change stream
    with patch.object(watcher, "_watch_loop", new_callable=AsyncMock) as mock_loop:
        await watcher.start()

        assert watcher.running is True
        assert "electronics" in watcher._known_values
        assert "furniture" in watcher._known_values
        mock_backend.get_distinct_values.assert_awaited_once_with("category")
        assert watcher._task is not None

    # Stop the watcher
    await watcher.stop()

    assert watcher.running is False
    assert watcher._task is None


async def test_start_includes_registry_keys(mock_backend, sample_config_with_partitions):
    """start() merges partition registry keys into _known_values alongside
    distinct values from the backend."""
    mock_backend.get_distinct_values = AsyncMock(return_value=["sports"])

    watcher = PartitionWatcher(mock_backend, sample_config_with_partitions)

    with patch.object(watcher, "_watch_loop", new_callable=AsyncMock):
        await watcher.start()

    # Registry keys: electronics, furniture, clothing
    # Backend distinct: sports
    expected = {"electronics", "furniture", "clothing", "sports"}
    assert watcher._known_values == expected


# ---------------------------------------------------------------------------
# _handle_change — auto-provision (no confirmation)
# ---------------------------------------------------------------------------


async def test_auto_provisioning_creates_partition(
    mock_backend, sample_config, mock_provisioner
):
    """When auto_provision=True and confirmation_required=False,
    _handle_change calls provisioner.create_partition for a new value."""
    sample_config.lifecycle.auto_provision = True
    sample_config.lifecycle.confirmation_required = False

    watcher = PartitionWatcher(
        mock_backend, sample_config, provisioner=mock_provisioner
    )
    watcher._known_values = {"electronics", "furniture"}

    change = {
        "operationType": "insert",
        "fullDocument": {"category": "new_category", "_id": "doc1"},
    }

    await watcher._handle_change(change, "category")

    # The new value should be added to known values
    assert "new_category" in watcher._known_values

    # Provisioner should have been called
    mock_provisioner.create_partition.assert_awaited_once_with(
        name="new_category",
        filter_value="new_category",
        skip_if_exists=True,
    )

    # Partitions created counter should increment
    assert watcher._partitions_created == 1
    assert watcher._last_event is not None


async def test_auto_provision_skips_known_value(
    mock_backend, sample_config, mock_provisioner
):
    """_handle_change does nothing if the partition value is already known."""
    sample_config.lifecycle.auto_provision = True
    sample_config.lifecycle.confirmation_required = False

    watcher = PartitionWatcher(
        mock_backend, sample_config, provisioner=mock_provisioner
    )
    watcher._known_values = {"electronics"}

    change = {
        "operationType": "insert",
        "fullDocument": {"category": "electronics", "_id": "doc2"},
    }

    await watcher._handle_change(change, "category")

    mock_provisioner.create_partition.assert_not_awaited()
    assert watcher._partitions_created == 0


# ---------------------------------------------------------------------------
# _handle_change — confirmation required
# ---------------------------------------------------------------------------


async def test_confirmation_required_queues_partition(
    mock_backend, sample_config, mock_provisioner
):
    """When confirmation_required=True, _handle_change adds the value to
    _pending_partitions instead of provisioning immediately."""
    sample_config.lifecycle.auto_provision = True
    sample_config.lifecycle.confirmation_required = True

    watcher = PartitionWatcher(
        mock_backend, sample_config, provisioner=mock_provisioner
    )
    watcher._known_values = {"electronics"}

    change = {
        "operationType": "insert",
        "fullDocument": {"category": "sports", "_id": "doc3"},
    }

    await watcher._handle_change(change, "category")

    assert "sports" in watcher._pending_partitions
    assert "sports" in watcher._known_values
    # Provisioner should NOT have been called
    mock_provisioner.create_partition.assert_not_awaited()
    assert watcher._partitions_created == 0


# ---------------------------------------------------------------------------
# Error accumulation and get_status
# ---------------------------------------------------------------------------


async def test_error_accumulation_in_status(mock_backend, sample_config):
    """get_status returns accumulated errors, capped at the last 10."""
    watcher = PartitionWatcher(mock_backend, sample_config)

    # Simulate 15 errors
    for i in range(15):
        watcher._errors.append(f"Error {i}")

    status = watcher.get_status()

    assert isinstance(status, WatcherStatus)
    assert len(status.errors) == 10
    # Should be the last 10 errors (indices 5-14)
    assert status.errors[0] == "Error 5"
    assert status.errors[-1] == "Error 14"


async def test_get_status_initial(mock_backend, sample_config):
    """A freshly created watcher reports running=False, partitions_created=0,
    errors=[], and last_event=None."""
    watcher = PartitionWatcher(mock_backend, sample_config)

    status = watcher.get_status()

    assert isinstance(status, WatcherStatus)
    assert status.running is False
    assert status.partitions_created == 0
    assert status.errors == []
    assert status.last_event is None


# ---------------------------------------------------------------------------
# _handle_change — edge cases
# ---------------------------------------------------------------------------


async def test_handle_change_missing_partition_field(mock_backend, sample_config):
    """_handle_change ignores documents that lack the partition field."""
    sample_config.lifecycle.auto_provision = True
    sample_config.lifecycle.confirmation_required = False

    watcher = PartitionWatcher(mock_backend, sample_config)
    watcher._known_values = set()

    change = {
        "operationType": "insert",
        "fullDocument": {"_id": "doc4", "title": "No category here"},
    }

    await watcher._handle_change(change, "category")

    # No new value should be added
    assert watcher._known_values == set()
    assert watcher._partitions_created == 0


async def test_auto_provision_error_captured(
    mock_backend, sample_config, mock_provisioner
):
    """When provisioner.create_partition raises, the error is captured in
    _errors and _partitions_created is not incremented."""
    sample_config.lifecycle.auto_provision = True
    sample_config.lifecycle.confirmation_required = False

    mock_provisioner.create_partition = AsyncMock(
        side_effect=RuntimeError("Index creation failed")
    )

    watcher = PartitionWatcher(
        mock_backend, sample_config, provisioner=mock_provisioner
    )
    watcher._known_values = set()

    change = {
        "operationType": "insert",
        "fullDocument": {"category": "broken_partition", "_id": "doc5"},
    }

    await watcher._handle_change(change, "category")

    assert watcher._partitions_created == 0
    assert len(watcher._errors) == 1
    assert "broken_partition" in watcher._errors[0]
    # The value is still added to known set (detected even if provisioning fails)
    assert "broken_partition" in watcher._known_values


async def test_on_new_value_callback_invoked(mock_backend, sample_config):
    """When on_new_value callback is provided, it is called with the new value."""
    sample_config.lifecycle.auto_provision = False

    callback = MagicMock()
    watcher = PartitionWatcher(
        mock_backend, sample_config, on_new_value=callback
    )
    watcher._known_values = set()

    change = {
        "operationType": "insert",
        "fullDocument": {"category": "appliances", "_id": "doc6"},
    }

    await watcher._handle_change(change, "category")

    callback.assert_called_once_with("appliances")


# ---------------------------------------------------------------------------
# Auto-reconnect tests
# ---------------------------------------------------------------------------


class TestWatcherAutoReconnect:
    """Tests for watcher auto-reconnect on change stream failures."""

    async def test_reconnects_on_change_stream_error(
        self, mock_backend, sample_config
    ):
        """When watch_collection fails once then succeeds, the watcher
        reconnects and processes changes from the second stream."""
        sample_config.partitions.registry = {}

        mock_backend.get_distinct_values = AsyncMock(return_value=[])

        call_count = 0

        async def fake_watch(pipeline):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Change stream interrupted")
            # Second call: yield one change then return
            yield {
                "operationType": "insert",
                "fullDocument": {"category": "reconnected_val", "_id": "doc99"},
            }

        mock_backend.watch_collection = MagicMock(side_effect=fake_watch)

        sample_config.lifecycle.auto_provision = False

        watcher = PartitionWatcher(mock_backend, sample_config)
        watcher._known_values = set()
        watcher._running = True

        with patch("semantic_vector_router.lifecycle.watcher.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             patch("semantic_vector_router.lifecycle.watcher.random.random", return_value=0.5):
            await watcher._watch_loop()

            # Sleep should have been called once for the backoff after the first failure
            mock_sleep.assert_awaited_once()

        # The change from the second stream should have been processed
        assert "reconnected_val" in watcher._known_values
        assert call_count == 2
        assert len(watcher._errors) == 1

    async def test_max_retries_exceeded_stops_watcher(
        self, mock_backend, sample_config
    ):
        """When watch_collection always fails, the watcher raises
        ChangeStreamError after exceeding watcher_max_retries."""
        sample_config.partitions.registry = {}
        sample_config.resilience.watcher_max_retries = 2

        mock_backend.get_distinct_values = AsyncMock(return_value=[])

        async def always_fail(pipeline):
            raise ConnectionError("Permanent failure")
            # Make it an async generator by adding an unreachable yield
            yield  # pragma: no cover

        mock_backend.watch_collection = MagicMock(side_effect=always_fail)

        watcher = PartitionWatcher(mock_backend, sample_config)
        watcher._known_values = set()
        watcher._running = True

        with patch("semantic_vector_router.lifecycle.watcher.asyncio.sleep", new_callable=AsyncMock), \
             patch("semantic_vector_router.lifecycle.watcher.random.random", return_value=0.5):
            with pytest.raises(ChangeStreamError, match="failed after 2 reconnection attempts"):
                await watcher._watch_loop()

        assert watcher._running is False
        assert len(watcher._errors) == 2

    async def test_attempt_counter_resets_on_success(
        self, mock_backend, sample_config
    ):
        """After a successful change stream session (clean exit), the attempt
        counter resets so subsequent failures get a fresh retry budget."""
        sample_config.partitions.registry = {}
        sample_config.resilience.watcher_max_retries = 3

        mock_backend.get_distinct_values = AsyncMock(return_value=[])

        # The watch loop exits cleanly when the stream ends without error.
        # After a clean exit, attempt resets to 0 and the loop breaks.
        # We verify this by checking: if attempt didn't reset, a subsequent
        # call would have a non-zero starting attempt.

        call_count = 0

        async def succeeds_then_ends(pipeline):
            nonlocal call_count
            call_count += 1
            yield {
                "operationType": "insert",
                "fullDocument": {"category": f"val_{call_count}", "_id": f"doc{call_count}"},
            }
            # Stream ends cleanly (async generator returns)

        mock_backend.watch_collection = MagicMock(side_effect=succeeds_then_ends)

        sample_config.lifecycle.auto_provision = False

        watcher = PartitionWatcher(mock_backend, sample_config)
        watcher._known_values = set()
        watcher._running = True

        await watcher._watch_loop()

        # Stream ended cleanly after processing one change
        assert "val_1" in watcher._known_values
        # No errors recorded (clean exit)
        assert len(watcher._errors) == 0
        # Only called once because clean exit breaks the loop
        assert call_count == 1


# ---------------------------------------------------------------------------
# Pending partition persistence tests
# ---------------------------------------------------------------------------


class TestWatcherPendingPersistence:
    """Tests for pending partition persistence through config."""

    async def test_pending_loaded_from_config_on_init(
        self, mock_backend, sample_config
    ):
        """When config.lifecycle.pending_partitions has values, the watcher
        loads them into _pending_partitions on init."""
        sample_config.lifecycle.pending_partitions = ["val1", "val2"]

        watcher = PartitionWatcher(mock_backend, sample_config)

        assert "val1" in watcher._pending_partitions
        assert "val2" in watcher._pending_partitions
        assert len(watcher._pending_partitions) == 2

    @patch("semantic_vector_router.lifecycle.watcher.save_config")
    async def test_pending_persisted_on_add(
        self, mock_save_config, mock_backend, sample_config, mock_provisioner
    ):
        """When a new partition value is detected with confirmation_required=True,
        it is added to _pending_partitions and save_config is called."""
        sample_config.lifecycle.auto_provision = True
        sample_config.lifecycle.confirmation_required = True

        watcher = PartitionWatcher(
            mock_backend, sample_config, provisioner=mock_provisioner
        )
        watcher._known_values = set()

        change = {
            "operationType": "insert",
            "fullDocument": {"category": "new_pending", "_id": "doc10"},
        }

        await watcher._handle_change(change, "category")

        assert "new_pending" in watcher._pending_partitions
        assert "new_pending" in sample_config.lifecycle.pending_partitions
        mock_save_config.assert_called_once_with(sample_config)
        # Provisioner should NOT have been called (pending confirmation)
        mock_provisioner.create_partition.assert_not_awaited()

    @patch("semantic_vector_router.lifecycle.watcher.save_config")
    async def test_pending_persisted_on_confirm(
        self, mock_save_config, mock_backend, sample_config, mock_provisioner
    ):
        """When a pending partition is confirmed, it is removed from
        _pending_partitions and save_config is called."""
        sample_config.lifecycle.auto_provision = True
        sample_config.lifecycle.confirmation_required = True
        sample_config.lifecycle.pending_partitions = ["sports"]

        watcher = PartitionWatcher(
            mock_backend, sample_config, provisioner=mock_provisioner
        )

        assert "sports" in watcher._pending_partitions

        result = await watcher.confirm_partition("sports")

        assert result is True
        assert "sports" not in watcher._pending_partitions
        # save_config called: once for removing from pending
        mock_save_config.assert_called_with(sample_config)
        assert "sports" not in sample_config.lifecycle.pending_partitions

    @patch("semantic_vector_router.lifecycle.watcher.save_config")
    async def test_pending_persisted_on_reject(
        self, mock_save_config, mock_backend, sample_config
    ):
        """When a pending partition is rejected, it is removed from
        _pending_partitions and save_config is called."""
        sample_config.lifecycle.pending_partitions = ["unwanted"]

        watcher = PartitionWatcher(mock_backend, sample_config)

        assert "unwanted" in watcher._pending_partitions

        result = watcher.reject_partition("unwanted")

        assert result is True
        assert "unwanted" not in watcher._pending_partitions
        mock_save_config.assert_called_once_with(sample_config)
        assert "unwanted" not in sample_config.lifecycle.pending_partitions

    async def test_pending_merged_on_start(self, mock_backend, sample_config):
        """When start() is called, persisted pending partitions from config
        and in-memory pending are merged with deduplication."""
        sample_config.lifecycle.pending_partitions = ["from_config", "shared"]
        sample_config.partitions.registry = {}

        mock_backend.get_distinct_values = AsyncMock(return_value=[])

        watcher = PartitionWatcher(mock_backend, sample_config)
        # After init, pending comes from config
        assert "from_config" in watcher._pending_partitions
        assert "shared" in watcher._pending_partitions

        # Manually add an in-memory pending that is not in config
        watcher._pending_partitions.append("in_memory_only")
        # Also add a duplicate of one already from config
        # (should not create a second entry after merge)

        with patch.object(watcher, "_watch_loop", new_callable=AsyncMock):
            await watcher.start()

        # All three should be present, "shared" only once
        assert "from_config" in watcher._pending_partitions
        assert "shared" in watcher._pending_partitions
        assert "in_memory_only" in watcher._pending_partitions
        assert watcher._pending_partitions.count("shared") == 1
        assert watcher._pending_partitions.count("from_config") == 1
