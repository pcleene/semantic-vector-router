"""Unit tests for CLI repartition commands."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from semantic_vector_router.cli.repartition import repartition_group
from semantic_vector_router.models import (
    DatabaseConfig,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingProvider,
    PartitioningConfig,
    SVRConfig,
    VectorSearchConfig,
)


@pytest.fixture
def runner():
    return CliRunner()


def _make_config():
    """Create a minimal SVRConfig for repartition CLI tests."""
    return SVRConfig(
        database=DatabaseConfig(
            connection_string_env="MONGODB_URI",
            database="test_db",
            source_collection="products",
        ),
        partitioning=PartitioningConfig(field="category"),
        vector_search=VectorSearchConfig(dimensions=1536),
        embedding=EmbeddingConfig(
            mode=EmbeddingMode.BYOM,
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
            dimensions=1536,
        ),
    )


def _run_async_side_effect(coro):
    """Side effect for _run_async mock that actually executes the coroutine."""
    return asyncio.run(coro)


class TestRepartitionPending:
    """Tests for 'repartition pending' command."""

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_pending_empty(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When no pending operations exist, output says 'No pending operations'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance
        mock_meta_instance.list_operations = AsyncMock(return_value=[])

        result = runner.invoke(repartition_group, ["pending"])
        assert result.exit_code == 0
        assert "No pending operations" in result.output

        # Verify cleanup
        mock_meta_instance.disconnect.assert_awaited_once()
        mock_backend.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_pending_with_ops(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When pending operations exist, a table is rendered with operation details."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        pending_ops = [
            {
                "_id": "op:split-electronics-2025-01-15",
                "action": "split",
                "target_partition": "electronics",
                "signal": "threshold_breach",
                "created_at": "2025-01-15T10:00:00",
            },
            {
                "_id": "op:split-furniture-2025-01-16",
                "action": "split",
                "partition": "furniture",
                "signal": "threshold_breach",
                "created_at": "2025-01-16T12:00:00",
            },
        ]
        mock_meta_instance.list_operations = AsyncMock(return_value=pending_ops)

        result = runner.invoke(repartition_group, ["pending"])
        assert result.exit_code == 0
        assert "Pending Operations" in result.output
        # Rich may truncate long IDs in narrow terminal, check partial matches
        assert "op:split-electr" in result.output
        assert "electronics" in result.output
        assert "furniture" in result.output
        assert "split" in result.output
        assert "threshold_breach" in result.output
        assert "2 pending operation(s)" in result.output


class TestRepartitionExecute:
    """Tests for 'repartition execute' command."""

    @patch("semantic_vector_router.cli.repartition.RepartitionEngine")
    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_execute_success(
        self, mock_run_async, mock_get_backend, MockMetadata, MockEngine, runner
    ):
        """When engine returns True, output says 'completed successfully'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_engine_instance = AsyncMock()
        mock_engine_instance.execute_operation = AsyncMock(return_value=True)
        MockEngine.return_value = mock_engine_instance

        result = runner.invoke(repartition_group, ["execute", "op:split-electronics-2025-01-15"])
        assert result.exit_code == 0
        assert "op:split-electronics-2025-01-15" in result.output
        assert "completed successfully" in result.output

        mock_engine_instance.execute_operation.assert_awaited_once_with(
            "op:split-electronics-2025-01-15"
        )

    @patch("semantic_vector_router.cli.repartition.RepartitionEngine")
    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_execute_failure(
        self, mock_run_async, mock_get_backend, MockMetadata, MockEngine, runner
    ):
        """When engine returns False, output says 'failed'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_engine_instance = AsyncMock()
        mock_engine_instance.execute_operation = AsyncMock(return_value=False)
        MockEngine.return_value = mock_engine_instance

        result = runner.invoke(repartition_group, ["execute", "op:split-fail-123"])
        assert result.exit_code == 0
        assert "failed" in result.output.lower()

        # Verify cleanup
        mock_meta_instance.disconnect.assert_awaited_once()
        mock_backend.disconnect.assert_awaited_once()


class TestRepartitionStatus:
    """Tests for 'repartition status' command."""

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_status(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """Shows operation details in a panel with steps table."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        operation = {
            "_id": "op:split-electronics-2025-01-15",
            "status": "pending",
            "action": "split",
            "target_partition": "electronics",
            "created_at": "2025-01-15T10:00:00",
            "error": None,
            "steps": [
                {
                    "step": 1,
                    "action": "create_children",
                    "status": "done",
                    "started_at": "2025-01-15T10:01:00",
                    "completed_at": "2025-01-15T10:02:00",
                },
                {
                    "step": 2,
                    "action": "build_indexes",
                    "status": "in_progress",
                    "started_at": "2025-01-15T10:02:00",
                    "completed_at": None,
                },
                {
                    "step": 3,
                    "action": "wait_indexes",
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                },
            ],
        }
        mock_meta_instance.get_operation = AsyncMock(return_value=operation)

        result = runner.invoke(
            repartition_group, ["status", "op:split-electronics-2025-01-15"]
        )
        assert result.exit_code == 0
        assert "Operation Details" in result.output
        assert "op:split-electronics-2025-01-15" in result.output
        assert "electronics" in result.output
        assert "split" in result.output
        assert "create_children" in result.output
        assert "build_indexes" in result.output
        assert "wait_indexes" in result.output

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_status_not_found(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When operation is not found, display error message."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance
        mock_meta_instance.get_operation = AsyncMock(return_value=None)

        result = runner.invoke(repartition_group, ["status", "op:nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_status_failed_with_error(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When operation has an error, display it."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        operation = {
            "_id": "op:split-fail-123",
            "status": "failed",
            "action": "split",
            "target_partition": "electronics",
            "created_at": "2025-01-15T10:00:00",
            "error": "Index creation timed out",
            "steps": [
                {
                    "step": 1,
                    "action": "create_children",
                    "status": "done",
                    "started_at": "2025-01-15T10:01:00",
                    "completed_at": "2025-01-15T10:02:00",
                },
                {
                    "step": 2,
                    "action": "build_indexes",
                    "status": "failed",
                    "started_at": "2025-01-15T10:02:00",
                    "completed_at": None,
                },
            ],
        }
        mock_meta_instance.get_operation = AsyncMock(return_value=operation)

        result = runner.invoke(repartition_group, ["status", "op:split-fail-123"])
        assert result.exit_code == 0
        assert "Index creation timed out" in result.output

    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_status_no_steps(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When operation has no steps, only the panel is shown."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        operation = {
            "_id": "op:split-new-123",
            "status": "pending",
            "action": "split",
            "target_partition": "clothing",
            "created_at": "2025-01-20T08:00:00",
            "error": None,
            "steps": [],
        }
        mock_meta_instance.get_operation = AsyncMock(return_value=operation)

        result = runner.invoke(repartition_group, ["status", "op:split-new-123"])
        assert result.exit_code == 0
        assert "Operation Details" in result.output
        assert "clothing" in result.output


class TestRepartitionRollback:
    """Tests for 'repartition rollback' command."""

    @patch("semantic_vector_router.cli.repartition.RepartitionEngine")
    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_rollback(
        self, mock_run_async, mock_get_backend, MockMetadata, MockEngine, runner
    ):
        """With --yes flag, rollback completes successfully."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_engine_instance = AsyncMock()
        mock_engine_instance.rollback_operation = AsyncMock()
        MockEngine.return_value = mock_engine_instance

        result = runner.invoke(
            repartition_group, ["rollback", "op:split-electronics-2025-01-15", "--yes"]
        )
        assert result.exit_code == 0
        assert "rolled back successfully" in result.output

        mock_engine_instance.rollback_operation.assert_awaited_once_with(
            "op:split-electronics-2025-01-15"
        )

    def test_repartition_rollback_cancelled(self, runner):
        """Without --yes flag, user can cancel the rollback (input 'n')."""
        result = runner.invoke(
            repartition_group,
            ["rollback", "op:split-electronics-2025-01-15"],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    @patch("semantic_vector_router.cli.repartition.RepartitionEngine")
    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_rollback_confirmed_interactively(
        self, mock_run_async, mock_get_backend, MockMetadata, MockEngine, runner
    ):
        """Without --yes flag, user confirms with 'y' and rollback proceeds."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_engine_instance = AsyncMock()
        mock_engine_instance.rollback_operation = AsyncMock()
        MockEngine.return_value = mock_engine_instance

        result = runner.invoke(
            repartition_group,
            ["rollback", "op:split-electronics-2025-01-15"],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "rolled back successfully" in result.output

        mock_engine_instance.rollback_operation.assert_awaited_once()

    @patch("semantic_vector_router.cli.repartition.RepartitionEngine")
    @patch("semantic_vector_router.cli.repartition.MetadataStore")
    @patch("semantic_vector_router.cli.repartition._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.repartition._run_async")
    def test_repartition_rollback_cleanup_on_success(
        self, mock_run_async, mock_get_backend, MockMetadata, MockEngine, runner
    ):
        """Verify metadata and backend disconnect are called after rollback."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_engine_instance = AsyncMock()
        mock_engine_instance.rollback_operation = AsyncMock()
        MockEngine.return_value = mock_engine_instance

        result = runner.invoke(
            repartition_group, ["rollback", "op:split-test-123", "-y"]
        )
        assert result.exit_code == 0

        mock_meta_instance.disconnect.assert_awaited_once()
        mock_backend.disconnect.assert_awaited_once()
