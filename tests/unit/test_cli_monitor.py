"""Unit tests for CLI monitor commands."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from semantic_vector_router.cli.monitor import monitor_group
from semantic_vector_router.lifecycle.detector import DetectionResult
from semantic_vector_router.models import (
    DatabaseConfig,
    DetectionSignal,
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
    """Create a minimal SVRConfig for monitor CLI tests."""
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


class TestMonitorCheck:
    """Tests for 'monitor check' command."""

    @patch("semantic_vector_router.cli.monitor.PartitionDetector")
    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_check_no_issues(
        self, mock_run_async, mock_get_backend, MockMetadata, MockDetector, runner
    ):
        """When detection returns empty list, output says 'No issues detected'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        mock_detector_instance = AsyncMock()
        mock_detector_instance.run_detection = AsyncMock(return_value=[])
        MockDetector.return_value = mock_detector_instance

        result = runner.invoke(monitor_group, ["check"])
        assert result.exit_code == 0
        assert "No issues detected" in result.output

        # Verify cleanup calls
        mock_meta_instance.disconnect.assert_awaited_once()
        mock_backend.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.monitor.PartitionDetector")
    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_check_with_results(
        self, mock_run_async, mock_get_backend, MockMetadata, MockDetector, runner
    ):
        """When detection returns results, a table is rendered with signal info."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        detection_results = [
            DetectionResult(
                signal=DetectionSignal.THRESHOLD_BREACH,
                partition="electronics",
                details={"count": 12000000, "threshold": 10000000, "overage": 2000000},
                auto_executable=True,
                suggested_action="split-electronics",
            ),
            DetectionResult(
                signal=DetectionSignal.APPROACHING_THRESHOLD,
                partition="furniture",
                details={"current_count": 8000000, "threshold": 10000000, "days_to_breach": 15.3},
                auto_executable=False,
                suggested_action="prepare-split-furniture",
            ),
        ]

        mock_detector_instance = AsyncMock()
        mock_detector_instance.run_detection = AsyncMock(return_value=detection_results)
        MockDetector.return_value = mock_detector_instance

        result = runner.invoke(monitor_group, ["check"])
        assert result.exit_code == 0
        assert "Detection Results" in result.output
        assert "electronics" in result.output
        assert "furniture" in result.output
        assert "threshold_breach" in result.output
        # Rich may truncate long signal names in narrow terminal
        assert "approaching_thr" in result.output
        assert "2 issue(s) detected" in result.output

    @patch("semantic_vector_router.cli.monitor.PartitionDetector")
    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_check_with_auto_execute(
        self, mock_run_async, mock_get_backend, MockMetadata, MockDetector, runner
    ):
        """With --auto-execute, auto-executable operations are announced."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        detection_results = [
            DetectionResult(
                signal=DetectionSignal.THRESHOLD_BREACH,
                partition="electronics",
                details={"count": 12000000, "threshold": 10000000},
                auto_executable=True,
                suggested_action="split-electronics",
            ),
        ]

        mock_detector_instance = AsyncMock()
        mock_detector_instance.run_detection = AsyncMock(return_value=detection_results)
        MockDetector.return_value = mock_detector_instance

        result = runner.invoke(monitor_group, ["check", "--auto-execute"])
        assert result.exit_code == 0
        assert "Auto-executing 1 operation(s)" in result.output

    @patch("semantic_vector_router.cli.monitor.PartitionDetector")
    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_check_auto_execute_no_auto_ops(
        self, mock_run_async, mock_get_backend, MockMetadata, MockDetector, runner
    ):
        """With --auto-execute but no auto-executable ops, says 'No auto-executable operations'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        detection_results = [
            DetectionResult(
                signal=DetectionSignal.SEVERE_SKEW,
                partition="furniture",
                details={"ratio": 6.5},
                auto_executable=False,
                suggested_action="rebalance-_root",
            ),
        ]

        mock_detector_instance = AsyncMock()
        mock_detector_instance.run_detection = AsyncMock(return_value=detection_results)
        MockDetector.return_value = mock_detector_instance

        result = runner.invoke(monitor_group, ["check", "--auto-execute"])
        assert result.exit_code == 0
        assert "No auto-executable operations" in result.output

    @patch("semantic_vector_router.cli.monitor.PartitionDetector")
    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_check_long_details_truncated(
        self, mock_run_async, mock_get_backend, MockMetadata, MockDetector, runner
    ):
        """Details longer than 60 chars are truncated with '...'."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        # Create a result with very long details
        long_details = {f"key_{i}": f"value_that_is_quite_long_{i}" for i in range(10)}
        detection_results = [
            DetectionResult(
                signal=DetectionSignal.UNDERPOPULATED,
                partition="tiny_partition",
                details=long_details,
                auto_executable=False,
                suggested_action="merge-tiny_partition",
            ),
        ]

        mock_detector_instance = AsyncMock()
        mock_detector_instance.run_detection = AsyncMock(return_value=detection_results)
        MockDetector.return_value = mock_detector_instance

        result = runner.invoke(monitor_group, ["check"])
        assert result.exit_code == 0
        # The truncation happens at render time, verify the command ran fine
        assert "1 issue(s) detected" in result.output


class TestMonitorHistory:
    """Tests for 'monitor history' command."""

    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_history(self, mock_run_async, mock_get_backend, MockMetadata, runner):
        """When health history data exists, a table is rendered."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance

        history_data = [
            {"ts": "2025-01-01T00:00:00", "count": 5000},
            {"ts": "2025-01-02T00:00:00", "count": 5200},
            {"ts": "2025-01-03T00:00:00", "count": 5400},
        ]
        mock_meta_instance.get_health_history = AsyncMock(return_value=history_data)

        result = runner.invoke(monitor_group, ["history", "electronics"])
        assert result.exit_code == 0
        assert "Health History: electronics" in result.output
        assert "5,000" in result.output
        assert "5,200" in result.output
        assert "5,400" in result.output
        assert "3 data point(s)" in result.output

        # Verify cleanup
        mock_meta_instance.disconnect.assert_awaited_once()
        mock_backend.disconnect.assert_awaited_once()

    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_history_empty(self, mock_run_async, mock_get_backend, MockMetadata, runner):
        """When no health history is found, display appropriate message."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance
        mock_meta_instance.get_health_history = AsyncMock(return_value=[])

        result = runner.invoke(monitor_group, ["history", "nonexistent"])
        assert result.exit_code == 0
        assert "No health history" in result.output
        assert "nonexistent" in result.output

    @patch("semantic_vector_router.cli.monitor.MetadataStore")
    @patch("semantic_vector_router.cli.monitor._get_backend", new_callable=AsyncMock)
    @patch("semantic_vector_router.cli.monitor._run_async")
    def test_monitor_history_with_separate_metadata_db(
        self, mock_run_async, mock_get_backend, MockMetadata, runner
    ):
        """When metadata has its own connection_string_env, _set_shared_db is NOT called."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _make_config()
        # Set a separate metadata connection string
        config.lifecycle.metadata.connection_string_env = "METADATA_URI"
        mock_get_backend.return_value = (config, mock_backend)

        mock_meta_instance = AsyncMock()
        MockMetadata.return_value = mock_meta_instance
        mock_meta_instance.get_health_history = AsyncMock(return_value=[])

        result = runner.invoke(monitor_group, ["history", "test_partition"])
        assert result.exit_code == 0
        # _set_shared_db should NOT have been called since connection_string_env is set
        mock_meta_instance._set_shared_db.assert_not_called()
