"""Unit tests for CLI schedule and webhooks commands."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from semantic_vector_router.cli.schedule import schedule_group
from semantic_vector_router.cli.webhooks import webhooks_group


@pytest.fixture
def runner():
    return CliRunner()


def _mock_config(scheduler_enabled=False, webhooks=None, maintenance_window=None):
    """Create a mock SVRConfig for scheduler/webhook CLI tests."""
    config = MagicMock()
    config.scheduler.enabled = scheduler_enabled
    config.scheduler.tick_interval_seconds = 30
    config.scheduler.worker_id = None
    config.scheduler.maintenance_window = maintenance_window
    config.scheduler.detection_interval = "1h"
    config.scheduler.centroid_refresh_interval = "6h"
    config.scheduler.count_update_interval = "1h"
    config.scheduler.repartition_check_interval = "30m"
    config.scheduler.index_health_interval = "6h"
    config.events.enabled = True
    config.events.webhooks = webhooks or []
    return config


def _mock_maintenance_window():
    """Create a mock maintenance window config section."""
    mw = MagicMock()
    mw.allowed_days = ["monday", "tuesday", "wednesday"]
    mw.allowed_hours = {"start": 2, "end": 6}
    mw.timezone = "US/Eastern"
    return mw


def _mock_webhook(url="https://example.com/hook", events=None, enabled=True, secret=None):
    """Create a mock webhook config entry."""
    wh = MagicMock()
    wh.url = url
    wh.events = events or []
    wh.enabled = enabled
    wh.secret = secret
    return wh


def _run_async_side_effect(coro):
    """Side effect for _run_async mock that actually executes the coroutine."""
    return asyncio.run(coro)


# ─── Schedule CLI Tests ───────────────────────────────────────────────────────


class TestScheduleList:
    """Tests for 'schedule list' command."""

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_list_enabled_shows_table(self, mock_load_config, runner):
        """When scheduler is enabled, shows a table of all registered jobs."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=True)

        result = runner.invoke(schedule_group, ["list"])

        assert result.exit_code == 0
        assert "Scheduled Jobs" in result.output
        assert "detection" in result.output
        assert "centroid_refresh" in result.output
        assert "count_update" in result.output
        assert "repartition_check" in result.output
        assert "index_health" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_list_disabled_shows_warning(self, mock_load_config, runner):
        """When scheduler is disabled, shows a warning message."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=False)

        result = runner.invoke(schedule_group, ["list"])

        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_list_shows_intervals(self, mock_load_config, runner):
        """The table displays interval values for each job."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=True)

        result = runner.invoke(schedule_group, ["list"])

        assert result.exit_code == 0
        assert "1h" in result.output
        assert "6h" in result.output
        assert "30m" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_list_disabled_job_shows_disabled(self, mock_load_config, runner):
        """Jobs with interval=None are displayed as disabled."""
        config = _mock_config(scheduler_enabled=True)
        config.scheduler.repartition_check_interval = None
        mock_load_config.return_value = config

        result = runner.invoke(schedule_group, ["list"])

        assert result.exit_code == 0
        assert "disabled" in result.output


class TestScheduleStatus:
    """Tests for 'schedule status' command."""

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_status_shows_enabled(self, mock_load_config, runner):
        """Shows enabled/disabled and tick interval in status output."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=True)

        result = runner.invoke(schedule_group, ["status"])

        assert result.exit_code == 0
        assert "Scheduler Status" in result.output
        assert "yes" in result.output
        assert "30" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_status_shows_disabled(self, mock_load_config, runner):
        """When scheduler is disabled, status shows 'no'."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=False)

        result = runner.invoke(schedule_group, ["status"])

        assert result.exit_code == 0
        assert "no" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_status_with_maintenance_window(self, mock_load_config, runner):
        """When a maintenance window is configured, shows window details."""
        mw = _mock_maintenance_window()
        mock_load_config.return_value = _mock_config(
            scheduler_enabled=True, maintenance_window=mw,
        )

        result = runner.invoke(schedule_group, ["status"])

        assert result.exit_code == 0
        assert "Maintenance Window" in result.output
        assert "monday" in result.output
        assert "US/Eastern" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_status_no_maintenance_window(self, mock_load_config, runner):
        """When no maintenance window is configured, window section is absent."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=True)

        result = runner.invoke(schedule_group, ["status"])

        assert result.exit_code == 0
        assert "Maintenance Window" not in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_status_worker_id(self, mock_load_config, runner):
        """Worker ID is shown, or auto-generated label when None."""
        mock_load_config.return_value = _mock_config(scheduler_enabled=True)

        result = runner.invoke(schedule_group, ["status"])

        assert result.exit_code == 0
        assert "Worker ID" in result.output
        assert "auto-generated" in result.output


class TestScheduleWindow:
    """Tests for 'schedule window' command."""

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_window_shows_config(self, mock_load_config, runner):
        """Shows window configuration (days, hours, timezone)."""
        mw = _mock_maintenance_window()
        mock_load_config.return_value = _mock_config(maintenance_window=mw)

        result = runner.invoke(schedule_group, ["window"])

        assert result.exit_code == 0
        assert "Maintenance Window" in result.output
        assert "monday" in result.output
        assert "US/Eastern" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_window_no_window_configured(self, mock_load_config, runner):
        """When no maintenance window is configured, shows a dim info message."""
        mock_load_config.return_value = _mock_config(maintenance_window=None)

        result = runner.invoke(schedule_group, ["window"])

        assert result.exit_code == 0
        assert "No maintenance window" in result.output

    @patch("semantic_vector_router.scheduler.window.is_within_window")
    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_window_check_open(
        self, mock_load_config, mock_is_within, runner
    ):
        """--check flag calls is_within_window and shows OPEN when True."""
        mw = _mock_maintenance_window()
        mock_load_config.return_value = _mock_config(maintenance_window=mw)
        mock_is_within.return_value = True

        result = runner.invoke(schedule_group, ["window", "--check"])

        assert result.exit_code == 0
        assert "OPEN" in result.output

    @patch("semantic_vector_router.scheduler.window.is_within_window")
    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_window_check_closed(
        self, mock_load_config, mock_is_within, runner
    ):
        """--check flag calls is_within_window and shows CLOSED when False."""
        mw = _mock_maintenance_window()
        mock_load_config.return_value = _mock_config(maintenance_window=mw)
        mock_is_within.return_value = False

        result = runner.invoke(schedule_group, ["window", "--check"])

        assert result.exit_code == 0
        assert "CLOSED" in result.output


class TestScheduleRun:
    """Tests for 'schedule run' command."""

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_run_shows_force_message(self, mock_load_config, runner):
        """Force-run command shows the force-running message with job ID."""
        mock_load_config.return_value = _mock_config()

        result = runner.invoke(schedule_group, ["run", "detection"])

        assert result.exit_code == 0
        assert "Force-running" in result.output
        assert "detection" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_run_sdk_hint(self, mock_load_config, runner):
        """Force-run command shows SDK usage hint."""
        mock_load_config.return_value = _mock_config()

        result = runner.invoke(schedule_group, ["run", "centroid_refresh"])

        assert result.exit_code == 0
        assert "run_now" in result.output


class TestSchedulePauseResume:
    """Tests for 'schedule pause' and 'schedule resume' commands."""

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_pause(self, mock_load_config, runner):
        """Pause command shows pausing message for the given job ID."""
        mock_load_config.return_value = _mock_config()

        result = runner.invoke(schedule_group, ["pause", "detection"])

        assert result.exit_code == 0
        assert "Pausing" in result.output
        assert "detection" in result.output

    @patch("semantic_vector_router.cli.schedule.load_config")
    def test_schedule_resume(self, mock_load_config, runner):
        """Resume command shows resuming message for the given job ID."""
        mock_load_config.return_value = _mock_config()

        result = runner.invoke(schedule_group, ["resume", "detection"])

        assert result.exit_code == 0
        assert "Resuming" in result.output
        assert "detection" in result.output


class TestScheduleHistory:
    """Tests for 'schedule history' command."""

    @patch("semantic_vector_router.cli.schedule._run_async")
    @patch("semantic_vector_router.cli.schedule._get_backend", new_callable=AsyncMock)
    def test_schedule_history_no_records(self, mock_get_backend, mock_run_async, runner):
        """When no job history is found, shows a yellow warning."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _mock_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        # Mock the metadata store cursor
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        # MetadataStore is late-imported inside the async function body,
        # so patch at canonical location.
        with patch("semantic_vector_router.backends.metadata.MetadataStore") as MockMeta:
            mock_meta = AsyncMock()
            MockMeta.return_value = mock_meta
            mock_meta._coll = MagicMock()
            mock_meta._coll.find.return_value = mock_cursor

            result = runner.invoke(schedule_group, ["history"])

        assert result.exit_code == 0
        assert "No job history" in result.output


# ─── Webhooks CLI Tests ──────────────────────────────────────────────────────


class TestWebhooksList:
    """Tests for 'webhooks list' command."""

    @patch("semantic_vector_router.cli.webhooks.load_config")
    def test_webhooks_list_with_webhooks(self, mock_load_config, runner):
        """When webhooks are configured, shows a table with URL, events, status."""
        wh1 = _mock_webhook(
            url="https://api.example.com/events",
            events=["partition.created"],
            enabled=True,
            secret="<redacted>",
        )
        wh2 = _mock_webhook(
            url="https://slack.example.com/webhook",
            events=[],
            enabled=False,
            secret=None,
        )
        mock_load_config.return_value = _mock_config(webhooks=[wh1, wh2])

        result = runner.invoke(webhooks_group, ["list"])

        assert result.exit_code == 0
        assert "Webhooks" in result.output
        assert "api.example.com" in result.output
        assert "slack.example.com" in result.output

    @patch("semantic_vector_router.cli.webhooks.load_config")
    def test_webhooks_list_no_webhooks(self, mock_load_config, runner):
        """When no webhooks are configured, shows a warning message."""
        mock_load_config.return_value = _mock_config(webhooks=[])

        result = runner.invoke(webhooks_group, ["list"])

        assert result.exit_code == 0
        assert "No webhooks configured" in result.output

    @patch("semantic_vector_router.cli.webhooks.load_config")
    def test_webhooks_list_long_url_truncated(self, mock_load_config, runner):
        """URLs longer than 50 chars are truncated."""
        long_url = "https://very-long-hostname.example.com/webhooks/events/receiver/endpoint"
        wh = _mock_webhook(url=long_url, enabled=True)
        mock_load_config.return_value = _mock_config(webhooks=[wh])

        result = runner.invoke(webhooks_group, ["list"])

        assert result.exit_code == 0
        # The full URL should NOT appear in output (it was truncated)
        assert long_url not in result.output
        # But the beginning should
        assert "very-long-hostname" in result.output

    @patch("semantic_vector_router.cli.webhooks.load_config")
    def test_webhooks_list_shows_secret_status(self, mock_load_config, runner):
        """Shows 'configured' when secret is set, 'none' when not."""
        wh_with_secret = _mock_webhook(secret="<redacted>")
        wh_without_secret = _mock_webhook(url="https://other.example.com/hook", secret=None)
        mock_load_config.return_value = _mock_config(
            webhooks=[wh_with_secret, wh_without_secret]
        )

        result = runner.invoke(webhooks_group, ["list"])

        assert result.exit_code == 0
        assert "configured" in result.output
        assert "none" in result.output


class TestWebhooksTest:
    """Tests for 'webhooks test' command."""

    @patch("semantic_vector_router.cli.webhooks._run_async")
    def test_webhooks_test_success(self, mock_run_async, runner):
        """Successful test webhook shows green 'Success!' message."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_test_result = MagicMock()
        mock_test_result.success = True
        mock_test_result.status_code = 200
        mock_test_result.response_time_ms = 42.5

        with patch(
            "semantic_vector_router.events.webhook.WebhookDispatcher.test_webhook",
            new_callable=AsyncMock,
            return_value=mock_test_result,
        ), patch(
            "semantic_vector_router.events.webhook.WebhookDispatcher.close",
            new_callable=AsyncMock,
        ):
            result = runner.invoke(webhooks_group, ["test", "https://httpbin.org/post"])

        assert result.exit_code == 0
        assert "Success" in result.output

    @patch("semantic_vector_router.cli.webhooks._run_async")
    def test_webhooks_test_failure(self, mock_run_async, runner):
        """Failed test webhook shows red 'Failed!' message."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_test_result = MagicMock()
        mock_test_result.success = False
        mock_test_result.status_code = None
        mock_test_result.error = "Connection refused"
        mock_test_result.response_time_ms = 10.0

        with patch(
            "semantic_vector_router.events.webhook.WebhookDispatcher.test_webhook",
            new_callable=AsyncMock,
            return_value=mock_test_result,
        ), patch(
            "semantic_vector_router.events.webhook.WebhookDispatcher.close",
            new_callable=AsyncMock,
        ):
            result = runner.invoke(webhooks_group, ["test", "https://bad-host.invalid/hook"])

        assert result.exit_code == 0
        assert "Failed" in result.output


class TestWebhooksHistory:
    """Tests for 'webhooks history' command."""

    @patch("semantic_vector_router.cli.webhooks._run_async")
    @patch("semantic_vector_router.cli.webhooks._get_backend", new_callable=AsyncMock)
    def test_webhooks_history_no_records(self, mock_get_backend, mock_run_async, runner):
        """When no webhook delivery history is found, shows a yellow warning."""
        mock_run_async.side_effect = _run_async_side_effect

        mock_backend = AsyncMock()
        mock_backend._db = MagicMock()
        config = _mock_config()
        config.lifecycle.metadata.connection_string_env = None
        mock_get_backend.return_value = (config, mock_backend)

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        # MetadataStore is late-imported inside the async function body,
        # so patch at canonical location.
        with patch("semantic_vector_router.backends.metadata.MetadataStore") as MockMeta:
            mock_meta = AsyncMock()
            MockMeta.return_value = mock_meta
            mock_meta._coll = MagicMock()
            mock_meta._coll.find.return_value = mock_cursor

            result = runner.invoke(webhooks_group, ["history"])

        assert result.exit_code == 0
        assert "No webhook delivery history" in result.output
