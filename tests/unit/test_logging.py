"""Unit tests for structured logging (utils/logging.py)."""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from semantic_vector_router.utils.logging import (
    SVRLogFormatter,
    configure_logging,
    correlation_id_var,
    get_correlation_id,
    get_logger,
    log_operation,
    new_correlation_id,
)


# ---------------------------------------------------------------------------
# SVRLogFormatter
# ---------------------------------------------------------------------------


class TestSVRLogFormatter:
    def test_format_produces_valid_json(self):
        formatter = SVRLogFormatter()
        record = logging.LogRecord(
            name="semantic_vector_router.client",
            level=logging.INFO,
            pathname="client.py",
            lineno=1,
            msg="Search completed",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "semantic_vector_router.client"
        assert parsed["msg"] == "Search completed"
        assert "ts" in parsed
        assert "correlation_id" in parsed

    def test_format_includes_extra_fields(self):
        formatter = SVRLogFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="msg",
            args=None,
            exc_info=None,
        )
        record.duration_ms = 42.5
        record.partitions = 3
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["duration_ms"] == 42.5
        assert parsed["partitions"] == 3

    def test_format_includes_exception_info(self):
        formatter = SVRLogFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="fail",
            args=None,
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_format_includes_correlation_id(self):
        token = correlation_id_var.set("test-cid-123")
        try:
            formatter = SVRLogFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=1, msg="msg", args=None, exc_info=None,
            )
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["correlation_id"] == "test-cid-123"
        finally:
            correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_json_format_true(self):
        logger = configure_logging(level="DEBUG", json_format=True)
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, SVRLogFormatter)

    def test_json_format_false(self):
        logger = configure_logging(level="WARNING", json_format=False)
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1
        assert not isinstance(logger.handlers[0].formatter, SVRLogFormatter)

    def test_propagate_disabled(self):
        logger = configure_logging()
        assert logger.propagate is False

    def test_clears_existing_handlers(self):
        logger = configure_logging()
        configure_logging()
        assert len(logger.handlers) == 1


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_named_logger(self):
        logger = get_logger("semantic_vector_router.client")
        assert logger.name == "semantic_vector_router.client"

    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# Correlation ID
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def test_new_correlation_id_sets_contextvar(self):
        cid = new_correlation_id()
        assert len(cid) == 12
        assert get_correlation_id() == cid

    def test_get_correlation_id_empty_default(self):
        token = correlation_id_var.set("")
        try:
            assert get_correlation_id() == ""
        finally:
            correlation_id_var.reset(token)

    @pytest.mark.asyncio
    async def test_correlation_id_propagates_across_await(self):
        cid = new_correlation_id()

        async def inner():
            return get_correlation_id()

        result = await inner()
        assert result == cid

    @pytest.mark.asyncio
    async def test_correlation_id_isolation_between_tasks(self):
        """Each asyncio task should get its own correlation ID context."""
        results = {}

        async def task_a():
            cid = new_correlation_id()
            await asyncio.sleep(0.01)
            results["a_set"] = cid
            results["a_get"] = get_correlation_id()

        async def task_b():
            cid = new_correlation_id()
            await asyncio.sleep(0.01)
            results["b_set"] = cid
            results["b_get"] = get_correlation_id()

        await asyncio.gather(
            asyncio.create_task(task_a()),
            asyncio.create_task(task_b()),
        )

        assert results["a_set"] == results["a_get"]
        assert results["b_set"] == results["b_get"]
        assert results["a_set"] != results["b_set"]


# ---------------------------------------------------------------------------
# log_operation decorator
# ---------------------------------------------------------------------------


class TestLogOperation:
    @pytest.mark.asyncio
    async def test_async_logs_start_and_complete(self):
        test_logger = logging.getLogger("test.log_operation")
        messages = []

        original_info = test_logger.info

        def capture_info(msg, *args, **kwargs):
            messages.append(msg)

        test_logger.info = capture_info

        @log_operation(test_logger, "search", partitions=3)
        async def do_search():
            return "result"

        result = await do_search()
        assert result == "result"
        assert "search.start" in messages
        assert "search.complete" in messages

        test_logger.info = original_info

    @pytest.mark.asyncio
    async def test_async_logs_error_on_exception(self):
        test_logger = logging.getLogger("test.log_operation_err")
        error_messages = []

        def capture_error(msg, *args, **kwargs):
            error_messages.append(msg)

        test_logger.error = capture_error

        @log_operation(test_logger, "embed")
        async def do_embed():
            raise RuntimeError("API timeout")

        with pytest.raises(RuntimeError, match="API timeout"):
            await do_embed()

        assert any("embed.error" in m for m in error_messages)

    def test_sync_logs_start_and_complete(self):
        test_logger = logging.getLogger("test.log_operation_sync")
        messages = []

        def capture_info(msg, *args, **kwargs):
            messages.append(msg)

        test_logger.info = capture_info

        @log_operation(test_logger, "validate")
        def do_validate():
            return True

        result = do_validate()
        assert result is True
        assert "validate.start" in messages
        assert "validate.complete" in messages

    def test_sync_logs_error_on_exception(self):
        test_logger = logging.getLogger("test.log_operation_sync_err")
        error_messages = []

        def capture_error(msg, *args, **kwargs):
            error_messages.append(msg)

        test_logger.error = capture_error

        @log_operation(test_logger, "parse")
        def do_parse():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            do_parse()

        assert any("parse.error" in m for m in error_messages)

    @pytest.mark.asyncio
    async def test_duration_ms_in_complete_extra(self):
        test_logger = logging.getLogger("test.duration")
        extras = []

        def capture_info(msg, *args, **kwargs):
            if "extra" in kwargs:
                extras.append(kwargs["extra"])

        test_logger.info = capture_info

        @log_operation(test_logger, "slow_op")
        async def slow():
            await asyncio.sleep(0.05)

        await slow()

        complete_extras = [e for e in extras if "duration_ms" in e]
        assert len(complete_extras) == 1
        assert complete_extras[0]["duration_ms"] > 0
