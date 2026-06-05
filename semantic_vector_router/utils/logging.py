"""Structured logging for Semantic Vector Router.

Provides JSON-formatted structured logging with correlation ID propagation,
operation timing decorators, and configurable log levels.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

# Context variable for correlation ID — propagates across await boundaries
correlation_id_var: ContextVar[str] = ContextVar("svr_correlation_id", default="")

# Standard LogRecord attributes to exclude from extra fields
_STANDARD_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "created", "relativeCreated", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName", "sinfo",
    "filename", "module", "pathname", "levelname", "levelno",
    "process", "processName", "thread", "threadName", "taskName",
    "message", "msecs",
})

# Human-readable format string
_HUMAN_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class SVRLogFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Output format:
    {"ts": "2026-02-12T14:30:00.123456+00:00", "level": "INFO",
     "logger": "semantic_vector_router.client", "msg": "Search completed",
     "correlation_id": "abc123def456", "duration_ms": 42.5,
     "partitions": 3, "results": 10}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with SVR context fields.

        Args:
            record: The log record to format.

        Returns:
            JSON string with structured log data.
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        # Include any extra fields set on the record
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    logger_name: str = "semantic_vector_router",
) -> logging.Logger:
    """Configure SVR logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, use JSON formatter. If False, use human-readable format.
        logger_name: Root logger name for SVR.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    logger.handlers.clear()

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(SVRLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT))

    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the SVR namespace.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A logger instance with the given name.
    """
    return logging.getLogger(name)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current async context.

    Returns:
        The newly generated 12-character hex correlation ID.
    """
    cid = uuid.uuid4().hex[:12]
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID, or empty string if none set.

    Returns:
        The current correlation ID string.
    """
    return correlation_id_var.get()


def log_operation(
    logger: logging.Logger,
    operation: str,
    **context: Any,
) -> Callable:
    """Decorator that logs operation start/end with timing and context.

    Works with both sync and async functions. Emits structured log messages
    at operation start, completion, and on error with duration timing.

    Args:
        logger: The logger instance to use for output.
        operation: Name of the operation (e.g., "search", "embed").
        **context: Additional key-value pairs included in start log entry.

    Returns:
        A decorator function.
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.info(f"{operation}.start", extra=context)
                start = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.info(
                        f"{operation}.complete",
                        extra={"duration_ms": round(elapsed_ms, 2)},
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.error(
                        f"{operation}.error",
                        extra={
                            "duration_ms": round(elapsed_ms, 2),
                            "error": type(exc).__name__,
                        },
                    )
                    raise

            return async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.info(f"{operation}.start", extra=context)
                start = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.info(
                        f"{operation}.complete",
                        extra={"duration_ms": round(elapsed_ms, 2)},
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.error(
                        f"{operation}.error",
                        extra={
                            "duration_ms": round(elapsed_ms, 2),
                            "error": type(exc).__name__,
                        },
                    )
                    raise

            return sync_wrapper

    return decorator
