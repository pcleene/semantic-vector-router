"""Retry decorator with exponential backoff and jitter."""

import asyncio
import functools
import inspect
import random
import time
from typing import Any, Callable, Optional, Type

from semantic_vector_router.utils.logging import get_logger

logger = get_logger(__name__)

# Default retryable exceptions for MongoDB operations
RETRYABLE_MONGODB: tuple[Type[Exception], ...] = ()
# Default retryable exceptions for HTTP API calls
RETRYABLE_HTTP: tuple[Type[Exception], ...] = ()

# Lazy-load to avoid import errors if pymongo/httpx not installed
_RETRYABLE_MONGODB_LOADED = False
_RETRYABLE_HTTP_LOADED = False


def _get_retryable_mongodb() -> tuple[Type[Exception], ...]:
    """Get retryable MongoDB exception types (lazy import)."""
    global RETRYABLE_MONGODB, _RETRYABLE_MONGODB_LOADED
    if not _RETRYABLE_MONGODB_LOADED:
        try:
            from pymongo.errors import (
                AutoReconnect,
                NetworkTimeout,
                ServerSelectionTimeoutError,
            )
            RETRYABLE_MONGODB = (AutoReconnect, NetworkTimeout, ServerSelectionTimeoutError)
        except ImportError:
            RETRYABLE_MONGODB = ()
        _RETRYABLE_MONGODB_LOADED = True
    return RETRYABLE_MONGODB


def _get_retryable_http() -> tuple[Type[Exception], ...]:
    """Get retryable HTTP exception types (lazy import)."""
    global RETRYABLE_HTTP, _RETRYABLE_HTTP_LOADED
    if not _RETRYABLE_HTTP_LOADED:
        try:
            import httpx
            RETRYABLE_HTTP = (httpx.TimeoutException, httpx.HTTPStatusError)
        except ImportError:
            RETRYABLE_HTTP = ()
        _RETRYABLE_HTTP_LOADED = True
    return RETRYABLE_HTTP


def _is_retryable_http_status(exc: Exception) -> bool:
    """Check if an HTTP error has a retryable status code."""
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
    except ImportError:
        pass
    return True  # Non-HTTPStatusError exceptions in the retryable set are always retryable


def _get_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After header value from HTTP 429 responses."""
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
    except ImportError:
        pass
    return None


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> Callable:
    """Retry decorator with exponential backoff and jitter.

    Can be used as a decorator or called programmatically:

        @with_retry(max_attempts=3)
        async def my_func():
            ...

        # Or programmatically:
        result = await with_retry(max_attempts=3)(my_func)(args)

    Args:
        max_attempts: Maximum number of attempts (0 = no retry, just run once).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types to retry on.
            Defaults to all exceptions if None.

    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_max = max(max_attempts, 1)
            last_exception: Optional[Exception] = None

            for attempt in range(effective_max):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this exception is retryable
                    if retryable_exceptions is not None:
                        if not isinstance(e, retryable_exceptions):
                            raise

                        # For HTTP status errors, check if the status code is retryable
                        if not _is_retryable_http_status(e):
                            raise

                    # Last attempt — don't sleep, just raise
                    if attempt >= effective_max - 1:
                        break

                    # Check for Retry-After header (HTTP 429)
                    retry_after = _get_retry_after(e)
                    if retry_after is not None:
                        delay = retry_after
                    else:
                        # Exponential backoff with jitter (±25%)
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = delay * 0.25 * (2 * random.random() - 1)
                        delay = delay + jitter

                    logger.warning(
                        f"Retry {attempt + 1}/{effective_max - 1} for {func.__qualname__}: "
                        f"{type(e).__name__}: {e} — retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)

            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_max = max(max_attempts, 1)
            last_exception: Optional[Exception] = None

            for attempt in range(effective_max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if retryable_exceptions is not None:
                        if not isinstance(e, retryable_exceptions):
                            raise
                        if not _is_retryable_http_status(e):
                            raise

                    if attempt >= effective_max - 1:
                        break

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * 0.25 * (2 * random.random() - 1)
                    delay = delay + jitter

                    logger.warning(
                        f"Retry {attempt + 1}/{effective_max - 1} for {func.__qualname__}: "
                        f"{type(e).__name__}: {e} — retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception  # type: ignore[misc]

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def async_retry(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> Any:
    """Execute an async function with retry logic (programmatic use).

    Args:
        func: Async function to call.
        args: Positional arguments.
        kwargs: Keyword arguments.
        max_attempts: Maximum attempts.
        base_delay: Base delay for backoff.
        max_delay: Maximum delay cap.
        retryable_exceptions: Exception types to retry.

    Returns:
        Result of the function call.
    """
    kwargs = kwargs or {}
    wrapped = with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
    )(func)
    return await wrapped(*args, **kwargs)
