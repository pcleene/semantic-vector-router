"""PostgreSQL + pgvector backend for vector search."""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import to avoid circular dependency with models package."""
    if name == "PostgresBackend":
        from semantic_vector_router.backends.postgres.backend import (  # noqa: PLC0415
            PostgresBackend,
        )

        return PostgresBackend
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["PostgresBackend"]
