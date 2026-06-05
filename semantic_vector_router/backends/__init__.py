"""Database backend implementations."""

from semantic_vector_router.backends.base import (
    AutoEmbeddingCapable,
    BaseBackend,
    ChangeStreamCapable,
)
from semantic_vector_router.backends.mongodb import MongoDBBackend
from semantic_vector_router.backends.postgres import PostgresBackend

__all__ = [
    "AutoEmbeddingCapable",
    "BaseBackend",
    "ChangeStreamCapable",
    "MongoDBBackend",
    "PostgresBackend",
]
