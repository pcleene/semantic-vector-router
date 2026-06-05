"""MongoDB Atlas backend package.

Re-exports MongoDBBackend, helper classes, and commonly-patched symbols
for backward compatibility with existing test patches.
"""

from pymongo import AsyncMongoClient  # noqa: F401 — patch target compat

from semantic_vector_router.backends.mongodb.backend import (
    RETRYABLE_MONGODB,
    MongoDBBackend,
)
from semantic_vector_router.backends.mongodb.indexes import MongoDBIndexOps
from semantic_vector_router.backends.mongodb.vectors import (
    bindata_to_vector,
    query_vector_for_search,
    vector_to_bindata,
)
from semantic_vector_router.backends.mongodb.views import MongoDBViewOps
from semantic_vector_router.config import get_connection_string  # noqa: F401 — patch target compat

__all__ = [
    "AsyncMongoClient",
    "MongoDBBackend",
    "MongoDBIndexOps",
    "MongoDBViewOps",
    "RETRYABLE_MONGODB",
    "bindata_to_vector",
    "get_connection_string",
    "query_vector_for_search",
    "vector_to_bindata",
]
