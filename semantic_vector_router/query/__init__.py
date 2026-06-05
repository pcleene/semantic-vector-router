"""Query abstraction layer for SVR.

Provides portable filter validation, operator constants, and backend-specific
filter translators. The query layer validates filters once at the SDK entry
point (``SVRClient.search()``) and passes validated filters through to
backend-specific translators.
"""

from semantic_vector_router.query.filters import (
    COMPARISON_OPS,
    EXISTENCE_OPS,
    LOGICAL_OPS,
    REJECTED_OPERATORS,
    SET_OPS,
    SUPPORTED_OPERATORS,
    validate_filters,
)
from semantic_vector_router.query.mongo_filters import MongoFilterTranslator

__all__ = [
    "COMPARISON_OPS",
    "EXISTENCE_OPS",
    "LOGICAL_OPS",
    "REJECTED_OPERATORS",
    "SET_OPS",
    "SUPPORTED_OPERATORS",
    "MongoFilterTranslator",
    "validate_filters",
]
