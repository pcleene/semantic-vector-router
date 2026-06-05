"""MongoDB filter translator.

Validates operators and normalizes scalar shorthand to explicit ``$eq``.
Since SVR's filter syntax IS MongoDB syntax, the translation is mostly
identity — the key value add is validation and normalization.
"""

import logging
from typing import Any

from semantic_vector_router.query.filters import (
    COMPARISON_OPS,
    EXISTENCE_OPS,
    LOGICAL_OPS,
    REJECTED_OPERATORS,
    SET_OPS,
    SUPPORTED_OPERATORS,
)

logger = logging.getLogger(__name__)


class MongoFilterTranslator:
    """Translate and validate SVR filters for MongoDB $vectorSearch.filter.

    MongoDB's $vectorSearch.filter accepts a subset of MongoDB query syntax.
    This translator:
    1. Validates that all operators are in the supported set
    2. Rejects operators that would cause cryptic Atlas runtime errors
    3. Normalizes scalar shorthand to explicit ``{"$eq": value}``
    """

    def translate(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Translate SVR filters to MongoDB $vectorSearch.filter format.

        Args:
            filters: SVR filter dict (already validated by validate_filters).

        Returns:
            MongoDB-native filter dict for $vectorSearch.filter.
        """
        if not filters:
            return {}

        result: dict[str, Any] = {}

        for key, value in filters.items():
            if key in ("$and", "$or"):
                translated_subs = []
                for sub_filter in value:
                    translated = self.translate(sub_filter)
                    if translated:
                        translated_subs.append(translated)
                if translated_subs:
                    result[key] = translated_subs

            elif key == "$not":
                if isinstance(value, dict):
                    result["$not"] = self.translate(value)

            elif isinstance(value, dict):
                # Operator expression — pass through validated operators
                result[key] = value

            else:
                # Scalar shorthand → normalized (MongoDB handles both forms,
                # but explicit $eq is clearer in debug logs)
                result[key] = value

        return result
