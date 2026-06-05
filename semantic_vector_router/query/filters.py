"""Shared filter validation and operator constants.

Defines the portable filter operator set that SVR supports across all backends.
Validation runs once at ``SVRClient.search()`` entry — before any backend call.
"""

from typing import Any


# ── Supported operators (portable across both backends) ──────────

COMPARISON_OPS: dict[str, str] = {
    "$eq": "Equals (implicit when value is scalar)",
    "$ne": "Not equals",
    "$gt": "Greater than",
    "$gte": "Greater than or equal",
    "$lt": "Less than",
    "$lte": "Less than or equal",
}

SET_OPS: dict[str, str] = {
    "$in": "Value in list",
    "$nin": "Value not in list",
}

LOGICAL_OPS: dict[str, str] = {
    "$and": "All conditions must match",
    "$or": "At least one condition must match",
    "$not": "Negation of a condition",
}

EXISTENCE_OPS: dict[str, str] = {
    "$exists": "Field exists (Mongo) / IS NOT NULL via JSONB ? operator (Postgres)",
}

SUPPORTED_OPERATORS: frozenset[str] = frozenset(
    {*COMPARISON_OPS, *SET_OPS, *LOGICAL_OPS, *EXISTENCE_OPS}
)

# ── Rejected operators (common mistakes with actionable messages) ─

REJECTED_OPERATORS: dict[str, str] = {
    "$match": "Use post_native for post-vector-search filtering",
    "$all": "Array operator — use post_native with $match (MongoDB) or pre_native (PostgreSQL)",
    "$elemMatch": "Array operator — use post_native with $match (MongoDB) or pre_native (PostgreSQL)",
    "$regex": "Use post_native with $match (MongoDB) or pre_native (PostgreSQL)",
    "$text": "Use post_native with $match (MongoDB) or pre_native with tsvector (PostgreSQL)",
    "$size": "Array operator — use post_native with $match (MongoDB) or pre_native (PostgreSQL)",
    "$type": "Use post_native with $match (MongoDB) or pre_native (PostgreSQL)",
    "$near": "Geospatial — use post_native",
    "$geoWithin": "Geospatial — use post_native",
}


def validate_filters(filters: dict[str, Any]) -> None:
    """Validate a filter dict against the SVR operator set.

    Checks all operators recursively. Raises ``ValueError`` for:
    - Operators in ``REJECTED_OPERATORS`` (with actionable redirect message)
    - Unknown operators not in ``SUPPORTED_OPERATORS``

    This runs once at ``SVRClient.search()`` entry point before any backend
    dispatch. Backend code trusts the already-validated filter dict.

    Args:
        filters: SVR filter dict to validate.

    Raises:
        ValueError: If an unsupported or rejected operator is found.
    """
    if not filters:
        return

    for key, value in filters.items():
        if key.startswith("$"):
            # It's an operator — check if it's valid
            if key in REJECTED_OPERATORS:
                raise ValueError(
                    f"Operator '{key}' is not allowed in filters. "
                    f"{REJECTED_OPERATORS[key]}"
                )
            if key not in SUPPORTED_OPERATORS:
                supported = ", ".join(sorted(SUPPORTED_OPERATORS))
                raise ValueError(
                    f"Unsupported filter operator: {key}. "
                    f"Supported operators: {supported}"
                )

            # Recurse into logical operators
            if key in ("$and", "$or"):
                if not isinstance(value, list):
                    raise ValueError(
                        f"Operator '{key}' requires a list of conditions, "
                        f"got {type(value).__name__}"
                    )
                for sub_filter in value:
                    if isinstance(sub_filter, dict):
                        validate_filters(sub_filter)

            elif key == "$not":
                if isinstance(value, dict):
                    validate_filters(value)

        elif isinstance(value, dict):
            # Field with operator expression: {"field": {"$gt": 5}}
            for op in value:
                if op.startswith("$"):
                    if op in REJECTED_OPERATORS:
                        raise ValueError(
                            f"Operator '{op}' is not allowed in filters. "
                            f"{REJECTED_OPERATORS[op]}"
                        )
                    if op not in SUPPORTED_OPERATORS:
                        supported = ", ".join(sorted(SUPPORTED_OPERATORS))
                        raise ValueError(
                            f"Unsupported filter operator: {op}. "
                            f"Supported operators: {supported}"
                        )
                    # Recurse into $not value
                    if op == "$not" and isinstance(value[op], dict):
                        validate_filters(value[op])
