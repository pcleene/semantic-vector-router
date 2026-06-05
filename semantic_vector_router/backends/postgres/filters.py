"""Translate SVR filter DSL (MongoDB query syntax subset) to SQL WHERE clauses.

All values are parameterized (%s placeholders with psycopg) — no string
interpolation of user values into SQL. Field names for ``$exists`` are
also parameterized via the JSONB ``?`` operator.
"""

import re
from typing import Any

# Regex for valid JSONB field names: must start with letter or underscore,
# then contain only alphanumeric, underscore, or dot characters.
_FIELD_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


# Supported comparison operators: SVR/MongoDB syntax → SQL
COMPARISON_OPS: dict[str, str] = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

# Fields that are top-level table columns (not inside JSONB ``content``)
COLUMN_FIELDS: frozenset[str] = frozenset(
    {"id", "partition_name", "created_at", "updated_at"}
)


def translate_filters(
    svr_filters: dict[str, Any],
) -> tuple[str, list[Any]]:
    """Translate SVR filter dict to (SQL WHERE fragment, params).

    Examples::

        {"field": "value"}        → ("content->>'field' = %s", ["value"])
        {"field": {"$gt": 5}}     → ("content->>'field'::numeric > %s", [5])
        {"$and": [{...}, {...}]}  → ("(... AND ...)", [...])
        {"$or": [{...}, {...}]}   → ("(... OR ...)", [...])
        {"partition_name": "x"}   → ("partition_name = %s", ["x"])

    Args:
        svr_filters: SVR filter dict in MongoDB query syntax subset.

    Returns:
        Tuple of (SQL WHERE fragment, list of parameters).
        Returns ``("", [])`` if filters are empty.

    Raises:
        ValueError: If an unsupported operator is encountered.
    """
    if not svr_filters:
        return ("", [])

    parts: list[str] = []
    params: list[Any] = []

    for key, value in svr_filters.items():
        if key == "$and":
            sub_parts: list[str] = []
            for sub_filter in value:
                sql, p = translate_filters(sub_filter)
                if sql:
                    sub_parts.append(sql)
                    params.extend(p)
            if sub_parts:
                parts.append("(" + " AND ".join(sub_parts) + ")")

        elif key == "$or":
            sub_parts = []
            for sub_filter in value:
                sql, p = translate_filters(sub_filter)
                if sql:
                    sub_parts.append(sql)
                    params.extend(p)
            if sub_parts:
                parts.append("(" + " OR ".join(sub_parts) + ")")

        elif key == "$not":
            if isinstance(value, dict):
                sql, p = translate_filters(value)
                if sql:
                    parts.append(f"NOT ({sql})")
                    params.extend(p)

        elif isinstance(value, dict):
            # Operator expression: {"field": {"$gt": 5}}
            for op, operand in value.items():
                if op in COMPARISON_OPS:
                    sql_op = COMPARISON_OPS[op]
                    col = _column_ref(key)
                    cast = _cast_suffix(operand)
                    parts.append(f"{col}{cast} {sql_op} %s")
                    params.append(operand)
                elif op == "$in":
                    col = _column_ref(key)
                    if not operand:
                        # Empty $in always matches nothing
                        parts.append("FALSE")
                    else:
                        placeholders = ", ".join(["%s"] * len(operand))
                        parts.append(f"{col} IN ({placeholders})")
                        params.extend(operand)
                elif op == "$nin":
                    col = _column_ref(key)
                    if not operand:
                        # Empty $nin matches everything — no filter needed
                        pass
                    else:
                        placeholders = ", ".join(["%s"] * len(operand))
                        parts.append(f"{col} NOT IN ({placeholders})")
                        params.extend(operand)
                elif op == "$exists":
                    if operand:
                        parts.append("content ? %s")
                        params.append(key)
                    else:
                        parts.append("NOT (content ? %s)")
                        params.append(key)
                elif op == "$not":
                    if isinstance(operand, dict):
                        # {"field": {"$not": {"$gt": 5}}} → NOT (field > 5)
                        inner_filter = {key: operand}
                        inner_sql, inner_params = translate_filters(inner_filter)
                        if inner_sql:
                            parts.append(f"NOT ({inner_sql})")
                            params.extend(inner_params)
                    else:
                        # {"field": {"$not": value}} → field != value
                        col = _column_ref(key)
                        parts.append(f"{col} != %s")
                        params.append(operand)
                else:
                    raise ValueError(f"Unsupported filter operator: {op}")

        else:
            # Exact match: {"field": "value"}
            col = _column_ref(key)
            parts.append(f"{col} = %s")
            params.append(value)

    sql = " AND ".join(parts) if parts else ""
    return (sql, params)


def validate_field_name(field: str) -> None:
    """Validate a field name for safe use in SQL expressions.

    Raises:
        ValueError: If field name contains invalid characters.
    """
    if not _FIELD_NAME_RE.match(field):
        raise ValueError(
            f"Invalid field name '{field}': must match "
            f"^[a-zA-Z_][a-zA-Z0-9_.]*$"
        )


def _column_ref(field: str) -> str:
    """Get the SQL column reference for a field name.

    Top-level columns (``id``, ``partition_name``, etc.) are referenced
    directly. All other fields are accessed via JSONB:
    ``content->>'field'``.

    Raises:
        ValueError: If field name contains invalid characters.
    """
    if field in COLUMN_FIELDS:
        return field
    validate_field_name(field)
    return f"content->>'{field}'"


def _cast_suffix(value: Any) -> str:
    """Get a SQL cast suffix for numeric comparisons on JSONB text extraction.

    ``content->>'price'`` returns TEXT. For numeric comparisons we need
    ``content->>'price'::numeric``.
    """
    if isinstance(value, (int, float)):
        return "::numeric"
    return ""
