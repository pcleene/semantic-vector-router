"""Serialize structured documents into field-labeled text for embedding APIs.

Both MongoDB (via view $addFields) and PostgreSQL (via content JSONB) produce
structured objects representing document fields. Embedding APIs (Voyage, OpenAI,
Cohere) require a single string input. This module bridges the gap with
field-labeled serialization that preserves semantic context for better embeddings.

Shared by both backends — the same input dict produces identical output
regardless of whether it came from MongoDB or PostgreSQL.
"""

from typing import Any


def serialize_for_embedding(obj: dict[str, Any], prefix: str = "") -> str:
    """Serialize a structured document into field-labeled text for embedding.

    Rules:
        - Scalars → ``"field_name: value"``
        - Arrays of scalars → ``"field_name: val1, val2, val3"``
        - Nested objects → recursive with dot notation: ``"parent.child: value"``
        - Arrays of objects → indexed: ``"field[0].key: value"``
        - Null / None / empty string → skipped
        - Separator between fields: ``"\\n"`` (newline)

    Examples::

        >>> serialize_for_embedding({"title": "Headphones", "price": 99.99})
        'title: Headphones\\nprice: 99.99'

        >>> serialize_for_embedding({"title": "Headphones", "tags": ["audio", "wireless"]})
        'title: Headphones\\ntags: audio, wireless'

        >>> serialize_for_embedding({"specs": {"weight": "250g", "battery": "30h"}})
        'specs.weight: 250g\\nspecs.battery: 30h'

        >>> serialize_for_embedding({"reviews": [{"text": "Great", "rating": 5}]})
        'reviews[0].text: Great\\nreviews[0].rating: 5'
    """
    if not obj:
        return ""

    lines: list[str] = []
    _serialize_recursive(obj, prefix, lines)
    return "\n".join(lines)


def _serialize_recursive(
    value: Any, prefix: str, lines: list[str]
) -> None:
    """Recursively serialize a value into field-labeled lines."""
    if isinstance(value, dict):
        for key, val in value.items():
            full_key = f"{prefix}.{key}" if prefix else key
            _serialize_recursive(val, full_key, lines)
    elif isinstance(value, list):
        if not value:
            return
        # Check if all items are scalars
        if all(_is_scalar(item) for item in value):
            formatted = ", ".join(_format_scalar(item) for item in value)
            if formatted:
                lines.append(f"{prefix}: {formatted}")
        else:
            # Array of objects (or mixed) — use indexed notation
            for i, item in enumerate(value):
                indexed_prefix = f"{prefix}[{i}]"
                _serialize_recursive(item, indexed_prefix, lines)
    elif _is_scalar(value):
        formatted = _format_scalar(value)
        if formatted:
            lines.append(f"{prefix}: {formatted}")
    # None / unsupported types are silently skipped


def _is_scalar(value: Any) -> bool:
    """Check if a value is a scalar (not dict/list/None)."""
    return value is not None and not isinstance(value, (dict, list))


def _format_scalar(value: Any) -> str:
    """Format a scalar value as a string, skipping empty strings.

    Booleans are lowercased ("true"/"false").
    Floats are formatted cleanly (no trailing zeros beyond reasonable precision).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Clean float formatting: 349.99 not 349.990000
        formatted = f"{value:g}"
        return formatted
    s = str(value)
    return s
