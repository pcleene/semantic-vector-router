"""Unit tests for serialize_for_embedding() — backend-agnostic text serializer.

Tests 1-12 from Phase 16 spec. All pure unit tests, no mocks needed.
"""

import pytest

from semantic_vector_router.utils.text_serializer import serialize_for_embedding


# ── Test 1: Flat scalar fields ──────────────────────────────────────

class TestSerializeFlatFields:
    def test_simple_string_fields(self):
        result = serialize_for_embedding({
            "title": "Sony WH-1000XM5",
            "description": "Noise cancelling headphones",
        })
        assert result == "title: Sony WH-1000XM5\ndescription: Noise cancelling headphones"

    def test_single_field(self):
        result = serialize_for_embedding({"title": "Headphones"})
        assert result == "title: Headphones"


# ── Test 2: Nested objects with dot notation ────────────────────────

class TestSerializeNestedObjects:
    def test_one_level_nesting(self):
        result = serialize_for_embedding({
            "specs": {"weight": "250g", "battery": "30h"},
        })
        assert result == "specs.weight: 250g\nspecs.battery: 30h"

    def test_nested_with_flat(self):
        result = serialize_for_embedding({
            "title": "Headphones",
            "specs": {"weight": "250g", "battery": "30h"},
        })
        assert result == (
            "title: Headphones\n"
            "specs.weight: 250g\n"
            "specs.battery: 30h"
        )


# ── Test 3: Arrays of scalars ──────────────────────────────────────

class TestSerializeArraysOfScalars:
    def test_string_array(self):
        result = serialize_for_embedding({
            "tags": ["audio", "wireless", "noise-cancelling"],
        })
        assert result == "tags: audio, wireless, noise-cancelling"

    def test_int_array(self):
        result = serialize_for_embedding({"sizes": [10, 20, 30]})
        assert result == "sizes: 10, 20, 30"

    def test_empty_array(self):
        result = serialize_for_embedding({"tags": []})
        assert result == ""


# ── Test 4: Arrays of objects with indexed notation ─────────────────

class TestSerializeArraysOfObjects:
    def test_array_of_dicts(self):
        result = serialize_for_embedding({
            "reviews": [
                {"text": "Great", "rating": 5},
                {"text": "Good", "rating": 4},
            ],
        })
        assert result == (
            "reviews[0].text: Great\n"
            "reviews[0].rating: 5\n"
            "reviews[1].text: Good\n"
            "reviews[1].rating: 4"
        )

    def test_single_object_in_array(self):
        result = serialize_for_embedding({
            "reviews": [{"text": "Amazing"}],
        })
        assert result == "reviews[0].text: Amazing"


# ── Test 5: Null/None values skipped ───────────────────────────────

class TestSerializeSkipsNulls:
    def test_none_value_skipped(self):
        result = serialize_for_embedding({
            "title": "Headphones",
            "subtitle": None,
            "price": 99.99,
        })
        assert result == "title: Headphones\nprice: 99.99"

    def test_all_none(self):
        result = serialize_for_embedding({"a": None, "b": None})
        assert result == ""


# ── Test 6: Empty string values skipped ────────────────────────────

class TestSerializeSkipsEmptyStrings:
    def test_empty_string_skipped(self):
        result = serialize_for_embedding({
            "title": "Headphones",
            "subtitle": "",
            "price": 99.99,
        })
        assert result == "title: Headphones\nprice: 99.99"


# ── Test 7: Mixed types — realistic document ──────────────────────

class TestSerializeMixedTypes:
    def test_realistic_product(self):
        result = serialize_for_embedding({
            "title": "Sony WH-1000XM5",
            "description": "Premium noise cancelling headphones",
            "tags": ["audio", "wireless", "noise-cancelling"],
            "specs": {"weight": "250g", "battery_life": "30h", "driver_size": "30mm"},
        })
        expected = (
            "title: Sony WH-1000XM5\n"
            "description: Premium noise cancelling headphones\n"
            "tags: audio, wireless, noise-cancelling\n"
            "specs.weight: 250g\n"
            "specs.battery_life: 30h\n"
            "specs.driver_size: 30mm"
        )
        assert result == expected


# ── Test 8: Deeply nested (3+ levels) ─────────────────────────────

class TestSerializeDeeplyNested:
    def test_three_levels(self):
        result = serialize_for_embedding({
            "a": {"b": {"c": "deep_value"}},
        })
        assert result == "a.b.c: deep_value"

    def test_four_levels(self):
        result = serialize_for_embedding({
            "a": {"b": {"c": {"d": "very_deep"}}},
        })
        assert result == "a.b.c.d: very_deep"


# ── Test 9: Empty dict ────────────────────────────────────────────

class TestSerializeEmptyDict:
    def test_returns_empty_string(self):
        assert serialize_for_embedding({}) == ""


# ── Test 10: Preserves field order ─────────────────────────────────

class TestSerializePreservesFieldOrder:
    def test_order_preserved(self):
        result = serialize_for_embedding({
            "z_field": "last",
            "a_field": "first",
            "m_field": "middle",
        })
        lines = result.split("\n")
        assert lines[0] == "z_field: last"
        assert lines[1] == "a_field: first"
        assert lines[2] == "m_field: middle"


# ── Test 11: Numeric values ───────────────────────────────────────

class TestSerializeNumericValues:
    def test_float_clean_formatting(self):
        result = serialize_for_embedding({"price": 349.99})
        assert result == "price: 349.99"

    def test_integer_value(self):
        result = serialize_for_embedding({"count": 42})
        assert result == "count: 42"

    def test_large_float_no_trailing_zeros(self):
        result = serialize_for_embedding({"value": 100.0})
        assert result == "value: 100"

    def test_zero(self):
        result = serialize_for_embedding({"count": 0})
        assert result == "count: 0"


# ── Test 12: Boolean values ───────────────────────────────────────

class TestSerializeBooleanValues:
    def test_true(self):
        result = serialize_for_embedding({"active": True})
        assert result == "active: true"

    def test_false(self):
        result = serialize_for_embedding({"deleted": False})
        assert result == "deleted: false"

    def test_bool_with_other_fields(self):
        result = serialize_for_embedding({
            "name": "Test",
            "active": True,
            "count": 5,
        })
        assert result == "name: Test\nactive: true\ncount: 5"


# ── Edge cases ────────────────────────────────────────────────────

class TestSerializeEdgeCases:
    def test_nested_null_in_object(self):
        """Null values inside nested objects are skipped."""
        result = serialize_for_embedding({
            "specs": {"weight": "250g", "color": None},
        })
        assert result == "specs.weight: 250g"

    def test_mixed_array_with_nested_null(self):
        """Array of objects where some values are null."""
        result = serialize_for_embedding({
            "items": [{"name": "A", "val": None}, {"name": "B"}],
        })
        assert result == "items[0].name: A\nitems[1].name: B"

    def test_nested_empty_dict(self):
        """Empty nested dict produces no output for that key."""
        result = serialize_for_embedding({
            "title": "Test",
            "empty": {},
        })
        assert result == "title: Test"

    def test_nested_array_in_object(self):
        """Array inside nested object."""
        result = serialize_for_embedding({
            "meta": {"tags": ["a", "b"]},
        })
        assert result == "meta.tags: a, b"
