"""Unit tests for PostgreSQL filter translator.

Tests the SVR filter DSL → SQL WHERE clause translator.
Pure logic tests — no database connection needed.
"""

import pytest

from semantic_vector_router.backends.postgres.filters import (
    COLUMN_FIELDS,
    translate_filters,
)


# ── Empty / edge-case inputs ─────────────────────────────────────────


class TestEmptyFilters:
    """Empty and edge-case filter inputs."""

    def test_empty_dict_returns_empty(self):
        sql, params = translate_filters({})
        assert sql == ""
        assert params == []

    def test_none_value_exact_match(self):
        sql, params = translate_filters({"field": None})
        assert sql == "content->>'field' = %s"
        assert params == [None]


# ── Exact match ──────────────────────────────────────────────────────


class TestExactMatch:
    """Exact match filters: {"field": "value"}."""

    def test_string_value(self):
        sql, params = translate_filters({"name": "test"})
        assert sql == "content->>'name' = %s"
        assert params == ["test"]

    def test_integer_value(self):
        sql, params = translate_filters({"count": 42})
        assert sql == "content->>'count' = %s"
        assert params == [42]

    def test_float_value(self):
        sql, params = translate_filters({"price": 9.99})
        assert sql == "content->>'price' = %s"
        assert params == [9.99]

    def test_boolean_value(self):
        sql, params = translate_filters({"active": True})
        assert sql == "content->>'active' = %s"
        assert params == [True]

    def test_multiple_fields_implicit_and(self):
        sql, params = translate_filters({"name": "test", "status": "active"})
        assert "content->>'name' = %s" in sql
        assert "content->>'status' = %s" in sql
        assert " AND " in sql
        assert params == ["test", "active"]


# ── Top-level columns ────────────────────────────────────────────────


class TestTopLevelColumns:
    """Fields that are top-level columns bypass JSONB accessor."""

    def test_partition_name_is_direct_column(self):
        sql, params = translate_filters({"partition_name": "electronics"})
        assert sql == "partition_name = %s"
        assert params == ["electronics"]

    def test_id_is_direct_column(self):
        sql, params = translate_filters({"id": "doc123"})
        assert sql == "id = %s"
        assert params == ["doc123"]

    def test_created_at_is_direct_column(self):
        sql, params = translate_filters({"created_at": {"$gt": "2024-01-01"}})
        assert sql == "created_at > %s"
        assert params == ["2024-01-01"]

    def test_updated_at_is_direct_column(self):
        sql, params = translate_filters({"updated_at": {"$lt": "2024-12-31"}})
        assert sql == "updated_at < %s"
        assert params == ["2024-12-31"]

    def test_column_fields_set_is_complete(self):
        assert COLUMN_FIELDS == frozenset(
            {"id", "partition_name", "created_at", "updated_at"}
        )


# ── Comparison operators ─────────────────────────────────────────────


class TestComparisonOperators:
    """Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte."""

    def test_eq_operator(self):
        sql, params = translate_filters({"status": {"$eq": "active"}})
        assert sql == "content->>'status' = %s"
        assert params == ["active"]

    def test_ne_operator(self):
        sql, params = translate_filters({"status": {"$ne": "inactive"}})
        assert sql == "content->>'status' != %s"
        assert params == ["inactive"]

    def test_gt_operator_numeric(self):
        sql, params = translate_filters({"price": {"$gt": 100}})
        assert sql == "content->>'price'::numeric > %s"
        assert params == [100]

    def test_gte_operator_numeric(self):
        sql, params = translate_filters({"price": {"$gte": 50.5}})
        assert sql == "content->>'price'::numeric >= %s"
        assert params == [50.5]

    def test_lt_operator_numeric(self):
        sql, params = translate_filters({"quantity": {"$lt": 10}})
        assert sql == "content->>'quantity'::numeric < %s"
        assert params == [10]

    def test_lte_operator_numeric(self):
        sql, params = translate_filters({"quantity": {"$lte": 0}})
        assert sql == "content->>'quantity'::numeric <= %s"
        assert params == [0]

    def test_string_comparison_no_cast(self):
        sql, params = translate_filters({"name": {"$gt": "m"}})
        assert sql == "content->>'name' > %s"
        assert params == ["m"]
        assert "::numeric" not in sql

    def test_multiple_operators_on_same_field(self):
        sql, params = translate_filters({"price": {"$gte": 10, "$lte": 100}})
        assert "content->>'price'::numeric >= %s" in sql
        assert "content->>'price'::numeric <= %s" in sql
        assert " AND " in sql
        assert 10 in params
        assert 100 in params


# ── Set operators ────────────────────────────────────────────────────


class TestSetOperators:
    """Set membership operators: $in, $nin."""

    def test_in_operator(self):
        sql, params = translate_filters({"status": {"$in": ["active", "pending"]}})
        assert sql == "content->>'status' IN (%s, %s)"
        assert params == ["active", "pending"]

    def test_in_operator_single_value(self):
        sql, params = translate_filters({"status": {"$in": ["active"]}})
        assert sql == "content->>'status' IN (%s)"
        assert params == ["active"]

    def test_in_operator_empty_list(self):
        sql, params = translate_filters({"status": {"$in": []}})
        assert sql == "FALSE"
        assert params == []

    def test_nin_operator(self):
        sql, params = translate_filters(
            {"status": {"$nin": ["deleted", "archived"]}}
        )
        assert sql == "content->>'status' NOT IN (%s, %s)"
        assert params == ["deleted", "archived"]

    def test_nin_operator_empty_list(self):
        sql, params = translate_filters({"status": {"$nin": []}})
        assert sql == ""
        assert params == []

    def test_in_with_top_level_column(self):
        sql, params = translate_filters(
            {"partition_name": {"$in": ["a", "b", "c"]}}
        )
        assert sql == "partition_name IN (%s, %s, %s)"
        assert params == ["a", "b", "c"]


# ── $exists operator ─────────────────────────────────────────────────


class TestExistsOperator:
    """$exists operator for JSONB key presence."""

    def test_exists_true(self):
        sql, params = translate_filters({"tags": {"$exists": True}})
        assert sql == "content ? %s"
        assert params == ["tags"]

    def test_exists_false(self):
        sql, params = translate_filters({"tags": {"$exists": False}})
        assert sql == "NOT (content ? %s)"
        assert params == ["tags"]


# ── Logical operators ────────────────────────────────────────────────


class TestLogicalOperators:
    """Logical operators: $and, $or."""

    def test_and_operator(self):
        sql, params = translate_filters(
            {"$and": [{"name": "test"}, {"status": "active"}]}
        )
        assert sql == "(content->>'name' = %s AND content->>'status' = %s)"
        assert params == ["test", "active"]

    def test_or_operator(self):
        sql, params = translate_filters(
            {"$or": [{"status": "active"}, {"status": "pending"}]}
        )
        assert sql == "(content->>'status' = %s OR content->>'status' = %s)"
        assert params == ["active", "pending"]

    def test_nested_and_in_or(self):
        sql, params = translate_filters(
            {"$or": [{"$and": [{"a": 1}, {"b": 2}]}, {"c": 3}]}
        )
        assert "OR" in sql
        assert "AND" in sql
        assert len(params) == 3

    def test_nested_or_in_and(self):
        sql, params = translate_filters(
            {"$and": [{"$or": [{"x": 1}, {"y": 2}]}, {"z": 3}]}
        )
        assert "AND" in sql
        assert "OR" in sql
        assert len(params) == 3

    def test_and_with_empty_subfilter(self):
        sql, params = translate_filters({"$and": [{}, {"name": "test"}]})
        assert "content->>'name' = %s" in sql
        assert params == ["test"]

    def test_or_with_empty_subfilter(self):
        sql, params = translate_filters({"$or": [{}, {"name": "test"}]})
        assert "content->>'name' = %s" in sql
        assert params == ["test"]


# ── Unsupported operators ────────────────────────────────────────────


class TestUnsupportedOperator:
    """Unsupported operators raise ValueError."""

    def test_regex_raises_error(self):
        with pytest.raises(ValueError, match=r"Unsupported filter operator: \$regex"):
            translate_filters({"name": {"$regex": "^test"}})

    def test_not_operator_field_level(self):
        """$not is now supported — negates a condition on a field."""
        sql, params = translate_filters({"name": {"$not": {"$eq": "test"}}})
        assert "NOT" in sql
        assert "content->>'name' = %s" in sql
        assert params == ["test"]

    def test_type_raises_error(self):
        with pytest.raises(ValueError, match=r"Unsupported filter operator: \$type"):
            translate_filters({"name": {"$type": "string"}})


# ── Parameterization safety ──────────────────────────────────────────


class TestParameterizationSafety:
    """Verify SQL injection prevention via parameterization."""

    def test_value_with_sql_injection_is_parameterized(self):
        sql, params = translate_filters({"name": "'; DROP TABLE users; --"})
        assert sql == "content->>'name' = %s"
        assert params == ["'; DROP TABLE users; --"]
        # The malicious value is a parameter, never interpolated

    def test_numeric_injection_is_parameterized(self):
        sql, params = translate_filters(
            {"price": {"$gt": "1; DROP TABLE users"}}
        )
        assert "%s" in sql
        assert "1; DROP TABLE users" in params

    def test_in_values_are_all_parameterized(self):
        sql, params = translate_filters(
            {"status": {"$in": ["active", "'; DROP TABLE --"]}}
        )
        assert sql.count("%s") == 2
        assert "'; DROP TABLE --" in params

    def test_exists_key_is_parameterized(self):
        """Even $exists keys go through %s, not string interpolation."""
        sql, params = translate_filters(
            {"'; DROP TABLE --": {"$exists": True}}
        )
        assert "content ? %s" in sql
        assert params == ["'; DROP TABLE --"]


# ── Complex real-world filters ───────────────────────────────────────


class TestComplexFilters:
    """Complex real-world filter combinations."""

    def test_ecommerce_product_filter(self):
        """Realistic e-commerce: category + price range + in stock."""
        sql, params = translate_filters(
            {
                "$and": [
                    {"partition_name": "electronics"},
                    {"price": {"$gte": 100, "$lte": 500}},
                    {"in_stock": True},
                ]
            }
        )
        assert "partition_name = %s" in sql
        assert "content->>'price'::numeric >= %s" in sql
        assert "content->>'price'::numeric <= %s" in sql
        assert "content->>'in_stock' = %s" in sql
        assert len(params) == 4

    def test_combined_implicit_and(self):
        """Multiple top-level keys form implicit AND."""
        sql, params = translate_filters(
            {"category": "laptop", "brand": "apple", "price": {"$lt": 2000}}
        )
        assert " AND " in sql
        assert len(params) == 3

    def test_deeply_nested_filter(self):
        """Three-level nesting: $or → $and → comparison."""
        sql, params = translate_filters(
            {
                "$or": [
                    {
                        "$and": [
                            {"category": "electronics"},
                            {"price": {"$lt": 100}},
                        ]
                    },
                    {
                        "$and": [
                            {"category": "clothing"},
                            {"price": {"$lt": 50}},
                        ]
                    },
                ]
            }
        )
        assert "OR" in sql
        assert sql.count("AND") == 2
        assert len(params) == 4

    def test_mixed_operators_and_exact_match(self):
        """Mix of exact match, comparison, set, and logical."""
        sql, params = translate_filters(
            {
                "brand": "apple",
                "status": {"$in": ["active", "pending"]},
                "price": {"$gte": 100},
            }
        )
        assert "content->>'brand' = %s" in sql
        assert "content->>'status' IN (%s, %s)" in sql
        assert "content->>'price'::numeric >= %s" in sql
        assert len(params) == 4
