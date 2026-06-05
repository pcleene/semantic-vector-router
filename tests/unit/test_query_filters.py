"""Unit tests for query filter abstraction layer.

Tests cover:
- ``semantic_vector_router.query.filters``: operator constants, validate_filters()
- ``semantic_vector_router.query.mongo_filters``: MongoFilterTranslator.translate()

All tests are pure-logic — no mocking, no I/O, no network.
"""

import re

import pytest

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


# ── Operator constant tests ─────────────────────────────────────────


class TestOperatorConstants:
    """Verify the operator dictionaries and frozenset are correctly defined."""

    def test_supported_operators_contains_all_expected(self) -> None:
        """SUPPORTED_OPERATORS should contain exactly 12 operators."""
        expected = {
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
            "$in", "$nin",
            "$and", "$or", "$not",
            "$exists",
        }
        assert SUPPORTED_OPERATORS == expected
        assert len(SUPPORTED_OPERATORS) == 12

    def test_rejected_operators_has_nine_entries(self) -> None:
        """REJECTED_OPERATORS should have exactly 9 entries."""
        expected_keys = {
            "$match", "$all", "$elemMatch", "$regex", "$text",
            "$size", "$type", "$near", "$geoWithin",
        }
        assert set(REJECTED_OPERATORS.keys()) == expected_keys
        assert len(REJECTED_OPERATORS) == 9

    def test_no_overlap_between_supported_and_rejected(self) -> None:
        """Supported and rejected operator sets must be disjoint."""
        overlap = SUPPORTED_OPERATORS & set(REJECTED_OPERATORS.keys())
        assert overlap == set(), f"Unexpected overlap: {overlap}"

    def test_all_comparison_ops_start_with_dollar(self) -> None:
        """Every key in COMPARISON_OPS must start with $."""
        for op in COMPARISON_OPS:
            assert op.startswith("$"), f"{op} does not start with $"

    def test_all_set_ops_start_with_dollar(self) -> None:
        """Every key in SET_OPS must start with $."""
        for op in SET_OPS:
            assert op.startswith("$"), f"{op} does not start with $"

    def test_all_logical_ops_start_with_dollar(self) -> None:
        """Every key in LOGICAL_OPS must start with $."""
        for op in LOGICAL_OPS:
            assert op.startswith("$"), f"{op} does not start with $"

    def test_all_existence_ops_start_with_dollar(self) -> None:
        """Every key in EXISTENCE_OPS must start with $."""
        for op in EXISTENCE_OPS:
            assert op.startswith("$"), f"{op} does not start with $"

    def test_supported_is_union_of_all_categories(self) -> None:
        """SUPPORTED_OPERATORS must equal the union of all category dicts."""
        union = frozenset(
            {*COMPARISON_OPS, *SET_OPS, *LOGICAL_OPS, *EXISTENCE_OPS}
        )
        assert SUPPORTED_OPERATORS == union

    def test_rejected_operators_have_actionable_messages(self) -> None:
        """Every rejected operator should have a non-empty message string."""
        for op, message in REJECTED_OPERATORS.items():
            assert isinstance(message, str), f"{op} message is not a string"
            assert len(message) > 0, f"{op} has empty message"

    def test_supported_operators_is_frozenset(self) -> None:
        """SUPPORTED_OPERATORS must be a frozenset (immutable)."""
        assert isinstance(SUPPORTED_OPERATORS, frozenset)


# ── validate_filters tests ──────────────────────────────────────────


class TestValidateFiltersValid:
    """Tests for filter expressions that should pass validation."""

    def test_empty_dict_no_error(self) -> None:
        """Empty filter dict should pass without error."""
        validate_filters({})

    def test_none_like_no_error(self) -> None:
        """Falsy values (empty dict, None-coerced) should pass."""
        validate_filters({})
        # None itself would fail with AttributeError on .items(),
        # but an empty dict is the canonical "no filter" value.

    @pytest.mark.parametrize("op", ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"])
    def test_comparison_operator_no_error(self, op: str) -> None:
        """Each comparison operator should validate successfully."""
        validate_filters({"price": {op: 100}})

    @pytest.mark.parametrize("op", ["$in", "$nin"])
    def test_set_operator_no_error(self, op: str) -> None:
        """Set operators ($in, $nin) should validate successfully."""
        validate_filters({"category": {op: ["electronics", "books"]}})

    def test_and_operator_no_error(self) -> None:
        """$and with list of valid conditions should pass."""
        validate_filters({
            "$and": [
                {"status": {"$eq": "active"}},
                {"price": {"$gte": 10}},
            ]
        })

    def test_or_operator_no_error(self) -> None:
        """$or with list of valid conditions should pass."""
        validate_filters({
            "$or": [
                {"category": {"$eq": "electronics"}},
                {"category": {"$eq": "books"}},
            ]
        })

    def test_not_operator_no_error(self) -> None:
        """$not with nested condition should pass."""
        validate_filters({"status": {"$not": {"$eq": "archived"}}})

    def test_exists_operator_no_error(self) -> None:
        """$exists operator should pass validation."""
        validate_filters({"description": {"$exists": True}})

    def test_scalar_shorthand_no_error(self) -> None:
        """Scalar shorthand {"field": "value"} should pass validation."""
        validate_filters({"category": "electronics"})

    def test_scalar_shorthand_numeric(self) -> None:
        """Numeric scalar shorthand should pass validation."""
        validate_filters({"price": 42})

    def test_scalar_shorthand_boolean(self) -> None:
        """Boolean scalar shorthand should pass validation."""
        validate_filters({"active": True})

    def test_nested_and_with_inner_operators(self) -> None:
        """$and with inner operator expressions should pass."""
        validate_filters({
            "$and": [
                {"price": {"$gte": 10, "$lte": 100}},
                {"status": {"$in": ["active", "pending"]}},
            ]
        })

    def test_nested_or_with_inner_operators(self) -> None:
        """$or with inner operator expressions should pass."""
        validate_filters({
            "$or": [
                {"category": {"$eq": "electronics"}},
                {"price": {"$lt": 5}},
            ]
        })

    def test_mixed_valid_operators(self) -> None:
        """Multiple fields with different valid operators should pass."""
        validate_filters({
            "category": {"$eq": "electronics"},
            "price": {"$gte": 10},
            "status": {"$in": ["active", "pending"]},
            "archived": {"$exists": False},
        })

    def test_complex_three_level_nested_filter(self) -> None:
        """Complex 3-level nested filter should pass validation."""
        validate_filters({
            "$and": [
                {
                    "$or": [
                        {"category": {"$eq": "electronics"}},
                        {"category": {"$eq": "books"}},
                    ]
                },
                {"price": {"$gte": 10, "$lte": 1000}},
                {
                    "$or": [
                        {"status": {"$eq": "active"}},
                        {
                            "$and": [
                                {"status": {"$eq": "pending"}},
                                {"priority": {"$gte": 5}},
                            ]
                        },
                    ]
                },
            ]
        })

    def test_not_at_top_level(self) -> None:
        """$not used at top level should pass validation."""
        validate_filters({"$not": {"status": {"$eq": "archived"}}})

    def test_multiple_operators_on_single_field(self) -> None:
        """Multiple operators on a single field (range) should pass."""
        validate_filters({"price": {"$gt": 0, "$lt": 999}})

    def test_exists_combined_with_comparison(self) -> None:
        """$exists combined with comparison operator on same field should pass."""
        validate_filters({"score": {"$exists": True, "$gte": 0}})


class TestValidateFiltersRejected:
    """Tests for operators in REJECTED_OPERATORS — must raise ValueError."""

    @pytest.mark.parametrize("op,expected_msg_fragment", [
        ("$match", "post_native"),
        ("$all", "Array operator"),
        ("$elemMatch", "Array operator"),
        ("$regex", "post_native"),
        ("$text", "post_native"),
        ("$size", "Array operator"),
        ("$type", "post_native"),
        ("$near", "Geospatial"),
        ("$geoWithin", "Geospatial"),
    ])
    def test_rejected_operator_raises_with_actionable_message(
        self, op: str, expected_msg_fragment: str
    ) -> None:
        """Each rejected operator should raise ValueError with guidance."""
        with pytest.raises(ValueError, match=rf"Operator '{re.escape(op)}' is not allowed"):
            validate_filters({"field": {op: "value"}})

    @pytest.mark.parametrize("op", list(REJECTED_OPERATORS.keys()))
    def test_rejected_operator_message_contains_redirect(self, op: str) -> None:
        """Rejected operator error should include the redirect message."""
        with pytest.raises(ValueError) as exc_info:
            validate_filters({"field": {op: "value"}})
        assert REJECTED_OPERATORS[op] in str(exc_info.value)

    def test_rejected_operator_at_top_level(self) -> None:
        """Rejected operator used as top-level key should raise ValueError."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters({"$match": {"status": "active"}})

    def test_rejected_operator_nested_inside_and(self) -> None:
        """Rejected operator nested inside $and should raise ValueError."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters({
                "$and": [
                    {"tags": {"$all": ["python", "ml"]}},
                    {"status": {"$eq": "active"}},
                ]
            })

    def test_rejected_operator_nested_inside_or(self) -> None:
        """Rejected operator nested inside $or should raise ValueError."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters({
                "$or": [
                    {"name": {"$regex": "test.*"}},
                    {"status": {"$eq": "active"}},
                ]
            })

    def test_rejected_operator_deeply_nested(self) -> None:
        """Rejected operator deep inside nested logical structure should raise."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_filters({
                "$and": [
                    {
                        "$or": [
                            {"tags": {"$elemMatch": {"$eq": "python"}}},
                        ]
                    },
                ]
            })


class TestValidateFiltersUnknown:
    """Tests for unknown operators — must raise ValueError listing supported."""

    def test_unknown_operator_raises_listing_supported(self) -> None:
        """Unknown operator $foo should raise ValueError with supported list."""
        with pytest.raises(ValueError, match="Unsupported filter operator: \\$foo"):
            validate_filters({"field": {"$foo": "value"}})

    def test_unknown_operator_message_includes_supported_list(self) -> None:
        """Error for unknown operator should include supported operators."""
        with pytest.raises(ValueError) as exc_info:
            validate_filters({"field": {"$bar": 42}})
        error_msg = str(exc_info.value)
        assert "Supported operators:" in error_msg
        # Spot-check a few operators are listed
        assert "$eq" in error_msg
        assert "$in" in error_msg
        assert "$and" in error_msg

    def test_unknown_operator_at_top_level(self) -> None:
        """Unknown operator at top level (e.g., $custom) should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            validate_filters({"$custom": [{"a": 1}]})

    def test_unknown_operator_nested_inside_field_expression(self) -> None:
        """Unknown operator inside a field expression should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported filter operator: \\$baz"):
            validate_filters({"price": {"$baz": 100}})

    def test_unknown_operator_nested_inside_and(self) -> None:
        """Unknown operator inside $and sub-filter should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            validate_filters({
                "$and": [
                    {"field": {"$bogus": "value"}},
                ]
            })


class TestValidateFiltersStructuralErrors:
    """Tests for structural misuse of operators."""

    def test_and_with_non_list_value_raises(self) -> None:
        """$and with non-list value should raise ValueError."""
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters({"$and": {"status": "active"}})

    def test_and_with_string_value_raises(self) -> None:
        """$and with string value should raise ValueError."""
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters({"$and": "invalid"})

    def test_or_with_non_list_value_raises(self) -> None:
        """$or with non-list value should raise ValueError."""
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters({"$or": {"category": "books"}})

    def test_or_with_integer_value_raises(self) -> None:
        """$or with integer value should raise ValueError."""
        with pytest.raises(ValueError, match="requires a list"):
            validate_filters({"$or": 42})

    def test_and_error_mentions_got_type(self) -> None:
        """$and error message should mention the actual type received."""
        with pytest.raises(ValueError, match="got dict"):
            validate_filters({"$and": {"a": 1}})

    def test_or_error_mentions_got_type(self) -> None:
        """$or error message should mention the actual type received."""
        with pytest.raises(ValueError, match="got str"):
            validate_filters({"$or": "bad"})


# ── MongoFilterTranslator tests ─────────────────────────────────────


class TestMongoFilterTranslatorTranslate:
    """Tests for MongoFilterTranslator.translate()."""

    @pytest.fixture()
    def translator(self) -> MongoFilterTranslator:
        return MongoFilterTranslator()

    def test_empty_dict_returns_empty_dict(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Empty filter input should produce empty dict output."""
        assert translator.translate({}) == {}

    def test_scalar_shorthand_preserved(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Scalar shorthand should be preserved (MongoDB handles both forms)."""
        result = translator.translate({"category": "electronics"})
        assert result == {"category": "electronics"}

    def test_scalar_shorthand_numeric(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Numeric scalar should be preserved."""
        result = translator.translate({"price": 42})
        assert result == {"price": 42}

    def test_scalar_shorthand_boolean(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Boolean scalar should be preserved."""
        result = translator.translate({"active": True})
        assert result == {"active": True}

    def test_operator_expression_passed_through(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Operator expression should be passed through unchanged."""
        filters = {"price": {"$gte": 10, "$lte": 100}}
        result = translator.translate(filters)
        assert result == {"price": {"$gte": 10, "$lte": 100}}

    def test_eq_operator_passed_through(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Explicit $eq expression should be passed through."""
        result = translator.translate({"status": {"$eq": "active"}})
        assert result == {"status": {"$eq": "active"}}

    def test_and_with_multiple_conditions(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$and with multiple conditions should translate sub-filters."""
        filters = {
            "$and": [
                {"category": "electronics"},
                {"price": {"$lte": 500}},
            ]
        }
        result = translator.translate(filters)
        assert result == {
            "$and": [
                {"category": "electronics"},
                {"price": {"$lte": 500}},
            ]
        }

    def test_or_with_multiple_conditions(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$or with multiple conditions should translate sub-filters."""
        filters = {
            "$or": [
                {"category": {"$eq": "electronics"}},
                {"category": {"$eq": "books"}},
            ]
        }
        result = translator.translate(filters)
        assert result == {
            "$or": [
                {"category": {"$eq": "electronics"}},
                {"category": {"$eq": "books"}},
            ]
        }

    def test_not_with_nested_condition(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$not with a nested dict condition should translate recursively."""
        filters = {"$not": {"status": {"$eq": "archived"}}}
        result = translator.translate(filters)
        assert result == {"$not": {"status": {"$eq": "archived"}}}

    def test_multiple_fields_combined(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Multiple fields in a single filter dict should all translate."""
        filters = {
            "category": "electronics",
            "price": {"$gte": 10},
            "status": {"$in": ["active", "pending"]},
        }
        result = translator.translate(filters)
        assert result == {
            "category": "electronics",
            "price": {"$gte": 10},
            "status": {"$in": ["active", "pending"]},
        }

    def test_nested_logical_operators(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Nested logical operators should translate recursively."""
        filters = {
            "$and": [
                {
                    "$or": [
                        {"category": "electronics"},
                        {"category": "books"},
                    ]
                },
                {"price": {"$lte": 100}},
            ]
        }
        result = translator.translate(filters)
        assert result == {
            "$and": [
                {
                    "$or": [
                        {"category": "electronics"},
                        {"category": "books"},
                    ]
                },
                {"price": {"$lte": 100}},
            ]
        }

    def test_complex_ecommerce_filter(
        self, translator: MongoFilterTranslator
    ) -> None:
        """Complex real-world e-commerce filter should translate correctly."""
        filters = {
            "$and": [
                {
                    "$or": [
                        {"category": {"$eq": "electronics"}},
                        {"category": {"$eq": "accessories"}},
                    ]
                },
                {"price": {"$gte": 10, "$lte": 500}},
                {"in_stock": {"$exists": True}},
                {"brand": {"$in": ["Apple", "Samsung", "Sony"]}},
                {"discontinued": {"$ne": True}},
            ]
        }
        result = translator.translate(filters)
        assert result == {
            "$and": [
                {
                    "$or": [
                        {"category": {"$eq": "electronics"}},
                        {"category": {"$eq": "accessories"}},
                    ]
                },
                {"price": {"$gte": 10, "$lte": 500}},
                {"in_stock": {"$exists": True}},
                {"brand": {"$in": ["Apple", "Samsung", "Sony"]}},
                {"discontinued": {"$ne": True}},
            ]
        }

    def test_and_with_empty_sub_filters_skipped(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$and sub-filters that translate to empty dicts should be skipped."""
        filters = {
            "$and": [
                {},
                {"price": {"$gt": 0}},
            ]
        }
        result = translator.translate(filters)
        # Empty sub-filter translates to {} which is falsy, so it is skipped
        assert result == {"$and": [{"price": {"$gt": 0}}]}

    def test_or_with_empty_sub_filters_skipped(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$or sub-filters that translate to empty dicts should be skipped."""
        filters = {
            "$or": [
                {},
                {"status": "active"},
            ]
        }
        result = translator.translate(filters)
        assert result == {"$or": [{"status": "active"}]}

    def test_not_with_non_dict_value_ignored(
        self, translator: MongoFilterTranslator
    ) -> None:
        """$not with a non-dict value should not appear in result."""
        # The translate method only processes $not when value is dict
        filters = {"$not": "not_a_dict"}
        result = translator.translate(filters)
        assert result == {}

    def test_translate_preserves_none_values(
        self, translator: MongoFilterTranslator
    ) -> None:
        """None as a scalar value should be preserved (maps to null in Mongo)."""
        result = translator.translate({"field": None})
        assert result == {"field": None}

    def test_translate_preserves_list_values(
        self, translator: MongoFilterTranslator
    ) -> None:
        """List as a field value should be preserved."""
        result = translator.translate({"tags": ["a", "b"]})
        assert result == {"tags": ["a", "b"]}
