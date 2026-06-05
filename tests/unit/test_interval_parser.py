"""Comprehensive unit tests for the interval parser."""

import pytest

from semantic_vector_router.scheduler.interval import parse_interval


# --- Basic single-unit intervals ---


class TestBasicSeconds:
    def test_parse_30_seconds(self) -> None:
        assert parse_interval("30s") == 30.0

    def test_parse_1_second(self) -> None:
        assert parse_interval("1s") == 1.0

    def test_parse_120_seconds(self) -> None:
        assert parse_interval("120s") == 120.0


class TestBasicMinutes:
    def test_parse_5_minutes(self) -> None:
        assert parse_interval("5m") == 300.0

    def test_parse_1_minute(self) -> None:
        assert parse_interval("1m") == 60.0

    def test_parse_90_minutes(self) -> None:
        assert parse_interval("90m") == 5400.0


class TestBasicHours:
    def test_parse_1_hour(self) -> None:
        assert parse_interval("1h") == 3600.0

    def test_parse_6_hours(self) -> None:
        assert parse_interval("6h") == 21600.0

    def test_parse_12_hours(self) -> None:
        assert parse_interval("12h") == 43200.0

    def test_parse_24_hours(self) -> None:
        assert parse_interval("24h") == 86400.0


# --- Named intervals ---


class TestNamedIntervals:
    def test_daily_equals_86400_seconds(self) -> None:
        assert parse_interval("daily") == 86400.0

    def test_weekly_equals_604800_seconds(self) -> None:
        assert parse_interval("weekly") == 604800.0


# --- Compound intervals ---


class TestCompoundIntervals:
    def test_1_hour_30_minutes(self) -> None:
        assert parse_interval("1h30m") == 5400.0

    def test_2_hours_15_minutes(self) -> None:
        assert parse_interval("2h15m") == 8100.0

    def test_1_hour_30_minutes_45_seconds(self) -> None:
        assert parse_interval("1h30m45s") == 5445.0

    def test_5_minutes_30_seconds(self) -> None:
        assert parse_interval("5m30s") == 330.0

    def test_3_hours_0_minutes_10_seconds(self) -> None:
        assert parse_interval("3h0m10s") == 10810.0

    def test_compound_with_spaces(self) -> None:
        assert parse_interval("1h 30m") == 5400.0

    def test_compound_three_units_with_spaces(self) -> None:
        assert parse_interval("2h 15m 30s") == 8130.0


# --- Case insensitivity ---


class TestCaseInsensitivity:
    def test_uppercase_h(self) -> None:
        assert parse_interval("1H") == 3600.0

    def test_uppercase_m(self) -> None:
        assert parse_interval("5M") == 300.0

    def test_uppercase_s(self) -> None:
        assert parse_interval("30S") == 30.0

    def test_mixed_case_compound(self) -> None:
        assert parse_interval("1H30m") == 5400.0

    def test_all_uppercase_compound(self) -> None:
        assert parse_interval("2H15M30S") == 8130.0


# --- Error cases ---


class TestInvalidFormats:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("")

    def test_alphabetic_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("abc")

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("10x")

    def test_word_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("hello")

    def test_negative_hour_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("-1h")

    def test_number_without_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("100")

    def test_unit_without_number_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("h")

    def test_trailing_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unexpected characters"):
            parse_interval("1hfoo")

    def test_leading_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unexpected characters"):
            parse_interval("foo1h")


# --- Zero / non-positive ---


class TestZeroInterval:
    def test_zero_seconds_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_interval("0s")

    def test_zero_minutes_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_interval("0m")

    def test_zero_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_interval("0h")

    def test_compound_all_zeros_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parse_interval("0h0m0s")


# --- Return type ---


class TestReturnType:
    def test_returns_float_for_seconds(self) -> None:
        result = parse_interval("10s")
        assert isinstance(result, float)

    def test_returns_float_for_named(self) -> None:
        result = parse_interval("daily")
        assert isinstance(result, float)

    def test_returns_float_for_compound(self) -> None:
        result = parse_interval("1h30m")
        assert isinstance(result, float)


# --- Named interval case sensitivity ---


class TestNamedIntervalCaseSensitivity:
    def test_daily_uppercase_raises(self) -> None:
        """Named intervals are case-sensitive (only lowercase 'daily' works)."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("Daily")

    def test_weekly_uppercase_raises(self) -> None:
        """Named intervals are case-sensitive (only lowercase 'weekly' works)."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            parse_interval("Weekly")


# --- Large values ---


class TestLargeValues:
    def test_100_hours(self) -> None:
        assert parse_interval("100h") == 360000.0

    def test_1000_seconds(self) -> None:
        assert parse_interval("1000s") == 1000.0

    def test_999_minutes(self) -> None:
        assert parse_interval("999m") == 59940.0
