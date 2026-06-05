"""Comprehensive unit tests for the maintenance window checker.

Tests is_within_window() with various day/hour/timezone configurations.
Uses 2026-02-14 (a Saturday) as the base date for deterministic tests.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from semantic_vector_router.scheduler.models import MaintenanceWindow
from semantic_vector_router.scheduler.window import is_within_window


# ---------------------------------------------------------------------------
# 1. All days allowed, within hours → True
# ---------------------------------------------------------------------------

class TestAllDaysWithinHours:
    """Window defaults (all days, 0-24) should always return True."""

    def test_default_window_midday(self) -> None:
        window = MaintenanceWindow()
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday 12:00
        assert is_within_window(window, now) is True

    def test_default_window_midnight(self) -> None:
        window = MaintenanceWindow()
        now = datetime(2026, 2, 14, 0, 0, 0)  # Saturday 00:00
        assert is_within_window(window, now) is True

    def test_default_window_just_before_midnight(self) -> None:
        window = MaintenanceWindow()
        now = datetime(2026, 2, 14, 23, 59, 0)  # Saturday 23:59
        assert is_within_window(window, now) is True

    def test_within_hours_morning(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 6, "end": 18})
        now = datetime(2026, 2, 14, 10, 0, 0)  # Saturday 10:00
        assert is_within_window(window, now) is True

    def test_within_hours_at_start_boundary(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 6, "end": 18})
        now = datetime(2026, 2, 14, 6, 0, 0)  # Saturday 06:00 (inclusive)
        assert is_within_window(window, now) is True


# ---------------------------------------------------------------------------
# 2. All days allowed, outside hours → False
# ---------------------------------------------------------------------------

class TestAllDaysOutsideHours:
    """All days allowed but current hour is outside the allowed range."""

    def test_before_start_hour(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 6, "end": 18})
        now = datetime(2026, 2, 14, 3, 0, 0)  # Saturday 03:00
        assert is_within_window(window, now) is False

    def test_after_end_hour(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 6, "end": 18})
        now = datetime(2026, 2, 14, 20, 0, 0)  # Saturday 20:00
        assert is_within_window(window, now) is False

    def test_at_end_boundary(self) -> None:
        """End hour is exclusive: hour == end should be False."""
        window = MaintenanceWindow(allowed_hours={"start": 6, "end": 18})
        now = datetime(2026, 2, 14, 18, 0, 0)  # Saturday 18:00 (exclusive)
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 3. Day not in allowed list → False
# ---------------------------------------------------------------------------

class TestDayNotAllowed:
    """When the current day is not in allowed_days, result is False."""

    def test_weekday_only_on_saturday(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["monday", "tuesday", "wednesday", "thursday", "friday"],
        )
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday
        assert is_within_window(window, now) is False

    def test_sunday_only_on_saturday(self) -> None:
        window = MaintenanceWindow(allowed_days=["sunday"])
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday
        assert is_within_window(window, now) is False

    def test_wednesday_only_on_saturday(self) -> None:
        window = MaintenanceWindow(allowed_days=["wednesday"])
        now = datetime(2026, 2, 14, 2, 0, 0)  # Saturday
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 4. Day in allowed list + within hours → True
# ---------------------------------------------------------------------------

class TestDayAllowedAndWithinHours:
    """Correct day AND correct hour → True."""

    def test_saturday_in_list_within_hours(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["saturday"],
            allowed_hours={"start": 8, "end": 20},
        )
        now = datetime(2026, 2, 14, 14, 0, 0)  # Saturday 14:00
        assert is_within_window(window, now) is True

    def test_multiple_days_within_hours(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["friday", "saturday", "sunday"],
            allowed_hours={"start": 0, "end": 6},
        )
        now = datetime(2026, 2, 14, 3, 0, 0)  # Saturday 03:00
        assert is_within_window(window, now) is True


# ---------------------------------------------------------------------------
# 5. Midnight crossing (start=22, end=6)
# ---------------------------------------------------------------------------

class TestMidnightCrossing:
    """Hours that span midnight: start > end (e.g., 22-06)."""

    def test_at_23_within_crossing_window(self) -> None:
        """23:00 is >= 22, should be True."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 23, 0, 0)  # Saturday 23:00
        assert is_within_window(window, now) is True

    def test_at_10_outside_crossing_window(self) -> None:
        """10:00 is between 6 and 22, should be False."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 10, 0, 0)  # Saturday 10:00
        assert is_within_window(window, now) is False

    def test_at_03_within_crossing_window(self) -> None:
        """03:00 is < 6 (the end), should be True."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 3, 0, 0)  # Saturday 03:00
        assert is_within_window(window, now) is True

    def test_at_22_start_boundary_crossing(self) -> None:
        """22:00 exactly, should be True (inclusive start)."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 22, 0, 0)  # Saturday 22:00
        assert is_within_window(window, now) is True

    def test_at_06_end_boundary_crossing(self) -> None:
        """06:00 exactly, end is exclusive, should be False."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 6, 0, 0)  # Saturday 06:00
        assert is_within_window(window, now) is False

    def test_at_12_midday_outside_crossing(self) -> None:
        """12:00 should be outside [22, 06)."""
        window = MaintenanceWindow(allowed_hours={"start": 22, "end": 6})
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday 12:00
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 6. Timezone handling: Window in US/Eastern, now in UTC
# ---------------------------------------------------------------------------

class TestTimezoneConversion:
    """Verify timezone conversion from UTC to the window's timezone."""

    def test_utc_noon_is_eastern_morning(self) -> None:
        """UTC 12:00 = Eastern 07:00 (EST, UTC-5). Window 6-10 Eastern → True."""
        window = MaintenanceWindow(
            allowed_hours={"start": 6, "end": 10},
            timezone="US/Eastern",
        )
        # UTC Saturday 12:00 → Eastern Saturday 07:00
        now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is True

    def test_utc_afternoon_is_eastern_morning_outside(self) -> None:
        """UTC 16:00 = Eastern 11:00 (EST). Window 6-10 Eastern → False."""
        window = MaintenanceWindow(
            allowed_hours={"start": 6, "end": 10},
            timezone="US/Eastern",
        )
        now = datetime(2026, 2, 14, 16, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is False

    def test_naive_utc_assumed(self) -> None:
        """Naive datetime is assumed UTC, then converted to Eastern."""
        window = MaintenanceWindow(
            allowed_hours={"start": 6, "end": 10},
            timezone="US/Eastern",
        )
        # Naive 12:00 → assumed UTC → Eastern 07:00 → within [6, 10) → True
        now = datetime(2026, 2, 14, 12, 0, 0)
        assert is_within_window(window, now) is True

    def test_utc_early_morning_previous_eastern_day(self) -> None:
        """UTC Saturday 03:00 = Eastern Friday 22:00. Window Saturday only → False."""
        window = MaintenanceWindow(
            allowed_days=["saturday"],
            allowed_hours={"start": 0, "end": 24},
            timezone="US/Eastern",
        )
        # UTC Saturday 03:00 → Eastern Friday 22:00 → day is Friday → False
        now = datetime(2026, 2, 14, 3, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 7. Timezone Europe/London with DST awareness
# ---------------------------------------------------------------------------

class TestDSTAwareness:
    """Europe/London is UTC+0 in winter, UTC+1 in summer (BST)."""

    def test_london_winter_same_as_utc(self) -> None:
        """February is GMT (UTC+0), so UTC 10:00 = London 10:00."""
        window = MaintenanceWindow(
            allowed_hours={"start": 9, "end": 12},
            timezone="Europe/London",
        )
        now = datetime(2026, 2, 14, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is True

    def test_london_summer_bst_offset(self) -> None:
        """July is BST (UTC+1), so UTC 10:00 = London 11:00."""
        window = MaintenanceWindow(
            allowed_hours={"start": 10, "end": 12},
            timezone="Europe/London",
        )
        # 2026-07-11 is a Saturday; UTC 10:00 → London 11:00 (BST)
        now = datetime(2026, 7, 11, 10, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is True

    def test_london_summer_boundary_outside(self) -> None:
        """UTC 08:00 in July = London 09:00. Window 10-12 → False."""
        window = MaintenanceWindow(
            allowed_hours={"start": 10, "end": 12},
            timezone="Europe/London",
        )
        # 2026-07-11 is a Saturday; UTC 08:00 → London 09:00 (BST)
        now = datetime(2026, 7, 11, 8, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 8. Edge: start=0, end=24 (all hours) → True
# ---------------------------------------------------------------------------

class TestAllHoursEdge:
    """start=0, end=24 means the entire day is allowed."""

    def test_all_hours_at_midnight(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 24})
        now = datetime(2026, 2, 14, 0, 0, 0)
        assert is_within_window(window, now) is True

    def test_all_hours_at_23(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 24})
        now = datetime(2026, 2, 14, 23, 0, 0)
        assert is_within_window(window, now) is True

    def test_all_hours_midday(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 24})
        now = datetime(2026, 2, 14, 12, 30, 0)
        assert is_within_window(window, now) is True


# ---------------------------------------------------------------------------
# 9. Edge: start=0, end=0 (no hours) → False
# ---------------------------------------------------------------------------

class TestNoHoursEdge:
    """start=0, end=0 means zero-width range — nothing is within it."""

    def test_no_hours_at_midnight(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 0})
        now = datetime(2026, 2, 14, 0, 0, 0)
        assert is_within_window(window, now) is False

    def test_no_hours_midday(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 0})
        now = datetime(2026, 2, 14, 12, 0, 0)
        assert is_within_window(window, now) is False

    def test_no_hours_at_23(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 0, "end": 0})
        now = datetime(2026, 2, 14, 23, 0, 0)
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 10. Weekend-only window (allowed_days=["saturday", "sunday"])
# ---------------------------------------------------------------------------

class TestWeekendOnly:
    """Window restricted to Saturday and Sunday."""

    def test_saturday_allowed(self) -> None:
        window = MaintenanceWindow(allowed_days=["saturday", "sunday"])
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday
        assert is_within_window(window, now) is True

    def test_sunday_allowed(self) -> None:
        window = MaintenanceWindow(allowed_days=["saturday", "sunday"])
        now = datetime(2026, 2, 15, 12, 0, 0)  # Sunday
        assert is_within_window(window, now) is True

    def test_monday_not_allowed(self) -> None:
        window = MaintenanceWindow(allowed_days=["saturday", "sunday"])
        now = datetime(2026, 2, 16, 12, 0, 0)  # Monday
        assert is_within_window(window, now) is False

    def test_friday_not_allowed(self) -> None:
        window = MaintenanceWindow(allowed_days=["saturday", "sunday"])
        now = datetime(2026, 2, 13, 12, 0, 0)  # Friday
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 11. Single day window
# ---------------------------------------------------------------------------

class TestSingleDayWindow:
    """Window restricted to exactly one day."""

    def test_single_day_match(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["saturday"],
            allowed_hours={"start": 2, "end": 4},
        )
        now = datetime(2026, 2, 14, 3, 0, 0)  # Saturday 03:00
        assert is_within_window(window, now) is True

    def test_single_day_wrong_day(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["wednesday"],
            allowed_hours={"start": 2, "end": 4},
        )
        now = datetime(2026, 2, 14, 3, 0, 0)  # Saturday, not Wednesday
        assert is_within_window(window, now) is False

    def test_single_day_right_day_wrong_hour(self) -> None:
        window = MaintenanceWindow(
            allowed_days=["saturday"],
            allowed_hours={"start": 2, "end": 4},
        )
        now = datetime(2026, 2, 14, 10, 0, 0)  # Saturday 10:00
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 12. Invalid timezone fallback to UTC
# ---------------------------------------------------------------------------

class TestInvalidTimezoneFallback:
    """Invalid timezone strings should fall back to UTC."""

    def test_invalid_timezone_falls_back_to_utc(self) -> None:
        window = MaintenanceWindow(
            allowed_hours={"start": 10, "end": 14},
            timezone="Invalid/NotATimezone",
        )
        now = datetime(2026, 2, 14, 12, 0, 0)  # Saturday 12:00 UTC
        # Falls back to UTC, 12 is in [10, 14) → True
        assert is_within_window(window, now) is True

    def test_invalid_timezone_outside_hours(self) -> None:
        window = MaintenanceWindow(
            allowed_hours={"start": 10, "end": 14},
            timezone="Fake/Zone",
        )
        now = datetime(2026, 2, 14, 8, 0, 0)  # Saturday 08:00 UTC
        # Falls back to UTC, 8 is NOT in [10, 14) → False
        assert is_within_window(window, now) is False

    def test_empty_timezone_string_falls_back(self) -> None:
        window = MaintenanceWindow(
            allowed_hours={"start": 10, "end": 14},
            timezone="",
        )
        now = datetime(2026, 2, 14, 12, 0, 0)
        # Empty string is invalid → falls back to UTC → 12 in [10,14) → True
        assert is_within_window(window, now) is True


# ---------------------------------------------------------------------------
# 13. Case-insensitivity for day names
# ---------------------------------------------------------------------------

class TestDayNameCaseInsensitivity:
    """Day matching should be case-insensitive."""

    def test_uppercase_day_in_config(self) -> None:
        window = MaintenanceWindow(allowed_days=["Saturday"])
        now = datetime(2026, 2, 14, 12, 0, 0)
        assert is_within_window(window, now) is True

    def test_mixed_case_day_in_config(self) -> None:
        window = MaintenanceWindow(allowed_days=["SATURDAY", "Sunday"])
        now = datetime(2026, 2, 14, 12, 0, 0)
        assert is_within_window(window, now) is True


# ---------------------------------------------------------------------------
# 14. now=None uses current time (integration-style sanity check)
# ---------------------------------------------------------------------------

class TestNowDefaulting:
    """When now is None, the function should use utcnow() internally."""

    def test_now_none_uses_current_time(self) -> None:
        """Default window (all days, 0-24, UTC) should always be True."""
        window = MaintenanceWindow()
        # No explicit now — relies on real clock; default window covers everything
        assert is_within_window(window) is True


# ---------------------------------------------------------------------------
# 15. Timezone-aware datetime input (not naive, not UTC)
# ---------------------------------------------------------------------------

class TestTimezoneAwareDatetimeInput:
    """Pass a timezone-aware datetime that isn't UTC."""

    def test_eastern_aware_input_converted_to_window_tz(self) -> None:
        """Pass Eastern-aware datetime, window is Pacific."""
        window = MaintenanceWindow(
            allowed_hours={"start": 6, "end": 10},
            timezone="US/Pacific",
        )
        # Eastern 12:00 = Pacific 09:00 → in [6, 10) → True
        eastern = ZoneInfo("US/Eastern")
        now = datetime(2026, 2, 14, 12, 0, 0, tzinfo=eastern)
        assert is_within_window(window, now) is True

    def test_eastern_aware_input_outside_pacific_window(self) -> None:
        """Eastern 15:00 = Pacific 12:00 → outside [6, 10)."""
        window = MaintenanceWindow(
            allowed_hours={"start": 6, "end": 10},
            timezone="US/Pacific",
        )
        eastern = ZoneInfo("US/Eastern")
        now = datetime(2026, 2, 14, 15, 0, 0, tzinfo=eastern)
        assert is_within_window(window, now) is False


# ---------------------------------------------------------------------------
# 16. Missing keys in allowed_hours dict (fallback to defaults)
# ---------------------------------------------------------------------------

class TestAllowedHoursDefaults:
    """allowed_hours dict missing 'start' or 'end' falls back to 0 / 24."""

    def test_missing_start_defaults_to_zero(self) -> None:
        window = MaintenanceWindow(allowed_hours={"end": 12})
        now = datetime(2026, 2, 14, 5, 0, 0)  # 05:00, in [0, 12)
        assert is_within_window(window, now) is True

    def test_missing_end_defaults_to_24(self) -> None:
        window = MaintenanceWindow(allowed_hours={"start": 12})
        now = datetime(2026, 2, 14, 20, 0, 0)  # 20:00, in [12, 24)
        assert is_within_window(window, now) is True

    def test_empty_dict_defaults_to_all_hours(self) -> None:
        window = MaintenanceWindow(allowed_hours={})
        now = datetime(2026, 2, 14, 15, 0, 0)  # 15:00, in [0, 24)
        assert is_within_window(window, now) is True
