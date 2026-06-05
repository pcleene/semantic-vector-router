"""Interval string parser for the job scheduler."""

import re


def parse_interval(interval: str) -> float:
    """Parse interval string to seconds.

    Supported formats:
      "30s", "5m", "1h", "6h", "daily", "weekly"
      "1h30m", "2h15m" (compound)

    Returns: interval in seconds.
    Raises: ValueError for invalid format.
    """
    named = {
        "daily": 86400.0,
        "weekly": 604800.0,
    }
    if interval in named:
        return named[interval]

    units = {"s": 1.0, "m": 60.0, "h": 3600.0}
    pattern = re.compile(r"(\d+)\s*([smh])", re.IGNORECASE)
    matches = pattern.findall(interval)

    if not matches:
        raise ValueError(
            f"Invalid interval format: '{interval}'. "
            f"Expected formats: '30s', '5m', '1h', '6h', 'daily', 'weekly', '1h30m'"
        )

    # Validate no extra characters
    stripped = re.sub(r"\d+\s*[smhSMH]", "", interval).strip()
    if stripped:
        raise ValueError(
            f"Invalid interval format: '{interval}'. "
            f"Unexpected characters: '{stripped}'"
        )

    total = 0.0
    for value_str, unit in matches:
        total += int(value_str) * units[unit.lower()]

    if total <= 0:
        raise ValueError(f"Interval must be positive, got: '{interval}'")

    return total
