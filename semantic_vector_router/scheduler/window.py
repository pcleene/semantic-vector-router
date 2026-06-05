"""Maintenance window checking with timezone support."""

from datetime import datetime
from typing import Optional

from semantic_vector_router.scheduler.models import MaintenanceWindow


def is_within_window(
    window: MaintenanceWindow,
    now: Optional[datetime] = None,
) -> bool:
    """Check if current time falls within the maintenance window.

    Args:
        window: MaintenanceWindow configuration.
        now: Override current time (for testing). Should be timezone-naive
             or UTC. If None, uses current UTC time.

    Returns:
        True if the current time is within the allowed window.
    """
    import zoneinfo

    if now is None:
        now = datetime.utcnow()

    # Convert to target timezone
    try:
        tz = zoneinfo.ZoneInfo(window.timezone)
    except (KeyError, Exception):
        # Fall back to UTC if timezone is invalid
        tz = zoneinfo.ZoneInfo("UTC")

    # If now is naive, assume UTC
    if now.tzinfo is None:
        utc = zoneinfo.ZoneInfo("UTC")
        now = now.replace(tzinfo=utc)

    local_now = now.astimezone(tz)

    # Check day of week
    day_name = local_now.strftime("%A").lower()
    if day_name not in [d.lower() for d in window.allowed_days]:
        return False

    # Check hours
    start_hour = window.allowed_hours.get("start", 0)
    end_hour = window.allowed_hours.get("end", 24)
    current_hour = local_now.hour

    # Handle midnight crossing (e.g., start: 22, end: 6)
    if start_hour <= end_hour:
        # Normal range: start <= current < end
        return start_hour <= current_hour < end_hour
    else:
        # Midnight crossing: current >= start OR current < end
        return current_hour >= start_hour or current_hour < end_hour
