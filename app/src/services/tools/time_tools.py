"""Time tools — current date/time queries."""
from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def current_datetime() -> str:
    """Get the current date, time, and timezone.
    Use when asked about today's date, current time, day of week, week number,
    or any time-relative question ('tomorrow', 'next week', etc.).
    """
    now = datetime.now()
    return (
        f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(Timezone: {now.astimezone().tzname()}, "
        f"Day: {now.strftime('%A')}, "
        f"Week: {now.isocalendar()[1]})"
    )


__all__ = ["current_datetime"]
