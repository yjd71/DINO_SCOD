"""Shared formatting helpers for human-readable runtime logs."""

from datetime import datetime


LOG_TIME_FORMAT = "%m-%d %H:%M:%S"


def current_time() -> str:
    """Return local time in the compact format used by CLI logs."""

    return datetime.now().strftime(LOG_TIME_FORMAT)
