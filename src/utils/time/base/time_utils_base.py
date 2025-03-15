"""
Core time utilities for the Autonomous Trading System.

This module provides basic time utility functions that are used across
the time utilities package. These functions are separated here to avoid
circular dependencies between time_utils.py, market_calendar.py, and
market_hours.py.
"""

from datetime import datetime
from typing import Optional

import pytz

from src.utils.logging import get_logger

logger = get_logger("utils.time.base.time_utils_base")

# Default timezone for US markets
DEFAULT_TIMEZONE = pytz.timezone("America/New_York")
UTC_TIMEZONE = pytz.UTC


def now(tz: Optional[pytz.timezone] = None) -> datetime:
    """
    Get the current datetime with timezone.
    
    Args:
        tz: Timezone (default: UTC)
        
    Returns:
        Current datetime with timezone
    """
    current_time = datetime.now(UTC_TIMEZONE)
    if tz is not None:
        current_time = current_time.astimezone(tz)
    return current_time


def get_day_of_week(dt: datetime) -> int:
    """
    Get the day of the week for a given datetime.
    
    Args:
        dt: Datetime object
        
    Returns:
        Day of the week (0 = Monday, 6 = Sunday)
    """
    return dt.weekday()


def is_weekend(dt: datetime) -> bool:
    """
    Check if a given datetime is on a weekend.
    
    Args:
        dt: Datetime object
        
    Returns:
        True if the datetime is on a weekend, False otherwise
    """
    return dt.weekday() >= 5


def get_start_of_day(dt: datetime) -> datetime:
    """
    Get the start of the day for a given datetime.
    
    Args:
        dt: Datetime object
        
    Returns:
        Datetime object at the start of the day
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)