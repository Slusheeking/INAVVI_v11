"""
Time utilities for the Autonomous Trading System.

This module provides utilities for working with time, dates, and timestamps,
including market-specific time functions that integrate with market_hours.py
and market_calendar.py.
"""

from src.utils.time.time_utils import (
    now,
    now_us_eastern,
    timestamp_to_datetime,
    datetime_to_timestamp,
    format_datetime,
    parse_datetime,
    convert_to_est,
    convert_to_utc,
    resample_to_timeframe,
)

from src.utils.time.market_hours import (
    is_market_open,
    get_next_market_open,
    get_next_market_close,
    get_market_status,
    get_trading_sessions,
    MarketStatus,
    MarketSession,
)

# Import the MarketCalendar class and create wrapper functions
from src.utils.time.market_calendar import MarketCalendar
from datetime import datetime
from typing import List, Optional


# Create wrapper functions for MarketCalendar methods
def get_trading_days(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    exchange: str = "NYSE"
) -> List[datetime]:
    """
    Get a list of trading days between two dates.
    
    This is a wrapper for MarketCalendar.get_trading_days.
    
    Args:
        start_date: Start date
        end_date: End date
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        List of trading days
    """
    calendar = MarketCalendar(exchange=exchange, start_date=start_date, end_date=end_date)
    return calendar.get_trading_days(start_date, end_date)


def is_trading_day(date: datetime, exchange: str = "NYSE") -> bool:
    """
    Check if a date is a trading day.
    
    This is a wrapper for MarketCalendar.is_trading_day.
    
    Args:
        date: Date to check
        exchange: Exchange name (default: "NYSE")
        
    Returns:
        True if the date is a trading day, False otherwise
    """
    calendar = MarketCalendar(exchange=exchange)
    return calendar.is_trading_day(date)


__all__ = [
    # time_utils
    "now",
    "now_us_eastern",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_datetime",
    "parse_datetime",
    "convert_to_est",
    "convert_to_utc",
    "resample_to_timeframe",
    
    # market_hours
    "is_market_open",
    "get_next_market_open",
    "get_next_market_close",
    "get_market_status",
    "get_trading_sessions",
    "MarketStatus",
    "MarketSession",
    
    # market_calendar
    "get_trading_days",
    "is_trading_day",
    "MarketCalendar",
]