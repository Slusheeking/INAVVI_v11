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
    MarketHours,
    MarketStatus,
    MarketSession,
)

from src.utils.time.market_calendar import (
    get_trading_days,
    is_trading_day,
)

__all__ = [
    # time_utils
    "now",
    "now_us_eastern",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_datetime",
    "parse_datetime",
    "is_market_open",
    "get_next_market_open",
    "get_next_market_close",
    "get_previous_trading_day",
    "get_next_trading_day",
    "get_trading_days_between",
    "time_until_market_open",
    "time_until_market_close",
    "wait_until_market_open",
    "wait_until_market_close",
    "convert_to_est",
    "convert_to_utc",
    "is_same_trading_day",
    "get_current_bar_timestamp",
    "get_trading_hours_today",
    "get_trading_sessions_today",
    "get_current_trading_session",
    "is_within_trading_hours",
    "is_within_extended_hours",
    "get_time_to_next_bar",
    "wait_for_next_bar",
    "resample_to_timeframe",
    
    # market_hours
    "MarketHours",
    "MarketStatus",
    "MarketSession",
    "get_market_hours",
    "get_market_status",
    "get_trading_sessions",
    
    # market_calendar
    "get_trading_days",
    "is_trading_day",
]