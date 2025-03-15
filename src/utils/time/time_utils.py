"""
Time utilities for the Autonomous Trading System.

This module provides utilities for working with time, dates, and timestamps,
including market-specific time functions that integrate with market_hours.py
and market_calendar.py.
"""

from datetime import datetime
from typing import Optional, Union

import pandas as pd
import pytz

from src.utils.logging import get_logger
from src.utils.time.base.time_utils_base import (
    DEFAULT_TIMEZONE,
    UTC_TIMEZONE,
    now,
)

logger = get_logger("utils.time.time_utils")


def now_us_eastern() -> datetime:
    """
    Get the current datetime in US Eastern timezone.
    
    Returns:
        Current datetime in US Eastern timezone
    """
    return now(DEFAULT_TIMEZONE)


def timestamp_to_datetime(
    timestamp: Union[int, float], tz: Optional[pytz.timezone] = None
) -> datetime:
    """
    Convert a timestamp to a datetime object.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
        tz: Timezone (default: UTC)
        
    Returns:
        Datetime object
    """
    dt = datetime.fromtimestamp(timestamp, UTC_TIMEZONE)
    if tz is not None:
        dt = dt.astimezone(tz)
    return dt


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert a datetime object to a timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        Unix timestamp (seconds since epoch)
    """
    return dt.timestamp()


def format_datetime(
    dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S %Z"
) -> str:
    """
    Format a datetime object as a string.
    
    Args:
        dt: Datetime object
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_datetime(
    datetime_str: str,
    format_str: str = "%Y-%m-%d %H:%M:%S",
    tz: Optional[pytz.timezone] = None,
) -> datetime:
    """
    Parse a datetime string.
    
    Args:
        datetime_str: Datetime string
        format_str: Format string
        tz: Timezone (default: UTC)
        
    Returns:
        Datetime object
    """
    dt = datetime.strptime(datetime_str, format_str)
    if tz is not None:
        dt = tz.localize(dt)
    else:
        dt = UTC_TIMEZONE.localize(dt)
    return dt


def convert_to_est(dt: datetime) -> datetime:
    """
    Convert a datetime to US Eastern time.
    
    Args:
        dt: Datetime object
        
    Returns:
        Datetime in US Eastern timezone
    """
    if dt.tzinfo is None:
        dt = UTC_TIMEZONE.localize(dt)
    return dt.astimezone(DEFAULT_TIMEZONE)


def convert_to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC.
    
    Args:
        dt: Datetime object
        
    Returns:
        Datetime in UTC timezone
    """
    if dt.tzinfo is None:
        dt = DEFAULT_TIMEZONE.localize(dt)
    return dt.astimezone(UTC_TIMEZONE)


def resample_to_timeframe(
    df: pd.DataFrame, timeframe: str, price_col: str = 'close'
) -> pd.DataFrame:
    """
    Resample a DataFrame to a different timeframe.
    
    Args:
        df: DataFrame with DatetimeIndex
        timeframe: Target timeframe (e.g., '1min', '5min', '1h', '1d')
        price_col: Column name for price data
        
    Returns:
        Resampled DataFrame
    """
    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Map timeframe to pandas frequency string
    freq_map = {
        '1min': '1T',
        '5min': '5T',
        '15min': '15T',
        '30min': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W',
    }
    
    # Get the pandas frequency string
    freq = freq_map.get(timeframe.lower())
    if freq is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Resample the DataFrame
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled.dropna()