"""
Market hours utilities for the Autonomous Trading System.

This module provides utilities for working with market hours and trading
sessions for different exchanges.
"""

import datetime
from enum import Enum
from typing import Dict, List, Tuple

import pytz

from .base.time_utils_base import get_day_of_week, is_weekend
from src.utils.logging import get_logger

# Configure logger
logger = get_logger("utils.time.market_hours")

class MarketStatus(Enum):
    """Enum representing the status of a market."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    POST_MARKET = "post_market"
    HOLIDAY = "holiday"

    def __str__(self):
        return self.value

class MarketSession(Enum):
    """Enum representing different market sessions."""

    REGULAR = "regular"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"
    CLOSED = "closed"

    def __str__(self):
        return self.value


# Market holidays for different exchanges
# Format: {exchange: {year: [(month, day), ...]}}
MARKET_HOLIDAYS = {
    "NYSE": {
        2023: [
            (1, 2),  # New Year's Day (observed)
            (1, 16),  # Martin Luther King Jr. Day
            (2, 20),  # Presidents' Day
            (4, 7),  # Good Friday
            (5, 29),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 4),  # Labor Day
            (11, 23),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2024: [
            (1, 1),  # New Year's Day
            (1, 15),  # Martin Luther King Jr. Day
            (2, 19),  # Presidents' Day
            (3, 29),  # Good Friday
            (5, 27),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 2),  # Labor Day
            (11, 28),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2025: [
            (1, 1),  # New Year's Day
            (1, 20),  # Martin Luther King Jr. Day
            (2, 17),  # Presidents' Day
            (4, 18),  # Good Friday
            (5, 26),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 1),  # Labor Day
            (11, 27),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2026: [
            (1, 1),  # New Year's Day
            (1, 19),  # Martin Luther King Jr. Day
            (2, 16),  # Presidents' Day
            (4, 3),  # Good Friday
            (5, 25),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 3),  # Independence Day (observed)
            (9, 7),  # Labor Day
            (11, 26),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
    },
    "NASDAQ": {
        2023: [
            (1, 2),  # New Year's Day (observed)
            (1, 16),  # Martin Luther King Jr. Day
            (2, 20),  # Presidents' Day
            (4, 7),  # Good Friday
            (5, 29),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 4),  # Labor Day
            (11, 23),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2024: [
            (1, 1),  # New Year's Day
            (1, 15),  # Martin Luther King Jr. Day
            (2, 19),  # Presidents' Day
            (3, 29),  # Good Friday
            (5, 27),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 2),  # Labor Day
            (11, 28),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2025: [
            (1, 1),  # New Year's Day
            (1, 20),  # Martin Luther King Jr. Day
            (2, 17),  # Presidents' Day
            (4, 18),  # Good Friday
            (5, 26),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 1),  # Labor Day
            (11, 27),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
        2026: [
            (1, 1),  # New Year's Day
            (1, 19),  # Martin Luther King Jr. Day
            (2, 16),  # Presidents' Day
            (4, 3),  # Good Friday
            (5, 25),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 3),  # Independence Day (observed)
            (9, 7),  # Labor Day
            (11, 26),  # Thanksgiving Day
            (12, 25),  # Christmas Day
        ],
    },
    "LSE": {
        2023: [
            (1, 2),  # New Year's Day (observed)
            (4, 7),  # Good Friday
            (4, 10),  # Easter Monday
            (5, 1),  # Early May Bank Holiday
            (5, 29),  # Spring Bank Holiday
            (8, 28),  # Summer Bank Holiday
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        ],
        2024: [
            (1, 1),  # New Year's Day
            (3, 29),  # Good Friday
            (4, 1),  # Easter Monday
            (5, 6),  # Early May Bank Holiday
            (5, 27),  # Spring Bank Holiday
            (8, 26),  # Summer Bank Holiday
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        ],
        2025: [
            (1, 1),  # New Year's Day
            (4, 18),  # Good Friday
            (4, 21),  # Easter Monday
            (5, 5),  # Early May Bank Holiday
            (5, 26),  # Spring Bank Holiday
            (8, 25),  # Summer Bank Holiday
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        ],
        2026: [
            (1, 1),  # New Year's Day
            (4, 3),  # Good Friday
            (4, 6),  # Easter Monday
            (5, 4),  # Early May Bank Holiday
            (5, 25),  # Spring Bank Holiday
            (8, 31),  # Summer Bank Holiday
            (12, 25),  # Christmas Day
            (12, 28),  # Boxing Day (observed)
        ],
    },
    "TSE": {
        2023: [
            (1, 2),  # New Year's Day (observed)
            (1, 3),  # Bank Holiday
            (1, 9),  # Coming of Age Day
            (2, 11),  # National Foundation Day
            (2, 23),  # Emperor's Birthday
            (3, 21),  # Vernal Equinox Day
            (4, 29),  # Showa Day
            (5, 3),  # Constitution Memorial Day
            (5, 4),  # Greenery Day
            (5, 5),  # Children's Day
            (7, 17),  # Marine Day
            (8, 11),  # Mountain Day
            (9, 18),  # Respect for the Aged Day
            (9, 23),  # Autumnal Equinox Day
            (10, 9),  # Sports Day
            (11, 3),  # Culture Day
            (11, 23),  # Labor Thanksgiving Day
            (12, 31),  # New Year's Eve
        ],
        2024: [
            (1, 1),  # New Year's Day
            (1, 2),  # Bank Holiday
            (1, 3),  # Bank Holiday
            (1, 8),  # Coming of Age Day
            (2, 11),  # National Foundation Day
            (2, 12),  # National Foundation Day (observed)
            (2, 23),  # Emperor's Birthday
            (3, 20),  # Vernal Equinox Day
            (4, 29),  # Showa Day
            (5, 3),  # Constitution Memorial Day
            (5, 4),  # Greenery Day
            (5, 5),  # Children's Day
            (5, 6),  # Children's Day (observed)
            (7, 15),  # Marine Day
            (8, 11),  # Mountain Day
            (8, 12),  # Mountain Day (observed)
            (9, 16),  # Respect for the Aged Day
            (9, 22),  # Autumnal Equinox Day
            (9, 23),  # Autumnal Equinox Day (observed)
            (10, 14),  # Sports Day
            (11, 3),  # Culture Day
            (11, 4),  # Culture Day (observed)
            (11, 23),  # Labor Thanksgiving Day
            (12, 31),  # New Year's Eve
        ],
        2025: [
            (1, 1),  # New Year's Day
            (1, 2),  # Bank Holiday
            (1, 3),  # Bank Holiday
            (1, 13),  # Coming of Age Day
            (2, 11),  # National Foundation Day
            (2, 23),  # Emperor's Birthday
            (2, 24),  # Emperor's Birthday (observed)
            (3, 20),  # Vernal Equinox Day
            (4, 29),  # Showa Day
            (5, 3),  # Constitution Memorial Day
            (5, 4),  # Greenery Day
            (5, 5),  # Children's Day
            (5, 6),  # Children's Day (observed)
            (7, 21),  # Marine Day
            (8, 11),  # Mountain Day
            (9, 15),  # Respect for the Aged Day
            (9, 23),  # Autumnal Equinox Day
            (10, 13),  # Sports Day
            (11, 3),  # Culture Day
            (11, 23),  # Labor Thanksgiving Day
            (11, 24),  # Labor Thanksgiving Day (observed)
            (12, 31),  # New Year's Eve
        ],
        2026: [
            (1, 1),  # New Year's Day
            (1, 2),  # Bank Holiday
            (1, 11),  # Coming of Age Day
            (2, 11),  # National Foundation Day
            (2, 23),  # Emperor's Birthday
            (3, 20),  # Vernal Equinox Day
            (4, 29),  # Showa Day
            (5, 3),  # Constitution Memorial Day
            (5, 4),  # Greenery Day
            (5, 5),  # Children's Day
            (5, 6),  # Children's Day (observed)
            (7, 20),  # Marine Day
            (8, 11),  # Mountain Day
            (9, 21),  # Respect for the Aged Day
            (9, 22),  # Autumnal Equinox Day
            (10, 12),  # Sports Day
            (11, 3),  # Culture Day
            (11, 23),  # Labor Thanksgiving Day
            (12, 31),  # New Year's Eve
        ],
    },
}

# Market hours for different exchanges
# Format: {exchange: {session: {day_of_week: (start_time, end_time)}}}
# Day of week: 0 = Monday, 6 = Sunday
# Times are in exchange local time zone
MARKET_HOURS = {
    "NYSE": {
        MarketSession.PRE_MARKET: {
            0: ("04:00", "09:30"),
            1: ("04:00", "09:30"),
            2: ("04:00", "09:30"),
            3: ("04:00", "09:30"),
            4: ("04:00", "09:30"),
        },
        MarketSession.REGULAR: {
            0: ("09:30", "16:00"),
            1: ("09:30", "16:00"),
            2: ("09:30", "16:00"),
            3: ("09:30", "16:00"),
            4: ("09:30", "16:00"),
        },
        MarketSession.AFTER_HOURS: {
            0: ("16:00", "20:00"),
            1: ("16:00", "20:00"),
            2: ("16:00", "20:00"),
            3: ("16:00", "20:00"),
            4: ("16:00", "20:00"),
        },
    },
    "NASDAQ": {
        MarketSession.PRE_MARKET: {
            0: ("04:00", "09:30"),
            1: ("04:00", "09:30"),
            2: ("04:00", "09:30"),
            3: ("04:00", "09:30"),
            4: ("04:00", "09:30"),
        },
        MarketSession.REGULAR: {
            0: ("09:30", "16:00"),
            1: ("09:30", "16:00"),
            2: ("09:30", "16:00"),
            3: ("09:30", "16:00"),
            4: ("09:30", "16:00"),
        },
        MarketSession.AFTER_HOURS: {
            0: ("16:00", "20:00"),
            1: ("16:00", "20:00"),
            2: ("16:00", "20:00"),
            3: ("16:00", "20:00"),
            4: ("16:00", "20:00"),
        },
    },
    "LSE": {
        MarketSession.PRE_MARKET: {
            0: ("07:00", "08:00"),
            1: ("07:00", "08:00"),
            2: ("07:00", "08:00"),
            3: ("07:00", "08:00"),
            4: ("07:00", "08:00"),
        },
        MarketSession.REGULAR: {
            0: ("08:00", "16:30"),
            1: ("08:00", "16:30"),
            2: ("08:00", "16:30"),
            3: ("08:00", "16:30"),
            4: ("08:00", "16:30"),
        },
        MarketSession.AFTER_HOURS: {
            0: ("16:30", "17:15"),
            1: ("16:30", "17:15"),
            2: ("16:30", "17:15"),
            3: ("16:30", "17:15"),
            4: ("16:30", "17:15"),
        },
    },
    "TSE": {
        MarketSession.REGULAR: {
            0: ("09:00", "15:00"),
            1: ("09:00", "15:00"),
            2: ("09:00", "15:00"),
            3: ("09:00", "15:00"),
            4: ("09:00", "15:00"),
        }
    },
}

# Time zones for different exchanges
EXCHANGE_TIMEZONES = {
    "NYSE": "America/New_York",
    "NASDAQ": "America/New_York",
    "LSE": "Europe/London",
    "TSE": "Asia/Tokyo",
}


def is_market_holiday(dt: datetime.datetime, exchange: str) -> bool:
    """
    Check if a date is a market holiday for a given exchange.

    Args:
        dt: Datetime object
        exchange: Exchange name

    Returns:
        True if the date is a market holiday, False otherwise
    """
    logger.debug(f"Checking if {dt.date()} is a holiday for {exchange}")
    
    if exchange not in MARKET_HOLIDAYS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    year = dt.year
    month = dt.month
    day = dt.day

    if year not in MARKET_HOLIDAYS[exchange]:
        # If we don't have holiday data for this year, assume it's not a holiday
        logger.warning(
            f"No holiday data available for {exchange} in {year}. "
            f"Assuming {dt.date()} is not a holiday."
        )
        return False

    is_holiday = (month, day) in MARKET_HOLIDAYS[exchange][year]
    if is_holiday:
        logger.debug(f"{dt.date()} is a holiday for {exchange}")
    
    return is_holiday


def get_market_status(
    dt: datetime.datetime, exchange: str, use_exchange_time: bool = True
) -> MarketStatus:
    """
    Get the market status for a given datetime and exchange.

    Args:
        dt: Datetime object
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetime to exchange local time

    Returns:
        Market status
    """
    logger.debug(f"Getting market status for {dt} in {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and dt.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        dt = dt.astimezone(exchange_tz)

    # Check if it's a weekend
    if is_weekend(dt):
        logger.debug(f"{dt.date()} is a weekend, market is closed")
        return MarketStatus.CLOSED

    # Check if it's a holiday
    if is_market_holiday(dt, exchange):
        logger.debug(f"{dt.date()} is a holiday for {exchange}")
        return MarketStatus.HOLIDAY

    # Get the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = get_day_of_week(dt)

    # Check if it's during regular market hours
    if (
        MarketSession.REGULAR in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.REGULAR]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.REGULAR][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market is open at {dt} for {exchange}")
            return MarketStatus.OPEN

    # Check if it's during pre-market hours
    if (
        MarketSession.PRE_MARKET in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.PRE_MARKET]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.PRE_MARKET][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market is in pre-market at {dt} for {exchange}")
            return MarketStatus.PRE_MARKET

    # Check if it's during after-hours
    if (
        MarketSession.AFTER_HOURS in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.AFTER_HOURS]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][
            MarketSession.AFTER_HOURS
        ][day_of_week]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market is in after-hours at {dt} for {exchange}")
            return MarketStatus.AFTER_HOURS

    # If none of the above, the market is closed
    logger.debug(f"Market is closed at {dt} for {exchange}")
    return MarketStatus.CLOSED


def get_market_session(
    dt: datetime.datetime, exchange: str, use_exchange_time: bool = True
) -> MarketSession:
    """
    Get the market session for a given datetime and exchange.

    Args:
        dt: Datetime object
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetime to exchange local time

    Returns:
        Market session
    """
    logger.debug(f"Getting market session for {dt} in {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and dt.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        dt = dt.astimezone(exchange_tz)

    # Check if it's a weekend
    if is_weekend(dt):
        logger.debug(f"{dt.date()} is a weekend, market session is closed")
        return MarketSession.CLOSED

    # Check if it's a holiday
    if is_market_holiday(dt, exchange):
        logger.debug(f"{dt.date()} is a holiday for {exchange}, market session is closed")
        return MarketSession.CLOSED

    # Get the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = get_day_of_week(dt)

    # Check if it's during regular market hours
    if (
        MarketSession.REGULAR in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.REGULAR]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.REGULAR][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market session is REGULAR at {dt} for {exchange}")
            return MarketSession.REGULAR

    # Check if it's during pre-market hours
    if (
        MarketSession.PRE_MARKET in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.PRE_MARKET]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.PRE_MARKET][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market session is PRE_MARKET at {dt} for {exchange}")
            return MarketSession.PRE_MARKET

    # Check if it's during after-hours
    if (
        MarketSession.AFTER_HOURS in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.AFTER_HOURS]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][
            MarketSession.AFTER_HOURS
        ][day_of_week]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = dt.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start_time <= dt < end_time:
            logger.debug(f"Market session is AFTER_HOURS at {dt} for {exchange}")
            return MarketSession.AFTER_HOURS

    # Check if it's during overnight hours (between after-hours and pre-market)
    if (
        MarketSession.AFTER_HOURS in MARKET_HOURS[exchange]
        and MarketSession.PRE_MARKET in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.AFTER_HOURS]
        and (day_of_week + 1) % 7 in MARKET_HOURS[exchange][MarketSession.PRE_MARKET]
    ):
        after_start_str, after_end_str = MARKET_HOURS[exchange][
            MarketSession.AFTER_HOURS
        ][day_of_week]
        pre_start_str, pre_end_str = MARKET_HOURS[exchange][MarketSession.PRE_MARKET][
            (day_of_week + 1) % 7
        ]

        after_end_hour, after_end_minute = map(int, after_end_str.split(":"))
        pre_start_hour, pre_start_minute = map(int, pre_start_str.split(":"))

        after_end_time = dt.replace(
            hour=after_end_hour, minute=after_end_minute, second=0, microsecond=0
        )
        pre_start_time = dt.replace(
            hour=pre_start_hour, minute=pre_start_minute, second=0, microsecond=0
        ) + datetime.timedelta(days=1)

        if after_end_time <= dt < pre_start_time:
            logger.debug(f"Market session is OVERNIGHT at {dt} for {exchange}")
            return MarketSession.OVERNIGHT

    # If none of the above, the market is closed
    logger.debug(f"Market session is CLOSED at {dt} for {exchange}")
    return MarketSession.CLOSED


def get_next_market_open(
    dt: datetime.datetime, exchange: str, use_exchange_time: bool = True
) -> datetime.datetime:
    """
    Get the next market open time after a given datetime.

    Args:
        dt: Datetime object
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetime to exchange local time

    Returns:
        Datetime object representing the next market open time
    """
    logger.debug(f"Getting next market open time after {dt} for {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and dt.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        dt = dt.astimezone(exchange_tz)

    # Start from the current datetime
    current_dt = dt

    # Look ahead up to 10 days (to handle weekends and holidays)
    for _ in range(10):
        # Get the day of the week (0 = Monday, 6 = Sunday)
        day_of_week = get_day_of_week(current_dt)

        # Check if it's a weekend
        if is_weekend(current_dt):
            # Move to the next day
            current_dt = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + datetime.timedelta(days=1)
            continue

        # Check if it's a holiday
        if is_market_holiday(current_dt, exchange):
            # Move to the next day
            current_dt = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + datetime.timedelta(days=1)
            continue

        # Check if regular market hours are defined for this day
        if (
            MarketSession.REGULAR in MARKET_HOURS[exchange]
            and day_of_week in MARKET_HOURS[exchange][MarketSession.REGULAR]
        ):
            start_time_str, _ = MARKET_HOURS[exchange][MarketSession.REGULAR][
                day_of_week
            ]

            start_hour, start_minute = map(int, start_time_str.split(":"))

            market_open = current_dt.replace(
                hour=start_hour, minute=start_minute, second=0, microsecond=0
            )

            # If market open is in the future, return it
            if market_open > dt:
                logger.debug(f"Next market open for {exchange} is at {market_open}")
                return market_open

        # Move to the next day
        current_dt = current_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + datetime.timedelta(days=1)

    # If we couldn't find a market open time within 10 days, raise an exception
    logger.error(f"Could not find next market open time for {exchange} within 10 days from {dt}")
    raise ValueError(
        f"Could not find next market open time for {exchange} within 10 days"
    )


def is_market_open(time_zone: str, current_date: datetime.datetime) -> bool:
    """
    Check if the market is open at the given datetime.

    Args:
        time_zone: Time zone string (e.g., "America/New_York")
        current_date: Datetime object

    Returns:
        True if the market is open, False otherwise
    """
    logger.debug(f"Checking if market is open at {current_date} in {time_zone}")
    
    # Find the exchange for the given time zone
    exchange = None
    for exch, tz in EXCHANGE_TIMEZONES.items():
        if tz == time_zone:
            exchange = exch
            break
    
    if exchange is None:
        logger.error(f"Unknown time zone: {time_zone}")
        raise ValueError(f"Unknown time zone: {time_zone}")
    
    # Get the market status
    status = get_market_status(current_date, exchange, use_exchange_time=True)
    
    # Return True if the market is open
    return status == MarketStatus.OPEN


def get_next_market_close(
    dt: datetime.datetime, exchange: str, use_exchange_time: bool = True
) -> datetime.datetime:
    """
    Get the next market close time after a given datetime.

    Args:
        dt: Datetime object
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetime to exchange local time

    Returns:
        Datetime object representing the next market close time
    """
    logger.debug(f"Getting next market close time after {dt} for {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and dt.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        dt = dt.astimezone(exchange_tz)

    # Start from the current datetime
    current_dt = dt

    # Look ahead up to 10 days (to handle weekends and holidays)
    for _ in range(10):
        # Get the day of the week (0 = Monday, 6 = Sunday)
        day_of_week = get_day_of_week(current_dt)

        # Check if it's a weekend
        if is_weekend(current_dt):
            # Move to the next day
            current_dt = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + datetime.timedelta(days=1)
            continue

        # Check if it's a holiday
        if is_market_holiday(current_dt, exchange):
            # Move to the next day
            current_dt = current_dt.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + datetime.timedelta(days=1)
            continue

        # Check if regular market hours are defined for this day
        if (
            MarketSession.REGULAR in MARKET_HOURS[exchange]
            and day_of_week in MARKET_HOURS[exchange][MarketSession.REGULAR]
        ):
            _, end_time_str = MARKET_HOURS[exchange][MarketSession.REGULAR][day_of_week]

            end_hour, end_minute = map(int, end_time_str.split(":"))

            market_close = current_dt.replace(
                hour=end_hour, minute=end_minute, second=0, microsecond=0
            )

            # If market close is in the future, return it
            if market_close > dt:
                logger.debug(f"Next market close for {exchange} is at {market_close}")
                return market_close

        # Move to the next day
        current_dt = current_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + datetime.timedelta(days=1)

    # If we couldn't find a market close time within 10 days, raise an exception
    logger.error(f"Could not find next market close time for {exchange} within 10 days from {dt}")
    raise ValueError(
        f"Could not find next market close time for {exchange} within 10 days"
    )


def get_trading_days(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    exchange: str,
    use_exchange_time: bool = True,
) -> List[datetime.datetime]:
    """
    Get a list of trading days between two dates.

    Args:
        start_date: Start date
        end_date: End date
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetimes to exchange local time

    Returns:
        List of trading days
    """
    logger.debug(f"Getting trading days between {start_date.date()} and {end_date.date()} for {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and start_date.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        start_date = start_date.astimezone(exchange_tz)
        end_date = end_date.astimezone(exchange_tz)

    # Normalize dates to midnight
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate all dates between start_date and end_date
    all_dates = []
    current_date = start_date

    while current_date <= end_date:
        all_dates.append(current_date)
        current_date += datetime.timedelta(days=1)

    # Filter out weekends and holidays
    trading_days = []

    for date in all_dates:
        # Skip weekends
        if is_weekend(date):
            continue

        # Skip holidays
        if is_market_holiday(date, exchange):
            continue

        trading_days.append(date)

    logger.debug(f"Found {len(trading_days)} trading days for {exchange} between {start_date.date()} and {end_date.date()}")
    return trading_days


def get_trading_sessions(
    date: datetime.datetime, exchange: str, use_exchange_time: bool = True
) -> Dict[MarketSession, Tuple[datetime.datetime, datetime.datetime]]:
    """
    Get the trading sessions for a given date and exchange.

    Args:
        date: Date
        exchange: Exchange name
        use_exchange_time: Whether to convert the datetime to exchange local time

    Returns:
        Dictionary mapping session types to (start_time, end_time) tuples
    """
    logger.debug(f"Getting trading sessions for {date.date()} in {exchange}")
    if exchange not in MARKET_HOURS:
        logger.error(f"Unknown exchange: {exchange}")
        raise ValueError(f"Unknown exchange: {exchange}")

    # Convert to exchange local time if needed
    if use_exchange_time and date.tzinfo is not None:
        exchange_tz = pytz.timezone(EXCHANGE_TIMEZONES[exchange])
        date = date.astimezone(exchange_tz)

    # Normalize date to midnight
    date = date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Check if it's a weekend
    if is_weekend(date):
        logger.debug(f"{date.date()} is a weekend, no trading sessions")
        return {}

    # Check if it's a holiday
    if is_market_holiday(date, exchange):
        logger.debug(f"{date.date()} is a holiday for {exchange}, no trading sessions")
        return {}

    # Get the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = get_day_of_week(date)

    # Get the trading sessions
    sessions = {}

    # Regular session
    if (
        MarketSession.REGULAR in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.REGULAR]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.REGULAR][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = date.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = date.replace(
            hour=end_hour, minute=end_minute, second=0, microsecond=0
        )

        sessions[MarketSession.REGULAR] = (start_time, end_time)

    # Pre-market session
    if (
        MarketSession.PRE_MARKET in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.PRE_MARKET]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][MarketSession.PRE_MARKET][
            day_of_week
        ]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = date.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = date.replace(
            hour=end_hour, minute=end_minute, second=0, microsecond=0
        )

        sessions[MarketSession.PRE_MARKET] = (start_time, end_time)

    # After-hours session
    if (
        MarketSession.AFTER_HOURS in MARKET_HOURS[exchange]
        and day_of_week in MARKET_HOURS[exchange][MarketSession.AFTER_HOURS]
    ):
        start_time_str, end_time_str = MARKET_HOURS[exchange][
            MarketSession.AFTER_HOURS
        ][day_of_week]

        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))

        start_time = date.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
        end_time = date.replace(
            hour=end_hour, minute=end_minute, second=0, microsecond=0
        )

        sessions[MarketSession.AFTER_HOURS] = (start_time, end_time)

    logger.debug(f"Found {len(sessions)} trading sessions for {date.date()} in {exchange}")
    return sessions
