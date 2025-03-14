"""
Market calendar for the Autonomous Trading System.

This module provides a calendar class for managing market trading days,
holidays, and sessions.
"""

import datetime
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from .market_hours import (
    EXCHANGE_TIMEZONES,
    MarketSession,
    MarketStatus,
    get_market_session,
    get_market_status,
    get_next_market_close,
    get_next_market_open,
    get_trading_sessions,
    is_market_holiday,
)
from .time_utils import (
    get_day_of_week,
    get_start_of_day,
    is_weekend,
    now,
)


# Configure logger
logger = logging.getLogger(__name__)


class MarketCalendar:
    """
    Market calendar for the Autonomous Trading System.

    This class provides methods for managing market trading days, holidays,
    and sessions for different exchanges.
    """

    def __init__(
        self,
        exchange: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        timezone: Optional[str] = None,
    ):
        """
        Initialize a new MarketCalendar.

        Args:
            exchange: Exchange name (e.g., 'NYSE', 'NASDAQ', 'LSE', 'TSE')
            start_date: Start date for the calendar
            end_date: End date for the calendar
            timezone: Timezone for the calendar (if None, use exchange timezone)
        """
        self.exchange = exchange

        # Validate exchange
        if exchange not in EXCHANGE_TIMEZONES:
            raise ValueError(f"Unknown exchange: {exchange}")

        # Set timezone
        self.timezone = timezone or EXCHANGE_TIMEZONES[exchange]
        self.tz = pytz.timezone(self.timezone)

        # Set date range
        today = now(self.timezone).replace(hour=0, minute=0, second=0, microsecond=0)
        self.start_date = start_date or today - datetime.timedelta(days=365)
        self.end_date = end_date or today + datetime.timedelta(days=365)

        # Ensure dates are timezone-aware
        if self.start_date.tzinfo is None:
            self.start_date = self.tz.localize(self.start_date)
        else:
            self.start_date = self.start_date.astimezone(self.tz)

        if self.end_date.tzinfo is None:
            self.end_date = self.tz.localize(self.end_date)
        else:
            self.end_date = self.end_date.astimezone(self.tz)

        # Initialize calendar
        self._initialize_calendar()

    def _initialize_calendar(self) -> None:
        """Initialize the calendar with trading days and sessions."""
        # Generate all dates in the range
        all_dates = []
        current_date = self.start_date

        while current_date <= self.end_date:
            all_dates.append(current_date)
            current_date += datetime.timedelta(days=1)

        # Create calendar DataFrame
        self.calendar = pd.DataFrame(index=all_dates)
        self.calendar.index.name = "date"

        # Add trading day indicator
        self.calendar["is_trading_day"] = [
            not is_weekend(date) and not is_market_holiday(date, self.exchange)
            for date in self.calendar.index
        ]

        # Add day of week
        self.calendar["day_of_week"] = [
            get_day_of_week(date) for date in self.calendar.index
        ]

        # Add session times for trading days
        self.calendar["market_open"] = None
        self.calendar["market_close"] = None

        for date in self.calendar[self.calendar["is_trading_day"]].index:
            sessions = get_trading_sessions(date, self.exchange)

            if MarketSession.REGULAR in sessions:
                self.calendar.at[date, "market_open"] = sessions[MarketSession.REGULAR][
                    0
                ]
                self.calendar.at[date, "market_close"] = sessions[
                    MarketSession.REGULAR
                ][1]

    def is_trading_day(self, date: datetime.datetime) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if the date is a trading day, False otherwise
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        # Check if date is in calendar
        if date not in self.calendar.index:
            return not is_weekend(date) and not is_market_holiday(date, self.exchange)

        return bool(self.calendar.loc[date, "is_trading_day"])

    def get_trading_days(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[datetime.datetime]:
        """
        Get a list of trading days between two dates.

        Args:
            start_date: Start date (if None, use calendar start date)
            end_date: End date (if None, use calendar end date)

        Returns:
            List of trading days
        """
        # Set default dates
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        # Convert to exchange timezone if needed
        if start_date.tzinfo is not None and start_date.tzinfo != self.tz:
            start_date = start_date.astimezone(self.tz)

        if end_date.tzinfo is not None and end_date.tzinfo != self.tz:
            end_date = end_date.astimezone(self.tz)

        # Normalize to midnight
        start_date = get_start_of_day(start_date)
        end_date = get_start_of_day(end_date)

        # Create a list of trading days manually if there's a timezone mismatch
        # This avoids pandas timezone comparison issues
        if self.calendar.index.tz is not None and (start_date.tzinfo is None or end_date.tzinfo is None):
            trading_days = []
            for date in self.calendar.index:
                # Convert timezone-aware date to naive for comparison if needed
                date_naive = date.replace(tzinfo=None) if date.tzinfo is not None else date
                start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo is not None else start_date
                end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo is not None else end_date
                
                # Compare naive datetimes
                if date_naive >= start_naive and date_naive <= end_naive and self.calendar.loc[date, "is_trading_day"]:
                    trading_days.append(date)
            return trading_days
        
        # Otherwise use pandas filtering
        mask = (self.calendar.index >= start_date) & (self.calendar.index <= end_date)
        trading_days = self.calendar[
            mask & self.calendar["is_trading_day"]
        ].index.tolist()

        return trading_days

    def get_next_trading_day(
        self, date: Optional[datetime.datetime] = None, n: int = 1
    ) -> datetime.datetime:
        """
        Get the next trading day after a given date.

        Args:
            date: Date (if None, use current date)
            n: Number of trading days to advance

        Returns:
            Next trading day
        """
        # Set default date
        date = date or now(self.timezone)

        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        # Get next trading day from calendar
        trading_days = self.calendar[self.calendar.index > date]
        trading_days = trading_days[trading_days["is_trading_day"]]

        if len(trading_days) >= n:
            return trading_days.index[n - 1]

        # If not found in calendar, calculate manually
        current_date = date
        count = 0

        while count < n:
            current_date += datetime.timedelta(days=1)

            if not is_weekend(current_date) and not is_market_holiday(
                current_date, self.exchange
            ):
                count += 1

        return current_date

    def get_previous_trading_day(
        self, date: Optional[datetime.datetime] = None, n: int = 1
    ) -> datetime.datetime:
        """
        Get the previous trading day before a given date.

        Args:
            date: Date (if None, use current date)
            n: Number of trading days to go back

        Returns:
            Previous trading day
        """
        # Set default date
        date = date or now(self.timezone)

        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        # Get previous trading day from calendar
        trading_days = self.calendar[self.calendar.index < date]
        trading_days = trading_days[trading_days["is_trading_day"]]

        if len(trading_days) >= n:
            return trading_days.index[-n]

        # If not found in calendar, calculate manually
        current_date = date
        count = 0

        while count < n:
            current_date -= datetime.timedelta(days=1)

            if not is_weekend(current_date) and not is_market_holiday(
                current_date, self.exchange
            ):
                count += 1

        return current_date

    def get_market_open(self, date: datetime.datetime) -> Optional[datetime.datetime]:
        """
        Get the market open time for a given date.

        Args:
            date: Date

        Returns:
            Market open time, or None if the market is closed on that date
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        # Check if date is in calendar
        if date in self.calendar.index:
            return self.calendar.loc[date, "market_open"]

        # If not in calendar, calculate manually
        if not self.is_trading_day(date):
            return None

        sessions = get_trading_sessions(date, self.exchange)

        if MarketSession.REGULAR in sessions:
            return sessions[MarketSession.REGULAR][0]

        return None

    def get_market_close(self, date: datetime.datetime) -> Optional[datetime.datetime]:
        """
        Get the market close time for a given date.

        Args:
            date: Date

        Returns:
            Market close time, or None if the market is closed on that date
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        # Check if date is in calendar
        if date in self.calendar.index:
            return self.calendar.loc[date, "market_close"]

        # If not in calendar, calculate manually
        if not self.is_trading_day(date):
            return None

        sessions = get_trading_sessions(date, self.exchange)

        if MarketSession.REGULAR in sessions:
            return sessions[MarketSession.REGULAR][1]

        return None

    def is_market_open(self, dt: Optional[datetime.datetime] = None) -> bool:
        """
        Check if the market is open at a given datetime.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            True if the market is open, False otherwise
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Convert to exchange timezone if needed
        if dt.tzinfo is not None and dt.tzinfo != self.tz:
            dt = dt.astimezone(self.tz)

        # Get market status
        status = get_market_status(dt, self.exchange)

        return status == MarketStatus.OPEN

    def get_current_session(self, dt: Optional[datetime.datetime] = None) -> MarketSession:
        """
        Get the current market session at a given datetime.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            Market session
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Convert to exchange timezone if needed
        if dt.tzinfo is not None and dt.tzinfo != self.tz:
            dt = dt.astimezone(self.tz)

        # Get market session
        return get_market_session(dt, self.exchange)

    def get_next_market_open(
        self, dt: Optional[datetime.datetime] = None
    ) -> datetime.datetime:
        """
        Get the next market open time after a given datetime.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            Next market open time
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Convert to exchange timezone if needed
        if dt.tzinfo is not None and dt.tzinfo != self.tz:
            dt = dt.astimezone(self.tz)

        # Get next market open
        return get_next_market_open(dt, self.exchange)

    def get_next_market_close(
        self, dt: Optional[datetime.datetime] = None
    ) -> datetime.datetime:
        """
        Get the next market close time after a given datetime.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            Next market close time
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Convert to exchange timezone if needed
        if dt.tzinfo is not None and dt.tzinfo != self.tz:
            dt = dt.astimezone(self.tz)

        # Get next market close
        return get_next_market_close(dt, self.exchange)

    def get_time_to_open(
        self, dt: Optional[datetime.datetime] = None
    ) -> datetime.timedelta:
        """
        Get the time until the next market open.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            Time until next market open
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Get next market open
        next_open = self.get_next_market_open(dt)

        # Calculate time difference
        return next_open - dt

    def get_time_to_close(
        self, dt: Optional[datetime.datetime] = None
    ) -> datetime.timedelta:
        """
        Get the time until the next market close.

        Args:
            dt: Datetime (if None, use current time)

        Returns:
            Time until next market close
        """
        # Set default datetime
        dt = dt or now(self.timezone)

        # Get next market close
        next_close = self.get_next_market_close(dt)

        # Calculate time difference
        return next_close - dt

    def get_trading_hours(
        self, date: datetime.datetime
    ) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        """
        Get the trading hours for a given date.

        Args:
            date: Date

        Returns:
            Tuple of (market_open, market_close), or None if the market is closed on that date
        """
        # Get market open and close
        market_open = self.get_market_open(date)
        market_close = self.get_market_close(date)

        if market_open is None or market_close is None:
            return None

        return (market_open, market_close)

    def get_trading_minutes(self, date: datetime.datetime) -> int:
        """
        Get the number of trading minutes for a given date.

        Args:
            date: Date

        Returns:
            Number of trading minutes, or 0 if the market is closed on that date
        """
        # Get trading hours
        trading_hours = self.get_trading_hours(date)

        if trading_hours is None:
            return 0

        # Calculate trading minutes
        market_open, market_close = trading_hours
        delta = market_close - market_open

        return int(delta.total_seconds() / 60)

    def get_holidays(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[datetime.datetime]:
        """
        Get a list of holidays between two dates.

        Args:
            start_date: Start date (if None, use calendar start date)
            end_date: End date (if None, use calendar end date)

        Returns:
            List of holidays
        """
        # Set default dates
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        # Convert to exchange timezone if needed
        if start_date.tzinfo is not None and start_date.tzinfo != self.tz:
            start_date = start_date.astimezone(self.tz)

        if end_date.tzinfo is not None and end_date.tzinfo != self.tz:
            end_date = end_date.astimezone(self.tz)

        # Normalize to midnight
        start_date = get_start_of_day(start_date)
        end_date = get_start_of_day(end_date)

        # Get all dates in range
        all_dates = []
        current_date = start_date

        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += datetime.timedelta(days=1)

        # Filter for holidays (weekdays that are not trading days)
        holidays = []

        for date in all_dates:
            if not is_weekend(date) and is_market_holiday(date, self.exchange):
                holidays.append(date)

        return holidays

    def get_early_closes(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[datetime.datetime, datetime.datetime]:
        """
        Get a dictionary of early closes between two dates.

        Args:
            start_date: Start date (if None, use calendar start date)
            end_date: End date (if None, use calendar end date)

        Returns:
            Dictionary mapping dates to early close times
        """
        # This is a placeholder for future implementation
        # Early closes are not currently tracked in the market_hours module
        return {}

    def get_late_opens(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[datetime.datetime, datetime.datetime]:
        """
        Get a dictionary of late opens between two dates.

        Args:
            start_date: Start date (if None, use calendar start date)
            end_date: End date (if None, use calendar end date)

        Returns:
            Dictionary mapping dates to late open times
        """
        # This is a placeholder for future implementation
        # Late opens are not currently tracked in the market_hours module
        return {}

    def get_calendar_df(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> pd.DataFrame:
        """
        Get the calendar as a DataFrame.

        Args:
            start_date: Start date (if None, use calendar start date)
            end_date: End date (if None, use calendar end date)

        Returns:
            Calendar DataFrame
        """
        # Set default dates
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date

        # Handle string dates
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            # Make naive datetime timezone-aware
            start_date = self.tz.localize(start_date)
            
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            # Make naive datetime timezone-aware
            end_date = self.tz.localize(end_date)

        # Convert to exchange timezone if needed
        if start_date.tzinfo is not None and start_date.tzinfo != self.tz:
            start_date = start_date.astimezone(self.tz)

        if end_date.tzinfo is not None and end_date.tzinfo != self.tz:
            end_date = end_date.astimezone(self.tz)

        # Normalize to midnight
        start_date = get_start_of_day(start_date)
        end_date = get_start_of_day(end_date)

        # Get calendar slice
        mask = (self.calendar.index >= start_date) & (self.calendar.index <= end_date)

        return self.calendar[mask].copy()

    def get_business_days_between(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> int:
        """
        Get the number of business days between two dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of business days
        """
        # Convert to exchange timezone if needed
        if start_date.tzinfo is not None and start_date.tzinfo != self.tz:
            start_date = start_date.astimezone(self.tz)

        if end_date.tzinfo is not None and end_date.tzinfo != self.tz:
            end_date = end_date.astimezone(self.tz)

        # Normalize to midnight
        start_date = get_start_of_day(start_date)
        end_date = get_start_of_day(end_date)

        # Get all dates in range
        all_dates = []
        current_date = start_date

        while current_date <= end_date:
            all_dates.append(current_date)
            current_date += datetime.timedelta(days=1)

        # Count business days (weekdays that are not holidays)
        business_days = 0

        for date in all_dates:
            if not is_weekend(date) and not is_market_holiday(date, self.exchange):
                business_days += 1

        return business_days

    def get_trading_days_between(
        self, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> int:
        """
        Get the number of trading days between two dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of trading days
        """
        # This is the same as get_business_days_between for now
        # In the future, this could account for partial trading days
        return self.get_business_days_between(start_date, end_date)

    def add_trading_days(self, date: datetime.datetime, days: int) -> datetime.datetime:
        """
        Add a number of trading days to a date.

        Args:
            date: Date
            days: Number of trading days to add (can be negative)

        Returns:
            New date
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Normalize to midnight
        date = get_start_of_day(date)

        if days == 0:
            return date

        if days > 0:
            # Add trading days
            current_date = date
            count = 0

            while count < days:
                current_date += datetime.timedelta(days=1)

                if self.is_trading_day(current_date):
                    count += 1

            return current_date
        else:
            # Subtract trading days
            current_date = date
            count = 0

            while count < abs(days):
                current_date -= datetime.timedelta(days=1)

                if self.is_trading_day(current_date):
                    count += 1

            return current_date

    def get_trading_day_offset(
        self, date: datetime.datetime, offset: int
    ) -> datetime.datetime:
        """
        Get a date offset by a number of trading days.

        Args:
            date: Date
            offset: Number of trading days to offset (can be negative)

        Returns:
            Offset date
        """
        # This is the same as add_trading_days for now
        return self.add_trading_days(date, offset)

    def get_month_start(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the first trading day of the month.

        Args:
            date: Date

        Returns:
            First trading day of the month
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get first day of month
        first_day = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # If first day is a trading day, return it
        if self.is_trading_day(first_day):
            return first_day

        # Otherwise, get next trading day
        return self.get_next_trading_day(first_day)

    def get_month_end(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the last trading day of the month.

        Args:
            date: Date

        Returns:
            Last trading day of the month
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get first day of next month
        if date.month == 12:
            next_month = date.replace(
                year=date.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            next_month = date.replace(
                month=date.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0
            )

        # Get previous trading day
        return self.get_previous_trading_day(next_month)

    def get_quarter_start(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the first trading day of the quarter.

        Args:
            date: Date

        Returns:
            First trading day of the quarter
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get quarter start month
        quarter = (date.month - 1) // 3
        month = quarter * 3 + 1

        # Get first day of quarter
        first_day = date.replace(
            month=month, day=1, hour=0, minute=0, second=0, microsecond=0
        )

        # If first day is a trading day, return it
        if self.is_trading_day(first_day):
            return first_day

        # Otherwise, get next trading day
        return self.get_next_trading_day(first_day)

    def get_quarter_end(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the last trading day of the quarter.

        Args:
            date: Date

        Returns:
            Last trading day of the quarter
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get quarter end month
        quarter = (date.month - 1) // 3
        month = quarter * 3 + 3

        # Get first day of next quarter
        if month == 12:
            next_quarter = date.replace(
                year=date.year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            next_quarter = date.replace(
                month=month + 1, day=1, hour=0, minute=0, second=0, microsecond=0
            )

        # Get previous trading day
        return self.get_previous_trading_day(next_quarter)

    def get_year_start(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the first trading day of the year.

        Args:
            date: Date

        Returns:
            First trading day of the year
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get first day of year
        first_day = date.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )

        # If first day is a trading day, return it
        if self.is_trading_day(first_day):
            return first_day

        # Otherwise, get next trading day
        return self.get_next_trading_day(first_day)

    def get_year_end(self, date: datetime.datetime) -> datetime.datetime:
        """
        Get the last trading day of the year.

        Args:
            date: Date

        Returns:
            Last trading day of the year
        """
        # Convert to exchange timezone if needed
        if date.tzinfo is not None and date.tzinfo != self.tz:
            date = date.astimezone(self.tz)

        # Get first day of next year
        next_year = date.replace(
            year=date.year + 1,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        # Get previous trading day
        return self.get_previous_trading_day(next_year)
