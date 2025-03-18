#!/usr/bin/env python3
"""
Market Hours Handler

This module provides utilities for handling market hours detection and fallback mechanisms
for when the market is closed or API services are unavailable.
"""

import os
import time
import logging
import datetime
import pytz
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_hours_handler')

class MarketHoursHandler:
    """
    Handler for market hours detection and fallback mechanisms
    """
    
    def __init__(self):
        """Initialize the market hours handler"""
        # US market holidays for 2025 (update annually)
        self.market_holidays_2025 = [
            datetime.date(2025, 1, 1),   # New Year's Day
            datetime.date(2025, 1, 20),  # Martin Luther King Jr. Day
            datetime.date(2025, 2, 17),  # Presidents' Day
            datetime.date(2025, 4, 18),  # Good Friday
            datetime.date(2025, 5, 26),  # Memorial Day
            datetime.date(2025, 6, 19),  # Juneteenth
            datetime.date(2025, 7, 4),   # Independence Day
            datetime.date(2025, 9, 1),   # Labor Day
            datetime.date(2025, 11, 27), # Thanksgiving Day
            datetime.date(2025, 12, 25)  # Christmas Day
        ]
        
        # Initialize timezone
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def is_market_open(self, timestamp: Optional[datetime.datetime] = None) -> bool:
        """
        Check if the market is open at the given timestamp
        
        Args:
            timestamp: The timestamp to check (default: current time)
            
        Returns:
            bool: True if the market is open, False otherwise
        """
        # Use current time if no timestamp is provided
        if timestamp is None:
            timestamp = datetime.datetime.now(self.eastern_tz)
        elif timestamp.tzinfo is None:
            # Assume UTC if no timezone is provided
            timestamp = pytz.utc.localize(timestamp).astimezone(self.eastern_tz)
            
        # Get date components
        date = timestamp.date()
        weekday = timestamp.weekday()
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Check if it's a weekend
        if weekday >= 5:  # Saturday or Sunday
            logger.debug(f"Market closed: Weekend ({weekday})")
            return False
            
        # Check if it's a holiday
        if date in self.market_holidays_2025:
            logger.debug(f"Market closed: Holiday ({date})")
            return False
            
        # Check if it's during market hours (9:30 AM - 4:00 PM ET)
        market_open = (hour > 9 or (hour == 9 and minute >= 30))
        market_closed = hour >= 16
        
        if market_open and not market_closed:
            logger.debug(f"Market open: {hour}:{minute:02d} ET")
            return True
        else:
            logger.debug(f"Market closed: {hour}:{minute:02d} ET (outside trading hours)")
            return False
            
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get the current market status
        
        Returns:
            dict: Market status information
        """
        now = datetime.datetime.now(self.eastern_tz)
        is_open = self.is_market_open(now)
        
        market_status = "open" if is_open else "closed"
        logger.info(f"Market status: {market_status}")
        
        return {
            "status": "OK",
            "market": market_status,
            "serverTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exchanges": {
                "nyse": market_status,
                "nasdaq": market_status,
                "otc": market_status
            },
            "currencies": {
                "fx": "open",  # Forex markets are open 24/5
                "crypto": "open"  # Crypto markets are open 24/7
            },
            "is_fallback": True  # Flag to indicate this is fallback data
        }
        
    def get_trading_date_range(self, days_back: int = 30) -> tuple:
        """
        Get a date range for trading data that includes only trading days
        
        Args:
            days_back: Number of calendar days to look back
            
        Returns:
            tuple: (from_date, to_date) as strings in YYYY-MM-DD format
        """
        # Start with today's date
        end_date = datetime.datetime.now(self.eastern_tz).date()
        
        # For the start date, go back more days to account for weekends and holidays
        # Approximately 5/7 of days are trading days, so add 40% more days
        calendar_days_back = int(days_back * 1.4)
        start_date = end_date - datetime.timedelta(days=calendar_days_back)
        
        # Format dates as strings
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")

        return (from_date, to_date)
        
    def get_previous_trading_day(self) -> str:
        """
        Get the previous trading day
        
        Returns:
            str: Previous trading day in YYYY-MM-DD format
        """
        # Start with yesterday
        date = datetime.datetime.now(self.eastern_tz).date() - datetime.timedelta(days=1)
        
        # Go back until we find a trading day
        for _ in range(10):  # Limit to 10 iterations to prevent infinite loop
            weekday = date.weekday()
            
            # Check if it's a weekday and not a holiday
            if weekday < 5 and date not in self.market_holidays_2025:
                return date.strftime("%Y-%m-%d")
                
            # Go back one more day
            date -= datetime.timedelta(days=1)
            
        # Fallback to yesterday if we couldn't find a trading day
        return (datetime.datetime.now(self.eastern_tz).date() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        

# Singleton instance
market_hours_handler = MarketHoursHandler()

# For direct testing
if __name__ == "__main__":
    handler = MarketHoursHandler()
    
    # Check current market status
    status = handler.get_market_status()
    print(f"Current market status: {status['market']}")
    
    # Check if market is open
    is_open = handler.is_market_open()
    print(f"Is market open: {is_open}")
    
    # Get trading date range
    from_date, to_date = handler.get_trading_date_range(30)
    print(f"Trading date range: {from_date} to {to_date}")
    
    # Get previous trading day
    prev_day = handler.get_previous_trading_day()
    print(f"Previous trading day: {prev_day}")