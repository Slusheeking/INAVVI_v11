"""
Trading Strategy Selection Package

This package contains components for selecting trading instruments and timeframes.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.selection.ticker_selector import TickerSelector
from src.trading_strategy.selection.timeframe_selector import TimeframeSelector

# Set up logger for this module
logger = get_logger("trading_strategy.selection")
logger.info("Initializing trading strategy selection package")

__all__ = ["TickerSelector", "TimeframeSelector"]

logger.debug(f"Loaded selection components: {', '.join(__all__)}")