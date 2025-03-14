"""
Trading Strategy Signals Package

This package contains components for generating trading signals and detecting market patterns.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.signals.entry_signal_generator import EntrySignalGenerator
from src.trading_strategy.signals.peak_detector import PeakDetector

# Set up logger for this module
logger = get_logger("trading_strategy.signals")
logger.info("Initializing trading strategy signals package")

__all__ = ["EntrySignalGenerator", "PeakDetector"]

logger.debug(f"Loaded signal components: {', '.join(__all__)}")