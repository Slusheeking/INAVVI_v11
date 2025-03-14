"""
Trading Strategy Execution Package

This package contains components for executing trading orders and managing trade execution.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.execution.order_generator import OrderGenerator

# Set up logger for this module
logger = get_logger("trading_strategy.execution")
logger.info("Initializing trading strategy execution package")

__all__ = ["OrderGenerator"]

logger.debug(f"Loaded execution components: {', '.join(__all__)}")