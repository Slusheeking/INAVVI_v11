"""
Trading Strategy Sizing Package

This package contains components for determining position sizes based on risk parameters.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.sizing.risk_based_position_sizer import RiskBasedPositionSizer

# Set up logger for this module
logger = get_logger("trading_strategy.sizing")
logger.info("Initializing trading strategy sizing package")

__all__ = ["RiskBasedPositionSizer"]

logger.debug(f"Loaded sizing components: {', '.join(__all__)}")