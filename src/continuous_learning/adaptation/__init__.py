"""
Continuous Learning Adaptation Package

This package contains components for adapting trading strategies based on performance
analysis and changing market conditions.
"""

from src.utils.logging import get_logger
from src.continuous_learning.adaptation.strategy_adapter import StrategyAdapter

# Set up logger for this module
logger = get_logger("continuous_learning.adaptation")
logger.info("Initializing continuous learning adaptation package")

__all__ = ["StrategyAdapter"]

logger.debug(f"Loaded components: {', '.join(__all__)}")