"""
Continuous Learning Retraining Package

This package contains components for retraining machine learning models based on new data
and performance feedback, enabling the system to adapt to changing market conditions.
"""

from src.utils.logging.logger import get_logger
from src.continuous_learning.retraining.model_retrainer import ModelRetrainer

# Set up logger for this module
logger = get_logger("continuous_learning.retraining")
logger.info("Initializing continuous learning retraining package")

__all__ = ["ModelRetrainer"]

logger.debug(f"Loaded components: {', '.join(__all__)}")