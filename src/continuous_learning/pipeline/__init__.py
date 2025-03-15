"""
Continuous Learning Pipeline Package

This package contains components for the continuous learning pipeline, which orchestrates
the process of analyzing performance, adapting strategies, and retraining models.
"""

from src.utils.logging import get_logger
from src.continuous_learning.pipeline.continuous_learning_pipeline import ContinuousLearningPipeline

# Set up logger for this module
logger = get_logger("continuous_learning.pipeline")
logger.info("Initializing continuous learning pipeline package")

__all__ = ["ContinuousLearningPipeline"]

logger.debug(f"Pipeline package initialized with components: {', '.join(__all__)}")