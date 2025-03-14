"""
Continuous Learning Pipeline Package

This package contains components for the continuous learning pipeline, which orchestrates
the process of analyzing performance, adapting strategies, and retraining models.
"""

from src.utils.logging.logger import get_logger

# Set up logger for this module
logger = get_logger("continuous_learning.pipeline")
logger.info("Initializing continuous learning pipeline package")

# Import pipeline components when implemented
# from src.continuous_learning.pipeline.continuous_learning_pipeline import ContinuousLearningPipeline

__all__ = []  # Add pipeline components when implemented

logger.debug("Pipeline package initialized (no components loaded yet)")