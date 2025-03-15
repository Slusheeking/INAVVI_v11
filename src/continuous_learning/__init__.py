"""
Continuous Learning Package

This package contains components for continuous learning in the autonomous trading system,
enabling the system to adapt to changing market conditions through performance analysis,
strategy adaptation, and model retraining.
"""

from src.utils.logging.logger import get_logger
from src.continuous_learning.analysis.performance_analyzer import PerformanceAnalyzer
from src.continuous_learning.adaptation.strategy_adapter import StrategyAdapter
from src.continuous_learning.pipeline.continuous_learning_pipeline import ContinuousLearningPipeline
from src.continuous_learning.retraining.model_retrainer import ModelRetrainer

# Set up logger for this module
logger = get_logger("continuous_learning")
logger.info("Initializing continuous learning package")

__all__ = [
    "PerformanceAnalyzer",
    "StrategyAdapter",
    "ContinuousLearningPipeline",
    "ModelRetrainer"
]

logger.debug(f"Continuous learning package initialized with components: {', '.join(__all__)}")