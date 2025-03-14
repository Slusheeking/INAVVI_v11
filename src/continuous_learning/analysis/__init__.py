"""
Continuous Learning Analysis Package

This package contains components for analyzing trading performance, identifying areas for
improvement, and generating recommendations for strategy adaptation.
"""

from src.utils.logging.logger import get_logger
from src.continuous_learning.analysis.performance_analyzer import PerformanceAnalyzer

# Set up logger for this module
logger = get_logger("continuous_learning.analysis")
logger.info("Initializing continuous learning analysis package")

__all__ = ["PerformanceAnalyzer"]

logger.debug(f"Loaded components: {', '.join(__all__)}")