"""
Backtesting Reporting Package

This package provides tools for generating detailed performance reports
from backtesting results, including visualizations and metrics summaries.
"""

from src.utils.logging import get_logger
from .performance_reporter import PerformanceReport, PerformanceReporter

# Set up logger for this module
logger = get_logger("backtesting.reporting")
logger.info("Backtesting Reporting package initialized")

__all__ = ['PerformanceReport', 'PerformanceReporter']