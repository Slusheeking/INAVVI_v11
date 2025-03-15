"""
Backtesting Package

This package provides tools for backtesting trading strategies, including
market simulation, execution simulation, performance analysis, and reporting.
"""

from src.utils.logging import get_logger
from src.backtesting.analysis import StrategyAnalyzer, BacktestAnalyzer, StrategyEvaluator
from src.backtesting.engine import BacktestEngine, MarketSimulator, ExecutionSimulator
from src.backtesting.reporting import PerformanceReport, PerformanceReporter

# Set up logger for this module
logger = get_logger("backtesting")
logger.info("Backtesting package initialized")

__all__ = [
    'StrategyAnalyzer',
    'BacktestAnalyzer',
    'StrategyEvaluator',
    'BacktestEngine',
    'MarketSimulator',
    'ExecutionSimulator',
    'PerformanceReport',
    'PerformanceReporter'
]

logger.debug(f"Loaded components: {', '.join(__all__)}")