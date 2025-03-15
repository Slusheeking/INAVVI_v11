"""
Backtesting Analysis Package

This package provides tools for analyzing backtesting results, including
performance metrics calculation, strategy evaluation, and visualization.
"""

from src.utils.logging import get_logger
from .strategy_analyzer import StrategyAnalyzer
from .backtest_analyzer import BacktestAnalyzer
from .strategy_evaluator import StrategyEvaluator

# Set up logger for this module
logger = get_logger("backtesting.analysis")
logger.info("Backtesting Analysis package initialized")

__all__ = ['StrategyAnalyzer', 'BacktestAnalyzer', 'StrategyEvaluator']