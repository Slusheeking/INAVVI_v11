"""
Backtesting Engine Package

This package provides the core components for running backtests, including
market simulation, execution simulation, and backtest orchestration.
"""

from src.utils.logging import get_logger
from .backtest_engine import BacktestEngine
from .market_simulator import MarketSimulator
from .execution_simulator import ExecutionSimulator

# Set up logger for this module
logger = get_logger("backtesting.engine")
logger.info("Backtesting Engine package initialized")

__all__ = ['BacktestEngine', 'MarketSimulator', 'ExecutionSimulator']