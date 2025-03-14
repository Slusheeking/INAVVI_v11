"""
Backtesting Analysis Package

This package provides tools for analyzing backtesting results, including
performance metrics calculation, strategy evaluation, and visualization.
"""

import logging
from pathlib import Path

from .strategy_analyzer import StrategyAnalyzer

# Set up logging
logger = logging.getLogger('backtesting.analysis')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path('/home/ubuntu/INAVVI_v11-1/src/logs')
log_dir.mkdir(parents=True, exist_ok=True)

# Create file handler
log_file = log_dir / 'backtesting_analysis.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

logger.info("Backtesting Analysis package initialized")

__all__ = ['StrategyAnalyzer']