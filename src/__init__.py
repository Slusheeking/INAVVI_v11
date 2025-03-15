"""
Autonomous Trading System

This package contains the implementation of an autonomous trading system
with machine learning-based prediction, risk management, and execution.
"""

import os

# Initialize logging system
from src.utils.logging import setup_logger, get_logger

# Create root logger
root_logger = get_logger("root")

# Log initialization message
root_logger.info("Initializing Autonomous Trading System")

# Log project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
root_logger.info(f"Project root directory: {project_root}")

# Log Python path
root_logger.debug(f"Python path: {os.environ.get('PYTHONPATH', '')}")

# Version information
__version__ = "1.1.0"
root_logger.info(f"Autonomous Trading System version: {__version__}")

# Export public API
__all__ = [
    "setup_logger",
    "get_logger",
]