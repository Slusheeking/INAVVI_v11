"""
INAVVI Autonomous Trading System

This is the root package for the INAVVI Autonomous Trading System.
It provides a high-performance, AI-driven trading platform with
advanced risk management and execution capabilities.
"""

# Import logging utilities for easy access
from src.utils.logging import (
    setup_logger,
    get_logger,
    log_to_file,
    get_all_loggers,
)

# Version information
__version__ = "1.1.0"

# Package name
__package_name__ = "INAVVI_v11-1"

# Make logging utilities available at the package level
__all__ = [
    "setup_logger",
    "get_logger",
    "log_to_file",
    "get_all_loggers",
]