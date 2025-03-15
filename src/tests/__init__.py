"""
Tests for the Autonomous Trading System.

This package contains tests for the various components of the trading system,
including unit tests, integration tests, and system tests.
"""

from src.tests.full_system_test import run_full_system_test
from src.tests.performance_metrics import calculate_performance_metrics
from src.tests.system_components import (
    setup_test_environment,
    teardown_test_environment,
    create_test_data,
)

__all__ = [
    "run_full_system_test",
    "calculate_performance_metrics",
    "setup_test_environment",
    "teardown_test_environment",
    "create_test_data",
]