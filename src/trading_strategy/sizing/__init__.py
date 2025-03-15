"""
Position sizing for the Autonomous Trading System.

This module provides utilities for determining position sizes in trading strategies,
including risk-based position sizing and other position sizing methods.
"""

from src.trading_strategy.sizing.risk_based_position_sizer import (
    RiskBasedPositionSizer,
    calculate_position_size,
    calculate_max_position_size,
    calculate_position_value,
    calculate_shares_from_risk,
)

__all__ = [
    "RiskBasedPositionSizer",
    "calculate_position_size",
    "calculate_max_position_size",
    "calculate_position_value",
    "calculate_shares_from_risk",
]