"""
Risk management for the Autonomous Trading System.

This module provides utilities for managing risk in trading strategies,
including stop loss management and profit target management.
"""

from src.trading_strategy.risk.stop_loss_manager import (
    StopLossManager,
    calculate_stop_loss,
    update_stop_loss,
    check_stop_loss,
)

from src.trading_strategy.risk.profit_target_manager import (
    ProfitTargetManager,
    calculate_profit_target,
    update_profit_target,
    check_profit_target,
)

__all__ = [
    "StopLossManager",
    "calculate_stop_loss",
    "update_stop_loss",
    "check_stop_loss",
    "ProfitTargetManager",
    "calculate_profit_target",
    "update_profit_target",
    "check_profit_target",
]