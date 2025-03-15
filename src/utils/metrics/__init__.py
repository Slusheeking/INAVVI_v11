"""
Metrics utilities for the Autonomous Trading System.

This module provides utilities for calculating trading metrics and performance
statistics for trading strategies.
"""

from src.utils.metrics.metrics_utils import (
    calculate_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_kelly_criterion,
    calculate_average_trade,
    calculate_average_win,
    calculate_average_loss,
    calculate_win_loss_ratio,
    calculate_max_consecutive_wins,
    calculate_max_consecutive_losses,
    calculate_volatility,
    calculate_var,
    calculate_cvar,
    calculate_beta,
    calculate_alpha,
    calculate_information_ratio,
    calculate_trading_metrics,
    calculate_trade_statistics,
)

__all__ = [
    "calculate_returns",
    "calculate_cumulative_returns",
    "calculate_drawdowns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_omega_ratio",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_expectancy",
    "calculate_kelly_criterion",
    "calculate_average_trade",
    "calculate_average_win",
    "calculate_average_loss",
    "calculate_win_loss_ratio",
    "calculate_max_consecutive_wins",
    "calculate_max_consecutive_losses",
    "calculate_volatility",
    "calculate_var",
    "calculate_cvar",
    "calculate_beta",
    "calculate_alpha",
    "calculate_information_ratio",
    "calculate_trading_metrics",
    "calculate_trade_statistics",
]