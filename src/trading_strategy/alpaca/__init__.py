"""
Alpaca integration for the Autonomous Trading System.

This module provides utilities for integrating with the Alpaca trading API,
including client, position management, and trade execution.
"""

from src.trading_strategy.alpaca.alpaca_client import (
    AlpacaClient,
    create_alpaca_client,
    get_account_info,
    get_positions,
    get_orders,
)

from src.trading_strategy.alpaca.alpaca_position_manager import (
    AlpacaPositionManager,
    get_position,
    update_position,
    close_position,
)

from src.trading_strategy.alpaca.alpaca_trade_executor import (
    AlpacaTradeExecutor,
    execute_trade,
    cancel_order,
    replace_order,
)

__all__ = [
    "AlpacaClient",
    "create_alpaca_client",
    "get_account_info",
    "get_positions",
    "get_orders",
    "AlpacaPositionManager",
    "get_position",
    "update_position",
    "close_position",
    "AlpacaTradeExecutor",
    "execute_trade",
    "cancel_order",
    "replace_order",
]