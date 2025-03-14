"""
Trading Strategy Alpaca Package

This package contains components for interacting with the Alpaca trading API.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.alpaca.alpaca_client import AlpacaClient
from src.trading_strategy.alpaca.alpaca_position_manager import AlpacaPositionManager
from src.trading_strategy.alpaca.alpaca_trade_executor import AlpacaTradeExecutor

# Set up logger for this module
logger = get_logger("trading_strategy.alpaca")
logger.info("Initializing trading strategy Alpaca package")

__all__ = ["AlpacaClient", "AlpacaPositionManager", "AlpacaTradeExecutor"]

logger.debug(f"Loaded Alpaca components: {', '.join(__all__)}")