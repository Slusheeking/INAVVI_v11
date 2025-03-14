"""
Trading Strategy Risk Package

This package contains components for managing trading risk, including stop losses and profit targets.
"""

from src.utils.logging.logger import get_logger
from src.trading_strategy.risk.stop_loss_manager import StopLossManager
from src.trading_strategy.risk.profit_target_manager import ProfitTargetManager

# Set up logger for this module
logger = get_logger("trading_strategy.risk")
logger.info("Initializing trading strategy risk package")

__all__ = ["StopLossManager", "ProfitTargetManager"]

logger.debug(f"Loaded risk components: {', '.join(__all__)}")