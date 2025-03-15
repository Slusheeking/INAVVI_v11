"""
Trading strategy for the Autonomous Trading System.

This package provides components for implementing trading strategies,
including signal generation, position sizing, risk management, and execution.
"""

# Alpaca imports
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

# Execution imports
from src.trading_strategy.execution.order_generator import (
    OrderGenerator,
    generate_order,
    generate_limit_order,
    generate_market_order,
    generate_stop_order,
    generate_stop_limit_order,
)

# Risk management imports
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

# Selection imports
from src.trading_strategy.selection.ticker_selector import (
    TickerSelector,
    select_tickers,
    filter_tickers,
    rank_tickers,
)

from src.trading_strategy.selection.timeframe_selector import (
    TimeframeSelector,
    select_timeframe,
    get_optimal_timeframe,
    get_timeframe_range,
)

# Signal generation imports
from src.trading_strategy.signals.entry_signal_generator import (
    EntrySignalGenerator,
    generate_entry_signals,
    filter_signals,
    rank_signals,
)

from src.trading_strategy.signals.peak_detector import (
    PeakDetector,
    detect_peaks,
    detect_troughs,
    calculate_peak_metrics,
    filter_peaks,
)

# Position sizing imports
from src.trading_strategy.sizing.risk_based_position_sizer import (
    RiskBasedPositionSizer,
    calculate_position_size,
    calculate_max_position_size,
    calculate_position_value,
    calculate_shares_from_risk,
)

__all__ = [
    # Alpaca
    "AlpacaClient", "create_alpaca_client", "get_account_info", "get_positions", "get_orders",
    "AlpacaPositionManager", "get_position", "update_position", "close_position",
    "AlpacaTradeExecutor", "execute_trade", "cancel_order", "replace_order",
    
    # Execution
    "OrderGenerator", "generate_order", "generate_limit_order", "generate_market_order",
    "generate_stop_order", "generate_stop_limit_order",
    
    # Risk management
    "StopLossManager", "calculate_stop_loss", "update_stop_loss", "check_stop_loss",
    "ProfitTargetManager", "calculate_profit_target", "update_profit_target", "check_profit_target",
    
    # Selection
    "TickerSelector", "select_tickers", "filter_tickers", "rank_tickers",
    "TimeframeSelector", "select_timeframe", "get_optimal_timeframe", "get_timeframe_range",
    
    # Signal generation
    "EntrySignalGenerator", "generate_entry_signals", "filter_signals", "rank_signals",
    "PeakDetector", "detect_peaks", "detect_troughs", "calculate_peak_metrics", "filter_peaks",
    
    # Position sizing
    "RiskBasedPositionSizer", "calculate_position_size", "calculate_max_position_size",
    "calculate_position_value", "calculate_shares_from_risk",
]