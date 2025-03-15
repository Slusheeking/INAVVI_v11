"""
Utilities for the Autonomous Trading System.

This package provides various utilities for the trading system, including:
- API utilities
- Concurrency utilities
- Database utilities
- Logging utilities
- Metrics utilities
- Serialization utilities
- Time utilities
"""

# API utilities
from src.utils.api.api_utils import (
    APIClient,
    AlpacaAPIClient,
    APICache,
    RateLimiter,
    RetryHandler,
    cached_api_call,
    create_api_client_from_env,
    create_alpaca_client_from_env,
    rate_limited,
)

# Concurrency utilities
from src.utils.concurrency.concurrency_utils import (
    run_in_thread,
    run_in_process,
    run_with_timeout,
    parallel_map,
    parallel_execute,
    throttle,
    debounce,
    retry,
    periodic,
    rate_limited as concurrency_rate_limited,
    gather_with_concurrency,
    run_async,
    ThreadSafeCounter,
    ThreadSafeDict,
    ThreadSafeList,
)

# Database utilities
from src.utils.database.database_utils import (
    get_connection_string,
    get_connection_string_from_env,
    get_connection_pool,
    get_connection_pool_from_env,
    get_connection,
    get_cursor,
    execute_query,
    execute_batch_query,
    execute_values_query,
    get_sqlalchemy_engine,
    get_sqlalchemy_engine_from_env,
    get_sqlalchemy_session,
    session_scope,
    execute_sqlalchemy_query,
    dataframe_to_sql,
    sql_to_dataframe,
    create_table_if_not_exists,
    create_hypertable,
    add_compression_policy,
    add_retention_policy,
    create_continuous_aggregate,
    get_table_columns,
    get_table_size,
    get_table_row_count,
    get_database_size,
    get_connection_info,
    wait_for_database,
    wait_for_database_from_env,
    close_all_connections,
)

# Logging utilities
from src.utils.logging.logger import (
    setup_logger,
    get_logger,
    log_to_file,
    get_all_loggers,
)

# Metrics utilities
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

# Serialization utilities
from src.utils.serialization.serialization_utils import (
    NumpyEncoder,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_dataframe,
    load_dataframe,
    dataframe_to_dict,
    dict_to_dataframe,
    serialize_numpy,
    deserialize_numpy,
    serialize_model,
    deserialize_model,
    object_to_json_string,
    json_string_to_object,
)

# Time utilities
from src.utils.time.time_utils import (
    now,
    now_us_eastern,
    timestamp_to_datetime,
    datetime_to_timestamp,
    format_datetime,
    parse_datetime,
    is_market_open,
    get_next_market_open,
    get_next_market_close,
    get_previous_trading_day,
    get_next_trading_day,
    get_trading_days_between,
    time_until_market_open,
    time_until_market_close,
    wait_until_market_open,
    wait_until_market_close,
    convert_to_est,
    convert_to_utc,
    is_same_trading_day,
    get_current_bar_timestamp,
    get_trading_hours_today,
    get_trading_sessions_today,
    get_current_trading_session,
    is_within_trading_hours,
    is_within_extended_hours,
    get_time_to_next_bar,
    wait_for_next_bar,
    resample_to_timeframe,
)

from src.utils.time.market_hours import (
    MarketHours,
    MarketStatus,
    get_market_hours,
    get_market_status,
)

from src.utils.time.market_calendar import (
    get_trading_days,
    is_trading_day,
)

__all__ = [
    # API utilities
    "APIClient", "AlpacaAPIClient", "APICache", "RateLimiter", "RetryHandler",
    "cached_api_call", "create_api_client_from_env", "create_alpaca_client_from_env",
    "rate_limited",
    
    # Concurrency utilities
    "run_in_thread", "run_in_process", "run_with_timeout", "parallel_map",
    "parallel_execute", "throttle", "debounce", "retry", "periodic",
    "concurrency_rate_limited", "gather_with_concurrency", "run_async",
    "ThreadSafeCounter", "ThreadSafeDict", "ThreadSafeList",
    
    # Database utilities
    "get_connection_string", "get_connection_string_from_env", "get_connection_pool",
    "get_connection_pool_from_env", "get_connection", "get_cursor", "execute_query",
    "execute_batch_query", "execute_values_query", "get_sqlalchemy_engine",
    "get_sqlalchemy_engine_from_env", "get_sqlalchemy_session", "session_scope",
    "execute_sqlalchemy_query", "dataframe_to_sql", "sql_to_dataframe",
    "create_table_if_not_exists", "create_hypertable", "add_compression_policy",
    "add_retention_policy", "create_continuous_aggregate", "get_table_columns",
    "get_table_size", "get_table_row_count", "get_database_size", "get_connection_info",
    "wait_for_database", "wait_for_database_from_env", "close_all_connections",
    
    # Logging utilities
    "setup_logger", "get_logger", "log_to_file", "get_all_loggers",
    
    # Metrics utilities
    "calculate_returns", "calculate_cumulative_returns", "calculate_drawdowns",
    "calculate_sharpe_ratio", "calculate_sortino_ratio", "calculate_calmar_ratio",
    "calculate_omega_ratio", "calculate_win_rate", "calculate_profit_factor",
    "calculate_expectancy", "calculate_kelly_criterion", "calculate_average_trade",
    "calculate_average_win", "calculate_average_loss", "calculate_win_loss_ratio",
    "calculate_max_consecutive_wins", "calculate_max_consecutive_losses",
    "calculate_volatility", "calculate_var", "calculate_cvar", "calculate_beta",
    "calculate_alpha", "calculate_information_ratio", "calculate_trading_metrics",
    "calculate_trade_statistics",
    
    # Serialization utilities
    "NumpyEncoder", "save_json", "load_json", "save_pickle", "load_pickle",
    "save_dataframe", "load_dataframe", "dataframe_to_dict", "dict_to_dataframe",
    "serialize_numpy", "deserialize_numpy", "serialize_model", "deserialize_model",
    "object_to_json_string", "json_string_to_object",
    
    # Time utilities
    "now", "now_us_eastern", "timestamp_to_datetime", "datetime_to_timestamp",
    "format_datetime", "parse_datetime", "is_market_open", "get_next_market_open",
    "get_next_market_close", "get_previous_trading_day", "get_next_trading_day",
    "get_trading_days_between", "time_until_market_open", "time_until_market_close",
    "wait_until_market_open", "wait_until_market_close", "convert_to_est",
    "convert_to_utc", "is_same_trading_day", "get_current_bar_timestamp",
    "get_trading_hours_today", "get_trading_sessions_today", "get_current_trading_session",
    "is_within_trading_hours", "is_within_extended_hours", "get_time_to_next_bar",
    "wait_for_next_bar", "resample_to_timeframe",
    
    # Market hours
    "MarketHours", "MarketStatus", "get_market_hours", "get_market_status",
    
    # Market calendar
    "get_trading_days", "is_trading_day",
]