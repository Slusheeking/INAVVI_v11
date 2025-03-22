#!/usr/bin/env python3
"""
Unified Trading System Configuration

This module centralizes all configuration parameters for the trading system:
- API keys and connection parameters (loaded from environment variables)
- Trading parameters (position limits, risk thresholds)
- Stock selection criteria (volatility thresholds, volume requirements)
- Model configuration (hyperparameters, feature importance)
- Scheduling parameters (update frequencies, retraining schedules)
- Monitoring thresholds (alerting criteria, performance metrics)

The configuration is organized into logical sections with default values
that can be overridden by environment variables or command line arguments.
"""

import os
import json
import logging
from typing import Dict, Any

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config')

# Import Prometheus client if available
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available for metrics collection")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "Prometheus client not available. Metrics collection will be limited.")

# Load API keys and sensitive configuration from environment variables
API_KEYS = {
    'polygon': os.environ.get('POLYGON_API_KEY', ''),
    'unusual_whales': os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
    'alpaca': os.environ.get('ALPACA_API_KEY', ''),
    'alpaca_secret': os.environ.get('ALPACA_API_SECRET', ''),
}

# Slack configuration
SLACK_CONFIG = {
    'bot_token': os.environ.get('SLACK_BOT_TOKEN', ''),
    'webhooks': {
        'notifications': os.environ.get('SLACK_WEBHOOK_NOTIFICATIONS', ''),
        'reports': os.environ.get('SLACK_WEBHOOK_REPORTS', ''),
        'portfolio': os.environ.get('SLACK_WEBHOOK_PORTFOLIO', ''),
        'positions': os.environ.get('SLACK_WEBHOOK_POSITIONS', ''),
    },
    'enabled': os.environ.get('SLACK_ALERTS_ENABLED', 'true').lower() == 'true',
    'max_notifications_per_hour': int(os.environ.get('MAX_SLACK_NOTIFICATIONS_PER_HOUR', 20)),
}

# Check for required API keys
if not API_KEYS['polygon']:
    logger.warning("POLYGON_API_KEY environment variable not set")

if not API_KEYS['unusual_whales']:
    logger.warning("UNUSUAL_WHALES_API_KEY environment variable not set")

# Redis configuration
REDIS_CONFIG = {
    'host': os.environ.get('REDIS_HOST', 'localhost'),
    'port': int(os.environ.get('REDIS_PORT', 6380)),
    'db': int(os.environ.get('REDIS_DB', 0)),
    'password': os.environ.get('REDIS_PASSWORD', 'trading_system_2025'),
    'username': os.environ.get('REDIS_USERNAME', 'default'),
    'ssl': os.environ.get('REDIS_SSL', 'false').lower() == 'true',
    'timeout': int(os.environ.get('REDIS_TIMEOUT', 5)),
    'ttl': int(os.environ.get('REDIS_TTL', 3600)),
}

# System configuration
SYSTEM_CONFIG = {
    'use_gpu': os.environ.get('USE_GPU', 'true').lower() == 'true',
    'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
    'metrics_port': int(os.environ.get('METRICS_PORT', 8000)),
    'max_connections': int(os.environ.get('MAX_CONNECTIONS', 50)),
    'max_pool_size': int(os.environ.get('MAX_POOL_SIZE', 20)),
    'connection_timeout': int(os.environ.get('CONNECTION_TIMEOUT', 10)),
    'max_retries': int(os.environ.get('MAX_RETRIES', 3)),
    'retry_backoff_factor': float(os.environ.get('RETRY_BACKOFF_FACTOR', 0.5)),
    'max_workers': int(os.environ.get('MAX_WORKERS', 10)),
    'batch_size': int(os.environ.get('BATCH_SIZE', 256)),
    'queue_size': int(os.environ.get('QUEUE_SIZE', 100)),
    'data_dir': os.environ.get('DATA_DIR', './data'),
    'models_dir': os.environ.get('MODELS_DIR', './models'),
    'logs_dir': os.environ.get('LOGS_DIR', './logs'),
    'prometheus': {
        'enabled': os.environ.get('PROMETHEUS_ENABLED', 'true').lower() == 'true',
        'port': int(os.environ.get('PROMETHEUS_PORT', 9090)),
        'scrape_interval': int(os.environ.get('PROMETHEUS_SCRAPE_INTERVAL', 15)),
        'evaluation_interval': int(os.environ.get('PROMETHEUS_EVALUATION_INTERVAL', 15)),
        'retention_time': os.environ.get('PROMETHEUS_RETENTION_TIME', '15d'),
        'storage_path': os.environ.get('PROMETHEUS_STORAGE_PATH', './data/prometheus'),
    },
    'gpu_config': {
        'tensorflow': {
            'allow_growth': os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'true').lower() == 'true',
            'gpu_allocator': os.environ.get('TF_GPU_ALLOCATOR', 'cuda_malloc_async'),
            'thread_mode': os.environ.get('TF_GPU_THREAD_MODE', 'gpu_private'),
            'thread_count': int(os.environ.get('TF_GPU_THREAD_COUNT', 8)),
            'use_cuda_graphs': os.environ.get('TF_USE_CUDA_GRAPHS', '1') == '1',
            'xla_flags': os.environ.get('TF_XLA_FLAGS', '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'),
            'host_mem_limit_mb': int(os.environ.get('TF_CUDA_HOST_MEM_LIMIT_IN_MB', 32000)),
            'enable_mlir_merge': os.environ.get('TF_MLIR_ENABLE_MERGE_CONTROL_FLOW_PASS', '1') == '1',
            'autotune_threshold': int(os.environ.get('TF_AUTOTUNE_THRESHOLD', 2)),
        },
        'tensorrt': {
            'precision_mode': os.environ.get('TENSORRT_PRECISION_MODE', 'FP16'),
            'allow_engine_native_segment': os.environ.get('TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT', 'true').lower() == 'true',
            'allow_custom_ops': os.environ.get('TF_TRT_ALLOW_CUSTOM_OPS', 'true').lower() == 'true',
            'use_implicit_batch': os.environ.get('TF_TRT_USE_IMPLICIT_BATCH', 'false').lower() == 'true',
            'allow_dynamic_shapes': os.environ.get('TF_TRT_ALLOW_DYNAMIC_SHAPES', 'true').lower() == 'true',
            'allow_reduced_precision': os.environ.get('TF_TRT_ALLOW_REDUCED_PRECISION', 'true').lower() == 'true',
        },
        'cupy': {
            'cache_dir': os.environ.get('CUPY_CACHE_DIR', './data/cache/cupy'),
            'save_cuda_source': os.environ.get('CUPY_CACHE_SAVE_CUDA_SOURCE', '1') == '1',
            'accelerators': os.environ.get('CUPY_ACCELERATORS', 'cub,cutensor').split(','),
            'tf32': os.environ.get('CUPY_TF32', '1') == '1',
            'compile_with_debug': os.environ.get('CUPY_CUDA_COMPILE_WITH_DEBUG', '0') == '1',
        },
        'cuda': {
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
            'device_max_connections': int(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 32)),
            'auto_boost': os.environ.get('CUDA_AUTO_BOOST', '1') == '1',
        },
    },
}

# Trading parameters
TRADING_CONFIG = {
    # Position and risk management
    'max_position_size': int(os.environ.get('MAX_POSITION_SIZE', 100000)),
    'max_position_pct': float(os.environ.get('MAX_POSITION_PCT', 0.05)),
    'max_total_positions': int(os.environ.get('MAX_TOTAL_POSITIONS', 10)),
    'stop_loss_pct': float(os.environ.get('STOP_LOSS_PCT', 0.02)),
    'take_profit_pct': float(os.environ.get('TAKE_PROFIT_PCT', 0.05)),
    'trailing_stop_pct': float(os.environ.get('TRAILING_STOP_PCT', 0.01)),
    'max_drawdown_pct': float(os.environ.get('MAX_DRAWDOWN_PCT', 0.10)),
    'risk_per_trade_pct': float(os.environ.get('RISK_PER_TRADE_PCT', 0.01)),

    # Trading schedule
    'trading_hours': {
        'start': os.environ.get('TRADING_HOURS_START', '09:30'),
        'end': os.environ.get('TRADING_HOURS_END', '16:00'),
        'timezone': os.environ.get('TRADING_TIMEZONE', 'America/New_York'),
    },
    'trading_days': [0, 1, 2, 3, 4],  # 0=Monday, 4=Friday (no weekend trading)

    # Trading strategies
    'strategies': {
        'momentum': {
            'enabled': os.environ.get('ENABLE_MOMENTUM_STRATEGY', 'true').lower() == 'true',
            'lookback_periods': [5, 10, 20],
            'threshold': float(os.environ.get('MOMENTUM_THRESHOLD', 0.02)),
        },
        'mean_reversion': {
            'enabled': os.environ.get('ENABLE_MEAN_REVERSION_STRATEGY', 'true').lower() == 'true',
            'lookback_periods': [5, 10, 20],
            'std_dev_threshold': float(os.environ.get('MEAN_REVERSION_STD_DEV', 2.0)),
        },
        'peak_prediction': {
            'enabled': os.environ.get('ENABLE_PEAK_PREDICTION_STRATEGY', 'true').lower() == 'true',
            'confidence_threshold': float(os.environ.get('PEAK_PREDICTION_CONFIDENCE', 0.75)),
            'lead_time_ms': int(os.environ.get('PEAK_PREDICTION_LEAD_TIME_MS', 150)),
        },
    },

    # Execution parameters
    'execution': {
        'order_type': os.environ.get('DEFAULT_ORDER_TYPE', 'market'),
        'time_in_force': os.environ.get('DEFAULT_TIME_IN_FORCE', 'day'),
        'max_slippage_pct': float(os.environ.get('MAX_SLIPPAGE_PCT', 0.001)),
        'min_price': float(os.environ.get('MIN_PRICE', 1.0)),
        'enable_fractional_shares': os.environ.get('ENABLE_FRACTIONAL_SHARES', 'true').lower() == 'true',
    },
}

# Stock selection criteria
STOCK_SELECTION_CONFIG = {
    'universe': {
        'source': os.environ.get('STOCK_UNIVERSE_SOURCE', 'polygon'),
        # $1B minimum
        'market_cap_min': float(os.environ.get('MARKET_CAP_MIN', 1e9)),
        # $1T maximum
        'market_cap_max': float(os.environ.get('MARKET_CAP_MAX', 1e12)),
        'price_min': float(os.environ.get('PRICE_MIN', 5.0)),
        'price_max': float(os.environ.get('PRICE_MAX', 1000.0)),
        'exclude_sectors': os.environ.get('EXCLUDE_SECTORS', '').split(','),
        'include_only': os.environ.get('INCLUDE_ONLY_TICKERS', '').split(','),
        'exclude': os.environ.get('EXCLUDE_TICKERS', '').split(','),
        'default_tickers': os.environ.get('DEFAULT_TICKERS', 'AAPL,MSFT,AMZN,GOOGL,TSLA,META,NVDA,JPM,V,PG').split(','),
    },

    'liquidity_criteria': {
        'min_avg_volume': int(os.environ.get('MIN_AVG_VOLUME', 500000)),
        'min_dollar_volume': float(os.environ.get('MIN_DOLLAR_VOLUME', 5e6)),
        'max_spread_pct': float(os.environ.get('MAX_SPREAD_PCT', 0.01)),
    },

    'volatility_criteria': {
        'min_atr_pct': float(os.environ.get('MIN_ATR_PCT', 0.01)),
        'max_atr_pct': float(os.environ.get('MAX_ATR_PCT', 0.05)),
        'min_option_implied_volatility': float(os.environ.get('MIN_IMPLIED_VOLATILITY', 0.20)),
    },

    'technical_criteria': {
        'rsi_oversold': float(os.environ.get('RSI_OVERSOLD', 30.0)),
        'rsi_overbought': float(os.environ.get('RSI_OVERBOUGHT', 70.0)),
        'volume_surge_threshold': float(os.environ.get('VOLUME_SURGE_THRESHOLD', 2.0)),
        'price_momentum_threshold': float(os.environ.get('PRICE_MOMENTUM_THRESHOLD', 0.03)),
    },

    'unusual_activity_criteria': {
        'min_option_flow_score': float(os.environ.get('MIN_OPTION_FLOW_SCORE', 0.7)),
        'min_unusual_volume_ratio': float(os.environ.get('MIN_UNUSUAL_VOLUME_RATIO', 3.0)),
        'min_dark_pool_significance': float(os.environ.get('MIN_DARK_POOL_SIGNIFICANCE', 0.5)),
    },
}

# ML model configuration
ML_CONFIG = {
    'models': {
        'xgboost': {
            'enabled': True,
            'hyperparameters': {
                'n_estimators': int(os.environ.get('XGB_N_ESTIMATORS', 100)),
                'max_depth': int(os.environ.get('XGB_MAX_DEPTH', 6)),
                'learning_rate': float(os.environ.get('XGB_LEARNING_RATE', 0.1)),
                'subsample': float(os.environ.get('XGB_SUBSAMPLE', 0.8)),
                'colsample_bytree': float(os.environ.get('XGB_COLSAMPLE_BYTREE', 0.8)),
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist' if SYSTEM_CONFIG['use_gpu'] else 'hist',
                'predictor': 'gpu_predictor' if SYSTEM_CONFIG['use_gpu'] else 'cpu_predictor',
                'gpu_id': 0 if SYSTEM_CONFIG['use_gpu'] else -1,
                'max_bin': 256,
                'grow_policy': 'lossguide',
                'max_leaves': 256,
                'sampling_method': 'gradient_based',
            },
        },
        'lstm': {
            'enabled': True,
            'hyperparameters': {
                'units': [64, 32],
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'optimizer': 'adam',
                'loss': 'binary_crossentropy',
                'batch_size': int(os.environ.get('LSTM_BATCH_SIZE', 32)),
                'epochs': int(os.environ.get('LSTM_EPOCHS', 50)),
                'patience': int(os.environ.get('LSTM_PATIENCE', 10)),
                'sequence_length': int(os.environ.get('LSTM_SEQUENCE_LENGTH', 20)),
                'use_cudnn': True if SYSTEM_CONFIG['use_gpu'] else False,
                'stateful': False,
                'return_sequences': True,
                'use_attention': True,
            },
        },
        'peak_prediction': {
            'enabled': True,
            'hyperparameters': {
                'time_steps': int(os.environ.get('PEAK_TIME_STEPS', 60)),
                'features': int(os.environ.get('PEAK_FEATURES', 20)),
                'hidden_units': int(os.environ.get('PEAK_HIDDEN_UNITS', 128)),
                'dropout': float(os.environ.get('PEAK_DROPOUT', 0.3)),
                'learn_rate': float(os.environ.get('PEAK_LEARN_RATE', 0.001)),
                'batch_size': int(os.environ.get('PEAK_BATCH_SIZE', 512)),
                'lookback_window': int(os.environ.get('PEAK_LOOKBACK', 10)),
                'prediction_horizon': int(os.environ.get('PEAK_HORIZON', 5)),
                'use_tensorrt': True if SYSTEM_CONFIG['use_gpu'] else False,
                'precision': 'FP16' if SYSTEM_CONFIG['use_gpu'] else 'FP32',
                'use_mixed_precision': True if SYSTEM_CONFIG['use_gpu'] else False,
            },
        },
        'transformer': {
            'enabled': True,
            'hyperparameters': {
                'num_layers': 4,
                'num_heads': 8,
                'd_model': 128,
                'dff': 512,
                'dropout_rate': 0.1,
                'batch_size': 64,
                'epochs': 50,
                'patience': 10,
                'use_tensorrt': True if SYSTEM_CONFIG['use_gpu'] else False,
                'precision': 'FP16' if SYSTEM_CONFIG['use_gpu'] else 'FP32',
            },
        },
    },

    'feature_engineering': {
        'price_features': [
            'close', 'open', 'high', 'low', 'volume',
            'vwap', 'price_change', 'price_pct_change',
        ],
        'technical_indicators': [
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr_14', 'adx_14', 'obv',
        ],
        'volatility_features': [
            'historical_volatility', 'implied_volatility',
            'volatility_skew', 'volatility_smile',
        ],
        'order_book_features': [
            'bid_ask_spread', 'bid_ask_imbalance',
            'order_book_pressure', 'liquidity_score',
        ],
        'market_features': [
            'market_breadth', 'sector_performance',
            'vix', 'yield_curve', 'market_sentiment',
        ],
        'gpu_acceleration': {
            'enabled': SYSTEM_CONFIG['use_gpu'],
            'use_cupy': True,
            'use_numba': True,
            'batch_processing': True,
            'parallel_features': True,
            'memory_optimization': {
                'max_cache_size_mb': 2048,
                'clear_cache_threshold': 0.8,
                'use_pinned_memory': True,
            },
        },
    },

    'model_evaluation': {
        'train_test_split': float(os.environ.get('TRAIN_TEST_SPLIT', 0.8)),
        'validation_split': float(os.environ.get('VALIDATION_SPLIT', 0.1)),
        'cross_validation_folds': int(os.environ.get('CV_FOLDS', 5)),
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'profit_factor'],
        'min_performance_threshold': float(os.environ.get('MIN_PERFORMANCE_THRESHOLD', 0.6)),
        'gpu_acceleration': {
            'parallel_evaluation': True if SYSTEM_CONFIG['use_gpu'] else False,
            'batch_inference': True if SYSTEM_CONFIG['use_gpu'] else False,
            'tensorrt_optimization': True if SYSTEM_CONFIG['use_gpu'] else False,
        },
    },

    'continual_learning': {
        'retraining_frequency': os.environ.get('RETRAINING_FREQUENCY', 'daily'),
        'model_version_storage': int(os.environ.get('MODEL_VERSION_STORAGE', 10)),
        'performance_decay_threshold': float(os.environ.get('PERFORMANCE_DECAY_THRESHOLD', 0.05)),
        'min_training_samples': int(os.environ.get('MIN_TRAINING_SAMPLES', 1000)),
        'adaptive_sample_weights': os.environ.get('ADAPTIVE_SAMPLE_WEIGHTS', 'true').lower() == 'true',
        'gpu_acceleration': {
            'parallel_training': True if SYSTEM_CONFIG['use_gpu'] else False,
            'mixed_precision': True if SYSTEM_CONFIG['use_gpu'] else False,
            'memory_optimization': {
                'gradient_accumulation': True if SYSTEM_CONFIG['use_gpu'] else False,
                'gradient_checkpointing': True if SYSTEM_CONFIG['use_gpu'] else False,
                'model_pruning': False,
            },
        },
    },

    'inference_optimization': {
        'tensorrt': {
            'enabled': SYSTEM_CONFIG['use_gpu'],
            'precision': os.environ.get('TENSORRT_PRECISION_MODE', 'FP16'),
            'workspace_size_mb': 2048,
            'max_batch_size': 64,
            'minimum_segment_size': 3,
            'dynamic_shapes': True,
            'cache_engines': True,
            'engine_cache_dir': os.path.join(SYSTEM_CONFIG['data_dir'], 'tensorrt_engines'),
        },
        'tensorflow': {
            'xla_compilation': True if SYSTEM_CONFIG['use_gpu'] else False,
            'mixed_precision': True if SYSTEM_CONFIG['use_gpu'] else False,
            'graph_optimization': True,
            'constant_folding': True,
            'layout_optimization': True,
            'remapping': True,
            'arithmetic_optimization': True,
            'dependency_optimization': True,
            'loop_optimization': True,
            'function_optimization': True,
            'debug_stripper': True,
            'scoped_allocator_optimization': True,
            'implementation_selector': True,
            'auto_mixed_precision': True if SYSTEM_CONFIG['use_gpu'] else False,
        },
        'cupy': {
            'enabled': SYSTEM_CONFIG['use_gpu'],
            'use_pinned_memory': True,
            'use_texture_memory': True,
            'use_unified_memory': True,
            'memory_pool_size_mb': 2048,
        },
    },
}

# Scheduling parameters
SCHEDULE_CONFIG = {
    'market_data_refresh': {
        # daily
        'ticker_list': os.environ.get('TICKER_LIST_REFRESH_INTERVAL', '86400'),
        # daily
        'ticker_details': os.environ.get('TICKER_DETAILS_REFRESH_INTERVAL', '86400'),
        'bars': {
            # every minute
            'minute': os.environ.get('MINUTE_BARS_REFRESH_INTERVAL', '60'),
            # hourly
            'hour': os.environ.get('HOUR_BARS_REFRESH_INTERVAL', '3600'),
            # daily
            'day': os.environ.get('DAY_BARS_REFRESH_INTERVAL', '86400'),
        },
        # every 5 seconds
        'quotes': os.environ.get('QUOTES_REFRESH_INTERVAL', '5'),
        # every 5 seconds
        'trades': os.environ.get('TRADES_REFRESH_INTERVAL', '5'),
        # every 30 seconds
        'unusual_activity': os.environ.get('UNUSUAL_ACTIVITY_REFRESH_INTERVAL', '30'),
        # every minute
        'darkpool_data': os.environ.get('DARKPOOL_REFRESH_INTERVAL', '60'),
    },

    'model_updates': {
        # every 5 minutes
        'feature_engineering': os.environ.get('FEATURE_ENGINEERING_UPDATE_INTERVAL', '300'),
        # every minute
        'model_prediction': os.environ.get('MODEL_PREDICTION_INTERVAL', '60'),
        # daily
        'model_retraining': os.environ.get('MODEL_RETRAINING_INTERVAL', '86400'),
        # twice daily
        'model_evaluation': os.environ.get('MODEL_EVALUATION_INTERVAL', '43200'),
    },

    'trading_updates': {
        # every minute
        'opportunity_scan': os.environ.get('OPPORTUNITY_SCAN_INTERVAL', '60'),
        # every 5 minutes
        'position_review': os.environ.get('POSITION_REVIEW_INTERVAL', '300'),
        # every 5 minutes
        'risk_assessment': os.environ.get('RISK_ASSESSMENT_INTERVAL', '300'),
        # every second
        'order_execution': os.environ.get('ORDER_EXECUTION_INTERVAL', '1'),
    },

    'system_updates': {
        # every 5 minutes
        'gpu_memory_cleanup': os.environ.get('GPU_MEMORY_CLEANUP_INTERVAL', '300'),
        # every minute
        'performance_monitoring': os.environ.get('PERFORMANCE_MONITORING_INTERVAL', '60'),
        # every minute
        'system_health_check': os.environ.get('SYSTEM_HEALTH_CHECK_INTERVAL', '60'),
        # hourly
        'log_rotation': os.environ.get('LOG_ROTATION_INTERVAL', '3600'),
    },
}

# Monitoring thresholds
MONITORING_CONFIG = {
    'system_metrics': {
        'cpu_usage_threshold': float(os.environ.get('CPU_USAGE_THRESHOLD', 80.0)),
        'memory_usage_threshold': float(os.environ.get('MEMORY_USAGE_THRESHOLD', 80.0)),
        'gpu_memory_threshold': float(os.environ.get('GPU_MEMORY_THRESHOLD', 90.0)),
        'disk_usage_threshold': float(os.environ.get('DISK_USAGE_THRESHOLD', 85.0)),
        'latency_threshold_ms': float(os.environ.get('LATENCY_THRESHOLD_MS', 100.0)),
    },

    'api_metrics': {
        'api_success_rate_threshold': float(os.environ.get('API_SUCCESS_RATE_THRESHOLD', 95.0)),
        'api_response_time_threshold': float(os.environ.get('API_RESPONSE_TIME_THRESHOLD', 500.0)),
        'websocket_reconnect_threshold': int(os.environ.get('WEBSOCKET_RECONNECT_THRESHOLD', 3)),
        'rate_limit_threshold_pct': float(os.environ.get('RATE_LIMIT_THRESHOLD_PCT', 80.0)),
    },

    'data_metrics': {
        'data_freshness_threshold_sec': int(os.environ.get('DATA_FRESHNESS_THRESHOLD_SEC', 300)),
        'data_completeness_threshold_pct': float(os.environ.get('DATA_COMPLETENESS_THRESHOLD_PCT', 95.0)),
        'data_quality_threshold': float(os.environ.get('DATA_QUALITY_THRESHOLD', 0.9)),
    },

    'ml_metrics': {
        'model_drift_threshold': float(os.environ.get('MODEL_DRIFT_THRESHOLD', 0.1)),
        'prediction_latency_threshold_ms': float(os.environ.get('PREDICTION_LATENCY_THRESHOLD_MS', 50.0)),
        'feature_importance_change_threshold': float(os.environ.get('FEATURE_IMPORTANCE_CHANGE_THRESHOLD', 0.2)),
    },

    'trading_metrics': {
        'max_drawdown_alert_threshold': float(os.environ.get('MAX_DRAWDOWN_ALERT_THRESHOLD', 0.05)),
        'win_rate_threshold': float(os.environ.get('WIN_RATE_THRESHOLD', 0.4)),
        'risk_reward_ratio_threshold': float(os.environ.get('RISK_REWARD_RATIO_THRESHOLD', 1.5)),
        'profit_factor_threshold': float(os.environ.get('PROFIT_FACTOR_THRESHOLD', 1.2)),
        'sharpe_ratio_threshold': float(os.environ.get('SHARPE_RATIO_THRESHOLD', 0.5)),
    },

    'alerts': {
        'enabled': os.environ.get('ALERTS_ENABLED', 'true').lower() == 'true',
        'email_alerts': os.environ.get('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
        'slack_alerts': os.environ.get('SLACK_ALERTS_ENABLED', 'false').lower() == 'true',
        'console_alerts': os.environ.get('CONSOLE_ALERTS_ENABLED', 'true').lower() == 'true',
        'max_alerts_per_hour': int(os.environ.get('MAX_ALERTS_PER_HOUR', 20)),
    },

    'prometheus': {
        'enabled': PROMETHEUS_AVAILABLE and os.environ.get('PROMETHEUS_ENABLED', 'true').lower() == 'true',
        'metrics': {
            'system': {
                'cpu_usage': 'system_cpu_usage_percent',
                'memory_usage': 'system_memory_usage_percent',
                'disk_usage': 'system_disk_usage_percent',
                'gpu_memory_usage': 'gpu_memory_usage_percent',
                'gpu_utilization': 'gpu_utilization_percent',
            },
            'api': {
                'request_count': 'api_client_request_count',
                'request_latency': 'api_client_request_latency_seconds',
                'error_count': 'api_client_error_count',
                'cache_hit_count': 'api_client_cache_hit_count',
                'cache_miss_count': 'api_client_cache_miss_count',
                'rate_limit_remaining': 'api_client_rate_limit_remaining',
                'websocket_reconnects': 'api_client_websocket_reconnects',
                'websocket_messages': 'api_client_websocket_messages',
            },
            'ml': {
                'prediction_count': 'ml_prediction_count',
                'prediction_latency': 'ml_prediction_latency_seconds',
                'model_accuracy': 'ml_model_accuracy',
                'model_f1_score': 'ml_model_f1_score',
                'training_duration': 'ml_training_duration_seconds',
                'feature_importance': 'ml_feature_importance',
                'gpu_processing_time': 'api_client_gpu_processing_time_seconds',
            },
            'trading': {
                'order_count': 'trading_order_count',
                'order_value': 'trading_order_value_dollars',
                'position_count': 'trading_position_count',
                'position_value': 'trading_position_value_dollars',
                'cash_balance': 'trading_cash_balance_dollars',
                'total_equity': 'trading_total_equity_dollars',
                'daily_pnl': 'trading_daily_pnl_dollars',
                'daily_pnl_percent': 'trading_daily_pnl_percent',
            },
        },
    },
}

# Error handling and retry configuration
ERROR_HANDLING_CONFIG = {
    'max_retries': int(os.environ.get('MAX_RETRIES', 3)),
    'retry_delay_base_ms': int(os.environ.get('RETRY_DELAY_BASE_MS', 1000)),
    'retry_delay_max_ms': int(os.environ.get('RETRY_DELAY_MAX_MS', 30000)),
    'circuit_breaker_threshold': int(os.environ.get('CIRCUIT_BREAKER_THRESHOLD', 5)),
    'circuit_breaker_reset_time_sec': int(os.environ.get('CIRCUIT_BREAKER_RESET_TIME_SEC', 60)),
    'error_reporting_level': os.environ.get('ERROR_REPORTING_LEVEL', 'WARNING'),
}

# Consolidate all configuration sections
config = {
    'api_keys': API_KEYS,
    'slack': SLACK_CONFIG,
    'redis': REDIS_CONFIG,
    'system': SYSTEM_CONFIG,
    'trading': TRADING_CONFIG,
    'stock_selection': STOCK_SELECTION_CONFIG,
    'ml': ML_CONFIG,
    'schedule': SCHEDULE_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'error_handling': ERROR_HANDLING_CONFIG,
}

# Function to load configuration from JSON file


def load_config_from_file(file_path: str) -> None:
    """
    Load configuration from a JSON file and update the current configuration

    Args:
        file_path: Path to the JSON configuration file
    """
    global config

    if not os.path.exists(file_path):
        logger.warning(
            f"Configuration file {file_path} not found, using defaults")
        return

    try:
        with open(file_path, 'r') as f:
            file_config = json.load(f)

        # Update configuration with values from file
        _update_config_recursive(config, file_config)
        logger.info(f"Loaded configuration from {file_path}")
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {e}")


def _update_config_recursive(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Recursively update a nested dictionary with values from another dictionary

    Args:
        target: Target dictionary to update
        source: Source dictionary with new values
    """
    for key, value in source.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                # If both values are dictionaries, update recursively
                _update_config_recursive(target[key], value)
            else:
                # Otherwise, update the value directly
                target[key] = value
        else:
            # If the key doesn't exist in the target, add it
            target[key] = value


# Try to load configuration from file if it exists
config_file_path = os.environ.get('CONFIG_FILE', 'config.json')
if os.path.exists(config_file_path):
    load_config_from_file(config_file_path)

# Verify critical configuration values


def verify_config() -> None:
    """Verify critical configuration values and log warnings for missing values"""
    if not config['api_keys']['polygon']:
        logger.warning(
            "Polygon API key is not set. Market data functionality will be limited.")

    if not config['api_keys']['unusual_whales']:
        logger.warning(
            "Unusual Whales API key is not set. Option flow analysis will be disabled.")

    # Verify GPU acceleration components
    if config['system']['use_gpu']:
        gpu_status = {
            'tensorflow': False,
            'tensorrt': False,
            'cupy': False,
            'cuda': False,
            'xgboost_gpu': False
        }

        # Check CUDA availability
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                gpu_status['cuda'] = True
                logger.info("NVIDIA GPU detected via nvidia-smi")
            else:
                logger.warning(
                    "nvidia-smi command failed, CUDA may not be available")
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")

        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                gpu_status['tensorflow'] = True
                logger.info(
                    f"TensorFlow {tf.__version__} detected with GPU support")

                # Log GPU device information
                for i, device in enumerate(physical_devices):
                    logger.info(f"TensorFlow GPU {i}: {device.name}")

                # Check TensorRT integration
                try:
                    from tensorflow.python.compiler.tensorrt import trt_convert as trt
                    gpu_status['tensorrt'] = True
                    tensorrt_version = trt.__version__ if hasattr(
                        trt, '__version__') else "Unknown"
                    logger.info(
                        f"TensorRT integration for TensorFlow detected (version: {tensorrt_version})")
                except ImportError:
                    logger.warning(
                        "TensorRT integration for TensorFlow not available")
            else:
                logger.warning(
                    "TensorFlow is available but no GPU devices were detected")
        except ImportError:
            logger.warning("TensorFlow not available")
        except Exception as e:
            logger.warning(f"Error checking TensorFlow GPU support: {e}")

        # Check CuPy
        try:
            import cupy as cp
            gpu_status['cupy'] = True
            logger.info(f"CuPy {cp.__version__} detected")

            # Test CuPy with a simple operation
            try:
                x = cp.array([1, 2, 3])
                y = cp.array([4, 5, 6])
                z = cp.add(x, y)
                logger.info("CuPy GPU operations test successful")

                # Configure CuPy memory pool
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=0.8)  # Use up to 80% of GPU memory
                logger.info("CuPy memory pool configured")

                # Create cache directory if it doesn't exist
                cache_dir = config['system']['gpu_config']['cupy']['cache_dir']
                import os
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"CuPy cache directory set to {cache_dir}")
            except Exception as e:
                logger.warning(f"CuPy GPU operations test failed: {e}")
        except ImportError:
            logger.warning("CuPy not available")
        except Exception as e:
            logger.warning(f"Error checking CuPy: {e}")

        # Check XGBoost GPU support
        try:
            import xgboost as xgb
            try:
                # Create a small test DMatrix
                import numpy as np
                X = np.random.rand(10, 5)
                y = np.random.randint(0, 2, 10)
                dtrain = xgb.DMatrix(X, label=y)

                # Try to train with GPU
                param = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                bst = xgb.train(param, dtrain, num_boost_round=1)
                gpu_status['xgboost_gpu'] = True
                logger.info(
                    f"XGBoost {xgb.__version__} with GPU support confirmed")
            except Exception as e:
                logger.warning(f"XGBoost GPU test failed: {e}")
        except ImportError:
            logger.warning("XGBoost not available")
        except Exception as e:
            logger.warning(f"Error checking XGBoost GPU support: {e}")

        # Update system config based on GPU component availability
        if not any(gpu_status.values()):
            logger.warning(
                "GPU acceleration is enabled but no GPU components are available")
            config['system']['use_gpu'] = False
        else:
            available_components = [k for k, v in gpu_status.items() if v]
            logger.info(
                f"GPU acceleration enabled with: {', '.join(available_components)}")

            # Update ML config based on available GPU components
            if not gpu_status['tensorflow']:
                logger.warning(
                    "Disabling TensorFlow-specific GPU optimizations")
                if 'tensorflow' in config['ml']['inference_optimization']:
                    config['ml']['inference_optimization']['tensorflow']['xla_compilation'] = False
                    config['ml']['inference_optimization']['tensorflow']['mixed_precision'] = False
                    config['ml']['inference_optimization']['tensorflow']['auto_mixed_precision'] = False

            if not gpu_status['tensorrt']:
                logger.warning("Disabling TensorRT optimizations")
                if 'tensorrt' in config['ml']['inference_optimization']:
                    config['ml']['inference_optimization']['tensorrt']['enabled'] = False

                # Disable TensorRT for peak prediction model
                if 'peak_prediction' in config['ml']['models']:
                    config['ml']['models']['peak_prediction']['hyperparameters']['use_tensorrt'] = False

                # Disable TensorRT for transformer model
                if 'transformer' in config['ml']['models']:
                    config['ml']['models']['transformer']['hyperparameters']['use_tensorrt'] = False

            if not gpu_status['cupy']:
                logger.warning("Disabling CuPy optimizations")
                if 'cupy' in config['ml']['inference_optimization']:
                    config['ml']['inference_optimization']['cupy']['enabled'] = False

                if 'gpu_acceleration' in config['ml']['feature_engineering']:
                    config['ml']['feature_engineering']['gpu_acceleration']['use_cupy'] = False

            if not gpu_status['xgboost_gpu']:
                logger.warning("Disabling XGBoost GPU optimizations")
                if 'xgboost' in config['ml']['models']:
                    config['ml']['models']['xgboost']['hyperparameters']['tree_method'] = 'hist'
                    config['ml']['models']['xgboost']['hyperparameters']['predictor'] = 'cpu_predictor'
                    config['ml']['models']['xgboost']['hyperparameters']['gpu_id'] = -1
    else:
        logger.info("GPU acceleration is disabled by configuration")


# Verify configuration on module import
verify_config()

# Export the consolidated configuration
__all__ = ['config', 'load_config_from_file', 'verify_config']
