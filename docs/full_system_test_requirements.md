# Full System Test Requirements

This document lists all the files needed to run the comprehensive full system test for the Autonomous Trading System.

## Test Files

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/tests/full_system_test.py` | Main test file that orchestrates the entire test process |
| `autonomous_trading_system/src/tests/performance_metrics.py` | Metrics collection for API response times, data processing, model training, etc. |
| `autonomous_trading_system/src/tests/system_components.py` | System controller, state manager, health monitor, and recovery components |
| `autonomous_trading_system/src/tests/run_full_system_test.sh` | Shell script to run the full system test with environment setup |

## API Client Files

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/data_acquisition/api/polygon_client.py` | Client for Polygon.io API for market data |
| `autonomous_trading_system/src/data_acquisition/api/unusual_whales_client.py` | Client for Unusual Whales API for options flow data |
| `autonomous_trading_system/src/trading_strategy/alpaca/alpaca_client.py` | Client for Alpaca API for trade execution |

## Data Acquisition Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/data_acquisition/storage/timescale_storage.py` | Storage interface for TimescaleDB |
| `autonomous_trading_system/src/data_acquisition/collectors/price_collector.py` | Collector for price data |
| `autonomous_trading_system/src/data_acquisition/collectors/quote_collector.py` | Collector for quote data |
| `autonomous_trading_system/src/data_acquisition/collectors/trade_collector.py` | Collector for trade data |
| `autonomous_trading_system/src/data_acquisition/collectors/options_collector.py` | Collector for options data |
| `autonomous_trading_system/src/data_acquisition/collectors/multi_timeframe_data_collector.py` | Collector for multi-timeframe data |
| `autonomous_trading_system/src/data_acquisition/pipeline/data_pipeline.py` | Pipeline for data acquisition and processing |
| `autonomous_trading_system/src/data_acquisition/transformation/data_transformer.py` | Transformer for data preprocessing |

## Feature Engineering Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/feature_engineering/pipeline/feature_pipeline.py` | Pipeline for feature generation |
| `autonomous_trading_system/src/feature_engineering/store/feature_store.py` | Storage for calculated features |
| `autonomous_trading_system/src/feature_engineering/analysis/feature_analyzer.py` | Analyzer for feature importance and correlation |

## Model Training Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/model_training/models/xgboost_model.py` | XGBoost model implementation |
| `autonomous_trading_system/src/model_training/models/lstm_model.py` | LSTM model implementation |
| `autonomous_trading_system/src/model_training/models/cnn_model.py` | CNN model implementation |
| `autonomous_trading_system/src/model_training/registry/model_registry.py` | Registry for trained models |
| `autonomous_trading_system/src/model_training/validation/model_validator.py` | Validator for model performance |
| `autonomous_trading_system/src/model_training/inference/model_inference.py` | Inference engine for model predictions |

## Trading Strategy Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/trading_strategy/selection/ticker_selector.py` | Selector for active tickers |
| `autonomous_trading_system/src/trading_strategy/selection/timeframe_selector.py` | Selector for optimal timeframes |
| `autonomous_trading_system/src/trading_strategy/signals/peak_detector.py` | Detector for price peaks and valleys |
| `autonomous_trading_system/src/trading_strategy/signals/entry_signal_generator.py` | Generator for entry signals |
| `autonomous_trading_system/src/trading_strategy/sizing/risk_based_position_sizer.py` | Position sizer based on risk parameters |
| `autonomous_trading_system/src/trading_strategy/risk/stop_loss_manager.py` | Manager for stop loss orders |
| `autonomous_trading_system/src/trading_strategy/risk/profit_target_manager.py` | Manager for profit target orders |
| `autonomous_trading_system/src/trading_strategy/execution/order_generator.py` | Generator for order parameters |
| `autonomous_trading_system/src/trading_strategy/alpaca/alpaca_trade_executor.py` | Executor for trades via Alpaca |
| `autonomous_trading_system/src/trading_strategy/alpaca/alpaca_position_manager.py` | Manager for open positions |

## Continuous Learning Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/continuous_learning/analysis/performance_analyzer.py` | Analyzer for trading performance |
| `autonomous_trading_system/src/continuous_learning/adaptation/strategy_adapter.py` | Adapter for strategy parameters |
| `autonomous_trading_system/src/continuous_learning/retraining/model_retrainer.py` | Retrainer for models with new data |

## Backtesting Components

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/backtesting/engine/backtest_engine.py` | Engine for backtesting strategies |
| `autonomous_trading_system/src/backtesting/analysis/strategy_analyzer.py` | Analyzer for strategy performance |
| `autonomous_trading_system/src/backtesting/reporting/performance_reporter.py` | Reporter for backtest results |

## Utility Files

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/time/market_hours.py` | Utilities for market hours |
| `autonomous_trading_system/src/utils/time/market_calendar.py` | Calendar for market trading days |

## Configuration Files

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/config/database_config.py` | Configuration for database connection |
| `autonomous_trading_system/.env` | Environment variables for API keys and configuration |

## External Dependencies

The following external dependencies are required:

1. **Python Packages**:
   - pandas
   - numpy
   - tensorflow (with GPU support)
   - xgboost
   - sqlalchemy
   - requests
   - asyncio

2. **Database**:
   - TimescaleDB (PostgreSQL extension)

3. **API Keys**:
   - Polygon.io API key
   - Unusual Whales API key
   - Alpaca API key ID and secret key

4. **Hardware**:
   - NVIDIA GPU with CUDA support (optional, for GPU acceleration)

## Directory Structure

The test expects the following directory structure:

```
autonomous_trading_system/
├── .env
├── docker-compose.yml
├── autonomous_trading_system/
│   ├── src/
│   │   ├── tests/
│   │   │   ├── full_system_test.py
│   │   │   ├── performance_metrics.py
│   │   │   ├── system_components.py
│   │   │   └── run_full_system_test.sh
│   │   ├── data_acquisition/
│   │   ├── feature_engineering/
│   │   ├── model_training/
│   │   ├── trading_strategy/
│   │   ├── continuous_learning/
│   │   ├── backtesting/
│   │   ├── utils/
│   │   └── config/
│   ├── data/
│   ├── models/
│   ├── results/
│   └── logs/
└── docs/
    └── full_system_test_requirements.md
```

## Running the Test

To run the full system test:

1. Ensure all required files are in place
2. Set up the environment variables in `.env`
3. Run the test script:

```bash
cd autonomous_trading_system
./src/tests/run_full_system_test.sh
```

The test will generate comprehensive logs and performance metrics in the `results/` directory.

## Additional Files for Full Production Operation

For running the system in full production mode (start, stop, monitor), the following additional files are required:

### System Startup and Shutdown

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/data_acquisition/pipeline/pipeline_scheduler.py` | Scheduler for data acquisition pipelines |
| `autonomous_trading_system/src/continuous_learning/pipeline/continuous_learning_pipeline.py` | Pipeline for continuous learning |
| `autonomous_trading_system/src/continuous_learning/retraining/retraining_scheduler.py` | Scheduler for model retraining |
| `autonomous_trading_system/src/model_training/inference/model_server.py` | Server for model inference |

### Database Management

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/database/connection_manager.py` | Manager for database connections |
| `autonomous_trading_system/src/utils/database/connection_pool.py` | Connection pooling for database |
| `autonomous_trading_system/src/utils/database/schema_manager.py` | Manager for database schema |
| `autonomous_trading_system/src/utils/database/timescale_manager.py` | Manager for TimescaleDB |
| `autonomous_trading_system/src/utils/database/db_tools.sh` | Tools for database management |
| `autonomous_trading_system/src/utils/database/install_dependencies.sh` | Script to install database dependencies |

### Concurrency and Process Management

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/concurrency/distributed_lock.py` | Distributed locking mechanism |
| `autonomous_trading_system/src/utils/concurrency/process_pool.py` | Process pool for parallel execution |
| `autonomous_trading_system/src/utils/concurrency/thread_pool.py` | Thread pool for concurrent operations |

### API Management

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/api/connection_pool.py` | Connection pooling for APIs |
| `autonomous_trading_system/src/utils/api/rate_limiter.py` | Rate limiting for API calls |
| `autonomous_trading_system/src/utils/api/retry_handler.py` | Retry handling for API calls |

### Monitoring and Metrics

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/metrics/performance_metrics.py` | Performance metrics collection |
| `autonomous_trading_system/src/utils/metrics/system_metrics.py` | System metrics collection |
| `autonomous_trading_system/src/tests/generate_test_report.py` | Generator for test reports |

### Serialization and Data Management

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/src/utils/serialization/json_serializer.py` | JSON serialization utilities |
| `autonomous_trading_system/src/utils/serialization/pickle_serializer.py` | Pickle serialization utilities |
| `autonomous_trading_system/src/data_acquisition/validation/data_validator.py` | Validator for data quality |
| `autonomous_trading_system/src/data_acquisition/validation/validation_rules.py` | Rules for data validation |

### Production Configuration

| File Path | Description |
|-----------|-------------|
| `autonomous_trading_system/docker-compose.yml` | Docker Compose configuration for containerized deployment |
| `autonomous_trading_system/src/model_training/optimization/gpu_accelerator.py` | GPU acceleration configuration |
| `autonomous_trading_system/src/model_training/optimization/cudnn_fixes.py` | CuDNN fixes for GPU acceleration |
| `autonomous_trading_system/src/model_training/optimization/mixed_precision_adapter.py` | Mixed precision training adapter |

These additional files provide the necessary components for running the system in a full production environment, including system startup and shutdown, monitoring, process management, and database management.