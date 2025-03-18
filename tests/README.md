# GPU-Accelerated Trading System

A high-performance, GPU-accelerated algorithmic trading system designed for institutional-grade trading with robust risk management, real-time data processing, and ML-based signal generation.

## System Architecture

The system is composed of several modular components that work together:

1. **Data Pipeline Integration** - Collects and processes market data from multiple sources
2. **Model Integration System** - Applies ML models to generate trading signals
3. **Execution System** - Handles order execution and position management
4. **Monitoring System** - Provides observability and alerting capabilities
5. **Integrated Trading System** - Coordinates all components and provides system management
6. **Continual Learning System** - Automatically updates ML models with new market data

## Components

### Data Pipeline Integration

The data pipeline integrates multiple data sources including:
- Polygon.io for market data (optimized with GPU acceleration)
- Unusual Whales for options flow data
- Custom data sources can be added through the modular architecture

Features:
- Real-time and historical data processing
- GPU-accelerated data transformations
- Efficient caching with Redis
- Fault-tolerant design with automatic reconnection

### Model Integration System

The model system applies machine learning models to generate trading signals:
- Supports multiple model types (trend following, mean reversion, etc.)
- GPU-accelerated inference for low latency
- Configurable signal thresholds and expiry
- Model performance tracking

### Execution System

The execution system handles order management and position tracking:
- Integration with Alpaca for order execution
- Sophisticated position sizing based on risk parameters
- Stop loss and take profit management
- Performance tracking and P&L calculation

### Monitoring System

The monitoring system provides comprehensive observability:
- Prometheus metrics for system health and performance
- Real-time alerting for critical issues
- Data quality and freshness monitoring
- Trading performance metrics

### Integrated Trading System

The main system that coordinates all components:
- Centralized configuration management
- System health monitoring
- Graceful startup and shutdown
- Fault tolerance and error recovery

### Continual Learning System

The continual learning system automatically updates ML models with new market data:
- Daily incremental model updates after market close
- Scheduled full model retraining during off-hours
- Model versioning and rollback capabilities
- Performance monitoring and validation
- Automatic model deployment

## Setup and Configuration

### Prerequisites

- Python 3.8+
- Redis server
- CUDA-compatible GPU (optional but recommended)
- API keys for:
  - Polygon.io
  - Unusual Whales
  - Alpaca

### Environment Variables

Configure the system using environment variables or the `.env` file:

```
# API Keys
POLYGON_API_KEY=your_polygon_api_key
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_API_URL=https://paper-api.alpaca.markets/v2

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=3600

# System Configuration
USE_GPU=true
LOG_LEVEL=INFO
```

### Docker Deployment

The system can be deployed using Docker Compose:

```bash
# Start the entire system
docker-compose up -d

# Start individual components
docker-compose up -d redis data_ingestion model_inference execution monitoring

# Start continual learning system
docker-compose up -d continual_learning

# View logs
docker-compose logs -f

# Stop the system
docker-compose down
```

## Running the System

### Command Line Options

The integrated trading system supports various command line options:

```bash
python -m tests.integrated_trading_system --help

# Options:
#   --config CONFIG         Path to configuration file
#   --redis-host HOST       Redis host (default: localhost)
#   --redis-port PORT       Redis port (default: 6379)
#   --redis-db DB           Redis database (default: 0)
#   --polygon-key KEY       Polygon.io API key
#   --unusual-whales-key KEY Unusual Whales API key
#   --alpaca-key KEY        Alpaca API key
#   --alpaca-secret SECRET  Alpaca API secret
#   --alpaca-url URL        Alpaca API URL
#   --no-gpu                Disable GPU acceleration
#   --enable-continual-learning Enable continual learning system
```

### Configuration File

The system can be configured using a JSON configuration file:

```json
{
  "system": {
    "name": "GPU-Accelerated Trading System",
    "version": "1.0.0",
    "health_check_interval": 60,
    "max_memory_usage": 0.8,
    "gpu_memory_limit": 0.9
  },
  "data": {
    "sources": ["polygon", "unusual_whales"],
    "cache_ttl": 300,
    "batch_size": 100,
    "update_interval": {
      "market_status": 60,
      "ticker_details": 86400,
      "aggregates": 60,
      "quotes": 5,
      "trades": 5
    }
  },
  "models": {
    "signal_threshold": 0.7,
    "signal_expiry": 300,
    "batch_size": 32,
    "update_interval": 86400,
      "continual_learning": {
        "enabled": true,
        "daily_update_time": "23:30",
        "full_retrain_time": "00:30",
        "update_window": 5,
        "performance_threshold": 0.8,
        "max_versions": 5
      }
  },
  "trading": {
    "max_positions": 5,
    "risk_per_trade": 0.005,
    "max_drawdown": 0.05,
    "trading_hours": {
      "start": "09:30",
      "end": "16:00",
      "timezone": "America/New_York"
    },
    "position_sizing": {
      "method": "risk_based",
      "max_position_size": 0.1
    }
  }
}
```

## Monitoring and Metrics

The monitoring system exposes Prometheus metrics on port 8000. Key metrics include:

- System metrics: CPU, memory, disk usage
- Data metrics: API request counts, error rates, data freshness
- Trading metrics: active positions, P&L, win rate, exposure
- Queue metrics: queue depths and processing latencies
- Model metrics: prediction counts, inference times, signal quality

### Continual Learning Metrics

The continual learning system exposes additional metrics:
- Model versions and update timestamps
- Training metrics (accuracy, loss, etc.)
- Update success/failure rates
- Model performance over time

## Development and Testing

### Adding New Data Sources

To add a new data source:

1. Create a new client class that implements the required interface
2. Register the data source in the data pipeline integration
3. Configure the data source in the system configuration

### Adding New Models

To add a new trading model:

1. Create a model class that implements the required interface
2. Register the model in the model integration system
3. Configure the model parameters in the system configuration

### Testing

Run individual component tests:

```bash
# Test data ingestion
python -m tests.test_data_ingestion_system

# Test model integration
python -m tests.test_ml_setup

# Test execution system
python -m tests.test_execution_system

# Test integrated system
python -m tests.test_integrated_trading_system

# Test continual learning system
python -m tests.run_continual_learning
```

## Performance Considerations

- The system is designed to leverage GPU acceleration for data processing and model inference
- Redis is used for efficient inter-component communication and caching
- Asynchronous processing is used for I/O-bound operations
- Thread pools are used for CPU-bound operations
- Memory usage is carefully monitored to prevent OOM conditions

## Risk Management

The system implements multiple layers of risk management:

- Position sizing based on account risk parameters
- Stop loss and take profit management
- Maximum drawdown controls
- Exposure limits
- Automatic position exit at market close
- Continuous monitoring and alerting

## License

This project is proprietary and confidential. Unauthorized copying, transfer, or use is strictly prohibited.