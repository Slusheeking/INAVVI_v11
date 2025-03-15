# Production Containerization Strategy for Autonomous Trading System

## Overview

This document outlines a simplified and robust containerization strategy for the Autonomous Trading System in a production environment, with detailed file structures for each container.

## Container File Structures

### 1. System Controller Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   └── setup_logs.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   └── database_utils.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   └── metrics_utils.py
│   │   └── time/
│   │       ├── __init__.py
│   │       ├── market_calendar.py
│   │       └── market_hours.py
│   └── system_controller/
│       ├── __init__.py
│       ├── main.py
│       └── system_components.py
├── config/
│   ├── logging.yaml
│   └── system_config.yaml
└── logs/
    └── system_controller.log
```

### 2. Data Acquisition Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   ├── database/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── api_utils.py
│   │   └── time/
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── polygon_client.py
│   │   │   └── unusual_whales_client.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── multi_timeframe_data_collector.py
│   │   │   ├── options_collector.py
│   │   │   ├── price_collector.py
│   │   │   ├── quote_collector.py
│   │   │   └── trade_collector.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   └── data_pipeline.py
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   └── timescale_storage.py
│   │   └── transformation/
│   │       ├── __init__.py
│   │       └── data_transformer.py
│   └── main.py
├── config/
│   ├── logging.yaml
│   └── collectors_config.yaml
└── logs/
    └── data_acquisition.log
```

### 3. Trading Strategy Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   ├── database/
│   │   ├── metrics/
│   │   └── time/
│   ├── trading_strategy/
│   │   ├── __init__.py
│   │   ├── alpaca/
│   │   │   ├── __init__.py
│   │   │   ├── alpaca_client.py
│   │   │   ├── alpaca_position_manager.py
│   │   │   └── alpaca_trade_executor.py
│   │   ├── execution/
│   │   │   ├── __init__.py
│   │   │   └── order_generator.py
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   ├── profit_target_manager.py
│   │   │   └── stop_loss_manager.py
│   │   ├── selection/
│   │   │   ├── __init__.py
│   │   │   ├── ticker_selector.py
│   │   │   └── timeframe_selector.py
│   │   ├── signals/
│   │   │   ├── __init__.py
│   │   │   ├── entry_signal_generator.py
│   │   │   └── peak_detector.py
│   │   └── sizing/
│   │       ├── __init__.py
│   │       └── risk_based_position_sizer.py
│   └── main.py
├── config/
│   ├── logging.yaml
│   └── trading_config.yaml
└── logs/
    └── trading_strategy.log
```

### 4. Model Training Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   ├── database/
│   │   ├── metrics/
│   │   └── time/
│   ├── model_training/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── cnn_model.py
│   │   │   ├── lstm_model.py
│   │   │   ├── xgboost_model.py
│   │   │   └── inference/
│   │   │       ├── __init__.py
│   │   │       └── model_inference.py
│   │   ├── registry/
│   │   │   ├── __init__.py
│   │   │   └── model_registry.py
│   │   └── validation/
│   │       ├── __init__.py
│   │       └── model_validator.py
│   └── main.py
├── config/
│   ├── logging.yaml
│   └── model_training_config.yaml
└── logs/
    └── model_training.log
```

### 5. Feature Engineering Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   ├── database/
│   │   ├── metrics/
│   │   └── time/
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   └── feature_analyzer.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   └── feature_pipeline.py
│   │   └── store/
│   │       ├── __init__.py
│   │       └── feature_store.py
│   └── main.py
├── config/
│   ├── logging.yaml
│   └── feature_engineering_config.yaml
└── logs/
    └── feature_engineering.log
```

### 6. Continuous Learning Container
```
/app/
├── src/
│   ├── utils/
│   │   ├── logging/
│   │   ├── database/
│   │   ├── metrics/
│   │   └── time/
│   ├── continuous_learning/
│   │   ├── __init__.py
│   │   ├── adaptation/
│   │   │   ├── __init__.py
│   │   │   └── strategy_adapter.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   └── performance_analyzer.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   └── continuous_learning_pipeline.py
│   │   └── retraining/
│   │       ├── __init__.py
│   │       └── model_retrainer.py
│   └── main.py
├── config/
│   ├── logging.yaml
│   └── continuous_learning_config.yaml
└── logs/
    └── continuous_learning.log
```

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  # Core system services
  system-controller:
    build:
      context: .
      dockerfile: docker/system_controller/Dockerfile
    image: autonomous_trading_system/system-controller:${VERSION}
    environment:
      # Connect to existing TimescaleDB container
      - TIMESCALEDB_HOST=${TIMESCALEDB_HOST}
      - TIMESCALEDB_PORT=${TIMESCALEDB_PORT}
      - TIMESCALEDB_DATABASE=${TIMESCALEDB_DATABASE}
      - TIMESCALEDB_USER=${TIMESCALEDB_USER}
      - TIMESCALEDB_PASSWORD=${TIMESCALEDB_PASSWORD}
      - LOG_LEVEL=${LOG_LEVEL}
    volumes:
      - logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    networks:
      - default
    external_links:
      - timescaledb-v1-1
      - model-training-tensorflow-v1-1
      - feature-engineering-tensorflow-v1-1
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  data-acquisition:
    build:
      context: .
      dockerfile: docker/data_acquisition/Dockerfile
    image: autonomous_trading_system/data-acquisition:${VERSION}
    depends_on:
      - system-controller
    environment:
      # Connect to existing TimescaleDB container
      - TIMESCALEDB_HOST=${TIMESCALEDB_HOST}
      - TIMESCALEDB_PORT=${TIMESCALEDB_PORT}
      - TIMESCALEDB_DATABASE=${TIMESCALEDB_DATABASE}
      - TIMESCALEDB_USER=${TIMESCALEDB_USER}
      - TIMESCALEDB_PASSWORD=${TIMESCALEDB_PASSWORD}
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - UNUSUAL_WHALES_API_KEY=${UNUSUAL_WHALES_API_KEY}
      - LOG_LEVEL=${LOG_LEVEL}
    volumes:
      - logs:/app/logs
      - ./config:/app/config
      - data:/app/data
    restart: unless-stopped
    networks:
      - default
    external_links:
      - timescaledb-v1-1
      - feature-engineering-tensorflow-v1-1
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  trading-strategy:
    build:
      context: .
      dockerfile: docker/trading_strategy/Dockerfile
    image: autonomous_trading_system/trading-strategy:${VERSION}
    depends_on:
      - system-controller
      - data-acquisition
    environment:
      # Connect to existing TimescaleDB container
      - TIMESCALEDB_HOST=${TIMESCALEDB_HOST}
      - TIMESCALEDB_PORT=${TIMESCALEDB_PORT}
      - TIMESCALEDB_DATABASE=${TIMESCALEDB_DATABASE}
      - TIMESCALEDB_USER=${TIMESCALEDB_USER}
      - TIMESCALEDB_PASSWORD=${TIMESCALEDB_PASSWORD}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_API_SECRET=${ALPACA_API_SECRET}
      - ALPACA_API_BASE_URL=${ALPACA_API_BASE_URL}
      - LOG_LEVEL=${LOG_LEVEL}
      - MAX_POSITION_SIZE=${MAX_POSITION_SIZE}
      - RISK_PERCENTAGE=${RISK_PERCENTAGE}
      - MAX_POSITIONS=${MAX_POSITIONS}
    volumes:
      - logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    networks:
      - default
    external_links:
      - timescaledb-v1-1
      - model-training-tensorflow-v1-1
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

volumes:
  logs:
  data:
```

## Dockerfile Configurations

### 1. System Controller Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/system_controller /app/src/system_controller
COPY --chown=appuser:appuser docker/system_controller/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

### 2. Data Acquisition Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/data_acquisition /app/src/data_acquisition
COPY --chown=appuser:appuser docker/data_acquisition/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

### 3. Trading Strategy Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/trading_strategy /app/src/trading_strategy
COPY --chown=appuser:appuser docker/trading_strategy/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

### 4. Model Training Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/model_training /app/src/model_training
COPY --chown=appuser:appuser docker/model_training/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

### 5. Feature Engineering Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/feature_engineering /app/src/feature_engineering
COPY --chown=appuser:appuser docker/feature_engineering/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

### 6. Continuous Learning Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/continuous_learning /app/src/continuous_learning
COPY --chown=appuser:appuser docker/continuous_learning/start.sh /app/start.sh

USER appuser

CMD ["/app/start.sh"]
```

## Startup Scripts

### 1. System Controller Start Script
```bash
#!/bin/bash
# docker/system_controller/start.sh

# Wait for TimescaleDB with timeout and backoff
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c '\q'; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to TimescaleDB after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for TimescaleDB... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

# Start the system controller
exec python -m src.system_controller.main
```

### 2. Data Acquisition Start Script
```bash
#!/bin/bash
# docker/data_acquisition/start.sh

# Wait for System Controller with timeout and backoff
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://system-controller:8000/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to System Controller after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for System Controller... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

# Start the data acquisition service
exec python -m src.data_acquisition.main
```

### 3. Trading Strategy Start Script
```bash
#!/bin/bash
# docker/trading_strategy/start.sh

# Wait for Data Acquisition with timeout and backoff
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://data-acquisition:8001/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to Data Acquisition after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for Data Acquisition... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

# Start the trading strategy service
exec python -m src.trading_strategy.main
```

## Configuration Files

### 1. Logging Configuration
```yaml
# config/logging.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    formatter: standard
    filename: /app/logs/service.log
root:
  level: INFO
  handlers: [console, file]
```

### 2. System Configuration
```yaml
# config/system_config.yaml
system:
  health_check_interval: 10
  metrics_port: 8000
database:
  max_connections: 20
  connection_timeout: 5
  retry_attempts: 3
monitoring:
  metrics_enabled: true
  log_level: INFO
```

## External Dependencies

### 1. TimescaleDB Integration
The system relies on an external TimescaleDB instance (`timescaledb-v1-1`) for time-series data storage. This approach offers several advantages:

- **Separation of Concerns**: Database management is handled independently from the application containers
- **Scalability**: The database can be scaled independently based on storage and performance needs
- **Persistence**: Data persists beyond the lifecycle of application containers
- **Backup Management**: Database backups can be managed separately from application deployments

Connection details are provided via environment variables:
```
TIMESCALEDB_HOST
TIMESCALEDB_PORT
TIMESCALEDB_DATABASE
TIMESCALEDB_USER
TIMESCALEDB_PASSWORD
```

### 2. Model Training Integration
The Model Training service (`model-training-tensorflow-v1-1`) is referenced by the System Controller and Trading Strategy containers. This service:

- Trains machine learning models using historical market data
- Manages model versioning and deployment
- Provides inference endpoints for real-time predictions
- Handles model validation and performance monitoring

### 3. Feature Engineering Integration
The Feature Engineering service (`feature-engineering-tensorflow-v1-1`) is referenced by the System Controller and Data Acquisition containers. This service:

- Processes raw market data into meaningful features
- Manages feature storage and retrieval
- Provides feature transformation pipelines
- Handles feature validation and quality monitoring

## Network Configuration

### 1. Service Discovery and Communication
The system uses Docker's built-in DNS for service discovery. Each container can communicate with others using their service names:

- `system-controller:8000` - System Controller API
- `data-acquisition:8001` - Data Acquisition API
- `trading-strategy:8002` - Trading Strategy API
- `timescaledb-v1-1:5432` - TimescaleDB
- `model-training-tensorflow-v1-1:8003` - Model Training API
- `feature-engineering-tensorflow-v1-1:8004` - Feature Engineering API

### 2. Port Exposure
The following ports are exposed for external communication:

- `8000` - System Controller API (health checks, metrics, system status)
- `8001` - Data Acquisition API (data collection status, metrics)
- `8002` - Trading Strategy API (trading status, position information)
- `8003` - Model Training API (model status, training metrics)
- `8004` - Feature Engineering API (feature status, transformation metrics)

### 3. Network Security
All inter-container communication occurs on a private Docker network. Only necessary ports are exposed to the host system. For production deployment, consider implementing:

- Network policies to restrict container-to-container communication
- TLS encryption for all API endpoints
- API authentication for sensitive endpoints
- Rate limiting to prevent abuse

## Resource Allocation

### 1. Container Resource Limits
Each container has resource limits defined to ensure fair resource allocation and prevent any single container from consuming all available resources:

- **System Controller**: 1.0 CPU, 1G memory
- **Data Acquisition**: 1.0 CPU, 1G memory
- **Trading Strategy**: 1.0 CPU, 1G memory
- **Model Training**: 2.0 CPU, 4G memory, 1 GPU
- **Feature Engineering**: 1.0 CPU, 2G memory
- **Continuous Learning**: 1.0 CPU, 2G memory

### 2. Scaling Considerations
For production deployment, consider implementing:

- Horizontal scaling for Data Acquisition to handle increased data volume
- Vertical scaling for Model Training to accommodate more complex models
- Load balancing for API endpoints to distribute traffic
- Auto-scaling based on resource utilization metrics

## Data Persistence Strategy

### 1. Volume Management
The system uses Docker volumes for data persistence:

- `timescaledb-data`: Stores TimescaleDB data files
- `logs`: Stores application logs from all containers
- `data`: Stores intermediate data files and cached data

### 2. Backup Strategy
Implement the following backup procedures:

- **TimescaleDB Backups**:
  - Daily full backups using `pg_dump`
  - Continuous WAL archiving for point-in-time recovery
  - Backup verification and restoration testing

- **Model Artifacts**:
  - Regular backups of trained models
  - Version control for model configurations
  - Backup of training datasets

- **Configuration Backups**:
  - Version control for all configuration files
  - Backup of environment variables and secrets

### 3. Retention Policy
Implement data retention policies:

- Raw market data: 90 days
- Processed features: 1 year
- Trading signals and decisions: 3 years
- Performance metrics: 5 years
- System logs: 30 days

## Implementation Notes

1. File Organization:
   - Each container has a clear, modular structure
   - Shared utilities are copied to each container
   - Configuration files are mounted as read-only
   - Logs are persisted in a shared volume

2. Dependencies:
   - Each container includes only necessary dependencies
   - Shared utilities prevent code duplication
   - Clear separation of concerns between services

3. Security:
   - Non-root user for all containers
   - Read-only configuration mounts
   - Environment variables for secrets
   - Minimal system dependencies

4. Monitoring:
   - Centralized logging
   - Health check endpoints
   - Basic metrics collection
   - Resource usage monitoring

5. Deployment:
   - Simple service startup
   - Clear dependency chain
   - Resource limits
   - Automatic restarts
