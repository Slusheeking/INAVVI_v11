# Autonomous Trading System - Containerization Strategy

## Overview

This repository contains the containerization strategy for the Autonomous Trading System. The system is designed to run in Docker containers, with a hybrid approach that leverages existing containers for some components and creates new containers for others.

## System Architecture

The system consists of the following components:

### Existing Containers (Used As-Is)

- **TimescaleDB** (`timescaledb-v1-1`): The time-series database that stores all market data, features, and trading information.
- **TensorFlow Model Training** (`model-training-tensorflow-v1-1`): The container responsible for training machine learning models.
- **TensorFlow Feature Engineering** (`feature-engineering-tensorflow-v1-1`): The container responsible for feature engineering and transformation.

### New Containers (Created by This Strategy)

- **System Controller**: Orchestrates the entire system, managing the lifecycle of other components.
- **Data Acquisition**: Collects market data from various sources (Polygon.io, Unusual Whales, etc.).
- **Trading Strategy**: Executes trading strategies based on signals from the model training component.

## Directory Structure

```
autonomous_trading_system/
├── .env                           # Environment variables
├── docker-compose.yml             # Docker Compose configuration
├── start_system.sh                # Script to start the system
├── stop_system.sh                 # Script to stop the system
├── docker/                        # Docker-related files
│   ├── system_controller/         # System Controller Docker files
│   │   └── Dockerfile             # System Controller Dockerfile
│   ├── data_acquisition/          # Data Acquisition Docker files
│   │   └── Dockerfile             # Data Acquisition Dockerfile
│   ├── trading_strategy/          # Trading Strategy Docker files
│   │   └── Dockerfile             # Trading Strategy Dockerfile
│   └── timescaledb/               # TimescaleDB Docker files
│       └── init/                  # TimescaleDB initialization scripts
│           └── init.sql           # Database initialization SQL
└── src/                           # Source code
    ├── backtesting/               # Backtesting components
    ├── continuous_learning/       # Continuous learning components
    ├── data_acquisition/          # Data acquisition components
    ├── feature_engineering/       # Feature engineering components
    ├── model_training/            # Model training components
    ├── tests/                     # Test components
    ├── trading_strategy/          # Trading strategy components
    └── utils/                     # Utility components
```

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- The following containers must be running:
  - `timescaledb-v1-1`
  - `model-training-tensorflow-v1-1`
  - `feature-engineering-tensorflow-v1-1`

### Configuration

The system is configured using environment variables in the `.env` file. The following variables are required:

```
# API Keys
POLYGON_API_KEY=your_polygon_api_key
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets/v2

# Database Configuration
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5434
TIMESCALEDB_DATABASE=ats_db
TIMESCALEDB_USER=ats_user
TIMESCALEDB_PASSWORD=ats_password

# System Configuration
LOG_LEVEL=INFO
MAX_POSITION_SIZE=2500
RISK_PERCENTAGE=0.02
MAX_POSITIONS=50
GPU_ENABLED=true
```

### Starting the System

To start the system, run the following command:

```bash
./start_system.sh
```

This script will:
1. Check if the required existing containers are running
2. Build and start the new containers
3. Check if the containers started successfully

### Stopping the System

To stop the system, run the following command:

```bash
./stop_system.sh
```

This script will stop the new containers, but leave the existing containers running.

## Monitoring the System

### Viewing Logs

To view the logs of all containers, run:

```bash
docker-compose logs -f
```

To view the logs of a specific container, run:

```bash
docker-compose logs -f <container_name>
```

For example, to view the logs of the system controller:

```bash
docker-compose logs -f system-controller
```

### Checking Container Status

To check the status of all containers, run:

```bash
docker-compose ps
```

## Troubleshooting

### Common Issues

#### Containers Won't Start

If the containers won't start, check the logs for error messages:

```bash
docker-compose logs
```

#### Database Connection Issues

If there are database connection issues, check:
1. The database container is running: `docker ps | grep timescaledb-v1-1`
2. The database connection parameters in the `.env` file are correct
3. The database is accessible from the containers: `docker exec -it system-controller ping timescaledb-v1-1`

#### API Connection Issues

If there are API connection issues, check:
1. The API keys in the `.env` file are correct
2. The API endpoints are accessible from the containers: `docker exec -it data-acquisition curl -I https://api.polygon.io`

## Advanced Configuration

### Using GPU Acceleration

To enable GPU acceleration for model training, set the `GPU_ENABLED` environment variable to `true` in the `.env` file.

### Customizing Container Resources

To customize the resources allocated to each container, modify the `docker-compose.yml` file. For example, to limit the CPU and memory usage of the system controller:

```yaml
system-controller:
  # ...
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 1G
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.