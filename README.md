# Autonomous Trading System

A high-performance, GPU-accelerated autonomous trading system leveraging AI/ML models for market prediction and trading strategy execution.

## System Architecture

The system is composed of several containerized microservices:

- **System Controller**: Central coordination service that manages the overall system
- **Data Pipeline**: Handles data acquisition, processing, and storage
- **Model Platform**: Manages model training, serving, and continuous learning
- **Feature Trading**: Combines feature engineering and trading strategy execution

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (NVIDIA GH200 Grace Hopper Superchip recommended)
- NVIDIA Container Toolkit installed
- At least 16GB RAM (32GB+ recommended)
- 100GB+ disk space

## Quick Start

### Setup

Run the setup script to create the necessary directory structure:

```bash
./setup.sh
```

### Starting the System

Use the start_system.sh script to start the system:

```bash
./start_system.sh
```

This script will:
1. Check for port availability and free up ports if needed
2. Start all system containers
3. Display the status of the running containers

Additional options:
- `--force`: Force kill processes using system ports
- `--skip-port-check`: Skip port availability check
- `--foreground`: Run in foreground (not detached)
- `--build`: Force rebuild of containers
- `--prune`: Prune unused Docker resources before starting

### Checking System Status

Use the system_status.sh script to check the status of the running system:

```bash
./system_status.sh
```

Additional options:
- `--detailed`: Show detailed status information
- `--logs`: Show recent logs from all containers
- `--lines N`: Show N lines of logs (default: 20)

### Stopping the System

Use the stop_system.sh script to stop the system:

```bash
./stop_system.sh
```

Additional options:
- `--remove-volumes`: Remove all volumes (WARNING: This will delete all data)
- `--prune`: Prune unused Docker resources after stopping

## Environment Configuration

The system uses environment variables for configuration. Copy the `.env.example` file to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key configuration parameters:
- API keys for data providers
- Database credentials
- Trading parameters
- GPU configuration

## System Components

### Data Acquisition

Collects market data from various sources:
- Price data
- Order book data
- Options data
- News and sentiment data

### Data Processing

Processes and normalizes raw data:
- Time series alignment
- Data cleaning
- Normalization
- Storage in TimescaleDB

### Feature Engineering

Extracts features from processed data:
- Technical indicators
- Statistical features
- Market microstructure features
- Sentiment analysis

### Model Training

Trains machine learning models:
- XGBoost for feature importance
- LSTM networks for time series prediction
- CNN models for pattern recognition
- Reinforcement learning for strategy optimization

### Model Services

Serves trained models for inference:
- Real-time prediction API
- Model versioning
- A/B testing

### Continuous Learning

Continuously improves models:
- Online learning
- Performance monitoring
- Automatic retraining
- Strategy adaptation

### Trading Strategy

Executes trading decisions:
- Signal generation
- Position sizing
- Risk management
- Order execution

## Monitoring

The system includes comprehensive monitoring:
- Prometheus for metrics collection
- Grafana for visualization
- Alertmanager for alerts
- Loki for log aggregation

Access the monitoring dashboard at http://localhost:3001 (default credentials: admin/admin)

## Troubleshooting

### Port Conflicts

If you encounter port conflicts, use the port_manager.sh script:

```bash
./docker/common/port_manager.sh --force
```

### Container Issues

Check container logs:

```bash
docker-compose logs -f [service_name]
```

### Database Issues

Connect to the TimescaleDB database:

```bash
psql -h localhost -p 5433 -U ats_user -d ats_db
```

## License

This project is proprietary and confidential.

## Contact

For support or inquiries, please contact the development team.