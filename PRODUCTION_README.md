# INAVVI Trading System Production Guide

This guide provides comprehensive instructions for setting up, managing, and maintaining the INAVVI Trading System in a production environment.

## System Architecture

The INAVVI Trading System is designed to run as a unified Docker container with the following components:

- **Trading Engine**: Core algorithmic trading system
- **ML Engine**: Machine learning models for market prediction
- **Stock Selection Engine**: GPU-accelerated stock universe building and filtering
- **Data Pipeline**: Data loading, preprocessing, and feature engineering
- **API Clients**: Interfaces to Polygon.io and Unusual Whales APIs
- **Redis**: In-memory database for caching and real-time data
- **Prometheus**: Monitoring and metrics collection

The system is optimized for NVIDIA GH200 Grace Hopper Superchips for maximum performance.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Valid API keys for Polygon.io and Unusual Whales

## Quick Start

1. Clone the repository
2. Create a `.env` file with your API keys
3. Run the production setup script:

```bash
./manage_production.sh setup
```

4. Start the system:

```bash
./manage_production.sh start
```

5. Monitor the system:

```bash
./manage_production.sh monitor
```

## Production Scripts

The following scripts are provided for managing the production system:

### Main Management Script

- `manage_production.sh`: Master script for managing the production system

```bash
# Show help
./manage_production.sh help

# Common commands
./manage_production.sh setup    # Set up the production environment
./manage_production.sh start    # Start the production system
./manage_production.sh stop     # Stop the production system
./manage_production.sh status   # Check system status
./manage_production.sh logs     # View system logs
./manage_production.sh monitor  # Monitor the system
./manage_production.sh backup   # Create a backup
```

### Setup and Verification

- `setup_production.sh`: Sets up the production environment
- `verify_production_readiness.sh`: Verifies that all files are production-ready

### Testing

- `run_tests_in_docker.sh`: Runs all tests inside the Docker container

### Monitoring

- `monitor_production.sh`: Monitors the health of the production system

```bash
# Basic monitoring
./monitor_production.sh

# Continuous monitoring (refreshes every 30 seconds)
./monitor_production.sh --watch
```

### Backup and Recovery

- `backup_production_data.sh`: Creates backups of all important data

```bash
# Create a backup
./backup_production_data.sh

# Create a backup and copy to remote storage (AWS S3 or Google Cloud Storage)
./backup_production_data.sh --remote

# Set up a daily backup cron job
./backup_production_data.sh --schedule
```

## Docker Configuration

The system uses Docker for containerization with the following key files:

- `Dockerfile.unified`: Defines the container image with all dependencies
- `docker-compose.unified.yml`: Defines the container configuration and volume mappings

## Environment Variables

Create a `.env` file with the following variables:

```
# API Keys
POLYGON_API_KEY=your_polygon_api_key
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_API_URL=https://paper-api.alpaca.markets

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0
REDIS_USERNAME=default
REDIS_PASSWORD=trading_system_2025
REDIS_SSL=false
REDIS_TIMEOUT=5

# GPU Configuration
USE_GPU=true
```

## Directory Structure

The system uses the following directory structure:

- `/app/project`: Source code (mapped from host)
- `/app/data`: Market data and processed data
- `/app/models`: ML models
- `/app/logs`: System logs
- `/etc/prometheus`: Prometheus configuration
- `/etc/redis`: Redis configuration

## Monitoring

The system exposes the following monitoring endpoints:

- Prometheus: http://localhost:9090
- Redis Metrics: http://localhost:9121/metrics
- Jupyter Lab: http://localhost:8888

## Backup and Recovery

Backups are stored in the `./backups` directory with the following naming convention:

```
trading_system_backup_YYYYMMDD_HHMMSS.tar.gz
```

Each backup includes:

- Redis data
- ML models
- Market data
- System logs
- Configuration files

To restore from a backup, use:

```bash
./manage_production.sh restore ./backups/restore_trading_system_backup_YYYYMMDD_HHMMSS.sh
```

## Production Readiness Checklist

Before deploying to production, ensure:

1. **Environment Variables**: All API keys and configuration are set in `.env`
2. **Redis Configuration**: Redis config is properly configured
3. **Network Setup**: The trading-network exists (`docker network create trading-network`)
4. **GPU Availability**: NVIDIA drivers and Docker runtime are properly configured
5. **Testing**: All tests pass in the Docker container
6. **Monitoring**: External monitoring is set up for the Prometheus endpoints
7. **Backup Strategy**: Regular backups are scheduled

## Troubleshooting

### Container Won't Start

Check Docker logs:

```bash
docker-compose -f docker-compose.unified.yml logs
```

### System Not Running

Check system logs:

```bash
./manage_production.sh logs trading
```

### GPU Not Detected

Verify NVIDIA drivers and Docker runtime:

```bash
nvidia-smi
docker info | grep -i runtime
```

### Redis Connection Issues

Check Redis logs:

```bash
./manage_production.sh logs redis
```

## Performance Tuning

The system is pre-configured for optimal performance on NVIDIA GH200 Grace Hopper Superchips. Key optimizations include:

- TF32 acceleration
- Unified memory for better performance
- NUMA-aware allocators
- XLA JIT compilation
- Mixed precision (float16)

## Security Considerations

- API keys are stored in the `.env` file and mounted into the container
- Redis is configured with a password
- The container runs with limited privileges
- All external ports are configurable

## Maintenance

Regular maintenance tasks:

1. **Backups**: Run daily backups
2. **Updates**: Regularly update the system with `./manage_production.sh update`
3. **Monitoring**: Regularly check system health with `./manage_production.sh monitor`
4. **Logs**: Regularly check logs for errors with `./manage_production.sh logs`

## Support

For issues or questions, please contact the development team.