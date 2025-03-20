#!/bin/bash

# Exit on any error
set -e

echo "Starting Trading System Tests..."

# Check if running in container
if [ ! -f /.dockerenv ]; then
    echo "Error: Tests must be run inside the trading system container"
    exit 1
fi

# Check GPU availability
nvidia-smi > /dev/null 2>&1 || { echo "Error: NVIDIA GPU not detected"; exit 1; }

# Verify environment variables
required_vars=("POLYGON_API_KEY" "UNUSUAL_WHALES_API_KEY" "REDIS_HOST" "REDIS_PORT" "REDIS_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Create required directories if they don't exist
mkdir -p data models logs

# Start Redis if not running
redis-cli ping > /dev/null 2>&1 || {
    echo "Starting Redis server..."
    redis-server /etc/redis/redis.conf &
    sleep 5  # Wait for Redis to start
}

# Start Prometheus if not running
pgrep prometheus > /dev/null || {
    echo "Starting Prometheus..."
    prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus &
    sleep 5  # Wait for Prometheus to start
}

# Install test dependencies
pip install pytest pytest-asyncio python-dotenv polygon-api-client redis pandas numpy tensorflow cupy-cuda12x websockets prometheus_client

# Run the tests
echo "Running system tests..."
cd /app/project && python3 -m pytest gpu_system/test_trading_system.py -v --asyncio-mode=auto

# Check test results
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Please check the output above for details."
    exit 1
fi
