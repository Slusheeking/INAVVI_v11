#!/bin/bash
# Trading System Starter Script
# This script starts the entire trading system with all components

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Ensure logs directory exists
mkdir -p logs

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

# Check if Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Redis is not running. Starting Redis..."
    if [ -f redis/redis.conf ]; then
        redis-server redis/redis.conf &
    else
        redis-server &
    fi
    sleep 2
fi

# Start the trading system
echo "Starting trading system..."
python3 scripts/start_system.py "$@"

# Exit with the same status as the Python script
exit $?