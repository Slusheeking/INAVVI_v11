#!/bin/bash
# Trading System Container Shutdown Script
# This script gracefully shuts down all services in the container

echo "Initiating graceful shutdown of trading system container..."

# Stop the trading system components first
if [ -f /app/scripts/stop_system.py ]; then
    echo "Stopping trading system components..."
    python3 /app/scripts/stop_system.py --force
else
    echo "Warning: stop_system.py not found, skipping component shutdown"
fi

# Stop Redis if running
if pgrep redis-server > /dev/null; then
    echo "Stopping Redis server..."
    redis-cli shutdown
    sleep 2
fi

# Stop Prometheus if running
if pgrep prometheus > /dev/null; then
    echo "Stopping Prometheus..."
    pkill -SIGTERM prometheus
    sleep 2
fi

# Stop Redis exporter if running
if pgrep redis_exporter > /dev/null; then
    echo "Stopping Redis exporter..."
    pkill -SIGTERM redis_exporter
    sleep 1
fi

# Stop supervisord last
if pgrep supervisord > /dev/null; then
    echo "Stopping supervisord..."
    supervisorctl shutdown
    sleep 2
fi

echo "Shutdown complete."