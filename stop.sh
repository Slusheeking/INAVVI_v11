#!/bin/bash
# Trading System Stopper Script
# This script stops the entire trading system with all components

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

# Check if --force flag is passed
FORCE_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        FORCE_FLAG="--force"
        break
    fi
done

# Stop the trading system
echo "Stopping trading system..."
python3 scripts/stop_system.py $FORCE_FLAG "$@"

# Check if Redis should be stopped
STOP_REDIS=true
for arg in "$@"; do
    if [ "$arg" == "--no-redis" ]; then
        STOP_REDIS=false
        break
    fi
done

# Stop Redis if needed
if [ "$STOP_REDIS" = true ]; then
    echo "Stopping Redis..."
    redis-cli shutdown
fi

echo "Trading system shutdown complete"
exit 0