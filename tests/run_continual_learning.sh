#!/bin/bash
# Run Continual Learning System
# This script starts the continual learning system for model updates

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
if [ -z "$POLYGON_API_KEY" ]; then
    echo "Error: POLYGON_API_KEY environment variable is required"
    exit 1
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
    echo "Error: UNUSUAL_WHALES_API_KEY environment variable is required"
    exit 1
fi

# Set default values for optional environment variables
export REDIS_HOST=${REDIS_HOST:-"localhost"}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export USE_GPU=${USE_GPU:-"true"}

echo "Starting Continual Learning System..."
echo "Redis: $REDIS_HOST:$REDIS_PORT DB:$REDIS_DB"
echo "GPU Enabled: $USE_GPU"

# Create directories if they don't exist
mkdir -p models
mkdir -p data

# Set environment variables for model and data directories
export MODELS_DIR=${MODELS_DIR:-"$(pwd)/models"}
export DATA_DIR=${DATA_DIR:-"$(pwd)/data"}

# Run the continual learning system
python3 -m tests.continual_learning_system

# Exit with the same code as the Python script
exit $?