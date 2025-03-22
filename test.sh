#!/bin/bash
# Trading System Test Script
# This script tests the startup and shutdown of the trading system

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

# Parse arguments
DEBUG_FLAG=""
NO_GPU_FLAG=""

for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        DEBUG_FLAG="--debug"
    elif [ "$arg" == "--no-gpu" ]; then
        NO_GPU_FLAG="--no-gpu"
    fi
done

# Run the test script
echo "Starting system test..."
python3 scripts/test_system.py $DEBUG_FLAG $NO_GPU_FLAG

# Get the exit code
EXIT_CODE=$?

# Print result
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n\033[0;32mSystem test PASSED\033[0m"
else
    echo -e "\n\033[0;31mSystem test FAILED\033[0m"
fi

exit $EXIT_CODE