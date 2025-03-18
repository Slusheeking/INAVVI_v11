#!/bin/bash
# Run Trading System
# This script starts the integrated trading system with the specified configuration

# Set default values
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_DB=${REDIS_DB:-0}
USE_GPU=${USE_GPU:-"true"}
CONFIG_PATH=${CONFIG_PATH:-""}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --redis-host)
      REDIS_HOST="$2"
      shift 2
      ;;
    --redis-port)
      REDIS_PORT="$2"
      shift 2
      ;;
    --redis-db)
      REDIS_DB="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --no-gpu)
      USE_GPU="false"
      shift
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --redis-host HOST     Redis host (default: localhost)"
      echo "  --redis-port PORT     Redis port (default: 6379)"
      echo "  --redis-db DB         Redis database (default: 0)"
      echo "  --config PATH         Path to configuration file"
      echo "  --no-gpu              Disable GPU acceleration"
      echo "  --log-level LEVEL     Set logging level (default: INFO)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

# Check for required API keys
if [ -z "$POLYGON_API_KEY" ]; then
  echo "Error: POLYGON_API_KEY is not set"
  echo "Please set it in the .env file or as an environment variable"
  exit 1
fi

if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
  echo "Error: ALPACA_API_KEY or ALPACA_API_SECRET is not set"
  echo "Please set them in the .env file or as environment variables"
  exit 1
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
  echo "Error: UNUSUAL_WHALES_API_KEY is not set"
  echo "Please set it in the .env file or as an environment variable"
  exit 1
fi

# Check for Slack integration (warning only, not required)
if [ -z "$SLACK_BOT_TOKEN" ] || [ -z "$SLACK_WEBHOOK_SYSTEM_NOTIFICATIONS" ]; then
  echo "Warning: Slack integration is not fully configured"
  echo "Reporting system will run with limited functionality"
  echo "Set SLACK_BOT_TOKEN and Slack webhook URLs in the .env file for full reporting capabilities"
fi

# Build command arguments
ARGS=""

if [ ! -z "$CONFIG_PATH" ]; then
  ARGS="$ARGS --config $CONFIG_PATH"
fi

if [ "$USE_GPU" = "false" ]; then
  ARGS="$ARGS --no-gpu"
fi

ARGS="$ARGS --redis-host $REDIS_HOST --redis-port $REDIS_PORT --redis-db $REDIS_DB"

# Set logging level
export LOG_LEVEL=$LOG_LEVEL

echo "Starting Integrated Trading System..."
echo "Redis: $REDIS_HOST:$REDIS_PORT/$REDIS_DB"
echo "GPU Acceleration: $USE_GPU"
echo "Log Level: $LOG_LEVEL"
echo "Reporting System: Enabled"

if [ ! -z "$CONFIG_PATH" ]; then
  echo "Configuration: $CONFIG_PATH"
fi

# Run the trading system
python -m tests.integrated_trading_system $ARGS

# Exit with the same status as the Python script
exit $?