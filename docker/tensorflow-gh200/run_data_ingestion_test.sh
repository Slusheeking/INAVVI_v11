#!/bin/bash
# Run Data Ingestion System Test
# This script runs the data ingestion system test with real API data

# Load environment variables from .env file
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
else
  echo "Warning: .env file not found"
fi

# Verify API keys are set
if [ -z "$POLYGON_API_KEY" ]; then
  echo "Error: POLYGON_API_KEY is not set"
  exit 1
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
  echo "Error: UNUSUAL_WHALES_API_KEY is not set"
  exit 1
fi

# Default values
WATCHLIST=${DEFAULT_WATCHLIST:-"AAPL,MSFT,GOOGL,AMZN,TSLA"}
DURATION=300
TEST_MODE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --watchlist)
      WATCHLIST="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --test)
      TEST_MODE=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--watchlist TICKERS] [--duration SECONDS] [--test]"
      exit 1
      ;;
  esac
done

# Print configuration
echo "Running Data Ingestion System with:"
echo "  Polygon API Key: ${POLYGON_API_KEY:0:4}****${POLYGON_API_KEY: -4}"
echo "  Unusual Whales API Key: ${UNUSUAL_WHALES_API_KEY:0:4}****${UNUSUAL_WHALES_API_KEY: -4}"
echo "  Watchlist: $WATCHLIST"
echo "  Duration: $DURATION seconds"
echo "  Test Mode: $([ $TEST_MODE -eq 1 ] && echo 'Yes' || echo 'No')"
echo ""

# Run the test
if [ $TEST_MODE -eq 1 ]; then
  echo "Running in test mode (unittest)"
  python3 -m unittest test_data_ingestion_system.py
else
  echo "Running data ingestion system for $DURATION seconds..."
  
  # Create a temporary Python script to run the system
  TMP_SCRIPT=$(mktemp)
  cat > $TMP_SCRIPT << EOF
#!/usr/bin/env python3
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_data_ingestion')

# Import our clients and system
from data_ingestion_api_clients import PolygonDataClient, UnusualWhalesClient, RedisCache
from test_data_ingestion_system import DataIngestionSystem, MockRedis
from load_env import load_env_file, get_env, get_env_list

# Load environment variables
load_env_file()

# Get configuration from environment
POLYGON_API_KEY = get_env('POLYGON_API_KEY')
UNUSUAL_WHALES_API_KEY = get_env('UNUSUAL_WHALES_API_KEY')
watchlist = "$WATCHLIST".split(',')  # Use command-line argument
logger.info(f"Using watchlist: {watchlist}")

# Create Redis client (using mock for simplicity)
redis_client = MockRedis()
logger.info("Using mock Redis")

# Create API clients
unusual_whales = UnusualWhalesClient(api_key=UNUSUAL_WHALES_API_KEY)
polygon = PolygonDataClient(api_key=POLYGON_API_KEY)

# Create data ingestion system
system = DataIngestionSystem(
    redis_client,
    unusual_whales,
    polygon
)

# Set active watchlist
system.active_watchlist = watchlist

# Start system
logger.info("Starting data ingestion system")
system.start()

try:
    # Run for specified duration
    logger.info(f"Running for $DURATION seconds")
    time.sleep($DURATION)
except KeyboardInterrupt:
    logger.info("Interrupted by user")
finally:
    # Stop system
    logger.info("Stopping data ingestion system")
    system.stop()
    
    # Close clients
    polygon.close()
    unusual_whales.close()
    
    # Print summary
    logger.info("\nRedis Data Summary:")
    for key in redis_client.data:
        logger.info(f"  {key}: {len(redis_client.data[key])} entries")

logger.info("Done")
EOF

  # Run the temporary script
  python3 $TMP_SCRIPT
  
  # Clean up
  rm $TMP_SCRIPT
fi

echo "Completed"