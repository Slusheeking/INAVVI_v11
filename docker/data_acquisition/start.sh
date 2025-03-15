#!/bin/bash
set -e

# Print environment information
echo "Starting Data Acquisition Service"
echo "Environment: $ENVIRONMENT"
echo "Log Level: $LOG_LEVEL"

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/data/raw
mkdir -p /app/data/processed

# Wait for TimescaleDB with timeout and backoff
echo "Waiting for TimescaleDB..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c '\q'; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to TimescaleDB after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for TimescaleDB... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "TimescaleDB connection successful"

# Wait for Redis with timeout and backoff
echo "Waiting for Redis..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

# Use redis-cli for Redis connectivity check
echo "Checking Redis connection using redis-cli..."
until redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping | grep -q "PONG"; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to Redis after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for Redis... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "Redis connection successful"

# Wait for System Controller with timeout and backoff
echo "Waiting for System Controller..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://ats-system-controller:8000/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to System Controller after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for System Controller... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "System Controller connection successful"

# Initialize database tables if needed
echo "Initializing database tables..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_prices (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    PRIMARY KEY (symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS market_quotes (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bid_price NUMERIC NOT NULL,
    ask_price NUMERIC NOT NULL,
    bid_size NUMERIC NOT NULL,
    ask_size NUMERIC NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS market_trades (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    conditions VARCHAR(100),
    PRIMARY KEY (symbol, timestamp)
);

-- Create hypertables if TimescaleDB extension is available
DO \$\$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('market_prices', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('market_quotes', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('market_trades', 'timestamp', if_not_exists => TRUE);
    END IF;
END
\$\$;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_prices_symbol ON market_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol ON market_quotes (symbol);
CREATE INDEX IF NOT EXISTS idx_market_trades_symbol ON market_trades (symbol);
"

echo "Database tables initialized"

# Verify API keys
if [ -z "$POLYGON_API_KEY" ]; then
    echo "WARNING: POLYGON_API_KEY is not set. Polygon data collection will be disabled."
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
    echo "WARNING: UNUSUAL_WHALES_API_KEY is not set. Unusual Whales data collection will be disabled."
fi

# Start the data acquisition service
echo "Starting data acquisition service..."
exec python -m src.data_acquisition.main