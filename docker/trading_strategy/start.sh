#!/bin/bash
set -e

# Print environment information
echo "Starting Trading Strategy Service"
echo "Environment: $ENVIRONMENT"
echo "Log Level: $LOG_LEVEL"

# Create necessary directories
mkdir -p /app/logs

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

# Always use netcat for Redis connectivity check instead of redis-cli
echo "Checking Redis connection using netcat..."
until nc -z $REDIS_HOST $REDIS_PORT; do
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
# If we reach here, Redis port is open
echo "Redis port is open, connection is successful"

echo "Redis connection successful"

# Wait for Data Acquisition with timeout and backoff
echo "Waiting for Data Acquisition..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://ats-data-acquisition:8001/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to Data Acquisition after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for Data Acquisition... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "Data Acquisition connection successful"

# Initialize database tables if needed
echo "Initializing database tables..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
-- Create tables for trading strategy
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    price NUMERIC NOT NULL,
    confidence NUMERIC,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    order_id VARCHAR(50) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity NUMERIC NOT NULL,
    price NUMERIC,
    status VARCHAR(20) NOT NULL,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS trading_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_timestamp TIMESTAMP WITH TIME ZONE,
    side VARCHAR(10) NOT NULL,
    entry_price NUMERIC NOT NULL,
    exit_price NUMERIC,
    quantity NUMERIC NOT NULL,
    profit_loss NUMERIC,
    profit_loss_percent NUMERIC,
    status VARCHAR(20) NOT NULL,
    stop_loss NUMERIC,
    take_profit NUMERIC,
    metadata JSONB
);

-- Create hypertables if TimescaleDB extension is available
DO \$\$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('trading_signals', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('trading_orders', 'timestamp', if_not_exists => TRUE);
    END IF;
END
\$\$;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals (symbol);
CREATE INDEX IF NOT EXISTS idx_trading_orders_symbol ON trading_orders (symbol);
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions (symbol);
"

echo "Database tables initialized"

# Verify Alpaca API keys
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
    echo "WARNING: Alpaca API credentials are not set. Trading execution will be disabled."
else
    echo "Verifying Alpaca API connection..."
    # Simple check to verify Alpaca API credentials
    if curl -s -X GET \
        -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
        "$ALPACA_API_BASE_URL/v2/account" | grep -q "account_number"; then
        echo "Alpaca API connection successful"
    else
        echo "WARNING: Failed to connect to Alpaca API. Please check your credentials."
    fi
fi

# Check trading parameters
echo "Trading parameters:"
echo "Max Position Size: $MAX_POSITION_SIZE"
echo "Risk Percentage: $RISK_PERCENTAGE"
echo "Max Positions: $MAX_POSITIONS"
echo "Max Position Value: $MAX_POSITION_VALUE"
echo "Max Daily Value: $MAX_DAILY_VALUE"

# Start the trading strategy service
echo "Starting trading strategy service..."
exec python -m src.trading_strategy.main