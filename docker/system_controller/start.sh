#!/bin/bash
set -e

# Print environment information
echo "Starting System Controller Service"
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

# Always use netcat for Redis connectivity check instead of relying on redis-cli
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

# Initialize system state in database if needed
echo "Initializing system state..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
CREATE TABLE IF NOT EXISTS system_state (
    id SERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB
);

CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    component VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    details JSONB,
    PRIMARY KEY (id, timestamp)
);

-- Create hypertable for system events if TimescaleDB extension is available
DO \$\$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('system_events', 'timestamp', if_not_exists => TRUE);
    END IF;
END
\$\$;
"

echo "System state initialized"

# Start the system controller service
echo "Starting system controller service..."
exec python -m src.system_controller.main