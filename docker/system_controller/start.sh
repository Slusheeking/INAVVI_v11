#!/bin/bash

# Set service name for logging
export SERVICE_NAME="System Controller"

# Source common scripts
source /app/docker/common/dependency_check.sh
source /app/docker/common/validate_env.sh

# Initialize health as starting
set_health_status "starting" 

# Check critical dependencies
echo "[INFO] Checking critical dependencies for $SERVICE_NAME..."
check_timescaledb true

# Check non-critical dependencies
echo "[INFO] Checking non-critical dependencies for $SERVICE_NAME..."
check_redis false
REDIS_AVAILABLE=$?

# Set feature flags based on dependency availability
if [ $REDIS_AVAILABLE -eq 0 ]; then
  set_feature_flag "redis_caching" 1 "/app/feature_flags.json" "Redis connection successful"
else
  set_feature_flag "redis_caching" 0 "/app/feature_flags.json" "Redis connection failed"
fi

set -e

# Print environment information
echo "[INFO] Starting $SERVICE_NAME"
echo "[INFO] Environment: $ENVIRONMENT"
echo "[INFO] Log Level: $LOG_LEVEL"

# Create necessary directories
mkdir -p /app/logs

# Initialize system state in database if needed
echo "[INFO] Initializing system state in database..."
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

echo "[INFO] System state initialized"

# Record system startup event
echo "[INFO] Recording system startup event..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
INSERT INTO system_events (component, event_type, severity, message, details)
VALUES ('$SERVICE_NAME', 'startup', 'info', 'Service starting', 
        '{\"environment\": \"$ENVIRONMENT\", \"redis_available\": $REDIS_AVAILABLE}'::jsonb);
"

# Start the system controller service
echo "[INFO] Starting $SERVICE_NAME main process..."
# If all critical dependencies are available, mark as healthy
# Otherwise, mark as degraded but continue
if [ $REDIS_AVAILABLE -eq 0 ]; then
  set_health_status "healthy"
else
  set_health_status "degraded"
  echo "[WARN] Service starting in degraded mode due to missing non-critical dependencies"
fi

echo "[INFO] Executing main Python module..."
exec python -m src.system_controller.main