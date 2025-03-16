#!/bin/bash

# Service-specific name for logging
export SERVICE_NAME="Data Pipeline"

# Source common scripts
source /app/docker/common/dependency_check.sh
source /app/docker/common/validate_env.sh

# Initialize health as starting
set_health_status "starting" "/app/health_status_data_acquisition.json"
set_health_status "starting" "/app/health_status_data_processing.json"

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

# Print environment information
echo "[INFO] Starting $SERVICE_NAME"
echo "[INFO] Environment: $ENVIRONMENT"
echo "[INFO] Log Level: $LOG_LEVEL"

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/data/raw
mkdir -p /app/data/processed
mkdir -p /app/features

# Wait for System Controller with timeout and backoff
echo "[INFO] Checking System Controller dependency..."
check_http_service "System Controller" "http://ats-system-controller:8000/health" true 30 5
echo "[SUCCESS] System Controller service is available"

# Initialize database tables if needed
echo "[INFO] Initializing database tables..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -f /app/docker/data_pipeline/init_db.sql || {
  echo "[ERROR] Failed to initialize database tables"
  exit 1
}
echo "[SUCCESS] Database tables initialized successfully"

# Check if TimescaleDB extension is installed
echo "[INFO] Checking TimescaleDB extension..."
if PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "SELECT extname FROM pg_extension WHERE extname = 'timescaledb';" | grep -q "timescaledb"; then
  echo "[SUCCESS] TimescaleDB extension is installed"
  set_feature_flag "timescaledb_extension" 1 "/app/feature_flags.json" "Extension installed"
else
  echo "[WARN] TimescaleDB extension is not installed. Hypertables will not be created."
  set_feature_flag "timescaledb_extension" 0 "/app/feature_flags.json" "Extension not installed"
fi

# Verify API keys
if [ -z "$POLYGON_API_KEY" ]; then
  echo "[WARN] POLYGON_API_KEY is not set. Polygon data collection will be disabled."
  set_feature_flag "polygon_data" 0 "/app/feature_flags.json" "API key not set"
else
  echo "[INFO] POLYGON_API_KEY is set. Polygon data collection will be enabled."
  set_feature_flag "polygon_data" 1 "/app/feature_flags.json" "API key set"
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
  echo "[WARN] UNUSUAL_WHALES_API_KEY is not set. Unusual Whales data collection will be disabled."
  set_feature_flag "unusual_whales_data" 0 "/app/feature_flags.json" "API key not set"
else
  echo "[INFO] UNUSUAL_WHALES_API_KEY is set. Unusual Whales data collection will be enabled."
  set_feature_flag "unusual_whales_data" 1 "/app/feature_flags.json" "API key set"
fi

# Record system startup event in database
echo "[INFO] Recording service startup in database..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
INSERT INTO system_events (component, event_type, severity, message, details)
VALUES ('$SERVICE_NAME', 'startup', 'info', 'Service starting', 
        '{\"environment\": \"$ENVIRONMENT\", \"redis_available\": $REDIS_AVAILABLE, \"polygon_api\": \"$([ -z "$POLYGON_API_KEY" ] && echo "missing" || echo "configured")\", \"unusual_whales_api\": \"$([ -z "$UNUSUAL_WHALES_API_KEY" ] && echo "missing" || echo "configured")\"}'::jsonb);
" || echo "[WARN] Failed to record startup event in database"

# Create HTTP health check endpoints
mkdir -p /app/health_endpoints
cat > /app/health_endpoints/data_acquisition_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_data_acquisition.json", "r") as f:
            health_data = json.load(f)
        
        if health_data["status"] == "healthy":
            return {"status": "healthy"}
        elif health_data["status"] == "degraded":
            return {"status": "degraded"}
        else:
            raise HTTPException(status_code=503, detail="Service unhealthy")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

cat > /app/health_endpoints/data_processing_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_data_processing.json", "r") as f:
            health_data = json.load(f)
        
        if health_data["status"] == "healthy":
            return {"status": "healthy"}
        elif health_data["status"] == "degraded":
            return {"status": "degraded"}
        else:
            raise HTTPException(status_code=503, detail="Service unhealthy")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
EOF

# Start health check endpoints
python /app/health_endpoints/data_acquisition_health.py &
python /app/health_endpoints/data_processing_health.py &

# If all critical dependencies are available, mark as healthy
# Otherwise, mark as degraded but continue
if [ $REDIS_AVAILABLE -eq 0 ]; then
  set_health_status "healthy" "/app/health_status_data_acquisition.json"
  set_health_status "healthy" "/app/health_status_data_processing.json"
else
  set_health_status "degraded" "/app/health_status_data_acquisition.json"
  set_health_status "degraded" "/app/health_status_data_processing.json"
  echo "[WARN] Service starting in degraded mode due to missing non-critical dependencies"
fi

echo "[INFO] Initialization complete. Services will be started by supervisord."
exit 0