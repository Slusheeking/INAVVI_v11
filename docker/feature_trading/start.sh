#!/bin/bash

# Service-specific name for logging
export SERVICE_NAME="Feature Trading Platform"

# Source common scripts
source /app/docker/common/dependency_check.sh
source /app/docker/common/validate_env.sh

# Initialize health as starting
set_health_status "starting" "/app/health_status_feature_engineering.json"
set_health_status "starting" "/app/health_status_trading_strategy.json"

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

# Critical check for TensorFlow - the container should not start if TensorFlow is not working
echo "[INFO] Performing critical TensorFlow check..."
if ! python -c 'import tensorflow as tf; print(f"TensorFlow Version: {tf.__version__}")'; then
  echo "[FATAL] TensorFlow import failed. This is a critical dependency. Exiting."
  exit 1
fi

# Verify TensorFlow is working correctly by running a simple model
echo "[INFO] Verifying TensorFlow functionality with a simple model test..."
if ! python -c '
import tensorflow as tf
import numpy as np

try:
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer="adam", loss="mse")
    
    # Generate some random data
    x = np.random.random((10, 5))
    y = np.random.random((10, 1))
    
    # Fit for just one step to verify everything works
    model.fit(x, y, epochs=1, verbose=0)
    
    print("[SUCCESS] TensorFlow model test completed successfully")
except Exception as e:
    print(f"[ERROR] TensorFlow model test failed: {e}")
    exit(1)
'; then
  echo "[FATAL] TensorFlow functionality test failed. Exiting."
  exit 1
fi

# Verify TA-Lib installation
echo "[INFO] Verifying TA-Lib installation..."
if python -c "import talib; print('TA-Lib functions available:', len(talib.get_functions()))"; then
    echo "[SUCCESS] TA-Lib installation verified successfully"
    set_feature_flag "talib" 1 "/app/feature_flags.json" "Installation verified"
else
    echo "[ERROR] TA-Lib installation verification failed"
    set_feature_flag "talib" 0 "/app/feature_flags.json" "Installation verification failed"
    exit 1
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/features

# Wait for System Controller with timeout and backoff
echo "[INFO] Checking System Controller dependency..."
check_http_service "System Controller" "http://ats-system-controller:8000/health" true 30 5
echo "[SUCCESS] System Controller service is available"

# Wait for Data Pipeline with timeout and backoff
echo "[INFO] Checking Data Pipeline dependency..."
check_http_service "Data Pipeline" "http://ats-data-pipeline:8001/health" true 30 5
echo "[SUCCESS] Data Pipeline service is available"

# Wait for Model Platform with timeout and backoff
echo "[INFO] Checking Model Platform dependency..."
check_http_service "Model Platform" "http://ats-model-platform:8003/health" true 30 5
echo "[SUCCESS] Model Platform service is available"

# Initialize feature store if it doesn't exist
echo "[INFO] Initializing feature store..."
if [ ! -d "$FEATURE_STORE_PATH" ] || [ -z "$(ls -A $FEATURE_STORE_PATH 2>/dev/null)" ]; then
    echo "[INFO] Feature store is empty, creating directory structure..."
    mkdir -p $FEATURE_STORE_PATH/price_features
    mkdir -p $FEATURE_STORE_PATH/volume_features
    mkdir -p $FEATURE_STORE_PATH/liquidity_features
    mkdir -p $FEATURE_STORE_PATH/sentiment_features
    mkdir -p $FEATURE_STORE_PATH/metadata
    echo "[SUCCESS] Feature store initialized"
    set_feature_flag "feature_store" 1 "/app/feature_flags.json" "Initialized empty store"
else
    echo "[INFO] Feature store already exists"
    set_feature_flag "feature_store" 1 "/app/feature_flags.json" "Store already exists"
fi

# Run a quick test to ensure TA-Lib is working properly with sample data
echo "[INFO] Running TA-Lib test with sample data..."
python -c "
import numpy as np
import talib
import pandas as pd

# Create sample price data
data = {
    'open': np.random.random(100) * 100 + 50,
    'high': np.random.random(100) * 100 + 60,
    'low': np.random.random(100) * 100 + 40,
    'close': np.random.random(100) * 100 + 50,
    'volume': np.random.random(100) * 1000000
}
df = pd.DataFrame(data)

# Calculate some indicators
df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)

print('TA-Lib test successful')
print('Sample indicators calculated:')
print(f'SMA(20): {df[\"sma_20\"].iloc[-1]:.2f}')
print(f'RSI(14): {df[\"rsi_14\"].iloc[-1]:.2f}')
print(f'MACD: {df[\"macd\"].iloc[-1]:.2f}')
print(f'Bollinger Bands: Upper={df[\"upper\"].iloc[-1]:.2f}, Middle={df[\"middle\"].iloc[-1]:.2f}, Lower={df[\"lower\"].iloc[-1]:.2f}')
"

# Verify Alpaca API keys
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
    echo "[WARN] Alpaca API credentials are not set. Trading execution will be disabled."
    set_feature_flag "alpaca_trading" 0 "/app/feature_flags.json" "API credentials not set"
else
    echo "[INFO] Verifying Alpaca API connection..."
    # Simple check to verify Alpaca API credentials
    if curl -s -X GET \
        -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
        "$ALPACA_API_BASE_URL/v2/account" | grep -q "account_number"; then
        echo "[SUCCESS] Alpaca API connection successful"
        set_feature_flag "alpaca_trading" 1 "/app/feature_flags.json" "API connection successful"
    else
        echo "[WARN] Failed to connect to Alpaca API. Please check your credentials."
        set_feature_flag "alpaca_trading" 0 "/app/feature_flags.json" "API connection failed"
    fi
fi

# Check trading parameters
echo "[INFO] Trading parameters:"
echo "[INFO] Max Position Size: $MAX_POSITION_SIZE"
echo "[INFO] Risk Percentage: $RISK_PERCENTAGE"
echo "[INFO] Max Positions: $MAX_POSITIONS"
echo "[INFO] Max Position Value: $MAX_POSITION_VALUE"
echo "[INFO] Max Daily Value: $MAX_DAILY_VALUE"

# Record system startup event in database
echo "[INFO] Recording service startup in database..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
INSERT INTO system_events (component, event_type, severity, message, details)
VALUES ('$SERVICE_NAME', 'startup', 'info', 'Service starting', 
        '{\"environment\": \"$ENVIRONMENT\", \"redis_available\": $REDIS_AVAILABLE, \"feature_store_path\": \"$FEATURE_STORE_PATH\", \"alpaca_api\": \"$([ -z "$ALPACA_API_KEY" ] && echo "missing" || echo "configured")\"}'::jsonb);
" || echo "[WARN] Failed to record startup event in database"

# Create HTTP health check endpoints
mkdir -p /app/health_endpoints
cat > /app/health_endpoints/feature_engineering_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_feature_engineering.json", "r") as f:
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
    uvicorn.run(app, host="0.0.0.0", port=8004)
EOF

cat > /app/health_endpoints/trading_strategy_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_trading_strategy.json", "r") as f:
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
EOF

# Start health check endpoints
python /app/health_endpoints/feature_engineering_health.py &
python /app/health_endpoints/trading_strategy_health.py &

# If all critical dependencies are available, mark as healthy
# Otherwise, mark as degraded but continue
if [ $REDIS_AVAILABLE -eq 0 ]; then
  set_health_status "healthy" "/app/health_status_feature_engineering.json"
  set_health_status "healthy" "/app/health_status_trading_strategy.json"
else
  set_health_status "degraded" "/app/health_status_feature_engineering.json"
  set_health_status "degraded" "/app/health_status_trading_strategy.json"
  echo "[WARN] Service starting in degraded mode due to missing non-critical dependencies"
fi

echo "[INFO] Initialization complete. Services will be started by supervisord."
exit 0