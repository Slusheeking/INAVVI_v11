#!/bin/bash
set -e

# Print environment information
echo "Starting Feature Engineering Service"
echo "Environment: $ENVIRONMENT"
echo "TA-Lib Version: $(python -c 'import talib; print(talib.__version__)')"
echo "Pandas Version: $(python -c 'import pandas; print(pandas.__version__)')"
echo "NumPy Version: $(python -c 'import numpy; print(numpy.__version__)')"

# Verify TA-Lib installation
echo "Verifying TA-Lib installation..."
if python -c "import talib; print('TA-Lib functions available:', len(talib.get_functions()))"; then
    echo "TA-Lib installation verified successfully"
else
    echo "ERROR: TA-Lib installation verification failed"
    exit 1
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/features

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

# Initialize feature store if it doesn't exist
echo "Initializing feature store..."
if [ ! -d "$FEATURE_STORE_PATH" ] || [ -z "$(ls -A $FEATURE_STORE_PATH 2>/dev/null)" ]; then
    echo "Feature store is empty, creating directory structure..."
    mkdir -p $FEATURE_STORE_PATH/price_features
    mkdir -p $FEATURE_STORE_PATH/volume_features
    mkdir -p $FEATURE_STORE_PATH/liquidity_features
    mkdir -p $FEATURE_STORE_PATH/sentiment_features
    mkdir -p $FEATURE_STORE_PATH/metadata
    echo "Feature store initialized"
else
    echo "Feature store already exists"
fi

# Run a quick test to ensure TA-Lib is working properly with sample data
echo "Running TA-Lib test with sample data..."
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

# Start the feature engineering service
echo "Starting feature engineering service..."
exec python -m src.feature_engineering.main