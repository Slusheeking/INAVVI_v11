#!/bin/bash
set -e

# Print environment information
echo "Starting Continuous Learning Service"
echo "Environment: $ENVIRONMENT"
echo "TensorFlow Version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "Scikit-learn Version: $(python -c 'import sklearn; print(sklearn.__version__)')"

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/models/registry
mkdir -p /app/data/backtest
mkdir -p /app/results/backtest

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

# Wait for Model Training service with timeout and backoff
echo "Waiting for Model Training service..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://ats-model-training-tf:8003/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to Model Training service after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for Model Training service... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "Model Training service connection successful"

# Wait for Feature Engineering service with timeout and backoff
echo "Waiting for Feature Engineering service..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRY_COUNT=0

until curl -s http://ats-feature-engineering:8004/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Failed to connect to Feature Engineering service after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
    echo "Waiting for Feature Engineering service... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
    # Exponential backoff
    RETRY_INTERVAL=$((RETRY_INTERVAL*2))
    if [ $RETRY_INTERVAL -gt 30 ]; then
        RETRY_INTERVAL=30
    fi
done

echo "Feature Engineering service connection successful"

# Initialize MLflow for experiment tracking if enabled
if [ "${MLFLOW_ENABLED:-false}" = "true" ]; then
    echo "Initializing MLflow for experiment tracking..."
    mkdir -p /app/mlflow
    export MLFLOW_TRACKING_URI="file:///app/mlflow"
    echo "MLflow initialized with tracking URI: $MLFLOW_TRACKING_URI"
fi

# Run a quick test to ensure HMM for regime detection is working
echo "Testing HMM for market regime detection..."
python -c "
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 300

# Create a Hidden Markov Model with 3 hidden states
model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)

# Set the model parameters
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.2, 0.3, 0.5]
])
model.means_ = np.array([
    [0.0],  # Bull market (positive returns)
    [0.0],  # Sideways market (flat returns)
    [-1.0]  # Bear market (negative returns)
])
model.covars_ = np.array([
    [[0.5]],  # Low volatility
    [[1.0]],  # Medium volatility
    [[2.0]]   # High volatility
])

# Sample from the HMM
X, Z = model.sample(n_samples)

# Fit the model to the data
model2 = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
model2.fit(X)

# Predict the hidden states
Z2 = model2.predict(X)

print('HMM test successful')
print(f'Number of samples: {n_samples}')
print(f'Number of regimes detected: {len(np.unique(Z2))}')
print(f'Regime distribution: {np.bincount(Z2)}')
"

# Start the continuous learning service
echo "Starting continuous learning service..."
exec python -m src.continuous_learning.main