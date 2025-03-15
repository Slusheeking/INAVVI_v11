#!/bin/bash
set -e

# Print environment information
echo "Starting Model Training Service"
echo "Environment: $ENVIRONMENT"
echo "GPU Enabled: $GPU_ENABLED"
echo "TensorFlow Version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"

# Check if GPU is available
if [ "$GPU_ENABLED" = "true" ]; then
    echo "Checking GPU availability..."
    if python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"; then
        echo "TensorFlow GPU setup successful"
    else
        echo "WARNING: GPU is enabled but TensorFlow cannot detect any GPU devices"
    fi
    
    if python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"; then
        echo "PyTorch CUDA setup successful"
        echo "CUDA Devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
        echo "CUDA Device Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
    else
        echo "WARNING: GPU is enabled but PyTorch cannot detect CUDA"
    fi
else
    echo "GPU is disabled, using CPU only"
    # Disable GPU for TensorFlow
    export CUDA_VISIBLE_DEVICES="-1"
    # Disable GPU for PyTorch
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/models/registry
mkdir -p /app/models/checkpoints

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

# Configure TensorFlow memory growth to avoid allocating all GPU memory at once
if [ "$GPU_ENABLED" = "true" ]; then
    echo "Configuring TensorFlow GPU memory growth..."
    python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Memory growth enabled for all GPUs')
    except RuntimeError as e:
        print('Error setting memory growth:', e)
"
fi

# Start TensorFlow Serving in the background if enabled
if [ -d "$MODEL_REGISTRY_PATH" ] && [ -n "$(ls -A $MODEL_REGISTRY_PATH 2>/dev/null)" ]; then
    echo "Starting TensorFlow Serving for model serving..."
    tensorflow_model_server \
        --port=$MODEL_SERVING_PORT \
        --model_name=trading_model \
        --model_base_path=$MODEL_REGISTRY_PATH \
        --rest_api_port=8501 \
        --enable_batching=true \
        --allow_version_labels_for_unavailable_models=true \
        --file_system_poll_wait_seconds=60 \
        --enable_model_warmup=true \
        > /app/logs/tf_serving.log 2>&1 &
    echo "TensorFlow Serving started on port $MODEL_SERVING_PORT"
else
    echo "No models found in registry, skipping TensorFlow Serving startup"
fi

# Start the model training service
echo "Starting model training service..."
exec python -m src.model_training.main