#!/bin/bash

# Service-specific name for logging
export SERVICE_NAME="Model Platform"

# Source common scripts
source /app/docker/common/dependency_check.sh
source /app/docker/common/validate_env.sh

# Initialize health as starting
set_health_status "starting" "/app/health_status_model_training.json"
set_health_status "starting" "/app/health_status_model_services.json"
set_health_status "starting" "/app/health_status_continuous_learning.json"

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
echo "[INFO] GPU Enabled: $GPU_ENABLED"

# Make test script executable
chmod +x /app/docker/model_platform/test_gpu_support.py

# Print NVIDIA GPU information if available
echo "[INFO] Checking NVIDIA GPU information..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || echo "[WARN] nvidia-smi command failed, but continuing"
    echo "[INFO] NVIDIA driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Unknown')"
fi

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

# Check PyTorch version (non-critical)
echo "[INFO] Checking PyTorch version..."
python -c 'import torch; print(f"PyTorch Version: {torch.__version__}")' || echo "[WARN] PyTorch import failed (non-critical)"

# Check if GPU is available, but don't fail if not
if [ "$GPU_ENABLED" = "true" ]; then
  echo "[INFO] Checking GPU availability..."

  # Set proper environment variables for GH200 Grace Hopper
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
  export TF_GPU_ALLOCATOR=cuda_malloc_async
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
  export CUDA_MODULE_LOADING=LAZY
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
  
  # Verify TensorFlow can see GPUs
  TF_GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")
  echo "[INFO] TensorFlow detected $TF_GPU_COUNT GPU(s)"
  
  if [ "$TF_GPU_COUNT" -gt 0 ]; then
    # Verify TensorFlow can actually use the GPU with a simple test
    echo "[INFO] Verifying TensorFlow GPU functionality..."
    if python -c '
import tensorflow as tf
import time

try:
    # Check if GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        print("[WARN] No GPUs detected by TensorFlow")
        exit(1)
        
    # Place tensors on GPU
    with tf.device("/GPU:0"):
        # Create large tensors to force GPU usage
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        
        # Time the matrix multiplication
        start = time.time()
        c = tf.matmul(a, b)
        # Force execution with a small access
        _ = c[0, 0].numpy()
        end = time.time()
        
    print(f"[SUCCESS] TensorFlow GPU test completed in {end - start:.2f} seconds")
    exit(0)
except Exception as e:
    print(f"[ERROR] TensorFlow GPU test failed: {e}")
    exit(1)
'; then
      echo "[SUCCESS] TensorFlow GPU functionality verified"
      set_feature_flag "gpu_tensorflow" 1 "/app/feature_flags.json" "TensorFlow GPU functional"
    else
      echo "[WARN] TensorFlow GPU functionality test failed. Will use CPU instead."
      set_feature_flag "gpu_tensorflow" 0 "/app/feature_flags.json" "TensorFlow GPU test failed"
      # Force CPU mode for TensorFlow
      export CUDA_VISIBLE_DEVICES="-1"
    fi
  else
    echo "[WARN] No GPUs detected by TensorFlow. Will use CPU instead."
    set_feature_flag "gpu_tensorflow" 0 "/app/feature_flags.json" "No GPUs detected"
    # Force CPU mode for TensorFlow
    export CUDA_VISIBLE_DEVICES="-1"
  fi
  
  # Check PyTorch GPU (non-critical)
  python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" || echo "[WARN] PyTorch GPU check failed"
  python -c "import torch; print('CUDA Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" || echo "[WARN] PyTorch device count check failed"
  python -c "import torch; print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" || echo "[WARN] PyTorch device name check failed"
  
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    set_feature_flag "gpu_pytorch" 1 "/app/feature_flags.json" "PyTorch GPU available"
  else
    set_feature_flag "gpu_pytorch" 0 "/app/feature_flags.json" "PyTorch GPU not available"
  fi
else
  echo "[INFO] GPU is disabled, using CPU only"
  # Disable GPU for TensorFlow
  export CUDA_VISIBLE_DEVICES="-1"
  # Disable GPU for PyTorch
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
  
  set_feature_flag "gpu_tensorflow" 0 "/app/feature_flags.json" "Disabled by configuration"
  set_feature_flag "gpu_pytorch" 0 "/app/feature_flags.json" "Disabled by configuration"
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/models/registry
mkdir -p /app/models/checkpoints
mkdir -p /app/data/backtest
mkdir -p /app/results/backtest
mkdir -p /tmp/matplotlib
chmod 777 /tmp/matplotlib
export MPLCONFIGDIR=/tmp/matplotlib
echo "[INFO] Created matplotlib config directory at /tmp/matplotlib"

# Initialize model registry with a placeholder model
echo "[INFO] Initializing model registry with a placeholder model..."
python /app/docker/model_platform/init_model_registry.py
if [ $? -eq 0 ]; then
  echo "[SUCCESS] Model registry initialized successfully"
else
  echo "[WARN] Failed to initialize model registry, but continuing startup"
fi

# Wait for System Controller with timeout and backoff
echo "[INFO] Checking System Controller dependency..."
check_http_service "System Controller" "http://ats-system-controller:8000/health" true 30 5
echo "[SUCCESS] System Controller service is available"

# Wait for Data Pipeline with timeout and backoff
echo "[INFO] Checking Data Pipeline dependency..."
check_http_service "Data Pipeline" "http://ats-data-pipeline:8001/health" true 30 5
echo "[SUCCESS] Data Pipeline service is available"

# Configure TensorFlow memory growth to avoid allocating all GPU memory at once
if [ "$GPU_ENABLED" = "true" ]; then
  echo "[INFO] Configuring TensorFlow GPU memory growth..."
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  
  # Run the comprehensive GPU test
  echo "[INFO] Running comprehensive GPU test..."
  if python /app/docker/model_platform/test_gpu_support.py > /app/logs/gpu_test_results.log 2>&1; then
    echo "[SUCCESS] GPU test completed. See /app/logs/gpu_test_results.log for details."
  else
    echo "[WARN] GPU test encountered issues. See /app/logs/gpu_test_results.log for details."
  fi
  if ! python -c "
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Memory growth enabled for GPU: {gpu}')
        print('Memory growth enabled for all GPUs')
    else:
        print('No GPUs found')
except Exception as e:
    print('[ERROR] Error setting memory growth:', e)
    exit(1)
"; then
    echo "[WARN] Failed to configure TensorFlow memory growth. This may cause memory issues."
  fi
fi

# Initialize MLflow for experiment tracking if enabled
if [ "${MLFLOW_ENABLED:-false}" = "true" ]; then
  echo "[INFO] Initializing MLflow for experiment tracking..."
  mkdir -p /app/mlflow
  export MLFLOW_TRACKING_URI="file:///app/mlflow"
  echo "[INFO] MLflow initialized with tracking URI: $MLFLOW_TRACKING_URI"
  set_feature_flag "mlflow" 1 "/app/feature_flags.json" "Enabled"
else
  set_feature_flag "mlflow" 0 "/app/feature_flags.json" "Disabled"
fi

# Record system startup event in database
echo "[INFO] Recording service startup in database..."
PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "
INSERT INTO system_events (component, event_type, severity, message, details)
VALUES ('$SERVICE_NAME', 'startup', 'info', 'Service starting', 
        '{\"environment\": \"$ENVIRONMENT\", \"redis_available\": $REDIS_AVAILABLE, \"gpu_enabled\": \"$GPU_ENABLED\", \"mlflow_enabled\": \"${MLFLOW_ENABLED:-false}\"}'::jsonb);
" || echo "[WARN] Failed to record startup event in database"

# Create HTTP health check endpoints
mkdir -p /app/health_endpoints
cat > /app/health_endpoints/model_training_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_model_training.json", "r") as f:
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
    uvicorn.run(app, host="0.0.0.0", port=8003)
EOF

cat > /app/health_endpoints/continuous_learning_health.py << EOF
from fastapi import FastAPI, HTTPException
import uvicorn
import json
import os

app = FastAPI()

@app.get("/health")
async def health():
    try:
        with open("/app/health_status_continuous_learning.json", "r") as f:
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
    uvicorn.run(app, host="0.0.0.0", port=8005)
EOF

# Start health check endpoints
python /app/health_endpoints/model_training_health.py &
python /app/health_endpoints/continuous_learning_health.py &

# If all critical dependencies are available, mark as healthy
# Otherwise, mark as degraded but continue
if [ $REDIS_AVAILABLE -eq 0 ]; then
  set_health_status "healthy" "/app/health_status_model_training.json"
  set_health_status "healthy" "/app/health_status_model_services.json"
  set_health_status "healthy" "/app/health_status_continuous_learning.json"
else
  set_health_status "degraded" "/app/health_status_model_training.json"
  set_health_status "degraded" "/app/health_status_model_services.json"
  set_health_status "degraded" "/app/health_status_continuous_learning.json"
  echo "[WARN] Service starting in degraded mode due to missing non-critical dependencies"
fi

echo "[INFO] Initialization complete. Services will be started by supervisord."
exit 0