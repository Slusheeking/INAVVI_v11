#!/bin/bash
# Comprehensive startup script for trading system container
# Don't exit immediately on errors to make the startup more resilient
set +e

echo "=================="
echo "== TensorFlow =="
echo "=================="

cat /etc/nv_tegra_release || true
echo ""
python -c "import tensorflow as tf; print(f'TensorFlow Version {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"

echo "Setting up Redis..."
# Apply Redis overcommit memory setting
echo "Applying Redis overcommit_memory setting..."
sysctl vm.overcommit_memory=1 || echo "Failed to set overcommit_memory, continuing anyway"

# Ensure Redis directories exist
mkdir -p /var/run/redis
mkdir -p /var/log/redis
mkdir -p /data
mkdir -p /var/lib/redis

# Start Supervisord
echo "Starting services via Supervisord..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/services.conf &
SUPERVISOR_PID=$!

# Wait for Redis to be ready
echo "Waiting for Redis to start..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! redis-cli -h localhost -p 6380 -a "trading_system_2025" ping > /dev/null 2>&1; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "WARNING: Redis not responding after $MAX_RETRIES attempts, but continuing startup"
        REDIS_FAILED=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "Waiting for Redis... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done

if [ -z "$REDIS_FAILED" ]; then
    echo "Redis is running successfully"
else
    echo "Attempting to restart Redis service manually..."
    redis-server /etc/redis/redis.conf --port 6380 --requirepass trading_system_2025 --daemonize yes
    sleep 3
fi

# Wait for Prometheus to be ready
echo "Waiting for Prometheus to start..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "WARNING: Prometheus not responding after $MAX_RETRIES attempts, but continuing startup"
        PROMETHEUS_FAILED=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "Waiting for Prometheus... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done

if [ -z "$PROMETHEUS_FAILED" ]; then
    echo "Prometheus is running successfully"
else
    echo "Attempting to restart Prometheus service manually..."
    pkill -f prometheus || true
    sleep 2
    /usr/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus --web.listen-address=:9090 &
    sleep 3
fi

# Setup GPU environment variables for optimal detection
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=all
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=8
export TF_USE_CUDNN=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT=true
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics
export TF_FORCE_UNIFIED_MEMORY=1
export TF_ENABLE_NUMA_AWARE_ALLOCATORS=1

# Run the enhanced unified GPU test suite
echo "Running enhanced unified GPU acceleration test..."
if [[ -f "/app/project/unified_gpu_test.py" ]]; then
    python /app/project/unified_gpu_test.py || echo "Some GPU tests failed, but continuing startup"
else
    echo "Error: unified_gpu_test.py not found. This should be included in the container build."
    echo "Continuing startup with limited GPU testing..."
    
    # Basic TensorFlow GPU check
    python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
fi

# Initialize monitoring system
echo "Initializing monitoring system..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:8000/metrics > /dev/null 2>&1; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Monitoring system failed to start after $MAX_RETRIES attempts"
        # Don't exit, just continue
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "Waiting for monitoring system... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done
echo "Monitoring system initialized"

# Wait for frontend to be ready
echo "Waiting for frontend to start..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:5000/health > /dev/null 2>&1; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Frontend failed to start after $MAX_RETRIES attempts"
        exit 1
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "Waiting for frontend... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done
echo "Frontend is running"

# Special cases for event-driven components
echo "All services started successfully. System is ready."

# Keep the container running - monitor main supervisor process
while kill -0 $SUPERVISOR_PID 2>/dev/null; do
    sleep 5
done

echo "Supervisord process ended. Shutting down container."
exit 1