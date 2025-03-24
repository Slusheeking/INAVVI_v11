#!/bin/bash
set -e

echo "Starting up trading system container with integrated fixes..."

# Ensure hosts file is correct
echo "127.0.0.1 localhost" > /etc/hosts

# Apply Redis overcommit_memory setting
echo "Applying Redis overcommit_memory setting..."
sysctl vm.overcommit_memory=1 || echo "Warning: Could not set vm.overcommit_memory=1"

# Verify Redis data directories exist and have proper permissions
echo "Verifying Redis directories and permissions..."
mkdir -p /data /var/lib/redis /app/data/redis
chmod -R 777 /data /var/lib/redis /app/data/redis

# Create required project directories
mkdir -p /app/project/frontend/templates /app/project/frontend/sessions

# Start Redis server explicitly first
echo "Starting Redis server..."
redis-server /etc/redis/redis.conf &
REDIS_PID=$!

# Give Redis time to start
echo "Waiting for Redis to initialize..."
sleep 5

# Test Redis connection
echo "Testing Redis connection..."
if redis-cli -p 6380 -a trading_system_2025 ping | grep -q PONG; then
  echo "Redis is running correctly."
else
  echo "Redis failed to start properly. Trying alternative method..."
  kill $REDIS_PID || true
  sleep 2
  redis-server --port 6380 --requirepass trading_system_2025 --daemonize no &
  REDIS_PID=$!
  sleep 3

  # Check again
  if redis-cli -p 6380 -a trading_system_2025 ping | grep -q PONG; then
    echo "Redis is now running correctly."
  else
    echo "Warning: Redis is still not responding. Will try to continue..."
  fi
fi

# Configure TensorFlow GPU
echo "Configuring TensorFlow GPU..."
python3 -c "
import os
import tensorflow as tf
print(\"TensorFlow version:\", tf.__version__)
os.environ[\"TF_FORCE_GPU_ALLOW_GROWTH\"] = \"true\"
os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"
os.environ[\"TF_GPU_THREAD_MODE\"] = \"gpu_private\"
gpus = tf.config.list_physical_devices(\"GPU\")
if gpus:
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
      print(f\"Memory growth set for GPU: {gpu}\")
    except Exception as e:
      print(f\"Error setting memory growth: {e}\")
  print(f\"GPU configuration applied. Found {len(gpus)} GPU(s).\")
else:
  print(\"No GPUs found to configure.\")
"

# Copy our custom supervisord config with 127.0.0.1 for Flask
if [ -f /app/project/services.conf ]; then
    echo "Using custom supervisord configuration with 127.0.0.1 host binding..."
    cp /app/project/services.conf /etc/supervisor/conf.d/services.conf
fi

echo "All initialization completed successfully."
echo "Starting services with supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/services.conf