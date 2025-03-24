#!/bin/bash
set -e

echo "Starting trading system with custom entrypoint..."

# Ensure hosts file is correct
echo "127.0.0.1 localhost" > /etc/hosts

# Apply Redis overcommit_memory setting - skip if in container with read-only /proc
echo "Applying Redis overcommit_memory setting..."
sysctl vm.overcommit_memory=1 2>/dev/null || echo "Note: vm.overcommit_memory setting skipped (expected in container environment)"

# Create a .profile file to set environment variables for GPU
cat > ~/.profile << EOF
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics
export TF_FORCE_UNIFIED_MEMORY=1
export TF_ENABLE_NUMA_AWARE_ALLOCATORS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=all
export TF_FORCE_GPU_ALLOW_GROWTH=true
EOF
source ~/.profile

# Verify Redis data directories exist and have proper permissions
echo "Verifying Redis directories and permissions..."
mkdir -p /data /var/lib/redis /app/data/redis
chmod -R 777 /data /var/lib/redis /app/data/redis

# Create required project directories
mkdir -p /app/project/frontend/templates /app/project/frontend/sessions
mkdir -p /var/log/supervisor /var/log/redis /var/log/prometheus

# Copy services.conf to correct location
if [ -f /app/project/services.conf ]; then
    echo "Using custom supervisord configuration..."
    cp /app/project/services.conf /etc/supervisor/conf.d/services.conf
fi

# Ensure the template directory exists but don't override the template file
mkdir -p /app/project/frontend/templates

# Print confirmation of template existence
if [ -f /app/project/frontend/templates/index.html ]; then
    echo "Using existing frontend template from host system"
    ls -la /app/project/frontend/templates/
else
    echo "WARNING: No index.html template found in /app/project/frontend/templates/"
    echo "The system may not function correctly without a template file"
fi

# Start Redis server explicitly with correct port
echo "Starting Redis server on port 6380..."
cp /app/project/redis/redis.conf /etc/redis/redis.conf
# Force correct port in case there's a mismatch
redis-server /etc/redis/redis.conf --port 6380 --requirepass trading_system_2025 &
REDIS_PID=$!

# Give Redis time to start
echo "Waiting for Redis to initialize..."
sleep 5

# Test Redis connection with explicit parameters
echo "Testing Redis connection..."
if redis-cli -h 127.0.0.1 -p 6380 -a trading_system_2025 ping | grep -q PONG; then
    echo "Redis is running correctly on port 6380."
else
    echo "Redis failed to start properly. Trying alternative method..."
    kill $REDIS_PID || true
    sleep 2
    
    # Start with minimal configuration
    echo "Starting Redis with minimal configuration..."
    redis-server --port 6380 --requirepass trading_system_2025 --daemonize no --bind 0.0.0.0 &
    REDIS_PID=$!
    sleep 5

    # Check again
    if redis-cli -h 127.0.0.1 -p 6380 -a trading_system_2025 ping | grep -q PONG; then
        echo "Redis is now running correctly on port 6380."
    else
        echo "ERROR: Redis is still not responding. This will cause system failure."
        echo "Continuing anyway to see if supervisord can handle it..."
    fi
fi

# Configure Flask to bind to all interfaces with original app
echo "Configuring Flask for external access with original app..."
export FLASK_RUN_HOST="0.0.0.0"
export FLASK_APP="/app/project/frontend/app.py"
export FLASK_ENV="development"
export FLASK_DEBUG=1
export FLASK_RUN_PORT=5000

# Verify access permissions for the original app
echo "Setting proper permissions on Flask app..."
chmod +x /app/project/frontend/app.py

# Set up directories for Prometheus
echo "Setting up Prometheus directories..."
mkdir -p /prometheus /etc/prometheus
cp /app/project/prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# Fix ownership and permissions for Prometheus
chmod -R 777 /prometheus /etc/prometheus

# Test Prometheus configuration
echo "Testing Prometheus configuration..."
/usr/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --test || echo "Warning: Prometheus configuration test failed, but continuing anyway"

# Ensure everything is ready before starting supervisord
echo "Fixing permissions for supervisord..."
chmod -R 777 /var/log/supervisor

echo "All initialization completed successfully."
echo "Starting services with supervisord..."
# Use supervisord without exec to allow shell to continue if it fails
/usr/bin/supervisord -n

# Check if supervisord started correctly
if [ $? -ne 0 ]; then
    echo "ERROR: Supervisord failed to start properly!"
    echo "Starting individual services manually as a fallback..."
    
    # Start Flask app manually as a fallback
    cd /app/project && python3 /app/project/frontend/app.py &
    
    # Keep container running
    tail -f /dev/null
else
    # Wait for a moment then check service status
    sleep 5
    supervisorctl status
fi