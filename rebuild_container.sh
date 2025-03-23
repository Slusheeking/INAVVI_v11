#!/bin/bash
set -e

echo "=== INAVVI Trading System Container Rebuild ==="
echo "This script will rebuild and restart the trading system container with optimized TensorFlow and CUDA settings."

# Stop any running containers
echo "Stopping existing containers..."
docker stop -t 5 trading-system || true
docker-compose -f docker-compose.unified.yml down -t 5

# Clean up any dangling images
echo "Cleaning up dangling images..."
docker image prune -f

# Rebuild the container
echo "Rebuilding the container with optimized settings..."
if ! docker-compose -f docker-compose.unified.yml build --no-cache; then
    echo "Build failed. Attempting fallback build with minimal dependencies..."
    # Create a backup of the Dockerfile
    cp Dockerfile.unified Dockerfile.unified.bak
    
    # Create a simplified version of the Dockerfile
    cat > Dockerfile.unified << EOF
# Pull the latest TensorFlow container with CUDA 12.4 support for NVIDIA GH200 Grace Hopper Superchips
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

# Install required system packages including Redis and Prometheus
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    wget \\
    python3-dev \\
    redis-server \\
    prometheus \\
    prometheus-node-exporter \\
    supervisor \\
    curl \\
    libpq-dev \\
    postgresql-client \\
    git \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Install Redis Exporter
RUN wget https://github.com/oliver006/redis_exporter/releases/download/v1.54.0/redis_exporter-v1.54.0.linux-arm64.tar.gz && \\
    tar xzf redis_exporter-v1.54.0.linux-arm64.tar.gz && \\
    mv redis_exporter-v1.54.0.linux-arm64/redis_exporter /usr/local/bin/ && \\
    rm -rf redis_exporter-v1.54.0.linux-arm64*

# Install core Python packages
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir \\
    pandas \\
    numpy \\
    requests \\
    redis \\
    urllib3 \\
    aiohttp \\
    websockets \\
    prometheus_client

# Create working directories
WORKDIR /app
RUN mkdir -p /app/models /app/data /app/logs /app/config

# Copy configuration files
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml
COPY redis/redis.conf /etc/redis/redis.conf

# Configure supervisord to manage services
COPY <<EOT /etc/supervisor/conf.d/services.conf
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
logfile_maxbytes=50MB
logfile_backups=5
loglevel=info

[program:redis]
command=/usr/bin/redis-server /etc/redis/redis.conf
autostart=true
autorestart=true
stderr_logfile=/var/log/redis/redis-server.err.log
stdout_logfile=/var/log/redis/redis-server.out.log
priority=10

[program:prometheus]
command=/usr/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus
autostart=true
autorestart=true
stderr_logfile=/var/log/prometheus.err.log
stdout_logfile=/var/log/prometheus.out.log
priority=20

[program:redis_exporter]
command=/usr/local/bin/redis_exporter
autostart=true
autorestart=true
stderr_logfile=/var/log/redis_exporter.err.log
stdout_logfile=/var/log/redis_exporter.out.log
priority=30
EOT

# Create directories and set permissions
RUN mkdir -p /var/log/supervisor && \\
    mkdir -p /var/log/redis && \\
    mkdir -p /var/log/prometheus && \\
    mkdir -p /var/run/redis && \\
    mkdir -p /data && \\
    touch /var/log/redis/redis-server.err.log && \\
    touch /var/log/redis/redis-server.out.log && \\
    touch /var/log/prometheus.err.log && \\
    touch /var/log/prometheus.out.log && \\
    touch /var/log/redis_exporter.err.log && \\
    touch /var/log/redis_exporter.out.log && \\
    chown -R root:root /var/log/redis && \\
    chown -R root:root /var/run/redis && \\
    chown -R root:root /data && \\
    chmod 755 /var/run/redis && \\
    chmod 755 /data

# Create startup script
COPY <<EOT /app/startup.sh
#!/bin/bash
set -e

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until redis-cli ping; do
  sleep 1
done
echo "Redis is ready!"

echo "Starting services with supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
EOT

RUN chmod +x /app/startup.sh

# Expose ports
EXPOSE 8888 6380 9090 9121

# Start all services using the startup script
CMD ["/app/startup.sh"]
EOF
    
    # Try building again with the simplified Dockerfile
    if ! docker-compose -f docker-compose.unified.yml build --no-cache; then
        echo "Fallback build also failed. Please check the logs for details."
        # Restore original Dockerfile
        mv Dockerfile.unified.bak Dockerfile.unified
        exit 1
    fi
    
    # Restore original Dockerfile
    mv Dockerfile.unified.bak Dockerfile.unified
    echo "Fallback build successful (with minimal dependencies)."
else
    echo "Build successful with all components."
fi

# Start the container
echo "Starting the container..."
docker-compose -f docker-compose.unified.yml up -d

# Wait for container to be ready
echo "Waiting for container to be ready..."
sleep 10

# Check if container is running
if docker ps | grep -q trading-system; then
    echo "Container is running successfully!"
    
    # Create directory for CuPy cache if it doesn't exist
    docker exec -it trading-system bash -c "mkdir -p /app/data/cache/cupy && chmod 777 /app/data/cache/cupy"
    
    # Run GPU status check
    echo "Running GPU status check..."
    docker exec -it trading-system bash -c "cd /app/project && python3 gpu_status.py"
    
    # Verify TensorFlow
    echo "Verifying TensorFlow..."
    if docker exec -it trading-system python3 /app/verify_tensorflow.py; then
        echo "TensorFlow verification successful!"
        
        # Verify TensorRT
        echo "Verifying TensorRT..."
        if docker exec -it trading-system python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"; then
            echo "TensorRT verification successful!"
        else
            echo "WARNING: TensorRT verification failed. The system will still function but without TensorRT acceleration."
        fi
    else
        echo "WARNING: TensorFlow verification failed. Check logs for details."
    fi
    
    echo "=== Rebuild Complete ==="
    echo "The trading system container has been rebuilt with optimized settings."
    echo "You can access Jupyter Lab at http://localhost:8888"
else
    echo "Error: Container failed to start properly."
    echo "Check logs with: docker-compose -f docker-compose.unified.yml logs"
fi