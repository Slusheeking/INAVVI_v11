#!/bin/bash
# Trading System Starter Script
# This script starts the entire trading system with all components

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Parse command line arguments
DOCKER_MODE=false
REBUILD=false
FORCE_FLAG=""

for arg in "$@"; do
    if [ "$arg" == "--docker" ]; then
        DOCKER_MODE=true
    fi
    if [ "$arg" == "--rebuild" ]; then
        REBUILD=true
    fi
    if [ "$arg" == "--force" ]; then
        FORCE_FLAG="--force"
    fi
done

# Set frontend environment variables
export FLASK_APP=frontend/app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
export FRONTEND_WEBSOCKET_ENABLED=true
export FRONTEND_REALTIME_UPDATES=true
export REDIS_PUBSUB_THREADS=4
export REDIS_NOTIFY_KEYSPACE_EVENTS=Kxe

# Ensure logs directory exists
mkdir -p logs
mkdir -p logs/frontend
mkdir -p logs/events
mkdir -p data
mkdir -p models

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

# Docker mode
if [ "$DOCKER_MODE" = true ]; then
    echo "Starting trading system in Docker mode..."
    
    # Check if container is already running
    if docker ps | grep -q trading-system; then
        if [ "$REBUILD" = true ]; then
            echo "Container is already running. Stopping it for rebuild..."
            ./stop.sh --docker $FORCE_FLAG
        else
            echo "Container is already running. Use --rebuild to force rebuild or use stop.sh first."
            exit 1
        fi
    fi
    
    # Rebuild if requested
    if [ "$REBUILD" = true ]; then
        echo "Rebuilding container..."
        ./rebuild_container.sh
    else
        # Just start the container if it's not running
        echo "Starting container without rebuild..."
        docker-compose -f docker-compose.unified.yml up -d
        
        # Wait for container to be ready
        echo "Waiting for container to be ready..."
        max_wait=120
        wait_interval=5
        elapsed=0
        
        while [ $elapsed -lt $max_wait ]; do
            # Check if container is running
            if ! docker ps | grep -q trading-system; then
                echo "Container failed to start. Checking logs..."
                docker-compose -f docker-compose.unified.yml logs --tail=50
                echo "Failed to start container. Exiting."
                exit 1
            fi
            
            # Check container health status
            health_status=$(docker inspect --format='{{.State.Health.Status}}' trading-system 2>/dev/null)
            
            if [ "$health_status" = "healthy" ]; then
                echo "Container is healthy and ready!"
                break
            fi
            
            echo "Waiting for container to be healthy... ($elapsed/$max_wait seconds)"
            sleep $wait_interval
            elapsed=$((elapsed + wait_interval))
        done
        
        if [ $elapsed -ge $max_wait ]; then
            echo "Timed out waiting for container to be healthy, but continuing anyway..."
            echo "Container status: $health_status"
        fi
    fi
    
    echo "Docker container is running. Access the frontend at http://localhost:5000"
    echo "Jupyter Lab is available at http://localhost:8888"
    echo "Prometheus is available at http://localhost:9090"
    
    # Exit since we're running in Docker mode
    exit 0
fi

# Local mode (non-Docker) continues below
echo "Starting trading system in local mode..."

# Check if Redis is running
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Redis is not running. Starting Redis..."
    if [ -f redis/redis.conf ]; then
        redis-server redis/redis.conf &
        REDIS_PID=$!
        echo "Redis started with PID $REDIS_PID"
    else
        redis-server &
        REDIS_PID=$!
        echo "Redis started with PID $REDIS_PID"
    fi
    sleep 2
fi

# Start the frontend event listener in the background
echo "Starting frontend event listener..."
cd frontend
python -m event_listener &
EVENT_LISTENER_PID=$!
echo "Frontend event listener started with PID $EVENT_LISTENER_PID"
cd ..

# Start the frontend server in the background
echo "Starting frontend server..."
cd frontend
python -m flask run --host=0.0.0.0 --port=5000 --with-threads &
FRONTEND_PID=$!
echo "Frontend server started with PID $FRONTEND_PID"
cd ..

# Start the trading system
echo "Starting trading system..."
python3 scripts/start_system.py "$@"
TRADING_SYSTEM_STATUS=$?

# Store PIDs for the stop script
echo "$REDIS_PID" > .redis.pid
echo "$EVENT_LISTENER_PID" > .event_listener.pid
echo "$FRONTEND_PID" > .frontend.pid

# Exit with the same status as the Python script
exit $TRADING_SYSTEM_STATUS