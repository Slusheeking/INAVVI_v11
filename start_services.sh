#!/bin/bash
# start_services.sh - Script to start Redis, Prometheus, Redis exporter, and Frontend services
# This script should be run inside the trading-system container

echo "=== Starting Trading System Services ==="

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null
    return $?
}

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

# Start Redis server if not already running
if ! is_running "redis-server"; then
    echo "Starting Redis server..."
    redis-server /etc/redis/redis.conf --port 6380 --requirepass trading_system_2025 --daemonize yes
    sleep 2
    if is_running "redis-server"; then
        echo "✅ Redis server started successfully"
    else
        echo "❌ Failed to start Redis server"
    fi
else
    echo "✅ Redis server is already running"
fi

# Start Prometheus if not already running
if ! is_running "prometheus"; then
    echo "Starting Prometheus..."
    prometheus --config.file=/etc/prometheus/prometheus.yml \
               --storage.tsdb.path=/prometheus \
               --web.console.libraries=/usr/share/prometheus/console_libraries \
               --web.console.templates=/usr/share/prometheus/consoles \
               --web.enable-lifecycle &
    sleep 2
    if is_running "prometheus"; then
        echo "✅ Prometheus started successfully"
    else
        echo "❌ Failed to start Prometheus"
    fi
else
    echo "✅ Prometheus is already running"
fi

# Start Redis exporter if not already running
if ! is_running "redis_exporter"; then
    echo "Starting Redis exporter..."
    redis_exporter --redis.addr=redis://localhost:6380 --redis.password=trading_system_2025 &
    sleep 2
    if is_running "redis_exporter"; then
        echo "✅ Redis exporter started successfully"
    else
        echo "❌ Failed to start Redis exporter"
    fi
else
    echo "✅ Redis exporter is already running"
fi

# Start the frontend event listener if not already running
if ! is_running "event_listener.py"; then
    echo "Starting frontend event listener..."
    cd frontend
    python -m event_listener &
    EVENT_LISTENER_PID=$!
    cd ..
    sleep 2
    if is_running "event_listener.py"; then
        echo "✅ Frontend event listener started successfully (PID: $EVENT_LISTENER_PID)"
        echo "$EVENT_LISTENER_PID" > .event_listener.pid
    else
        echo "❌ Failed to start frontend event listener"
    fi
else
    echo "✅ Frontend event listener is already running"
fi

# Start the frontend server if not already running
if ! is_running "flask run"; then
    echo "Starting frontend server..."
    cd frontend
    python -m flask run --host=0.0.0.0 --port=5000 --with-threads &
    FRONTEND_PID=$!
    cd ..
    sleep 2
    if is_running "flask run"; then
        echo "✅ Frontend server started successfully (PID: $FRONTEND_PID)"
        echo "$FRONTEND_PID" > .frontend.pid
    else
        echo "❌ Failed to start frontend server"
    fi
else
    echo "✅ Frontend server is already running"
fi

echo "=== Service Status ==="
echo "Redis server: $(pgrep -f redis-server > /dev/null && echo "Running" || echo "Not running")"
echo "Prometheus: $(pgrep -f prometheus > /dev/null && echo "Running" || echo "Not running")"
echo "Redis exporter: $(pgrep -f redis_exporter > /dev/null && echo "Running" || echo "Not running")"
echo "Frontend event listener: $(pgrep -f event_listener.py > /dev/null && echo "Running" || echo "Not running")"
echo "Frontend server: $(pgrep -f "flask run" > /dev/null && echo "Running" || echo "Not running")"

echo "=== All services started ==="
echo "You can now run the test scripts to verify functionality:"
echo "- python3 /app/test_redis.py"
echo "- python3 /app/test_prometheus.py"
echo "- python3 /app/test_tensorrt.py"
echo "- Visit http://localhost:5000 for the frontend dashboard"