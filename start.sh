#!/bin/bash
# Trading System Starter Script
# This script starts the entire trading system with all components

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

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

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

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