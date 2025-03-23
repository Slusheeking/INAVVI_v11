#!/bin/bash
# Trading System Stopper Script
# This script stops the entire trading system with all components

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

# Check if --force flag is passed
FORCE_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        FORCE_FLAG="--force"
        break
    fi
done

# Stop the trading system
echo "Stopping trading system..."
python3 scripts/stop_system.py $FORCE_FLAG "$@"

# Stop the frontend server
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        rm .frontend.pid
    else
        echo "Frontend server not running (PID: $FRONTEND_PID)"
        rm .frontend.pid
    fi
else
    echo "Stopping all Flask processes..."
    pkill -f "flask run" || true
fi

# Stop the event listener
if [ -f .event_listener.pid ]; then
    EVENT_LISTENER_PID=$(cat .event_listener.pid)
    if ps -p $EVENT_LISTENER_PID > /dev/null; then
        echo "Stopping event listener (PID: $EVENT_LISTENER_PID)..."
        kill $EVENT_LISTENER_PID
        rm .event_listener.pid
    else
        echo "Event listener not running (PID: $EVENT_LISTENER_PID)"
        rm .event_listener.pid
    fi
else
    echo "Stopping all event listener processes..."
    pkill -f "event_listener" || true
fi

# Check if Redis should be stopped
STOP_REDIS=true
for arg in "$@"; do
    if [ "$arg" == "--no-redis" ]; then
        STOP_REDIS=false
        break
    fi
done

# Stop Redis if needed
if [ "$STOP_REDIS" = true ]; then
    echo "Stopping Redis..."
    if [ -f .redis.pid ]; then
        REDIS_PID=$(cat .redis.pid)
        if ps -p $REDIS_PID > /dev/null; then
            kill $REDIS_PID
            rm .redis.pid
        else
            redis-cli shutdown || true
            rm .redis.pid
        fi
    else
        redis-cli shutdown || true
    fi
fi

echo "Trading system shutdown complete"
exit 0