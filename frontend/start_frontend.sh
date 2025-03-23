#!/bin/bash
# Start the frontend server for the trading system

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
export FRONTEND_WEBSOCKET_ENABLED=true
export FRONTEND_REALTIME_UPDATES=true
export REDIS_PUBSUB_THREADS=4
export REDIS_NOTIFY_KEYSPACE_EVENTS=Kxe

# Set Redis connection variables (use environment variables if available)
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6380}
export REDIS_DB=${REDIS_DB:-0}
export REDIS_USERNAME=${REDIS_USERNAME:-default}
export REDIS_PASSWORD=${REDIS_PASSWORD:-trading_system_2025}
export REDIS_TIMEOUT=${REDIS_TIMEOUT:-5}

# Print connection info
echo "Connecting to Redis at $REDIS_HOST:$REDIS_PORT"

# Create required directories
mkdir -p ../logs/frontend
mkdir -p ../logs/events
mkdir -p sessions

# Install required packages using system pip
echo "Installing required packages..."
pip3 install --user flask flask-cors flask-socketio redis eventlet gevent flask-redis flask-session

# Verify installations
echo "Verifying installations..."
python3 -c "import flask; print(f'Flask version: {flask.__version__}')"
python3 -c "import flask_socketio; print(f'Flask-SocketIO version: {flask_socketio.__version__}')"
python3 -c "import redis; print(f'Redis version: {redis.__version__}')"

# Start the event listener in the background
echo "Starting Redis event listener..."
python3 -m event_listener &
EVENT_LISTENER_PID=$!

# Start the frontend server
echo "Starting frontend server..."
python3 -m flask run --host=0.0.0.0 --port=5000 --with-threads

# Cleanup on exit
function cleanup {
    echo "Stopping Redis event listener..."
    kill $EVENT_LISTENER_PID
    echo "Frontend server stopped"
}

trap cleanup EXIT