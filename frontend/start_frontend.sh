#!/bin/bash
# Start the frontend server for the trading system

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
export FLASK_RUN_PORT=5000
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

# Setup and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages from requirements.txt
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo "Verifying installations..."
python -c "import werkzeug; print(f'Werkzeug version: {werkzeug.__version__}')" || echo "Werkzeug not installed correctly"
python -c "import flask; print(f'Flask version: {flask.__version__}')" || echo "Flask not installed correctly"
python -c "import flask_socketio; print(f'Flask-SocketIO version: {flask_socketio.__version__}')" || echo "Flask-SocketIO not installed correctly"
python -c "import redis; print(f'Redis version: {redis.__version__}')" || echo "Redis not installed correctly"

# Fix Redis configuration issue
echo "Configuring Redis..."
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD CONFIG SET stop-writes-on-bgsave-error no || echo "Could not configure Redis - continuing anyway"

# Start the event listener in the background
echo "Starting Redis event listener..."
python event_listener.py &
EVENT_LISTENER_PID=$!

# Start the frontend server
echo "Starting frontend server..."
python app.py

# Cleanup on exit
function cleanup {
    echo "Stopping Redis event listener..."
    kill $EVENT_LISTENER_PID
    echo "Frontend server stopped"
    deactivate
}

trap cleanup EXIT