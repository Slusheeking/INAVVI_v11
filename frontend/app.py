#!/usr/bin/env python3
"""
Frontend Application for Trading System with full dashboard support
"""

import json
import os
import time
import threading
import socket
import subprocess
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from flask import Flask, jsonify, render_template, request, session, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY', 'trading_system_frontend_secret')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sessions')
app.config['SESSION_FILE_THRESHOLD'] = 500

# Enable CORS for all routes and origins with extensive permissive settings
CORS(app,
     resources={r"/*": {"origins": "*"}},
     supports_credentials=True,
     allow_headers="*",
     expose_headers="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_ALLOW_ALL_ORIGINS'] = True
app.config['CORS_SUPPORTS_CREDENTIALS'] = True

# Initialize Socket.IO with proper settings
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Add additional CORS response headers to every request


@app.after_request
def after_request(response):
    """Add headers to every response to ensure no CORS issues"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Configure Redis connection
try:
    import redis
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6380)),
        db=int(os.environ.get("REDIS_DB", 0)),
        password=os.environ.get("REDIS_PASSWORD", "trading_system_2025"),
        username=os.environ.get("REDIS_USERNAME", "default"),
        socket_timeout=int(os.environ.get("REDIS_TIMEOUT", 5)),
        decode_responses=True,
    )
    redis_available = True
    print("Redis connection established for frontend")
except (ImportError, Exception) as e:
    redis_available = False
    redis_client = None
    print(f"Redis not available for frontend: {e!s}")


@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')


@app.route('/api/status')
def system_status():
    """Get the current system status with live data"""
    return jsonify(system_status_data())


@app.route('/api/notifications')
def get_notifications():
    """Get real system notifications from Redis or logs"""
    return jsonify(get_notifications_data())


@app.route('/api/portfolio')
def get_portfolio():
    """Get real portfolio information from Redis"""
    return jsonify(get_portfolio_data())


@app.route('/api/positions')
def get_positions():
    """Get real active positions from Redis"""
    return jsonify(get_positions_data())


@app.route('/api/patterns')
def get_patterns():
    """Get real detected patterns from Redis"""
    return jsonify(get_patterns_data())


@app.route('/api/api_status')
def get_api_status():
    """Get status of external APIs"""
    return jsonify(get_api_status_data())


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/test')
def test_route():
    """Test route that should always be accessible"""
    return Response('Access test successful - you can access this server!',
                    mimetype='text/plain',
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': '*',
                        'Access-Control-Allow-Methods': '*',
                    })


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial data upon connection
    emit('system_status', system_status_data())
    emit('portfolio_update', get_portfolio_data())
    emit('api_status', get_api_status_data())

    # Send positions
    positions = get_positions_data()
    if positions:
        for position in positions:
            emit('position_update', position)

    # Send patterns
    patterns = get_patterns_data()
    if patterns:
        for pattern in patterns:
            emit('pattern_alert', pattern)

    # Send a notification
    notifications = get_notifications_data()
    if notifications:
        emit('notification', notifications[0])


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('subscribe')
def handle_subscribe(data):
    print(f'Client subscribed to: {data}')


@socketio.on('request_update')
def handle_request_update():
    """Handle manual refresh requests"""
    emit('system_status', system_status_data())
    emit('portfolio_update', get_portfolio_data())
    emit('api_status', get_api_status_data())


# Background thread for real-time updates
update_thread = None


def background_update_task():
    """Background thread function for sending real-time updates"""
    print("Starting real-time update thread")
    while True:
        # Send system status updates every 2 seconds
        with app.app_context():
            try:
                # System status update
                status_data = system_status_data()
                socketio.emit('system_status', status_data)

                # GPU memory update
                if 'gpu' in status_data and 'memory' in status_data['gpu']:
                    socketio.emit('gpu_memory', status_data['gpu']['memory'])

                # Portfolio update (every 5 seconds)
                if int(time.time()) % 5 == 0:
                    portfolio_data = get_portfolio_data()
                    socketio.emit('portfolio_update', portfolio_data)

                # API status update (every 30 seconds)
                if int(time.time()) % 30 == 0:
                    api_status_data = get_api_status_data()
                    socketio.emit('api_status', api_status_data)

                # Send periodic notification updates (every 10 seconds)
                if int(time.time()) % 10 == 0:
                    notifications = get_notifications_data()
                    if notifications:
                        socketio.emit('notification', notifications[0])

                # Position updates (every 3 seconds)
                if int(time.time()) % 3 == 0:
                    positions = get_positions_data()
                    if positions:
                        for position in positions:
                            socketio.emit('position_update', position)

                # Pattern updates (every 30 seconds)
                if int(time.time()) % 30 == 0:
                    patterns = get_patterns_data()
                    if patterns:
                        for pattern in patterns:
                            socketio.emit('pattern_alert', pattern)

            except Exception as e:
                print(f"Error in background update thread: {e}")

        # Sleep for 2 seconds
        time.sleep(2)


def system_status_data():
    """Get system status data for both API and Socket.IO"""
    # Get real GPU memory information
    try:
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"]).decode('utf-8').strip()
        memory_values = nvidia_output.split(',')
        used_mb = int(memory_values[0].strip())
        total_mb = int(memory_values[1].strip())
        free_mb = int(memory_values[2].strip())
        usage_pct = (used_mb / total_mb) * 100

        gpu_memory = {
            "used_mb": used_mb,
            "total_mb": total_mb,
            "free_mb": free_mb,
            "usage_pct": usage_pct,
            "timestamp": int(time.time())
        }
    except Exception as e:
        print(f"Error fetching GPU data: {e}")
        # Return empty GPU data if real data can't be fetched
        gpu_memory = {
            "used_mb": 0,
            "total_mb": 0,
            "free_mb": 0,
            "usage_pct": 0,
            "timestamp": int(time.time())
        }

    # Check real component statuses
    components = {}

    # Check Redis status
    if redis_available:
        try:
            redis_ping = redis_client.ping()
            components["Redis"] = "running" if redis_ping else "error"
        except:
            components["Redis"] = "error"
    else:
        components["Redis"] = "stopped"

    # Check TensorFlow status
    try:
        import tensorflow as tf
        components["TensorFlow"] = "running"
    except:
        components["TensorFlow"] = "not available"

    # Check for running processes
    process_checks = {
        "Trading Engine": "trading_engine.py",
        "Data Pipeline": "data_pipeline.py",
        "Learning Engine": "learning_engine.py",
        "Prometheus": "prometheus"
    }

    try:
        process_list = subprocess.check_output(["ps", "-ef"]).decode('utf-8')
        for component, process_name in process_checks.items():
            components[component] = "running" if process_name in process_list else "stopped"
    except:
        for component, process_name in process_checks.items():
            components[component] = "unknown"

    # Get system uptime
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
    except:
        uptime_seconds = 0

    # Get active positions count from Redis
    active_positions = 0
    pending_orders = 0
    day_trading_candidates = 0

    if redis_available:
        try:
            active_positions = redis_client.scard('active_positions') or 0
            pending_orders = redis_client.scard('pending_orders') or 0
            day_trading_candidates = redis_client.scard(
                'trading_candidates') or 0
        except Exception as e:
            print(f"Error fetching counts from Redis: {e}")

    return {
        "status": "operational",
        "running": True,
        "timestamp": int(time.time()),
        "system_info": {
            "uptime_seconds": uptime_seconds,
        },
        "message": "System is running",
        "redis_available": redis_available,
        "components": components,
        "gpu": {
            "memory": gpu_memory
        },
        "active_positions": active_positions,
        "pending_orders": pending_orders,
        "day_trading_candidates": day_trading_candidates
    }


def get_api_status_data():
    """Get status of external APIs"""
    api_statuses = {}

    # Check Polygon REST API
    try:
        polygon_status = "stopped"
        if redis_available:
            polygon_health = redis_client.get('api:polygon:health')
            if polygon_health:
                polygon_status = "running" if polygon_health == "ok" else "error"
            else:
                # Try to check directly
                try:
                    polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
                    if polygon_api_key:
                        response = requests.get(
                            f"https://api.polygon.io/v1/marketstatus/now?apiKey={polygon_api_key}",
                            timeout=5
                        )
                        polygon_status = "running" if response.status_code == 200 else "error"
                except:
                    pass
        api_statuses["Polygon API"] = polygon_status
    except Exception as e:
        print(f"Error checking Polygon API status: {e}")
        api_statuses["Polygon API"] = "unknown"

    # Check Polygon WebSocket
    try:
        polygon_ws_status = "stopped"
        if redis_available:
            polygon_ws_health = redis_client.get('api:polygon_ws:health')
            if polygon_ws_health:
                polygon_ws_status = "running" if polygon_ws_health == "ok" else "error"
            else:
                # Check if there are any recent messages
                recent_messages = redis_client.exists(
                    'polygon:websocket:last_message')
                if recent_messages:
                    last_message_time = redis_client.get(
                        'polygon:websocket:last_message')
                    if last_message_time:
                        try:
                            last_time = float(last_message_time)
                            # If message received in last 5 minutes, consider it running
                            if time.time() - last_time < 300:
                                polygon_ws_status = "running"
                        except:
                            pass
        api_statuses["Polygon WebSocket"] = polygon_ws_status
    except Exception as e:
        print(f"Error checking Polygon WebSocket status: {e}")
        api_statuses["Polygon WebSocket"] = "unknown"

    # Check Unusual Whales API
    try:
        unusual_whales_status = "stopped"
        if redis_available:
            unusual_whales_health = redis_client.get(
                'api:unusual_whales:health')
            if unusual_whales_health:
                unusual_whales_status = "running" if unusual_whales_health == "ok" else "error"
            else:
                # Check if there are any recent data
                recent_data = redis_client.exists(
                    'unusual_whales:alerts:last_update')
                if recent_data:
                    last_update = redis_client.get(
                        'unusual_whales:alerts:last_update')
                    if last_update:
                        try:
                            # If data updated in last hour, consider it running
                            last_time = datetime.fromisoformat(last_update)
                            if (datetime.now() - last_time).total_seconds() < 3600:
                                unusual_whales_status = "running"
                        except:
                            pass
        api_statuses["Unusual Whales API"] = unusual_whales_status
    except Exception as e:
        print(f"Error checking Unusual Whales API status: {e}")
        api_statuses["Unusual Whales API"] = "unknown"

    # Check Alpaca API
    try:
        alpaca_status = "stopped"
        if redis_available:
            alpaca_health = redis_client.get('api:alpaca:health')
            if alpaca_health:
                alpaca_status = "running" if alpaca_health == "ok" else "error"
            else:
                # Try to check directly
                try:
                    alpaca_api_key = os.environ.get("ALPACA_API_KEY", "")
                    alpaca_api_secret = os.environ.get("ALPACA_API_SECRET", "")
                    if alpaca_api_key and alpaca_api_secret:
                        headers = {
                            'APCA-API-KEY-ID': alpaca_api_key,
                            'APCA-API-SECRET-KEY': alpaca_api_secret
                        }
                        response = requests.get(
                            "https://api.alpaca.markets/v2/account",
                            headers=headers,
                            timeout=5
                        )
                        alpaca_status = "running" if response.status_code == 200 else "error"
                except:
                    pass
        api_statuses["Alpaca API"] = alpaca_status
    except Exception as e:
        print(f"Error checking Alpaca API status: {e}")
        api_statuses["Alpaca API"] = "unknown"

    return {
        "timestamp": int(time.time()),
        "api_statuses": api_statuses
    }


def get_notifications_data():
    """Get notifications data for both API and Socket.IO"""
    notifications = []
    current_time = int(time.time())

    # Try to get notifications from Redis
    if redis_available:
        try:
            # Fetch the latest notifications from Redis
            redis_notifications = redis_client.lrange(
                'system:notifications', 0, 9)  # Get latest 10

            for notif_json in redis_notifications:
                try:
                    notif = json.loads(notif_json)
                    notifications.append(notif)
                except:
                    continue
        except Exception as e:
            print(f"Error fetching notifications from Redis: {e}")

    # If no notifications from Redis, try to read from log files
    if not notifications:
        try:
            log_path = os.path.join('..', 'logs', 'system.log')
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    # Get last 20 lines
                    last_lines = log_file.readlines()[-20:]

                    for line in last_lines:
                        if '[INFO]' in line:
                            level = 'info'
                        elif '[WARNING]' in line or '[WARN]' in line:
                            level = 'warning'
                        elif '[ERROR]' in line:
                            level = 'error'
                        elif '[SUCCESS]' in line:
                            level = 'success'
                        else:
                            continue

                        # Extract timestamp and message
                        parts = line.split(' ', 2)
                        if len(parts) >= 3:
                            try:
                                timestamp = int(datetime.strptime(
                                    parts[0], '%Y-%m-%d-%H:%M:%S').timestamp())
                                message = parts[2].strip()
                                notifications.append({
                                    "level": level,
                                    "message": message,
                                    "timestamp": timestamp
                                })
                            except:
                                continue
        except Exception as e:
            print(f"Error reading from log file: {e}")

    # Return empty list if no notifications found
    return notifications


def get_portfolio_data():
    """Get portfolio data for both API and Socket.IO"""
    portfolio_data = {
        "total_equity": 0.0,
        "cash": 0.0,
        "total_pnl": 0.0,
        "current_exposure": 0.0,
        "buying_power": 0.0,
        "account_status": "UNKNOWN",
        "data_source": "Alpaca API",
        "timestamp": int(time.time())
    }

    if redis_available:
        try:
            # Try to get portfolio data from Redis
            portfolio_json = redis_client.get('portfolio:summary')
            if portfolio_json:
                try:
                    redis_portfolio = json.loads(portfolio_json)
                    # Update with real values
                    for key in ['total_equity', 'cash', 'total_pnl', 'current_exposure', 'buying_power']:
                        if key in redis_portfolio:
                            portfolio_data[key] = float(redis_portfolio[key])
                except:
                    pass

            # If individual keys exist, try those too
            for key in ['total_equity', 'cash', 'total_pnl', 'current_exposure', 'buying_power']:
                try:
                    value = redis_client.get(f'portfolio:{key}')
                    if value:
                        portfolio_data[key] = float(value)
                except:
                    continue

            # Check for account status
            account_status = redis_client.get('api:alpaca:health')
            if account_status:
                portfolio_data["account_status"] = "ACTIVE" if account_status == "ok" else "INACTIVE"

            # Check for last update time from Alpaca
            last_update = redis_client.get('api:alpaca:last_update')
            if last_update:
                portfolio_data["last_alpaca_update"] = last_update

        except Exception as e:
            print(f"Error fetching portfolio data from Redis: {e}")

    return portfolio_data


def get_positions_data():
    """Get positions data for both API and Socket.IO"""
    positions = []

    if redis_available:
        try:
            # Get position keys
            position_keys = redis_client.smembers('active_positions')

            for pos_key in position_keys:
                position_data = redis_client.hgetall(f'position:{pos_key}')
                if position_data:
                    try:
                        # Convert string values to appropriate types
                        position = {
                            "ticker": position_data.get('ticker', 'UNKNOWN'),
                            "quantity": float(position_data.get('quantity', 0)),
                            "entry_price": float(position_data.get('entry_price', 0)),
                            "current_price": float(position_data.get('current_price', 0)),
                            "unrealized_pnl": float(position_data.get('unrealized_pnl', 0)),
                            "unrealized_pnl_pct": float(position_data.get('unrealized_pnl_pct', 0))
                        }
                        positions.append(position)
                    except:
                        continue
        except Exception as e:
            print(f"Error fetching positions from Redis: {e}")

    return positions


def get_patterns_data():
    """Get patterns data for both API and Socket.IO"""
    patterns = []

    if redis_available:
        try:
            # Get pattern keys
            pattern_keys = redis_client.smembers('detected_patterns')

            for pattern_key in pattern_keys:
                pattern_data = redis_client.hgetall(f'pattern:{pattern_key}')
                if pattern_data:
                    try:
                        # Convert string values to appropriate types
                        pattern = {
                            "ticker": pattern_data.get('ticker', 'UNKNOWN'),
                            "pattern": pattern_data.get('pattern', 'unknown'),
                            "confidence": float(pattern_data.get('confidence', 0.5)),
                            "timestamp": int(pattern_data.get('timestamp', int(time.time()) - 600))
                        }
                        patterns.append(pattern)
                    except:
                        continue

            # Sort by timestamp, newest first
            patterns.sort(key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            print(f"Error fetching patterns from Redis: {e}")

    return patterns


if __name__ == '__main__':
    # Create required directories
    template_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'templates')
    os.makedirs(template_dir, exist_ok=True)

    session_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'sessions')
    os.makedirs(session_dir, exist_ok=True)

    logs_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    events_dir = os.path.join(logs_dir, 'events')
    os.makedirs(events_dir, exist_ok=True)

    # Start background update thread
    if not update_thread or not update_thread.is_alive():
        update_thread = threading.Thread(target=background_update_task)
        update_thread.daemon = True
        update_thread.start()
        print("Started background update thread")

    # Start the Flask app with Socket.IO
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=True, allow_unsafe_werkzeug=True)
