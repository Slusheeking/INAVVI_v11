#!/usr/bin/env python3
"""
Real-time Updates Service for Trading System
This service provides Socket.IO real-time updates for the frontend, running as a separate service.
"""

import json
import os
import time
import threading
import random
import socket
import subprocess
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Initialize Flask app for Socket.IO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY', 'trading_system_realtime_secret')

# Enable CORS for all routes and origins
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
    print("Redis connection established for real-time updates service")
except (ImportError, Exception) as e:
    redis_available = False
    redis_client = None
    print(f"Redis not available for real-time updates service: {e!s}")

# Socket.IO event handlers


@socketio.on('connect')
def handle_connect():
    print('Client connected to real-time updates service')
    # Send initial data upon connection
    emit('system_status', system_status_data())
    emit('portfolio_update', get_portfolio_data())

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
    print('Client disconnected from real-time updates service')


@socketio.on('subscribe')
def handle_subscribe(data):
    print(f'Client subscribed to: {data}')


@socketio.on('request_update')
def handle_request_update():
    """Handle manual refresh requests"""
    emit('system_status', system_status_data())
    emit('portfolio_update', get_portfolio_data())


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
        # Fallback GPU data if real data can't be fetched
        gpu_memory = {
            "used_mb": 2048,
            "total_mb": 48000,
            "free_mb": 46000,
            "usage_pct": 4.2,
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
        uptime_seconds = 3600  # Fallback value

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

    # If still no notifications, provide some defaults based on actual system state
    if not notifications:
        # Add Redis status notification
        if redis_available:
            notifications.append({
                "level": "success",
                "message": "Redis connection established",
                "timestamp": current_time - 60
            })
        else:
            notifications.append({
                "level": "error",
                "message": "Redis connection failed",
                "timestamp": current_time - 60
            })

        # Add GPU notification
        try:
            import subprocess
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode('utf-8').strip()
            gpu_util = float(gpu_info)

            if gpu_util > 80:
                notifications.append({
                    "level": "warning",
                    "message": f"High GPU utilization: {gpu_util}%",
                    "timestamp": current_time - 30
                })
            else:
                notifications.append({
                    "level": "info",
                    "message": f"GPU utilization: {gpu_util}%",
                    "timestamp": current_time - 30
                })
        except:
            notifications.append({
                "level": "info",
                "message": "System started",
                "timestamp": current_time - 3600
            })

    return notifications


def get_portfolio_data():
    """Get portfolio data for both API and Socket.IO"""
    portfolio_data = {
        "total_equity": 250000.00,  # Default value
        "cash": 150000.00,          # Default value
        "total_pnl": 15000.00,      # Default value
        "current_exposure": 100000.00,  # Default value
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
                    for key in ['total_equity', 'cash', 'total_pnl', 'current_exposure']:
                        if key in redis_portfolio:
                            portfolio_data[key] = float(redis_portfolio[key])
                except:
                    pass

            # If individual keys exist, try those too
            for key in ['total_equity', 'cash', 'total_pnl', 'current_exposure']:
                try:
                    value = redis_client.get(f'portfolio:{key}')
                    if value:
                        portfolio_data[key] = float(value)
                except:
                    continue
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

    # If no positions found, return default example data
    if not positions:
        positions = [
            {
                "ticker": "AAPL",
                "quantity": 100,
                "entry_price": 175.50,
                "current_price": 180.25,
                "unrealized_pnl": 475.00,
                "unrealized_pnl_pct": 2.71
            },
            {
                "ticker": "MSFT",
                "quantity": 50,
                "entry_price": 350.00,
                "current_price": 345.75,
                "unrealized_pnl": -212.50,
                "unrealized_pnl_pct": -1.21
            }
        ]

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

    # If no patterns found, return default example data
    if not patterns:
        patterns = [
            {
                "ticker": "TSLA",
                "pattern": "bull_flag",
                "confidence": 0.85,
                "timestamp": int(time.time()) - 1200
            },
            {
                "ticker": "AMZN",
                "pattern": "double_bottom",
                "confidence": 0.92,
                "timestamp": int(time.time()) - 600
            }
        ]

    return patterns


# Simple health check endpoint
@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == '__main__':
    # Start background update thread
    if not update_thread or not update_thread.is_alive():
        update_thread = threading.Thread(target=background_update_task)
        update_thread.daemon = True
        update_thread.start()
        print("Started background update thread")

    # Start the Socket.IO server on port 5001 (different from the main frontend)
    socketio.run(app, host='0.0.0.0', port=5001,
                 debug=True, allow_unsafe_werkzeug=True)
