#!/usr/bin/env python3
"""
Frontend Application for Trading System

This module provides a web-based frontend for the trading system:
1. Dashboard for system metrics and performance
2. Real-time notifications and alerts
3. Portfolio and position visualization
4. Trading system control interface
5. ML model performance monitoring

The frontend integrates with the monitoring system via Redis
to display real-time data and notifications.
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from flask import Flask, jsonify, render_template, request, session
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

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode='eventlet', message_queue='redis://')

# Configure Redis connection
try:
    import redis
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6380)),
        db=int(os.environ.get("REDIS_DB", 0)),
        password=os.environ.get("REDIS_PASSWORD", ""),
        username=os.environ.get("REDIS_USERNAME", "default"),
        socket_timeout=int(os.environ.get("REDIS_TIMEOUT", 5)),
        decode_responses=True,
    )

    # Create a separate connection for pub/sub to avoid blocking the main connection
    redis_pubsub = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6380)),
        db=int(os.environ.get("REDIS_DB", 0)),
        password=os.environ.get("REDIS_PASSWORD", ""),
        username=os.environ.get("REDIS_USERNAME", "default"),
        socket_timeout=int(os.environ.get("REDIS_TIMEOUT", 5)),
        decode_responses=True,
    )

    # Initialize the pubsub object
    pubsub = redis_pubsub.pubsub()

    redis_available = True
    app.logger.info("Redis connection established for frontend")
except (ImportError, Exception) as e:
    redis_available = False
    redis_pubsub = None
    pubsub = None
    app.logger.warning(f"Redis not available for frontend: {e!s}")


@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')


@app.route('/api/status')
def system_status():
    """Get the current system status"""
    if not redis_available:
        return jsonify({
            "status": "limited",
            "message": "Redis not available, limited functionality",
            "timestamp": datetime.now().isoformat()
        })

    try:
        # Get system metrics from Redis
        system_metrics = redis_client.hgetall("system:metrics")

        # Format the response
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": float(system_metrics.get("cpu_percent", 0)),
                "memory_percent": float(system_metrics.get("memory_percent", 0)),
                "disk_percent": float(system_metrics.get("disk_percent", 0)),
                "uptime": float(system_metrics.get("uptime", 0)),
            }
        }

        # Add GPU metrics if available
        gpu_keys = redis_client.keys("gpu:*:metrics")
        if gpu_keys:
            status["gpu"] = []
            for key in gpu_keys:
                gpu_id = key.split(":")[1]
                gpu_metrics = redis_client.hgetall(key)
                if gpu_metrics:
                    status["gpu"].append({
                        "id": gpu_id,
                        "name": gpu_metrics.get("name", "Unknown"),
                        "memory_percent": float(gpu_metrics.get("memory_percent", 0)),
                        "gpu_utilization": float(gpu_metrics.get("gpu_utilization", 0)),
                        "temperature": float(gpu_metrics.get("temperature", 0)),
                    })

        return jsonify(status)
    except Exception as e:
        app.logger.error(f"Error getting system status: {e!s}")
        return jsonify({
            "status": "error",
            "message": f"Error getting system status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/notifications')
def get_notifications():
    """Get recent notifications"""
    if not redis_available:
        return jsonify([])

    try:
        # Get notifications from Redis
        channel = request.args.get('channel', 'general')
        limit = int(request.args.get('limit', 50))

        # Get notifications for the specified channel
        key = f"frontend:notifications:{channel}"
        notifications_json = redis_client.lrange(key, 0, limit - 1)

        # Parse JSON strings to objects
        notifications = [json.loads(n) for n in notifications_json]

        return jsonify(notifications)
    except Exception as e:
        app.logger.error(f"Error getting notifications: {e!s}")
        return jsonify([])


@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio data"""
    if not redis_available:
        return jsonify({})

    try:
        # Get latest portfolio data
        portfolio_json = redis_client.get("frontend:portfolio:latest")
        if not portfolio_json:
            return jsonify({})

        portfolio = json.loads(portfolio_json)

        # Get equity curve data if requested
        if request.args.get('include_history', 'false').lower() == 'true':
            equity_curve_json = redis_client.lrange(
                "frontend:portfolio:equity_curve", 0, 499)
            if equity_curve_json:
                portfolio["equity_curve"] = [json.loads(
                    point) for point in equity_curve_json]

        return jsonify(portfolio)
    except Exception as e:
        app.logger.error(f"Error getting portfolio data: {e!s}")
        return jsonify({})


@app.route('/api/positions')
def get_positions():
    """Get position data"""
    if not redis_available:
        return jsonify([])

    try:
        # Get latest position data
        positions_json = redis_client.get("frontend:positions:latest")
        if not positions_json:
            return jsonify([])

        positions = json.loads(positions_json)

        # Get position history if requested
        if request.args.get('include_history', 'false').lower() == 'true':
            history_json = redis_client.lrange(
                "frontend:positions:history", 0, 99)
            if history_json:
                positions["history"] = [json.loads(
                    point) for point in history_json]

        return jsonify(positions)
    except Exception as e:
        app.logger.error(f"Error getting position data: {e!s}")
        return jsonify([])


@app.route('/api/reports')
def get_reports():
    """Get available reports"""
    if not redis_available:
        return jsonify([])

    try:
        # Get report index
        report_index_json = redis_client.lrange(
            "frontend:reports:index", 0, 99)
        if not report_index_json:
            return jsonify([])

        # Parse JSON strings to objects
        reports = [json.loads(r) for r in report_index_json]

        return jsonify(reports)
    except Exception as e:
        app.logger.error(f"Error getting reports: {e!s}")
        return jsonify([])


@app.route('/api/reports/<report_id>')
def get_report(report_id):
    """Get a specific report by ID"""
    if not redis_available:
        return jsonify({})

    try:
        # Get report data
        report_json = redis_client.get(f"frontend:reports:{report_id}")
        if not report_json:
            return jsonify({"error": "Report not found"}), 404

        report = json.loads(report_json)

        return jsonify(report)
    except Exception as e:
        app.logger.error(f"Error getting report {report_id}: {e!s}")
        return jsonify({"error": f"Error getting report: {str(e)}"}), 500


# Redis event listener for real-time updates
class RedisEventListener:
    """Redis event listener for real-time updates"""

    def __init__(self, redis_client, socketio_instance):
        """Initialize the Redis event listener"""
        self.redis = redis_client
        self.pubsub = self.redis.pubsub()
        self.socketio = socketio_instance
        self.thread = None
        self.running = False
        self.event_handlers = {
            "system_status_update": self.handle_system_status,
            "market_status": self.handle_market_status,
            "portfolio_update": self.handle_portfolio_update,
            "equity_update": self.handle_equity_update,
            "position_update": self.handle_position_update,
            "position_closed": self.handle_position_closed,
            "system_startup": self.handle_system_startup,
            "system_shutdown_initiated": self.handle_system_shutdown,
            "system_shutdown_complete": self.handle_system_shutdown,
            "component_restart": self.handle_component_restart,
            "gpu_memory_warning": self.handle_gpu_memory_warning,
            "gpu_memory_critical": self.handle_gpu_memory_critical,
            "gpu_memory_update": self.handle_gpu_memory_update,
            "component_status_update": self.handle_component_status
        }

    def start(self):
        """Start listening for Redis events"""
        if self.running:
            return

        self.running = True
        self.pubsub.subscribe("frontend:events")
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        app.logger.info("Redis event listener started")

    def stop(self):
        """Stop listening for Redis events"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pubsub.unsubscribe()
        app.logger.info("Redis event listener stopped")

    def listen(self):
        """Listen for Redis events"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    self.process_message(message)
            except Exception as e:
                app.logger.error(f"Error in Redis event listener: {e!s}")
                time.sleep(1.0)

    def process_message(self, message):
        """Process a Redis message"""
        try:
            data = json.loads(message["data"])
            event_type = data.get("type")
            event_data = data.get("data", {})

            # Call the appropriate handler based on event type
            handler = self.event_handlers.get(event_type)
            if handler:
                handler(event_data)
            else:
                # Default handler for unknown event types
                self.handle_default(event_type, event_data)

        except Exception as e:
            app.logger.error(f"Error processing Redis message: {e!s}")

    def handle_system_status(self, data):
        """Handle system status update event"""
        self.socketio.emit("system_status", data)
        app.logger.debug(f"Emitted system_status: {data}")

    def handle_market_status(self, data):
        """Handle market status update event"""
        self.socketio.emit("market_status", data)
        app.logger.debug(f"Emitted market_status: {data}")

    def handle_portfolio_update(self, data):
        """Handle portfolio update event"""
        self.socketio.emit("portfolio_update", data)
        app.logger.debug(f"Emitted portfolio_update")

    def handle_equity_update(self, data):
        """Handle equity update event"""
        self.socketio.emit("equity_update", data)
        app.logger.debug(f"Emitted equity_update")

    def handle_position_update(self, data):
        """Handle position update event"""
        self.socketio.emit("position_update", data)
        app.logger.debug(f"Emitted position_update: {data.get('symbol')}")

    def handle_position_closed(self, data):
        """Handle position closed event"""
        self.socketio.emit("position_closed", data)
        app.logger.debug(f"Emitted position_closed: {data.get('symbol')}")

    def handle_system_startup(self, data):
        """Handle system startup event"""
        self.socketio.emit("system_event", {
            "type": "startup",
            "message": "System starting up",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.info(f"System startup event: {data}")

    def handle_system_shutdown(self, data):
        """Handle system shutdown event"""
        self.socketio.emit("system_event", {
            "type": "shutdown",
            "message": "System shutting down",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.info(f"System shutdown event: {data}")

    def handle_component_restart(self, data):
        """Handle component restart event"""
        self.socketio.emit("system_event", {
            "type": "component_restart",
            "message": f"Component {data.get('component')} restarting",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.info(f"Component restart event: {data}")

    def handle_gpu_memory_warning(self, data):
        """Handle GPU memory warning event"""
        self.socketio.emit("system_event", {
            "type": "gpu_warning",
            "message": f"GPU memory usage high: {data.get('usage_pct', 0):.1f}%",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.warning(f"GPU memory warning: {data}")

    def handle_gpu_memory_critical(self, data):
        """Handle GPU memory critical event"""
        self.socketio.emit("system_event", {
            "type": "gpu_critical",
            "message": f"GPU memory usage critical: {data.get('usage_pct', 0):.1f}%",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.error(f"GPU memory critical: {data}")

    def handle_gpu_memory_update(self, data):
        """Handle GPU memory update event"""
        self.socketio.emit("gpu_memory", data)
        app.logger.debug(f"Emitted gpu_memory update")

    def handle_component_status(self, data):
        """Handle component status update event"""
        self.socketio.emit("component_status", data)
        app.logger.debug(f"Emitted component_status update")

    def handle_default(self, event_type, data):
        """Handle unknown event types"""
        self.socketio.emit("system_event", {
            "type": event_type,
            "message": f"Event: {event_type}",
            "timestamp": data.get("timestamp", time.time()),
            "details": data
        })
        app.logger.debug(f"Emitted unknown event type: {event_type}")


# Initialize Redis event listener if Redis is available
event_listener = None
if redis_available and redis_pubsub:
    event_listener = RedisEventListener(redis_pubsub, socketio)


# WebSocket for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    app.logger.info(f"Client connected: {request.sid}")

    # Start the Redis event listener if available
    if event_listener and not event_listener.running:
        event_listener.start()

    # Send initial system status to the client
    if redis_available:
        try:
            # Get system status
            system_status = redis_client.hgetall("frontend:status")
            if system_status:
                socketio.emit('system_status', system_status, room=request.sid)

            # Get latest notifications
            notifications = redis_client.lrange(
                "frontend:notifications:general", 0, 9)
            if notifications:
                for notification_json in notifications:
                    notification = json.loads(notification_json)
                    socketio.emit('notification', notification,
                                  room=request.sid)
        except Exception as e:
            app.logger.error(f"Error sending initial data to client: {e!s}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    app.logger.info(f"Client disconnected: {request.sid}")


@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription to specific event types"""
    event_type = data.get('type')
    if event_type:
        app.logger.info(f"Client {request.sid} subscribed to {event_type}")
        # Store subscription in session
        if 'subscriptions' not in session:
            session['subscriptions'] = []
        if event_type not in session['subscriptions']:
            session['subscriptions'].append(event_type)
            session.modified = True


@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Handle unsubscription from specific event types"""
    event_type = data.get('type')
    if event_type and 'subscriptions' in session and event_type in session['subscriptions']:
        session['subscriptions'].remove(event_type)
        session.modified = True
        app.logger.info(f"Client {request.sid} unsubscribed from {event_type}")


# Legacy background thread for systems without Redis pub/sub
def background_thread():
    """Background thread for pushing updates to clients (legacy mode)"""
    last_notification_time = 0

    while True:
        if redis_available and not event_listener:
            try:
                # Check for new notifications
                notifications = redis_client.lrange(
                    "frontend:notifications:general", 0, 0)
                if notifications:
                    notification = json.loads(notifications[0])
                    if notification["timestamp"] > last_notification_time:
                        last_notification_time = notification["timestamp"]
                        socketio.emit('notification', notification)

                # Send system status updates every 5 seconds
                system_metrics = redis_client.hgetall("system:metrics")
                if system_metrics:
                    socketio.emit('system_update', {
                        "cpu_percent": float(system_metrics.get("cpu_percent", 0)),
                        "memory_percent": float(system_metrics.get("memory_percent", 0)),
                        "timestamp": time.time()
                    })
            except Exception as e:
                app.logger.error(f"Error in background thread: {e!s}")

        # Sleep for 5 seconds
        socketio.sleep(5)


@socketio.on('start_updates')
def start_updates():
    """Start sending updates to the client (legacy mode)"""
    # Only use background thread if Redis event listener is not available
    if not event_listener:
        socketio.start_background_task(background_thread)
    else:
        # Make sure the event listener is running
        if not event_listener.running:
            event_listener.start()


# Create the Redis event listener module file if it doesn't exist
def create_event_listener_module():
    """Create the event listener module file if it doesn't exist"""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, 'event_listener.py')

    if not os.path.exists(module_path):
        app.logger.info(f"Creating event listener module at {module_path}")
        with open(module_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Redis Event Listener for Trading System Frontend

This module listens for events published to Redis channels and forwards them
to connected WebSocket clients. It runs as a separate process to avoid blocking
the main Flask application.
\"\"\"

import json
import os
import sys
import time
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/events.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("event_listener")

# Import Redis
try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False
    logger.error("Redis not available, cannot start event listener")
    sys.exit(1)

class EventListener:
    \"\"\"Redis event listener for real-time updates\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the Redis event listener\"\"\"
        # Configure Redis connection
        self.redis = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6380)),
            db=int(os.environ.get("REDIS_DB", 0)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            username=os.environ.get("REDIS_USERNAME", "default"),
            socket_timeout=int(os.environ.get("REDIS_TIMEOUT", 5)),
            decode_responses=True,
        )
        self.pubsub = self.redis.pubsub()
        self.running = False
        self.thread = None
        
        # Test Redis connection
        try:
            self.redis.ping()
            logger.info("Redis connection established for event listener")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            sys.exit(1)
    
    def start(self):
        \"\"\"Start listening for Redis events\"\"\"
        if self.running:
            return
            
        self.running = True
        self.pubsub.subscribe("frontend:events")
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        logger.info("Redis event listener started")
    
    def stop(self):
        \"\"\"Stop listening for Redis events\"\"\"
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pubsub.unsubscribe()
        logger.info("Redis event listener stopped")
    
    def listen(self):
        \"\"\"Listen for Redis events\"\"\"
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    self.process_message(message)
            except Exception as e:
                logger.error(f"Error in Redis event listener: {e}")
                time.sleep(1.0)
    
    def process_message(self, message):
        \"\"\"Process a Redis message\"\"\"
        try:
            data = json.loads(message["data"])
            event_type = data.get("type")
            event_data = data.get("data", {})
            
            # Store the event in Redis for retrieval by the frontend
            if event_type:
                # Store in the appropriate channel
                channel = event_data.get("category", "general")
                event_key = f"frontend:events:{channel}:{event_type}"
                
                # Store the event with a timestamp
                if "timestamp" not in event_data:
                    event_data["timestamp"] = time.time()
                
                # Store the full event data
                event_json = json.dumps({
                    "type": event_type,
                    "data": event_data,
                    "received_at": time.time()
                })
                
                # Store in the event history (keep last 100 events per type)
                self.redis.lpush(event_key, event_json)
                self.redis.ltrim(event_key, 0, 99)
                
                # Also store in the general event history
                self.redis.lpush("frontend:events:all", event_json)
                self.redis.ltrim("frontend:events:all", 0, 499)
                
                logger.debug(f"Processed event: {event_type}")
        except Exception as e:
            logger.error(f"Error processing Redis message: {e}")

def main():
    \"\"\"Main entry point\"\"\"
    listener = EventListener()
    listener.start()
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()
""")
        app.logger.info(f"Created event listener module at {module_path}")


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

    # Create the event listener module
    create_event_listener_module()

    # Create a basic index.html template if it doesn't exist
    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Trading System Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .notification {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .notification.info {
            background-color: #e3f2fd;
        }
        .notification.success {
            background-color: #e8f5e9;
        }
        .notification.warning {
            background-color: #fff8e1;
        }
        .notification.error {
            background-color: #ffebee;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div class="container">
        <h1>Trading System Dashboard</h1>
        <p>This is a placeholder for the future frontend dashboard. The backend is ready to serve data to this interface.</p>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">System Status</div>
                <div id="system-status">Loading...</div>
            </div>
            
            <div class="card">
                <div class="card-title">Recent Notifications</div>
                <div id="notifications">Loading...</div>
            </div>
            
            <div class="card">
                <div class="card-title">Portfolio Summary</div>
                <div id="portfolio">Loading...</div>
            </div>
            
            <div class="card">
                <div class="card-title">Active Positions</div>
                <div id="positions">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        // Connect to WebSocket server
        const socket = io();
        
        // Handle connection events
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('system-status').textContent = 'Connected to server';
            
            // Subscribe to events
            socket.emit('subscribe', {type: 'system_status'});
            socket.emit('subscribe', {type: 'notification'});
            socket.emit('subscribe', {type: 'portfolio_update'});
            socket.emit('subscribe', {type: 'position_update'});
            
            // Start updates
            socket.emit('start_updates');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('system-status').textContent = 'Disconnected from server';
        });
        
        // Handle system status updates
        socket.on('system_status', function(data) {
            console.log('System status update:', data);
            document.getElementById('system-status').textContent =
                `Status: ${data.status}, Last update: ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;
        });
        
        // Handle notifications
        socket.on('notification', function(data) {
            console.log('Notification:', data);
            const notificationsDiv = document.getElementById('notifications');
            const notification = document.createElement('div');
            notification.className = `notification ${data.level || 'info'}`;
            notification.textContent = data.message;
            
            // Add to the top
            if (notificationsDiv.firstChild) {
                notificationsDiv.insertBefore(notification, notificationsDiv.firstChild);
            } else {
                notificationsDiv.appendChild(notification);
            }
            
            // Limit to 5 notifications
            while (notificationsDiv.children.length > 5) {
                notificationsDiv.removeChild(notificationsDiv.lastChild);
            }
        });
        
        // Handle portfolio updates
        socket.on('portfolio_update', function(data) {
            console.log('Portfolio update:', data);
            document.getElementById('portfolio').textContent =
                `Equity: $${data.total_equity.toFixed(2)}, Cash: $${data.cash.toFixed(2)}, P&L: $${data.total_pnl.toFixed(2)}`;
        });
        
        // Handle position updates
        socket.on('position_update', function(data) {
            console.log('Position update:', data);
            document.getElementById('positions').textContent =
                `Updated position: ${data.symbol}, Quantity: ${data.quantity}, Entry: $${data.entry_price.toFixed(2)}`;
        });
        
        // Handle system events
        socket.on('system_event', function(data) {
            console.log('System event:', data);
            const notificationsDiv = document.getElementById('notifications');
            const notification = document.createElement('div');
            
            // Set class based on event type
            let className = 'info';
            if (data.type.includes('error') || data.type.includes('critical')) {
                className = 'error';
            } else if (data.type.includes('warning')) {
                className = 'warning';
            } else if (data.type.includes('success')) {
                className = 'success';
            }
            
            notification.className = `notification ${className}`;
            notification.textContent = data.message;
            
            // Add to the top
            if (notificationsDiv.firstChild) {
                notificationsDiv.insertBefore(notification, notificationsDiv.firstChild);
            } else {
                notificationsDiv.appendChild(notification);
            }
            
            // Limit to 5 notifications
            while (notificationsDiv.children.length > 5) {
                notificationsDiv.removeChild(notificationsDiv.lastChild);
            }
        });
    </script>
</body>
</html>
            """)

    # Start the Redis event listener if available
    if redis_available and event_listener:
        event_listener.start()

    # Start the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
