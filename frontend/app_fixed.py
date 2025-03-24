#!/usr/bin/env python3
"""
Frontend Application for Trading System with Unrestricted Access
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from flask import Flask, jsonify, render_template, request, session, Response

# Initialize Flask app with completely permissive settings
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

# Set extremely permissive access


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
    """Get the current system status"""
    return jsonify({
        "status": "operational",
        "message": "System is running",
        "timestamp": datetime.now().isoformat(),
        "redis_available": redis_available
    })


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# Add a completely open test route


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

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
