# INAVVI Trading System Frontend

This directory contains the frontend components for the INAVVI trading system, which provides a modern, dark-themed web-based dashboard for monitoring and controlling the system.

## Overview

The frontend is built using:
- Flask for the backend API
- Flask-SocketIO for real-time updates via WebSockets
- Redis Pub/Sub for real-time event notifications
- Eventlet for asynchronous event handling
- HTML/CSS/JavaScript for the frontend UI
- Chart.js for data visualization
- Font Awesome for icons

## Features

- **Dark-themed modern UI** optimized for trading environments
- Real-time system status monitoring with push notifications
- Component health tracking and automatic restart notifications
- **GPU resource utilization monitoring** with warning and critical alerts
- **Pattern detection visualization** for technical analysis signals
- Real-time notification display with priority levels
- Portfolio and position visualization with live updates
- Trading system control interface
- Market status monitoring
- System startup and shutdown event tracking

## Architecture

The frontend integrates with the trading system through Redis Pub/Sub:

```
┌─────────────────┐      ┌─────────────────────┐      ┌───────────────────┐
│                 │      │                     │      │                   │
│  Trading System ├──────► Redis Pub/Sub Events◄──────┤ Frontend Web App  │
│                 │      │                     │      │                   │
└─────────────────┘      └─────────────────────┘      └───────────────────┘
                                   │                          ▲
                                   │                          │
                                   ▼                          │
                         ┌─────────────────────┐             │
                         │                     │             │
                         │ Redis Event Listener├─────────────┘
                         │                     │
                         └─────────────────────┘
```

- The trading system publishes events to Redis "frontend:events" channel
- The Redis Event Listener processes events in real-time
- The frontend receives events via WebSockets
- The frontend can send commands to the trading system via Redis

## Setup

The frontend is automatically started as part of the Docker container. It runs on port 5000 and can be accessed at http://localhost:5000.

### Components

The frontend consists of two main components:

1. **Flask Web Application** - Serves the web interface and handles API requests
2. **Redis Event Listener** - Background process that listens for real-time events

## Development

To run the frontend separately for development:

```bash
cd frontend
./start_frontend.sh
```

This script will:
1. Set up the required environment variables
2. Create necessary directories
3. Install dependencies
4. Start the Redis event listener in the background
5. Start the Flask web server

## UI Features

The dark-themed dashboard includes:

- **System Overview Card**: Displays uptime, active positions, pending orders, and day trading candidates
- **GPU Status Card**: Real-time GPU memory monitoring with visual indicators for usage levels
- **Components Card**: Status of all system components with visual indicators
- **Portfolio Card**: Financial metrics including equity, cash balance, P&L, and exposure
- **Active Positions Card**: Current positions with real-time P&L tracking
- **Pattern Detection Card**: Technical analysis patterns detected by the system
- **Notifications Card**: Real-time system notifications with priority levels

## API Endpoints

The frontend provides the following API endpoints:

- `/api/status` - Get the current system status
- `/api/notifications` - Get recent notifications
- `/api/portfolio` - Get portfolio data
- `/api/positions` - Get position data
- `/api/patterns` - Get detected patterns
- `/api/reports` - Get available reports
- `/api/reports/<report_id>` - Get a specific report

## WebSocket Events

The frontend provides real-time updates via WebSocket:

- `notification` - New notification
- `system_status` - System status update
- `portfolio_update` - Portfolio update
- `position_update` - Position update
- `position_closed` - Position closed
- `market_status` - Market status update
- `equity_update` - Equity curve update
- `gpu_memory` - GPU memory usage update
- `component_status` - Component status update
- `system_event` - System events (startup, shutdown, warnings, etc.)
- `pattern_alert` - Technical analysis pattern detection
- `gpu_memory_warning` - GPU memory usage warning
- `gpu_memory_critical` - Critical GPU memory usage alert

## Event Subscription

Clients can subscribe to specific event types:

```javascript
// Subscribe to specific event types
socket.emit('subscribe', {type: 'system_status'});
socket.emit('subscribe', {type: 'notification'});
```

## Dark Theme

The dashboard features a modern dark theme optimized for trading environments:

- **Reduced Eye Strain**: Dark backgrounds with carefully chosen contrast levels reduce eye fatigue during extended trading sessions
- **Focus on Critical Data**: Color-coded indicators highlight important information (green for success, amber for warnings, red for errors)
- **GPU Status Visualization**: Real-time GPU memory usage with color-coded progress bars
- **Pattern Detection Cards**: Visual representation of detected trading patterns with confidence levels
- **Responsive Design**: Adapts to different screen sizes while maintaining readability
- **Consistent Visual Hierarchy**: Information is organized by importance with clear visual separation

### Theme Colors

The dark theme uses a carefully selected color palette:

- Background: Dark gray/black (#121212, #1e1e1e, #2d2d2d)
- Text: Light gray/white (#e0e0e0, #a0a0a0)
- Accent: Blue (#3d5afe, #536dfe)
- Status indicators: Green (#4caf50), Amber (#ff9800), Red (#f44336), Blue (#2196f3)

## Configuration

The frontend is configured through environment variables:

- `FLASK_APP` - Set to `frontend/app.py`
- `FLASK_ENV` - Set to `development` or `production`
- `FLASK_DEBUG` - Enable debug mode (1 for enabled)
- `FRONTEND_WEBSOCKET_ENABLED` - Enable WebSocket (true/false)
- `FRONTEND_REALTIME_UPDATES` - Enable real-time updates (true/false)
- `FRONTEND_REFRESH_INTERVAL` - UI refresh interval in milliseconds
- `FRONTEND_MAX_EVENTS` - Maximum number of events to store
- `REDIS_HOST` - Redis host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6380)
- `REDIS_PASSWORD` - Redis password
- `REDIS_PUBSUB_THREADS` - Number of threads for Redis pub/sub
- `REDIS_NOTIFY_KEYSPACE_EVENTS` - Redis keyspace notification settings