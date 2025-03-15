"""
Database operations for the System Controller.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psycopg2
import redis
from psycopg2.extras import Json, RealDictCursor

from src.system_controller.models import (
    ComponentStatus,
    EventSeverity,
    EventType,
    SystemComponent,
    SystemEvent,
)

logger = logging.getLogger(__name__)

# Database connection parameters
DB_HOST = os.getenv("TIMESCALEDB_HOST", "localhost")
DB_PORT = os.getenv("TIMESCALEDB_PORT", "5435")  # Updated port to 5435
DB_NAME = os.getenv("TIMESCALEDB_DATABASE", "ats_db")
DB_USER = os.getenv("TIMESCALEDB_USER", "ats_user")
DB_PASSWORD = os.getenv("TIMESCALEDB_PASSWORD", "ats_password")

# Redis connection parameters
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis_password")


def get_db_connection():
    """Get a connection to the database."""
    try:
        # Read environment variables directly to ensure latest values
        host = os.getenv("TIMESCALEDB_HOST", "localhost")
        port = os.getenv("TIMESCALEDB_PORT", "5435")  # Updated default port to 5435
        dbname = os.getenv("TIMESCALEDB_DATABASE", "ats_db")
        user = os.getenv("TIMESCALEDB_USER", "ats_user")
        password = os.getenv("TIMESCALEDB_PASSWORD", "ats_password")
        
        logger.debug(f"Connecting to database at {host}:{port}/{dbname}")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def get_redis_connection():
    """Get a connection to Redis."""
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
        return r
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        raise


def register_component(component: SystemComponent) -> Tuple[bool, str]:
    """Register a component in the system state table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if component already exists
        cursor.execute(
            "SELECT id FROM system_state WHERE component = %s",
            (component.name,)
        )
        result = cursor.fetchone()
        
        if result:
            # Update existing component
            cursor.execute(
                """
                UPDATE system_state 
                SET status = %s, last_heartbeat = %s, details = %s
                WHERE component = %s
                """,
                (
                    component.status.value,
                    datetime.now(),
                    Json(component.details) if component.details else None,
                    component.name,
                ),
            )
        else:
            # Insert new component
            cursor.execute(
                """
                INSERT INTO system_state (component, status, last_heartbeat, details)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    component.name,
                    component.status.value,
                    datetime.now(),
                    Json(component.details) if component.details else None,
                ),
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Also update Redis for real-time access
        redis_conn = get_redis_connection()
        component_data = {
            "name": component.name,
            "status": component.status.value,
            "last_heartbeat": datetime.now().isoformat(),
            "details": json.dumps(component.details) if component.details else None,
        }
        redis_conn.hset("system_state", component.name, json.dumps(component_data))
        
        return True, f"Component {component.name} registered successfully"
    except Exception as e:
        logger.error(f"Error registering component: {e}")
        return False, f"Error registering component: {str(e)}"


def update_component_status(
    component_name: str, status: ComponentStatus, details: Optional[Dict] = None
) -> Tuple[bool, str, Optional[SystemComponent]]:
    """Update the status of a component."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Update the component status
        cursor.execute(
            """
            UPDATE system_state 
            SET status = %s, last_heartbeat = %s, details = %s
            WHERE component = %s
            RETURNING id, component, status, last_heartbeat, details
            """,
            (
                status.value,
                datetime.now(),
                Json(details) if details else None,
                component_name,
            ),
        )
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        if not result:
            return False, f"Component {component_name} not found", None
        
        # Update Redis
        redis_conn = get_redis_connection()
        component_data = {
            "name": component_name,
            "status": status.value,
            "last_heartbeat": datetime.now().isoformat(),
            "details": json.dumps(details) if details else None,
        }
        redis_conn.hset("system_state", component_name, json.dumps(component_data))
        
        # Convert to SystemComponent model
        component = SystemComponent(
            name=result["component"],
            status=ComponentStatus(result["status"]),
            last_heartbeat=result["last_heartbeat"],
            details=result["details"],
        )
        
        return True, f"Component {component_name} status updated", component
    except Exception as e:
        logger.error(f"Error updating component status: {e}")
        return False, f"Error updating component status: {str(e)}", None


def get_component(component_name: str) -> Optional[SystemComponent]:
    """Get a component from the system state table."""
    try:
        # Try Redis first for faster access
        redis_conn = get_redis_connection()
        component_data = redis_conn.hget("system_state", component_name)
        
        if component_data:
            data = json.loads(component_data)
            return SystemComponent(
                name=data["name"],
                status=ComponentStatus(data["status"]),
                last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
                details=json.loads(data["details"]) if data["details"] else None,
            )
        
        # Fall back to database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            SELECT component, status, last_heartbeat, details
            FROM system_state
            WHERE component = %s
            """,
            (component_name,),
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            return None
        
        return SystemComponent(
            name=result["component"],
            status=ComponentStatus(result["status"]),
            last_heartbeat=result["last_heartbeat"],
            details=result["details"],
        )
    except Exception as e:
        logger.error(f"Error getting component: {e}")
        return None


def get_all_components() -> List[SystemComponent]:
    """Get all components from the system state table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            SELECT component, status, last_heartbeat, details
            FROM system_state
            """
        )
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        components = []
        for result in results:
            component = SystemComponent(
                name=result["component"],
                status=ComponentStatus(result["status"]),
                last_heartbeat=result["last_heartbeat"],
                details=result["details"],
            )
            components.append(component)
        
        return components
    except Exception as e:
        logger.error(f"Error getting all components: {e}")
        return []


def log_event(event: SystemEvent) -> Tuple[bool, str, Optional[SystemEvent]]:
    """Log an event to the system events table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            INSERT INTO system_events (component, event_type, severity, message, details)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, timestamp, component, event_type, severity, message, details
            """,
            (
                event.component,
                event.event_type.value,
                event.severity.value,
                event.message,
                Json(event.details) if event.details else None,
            ),
        )
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        # Also publish to Redis for real-time notifications
        redis_conn = get_redis_connection()
        event_data = {
            "id": result["id"],
            "timestamp": result["timestamp"].isoformat(),
            "component": result["component"],
            "event_type": result["event_type"],
            "severity": result["severity"],
            "message": result["message"],
            "details": json.dumps(result["details"]) if result["details"] else None,
        }
        redis_conn.publish("system_events", json.dumps(event_data))
        
        # Store recent events in Redis for quick access
        redis_conn.lpush("recent_events", json.dumps(event_data))
        redis_conn.ltrim("recent_events", 0, 99)  # Keep only the 100 most recent events
        
        # Create SystemEvent object from result
        logged_event = SystemEvent(
            id=result["id"],
            timestamp=result["timestamp"],
            component=result["component"],
            event_type=EventType(result["event_type"]),
            severity=EventSeverity(result["severity"]),
            message=result["message"],
            details=result["details"],
        )
        
        return True, "Event logged successfully", logged_event
    except Exception as e:
        logger.error(f"Error logging event: {e}")
        return False, f"Error logging event: {str(e)}", None


def get_recent_events(limit: int = 100) -> List[SystemEvent]:
    """Get recent events from the system events table."""
    try:
        # Try Redis first for faster access
        redis_conn = get_redis_connection()
        events_data = redis_conn.lrange("recent_events", 0, limit - 1)
        
        if events_data and len(events_data) >= limit:
            events = []
            for event_data in events_data:
                data = json.loads(event_data)
                event = SystemEvent(
                    id=data["id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    component=data["component"],
                    event_type=EventType(data["event_type"]),
                    severity=EventSeverity(data["severity"]),
                    message=data["message"],
                    details=json.loads(data["details"]) if data["details"] else None,
                )
                events.append(event)
            return events
        
        # Fall back to database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            SELECT id, timestamp, component, event_type, severity, message, details
            FROM system_events
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit,),
        )
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        events = []
        for result in results:
            event = SystemEvent(
                id=result["id"],
                timestamp=result["timestamp"],
                component=result["component"],
                event_type=EventType(result["event_type"]),
                severity=EventSeverity(result["severity"]),
                message=result["message"],
                details=result["details"],
            )
            events.append(event)
        
        return events
    except Exception as e:
        logger.error(f"Error getting recent events: {e}")
        return []


def get_component_events(component_name: str, limit: int = 100) -> List[SystemEvent]:
    """Get events for a specific component."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            SELECT id, timestamp, component, event_type, severity, message, details
            FROM system_events
            WHERE component = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (component_name, limit),
        )
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        events = []
        for result in results:
            event = SystemEvent(
                id=result["id"],
                timestamp=result["timestamp"],
                component=result["component"],
                event_type=EventType(result["event_type"]),
                severity=EventSeverity(result["severity"]),
                message=result["message"],
                details=result["details"],
            )
            events.append(event)
        
        return events
    except Exception as e:
        logger.error(f"Error getting component events: {e}")
        return []