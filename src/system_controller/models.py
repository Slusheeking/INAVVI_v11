"""
Data models for the System Controller.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ComponentStatus(str, Enum):
    """Status of a system component."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class EventSeverity(str, Enum):
    """Severity level of a system event."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(str, Enum):
    """Type of system event."""
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    HEARTBEAT = "heartbeat"
    STATUS_CHANGE = "status_change"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SystemComponent(BaseModel):
    """Model representing a system component."""
    name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    details: Optional[Dict] = None


class SystemEvent(BaseModel):
    """Model representing a system event."""
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    component: str
    event_type: EventType
    severity: EventSeverity
    message: str
    details: Optional[Dict] = None


class SystemState(BaseModel):
    """Model representing the overall system state."""
    components: Dict[str, SystemComponent]
    last_updated: datetime = Field(default_factory=datetime.now)


class HealthStatus(BaseModel):
    """Model representing the health status of a component or the system."""
    status: str
    details: Optional[Dict] = None


class ComponentRegistration(BaseModel):
    """Model for registering a component with the system controller."""
    name: str
    status: ComponentStatus = ComponentStatus.STARTING
    details: Optional[Dict] = None


class HeartbeatRequest(BaseModel):
    """Model for a heartbeat request from a component."""
    component: str
    status: ComponentStatus
    details: Optional[Dict] = None


class StatusUpdateRequest(BaseModel):
    """Model for a status update request from a component."""
    component: str
    status: ComponentStatus
    details: Optional[Dict] = None


class EventRequest(BaseModel):
    """Model for an event request from a component."""
    component: str
    event_type: EventType
    severity: EventSeverity
    message: str
    details: Optional[Dict] = None


class ComponentResponse(BaseModel):
    """Response model for component operations."""
    success: bool
    message: str
    component: Optional[SystemComponent] = None


class EventResponse(BaseModel):
    """Response model for event operations."""
    success: bool
    message: str
    event: Optional[SystemEvent] = None


class SystemStateResponse(BaseModel):
    """Response model for system state operations."""
    success: bool
    message: str
    state: Optional[SystemState] = None


class EventsResponse(BaseModel):
    """Response model for retrieving multiple events."""
    success: bool
    message: str
    events: List[SystemEvent] = []