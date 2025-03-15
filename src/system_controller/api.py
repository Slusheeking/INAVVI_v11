"""
API endpoints for the System Controller.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from prometheus_client import Counter, Gauge, Histogram

from src.system_controller.db import (
    get_all_components,
    get_component,
    get_component_events,
    get_recent_events,
    log_event,
    register_component,
    update_component_status,
)
from src.system_controller.models import (
    ComponentRegistration,
    ComponentResponse,
    EventRequest,
    EventResponse,
    EventSeverity,
    EventType,
    EventsResponse,
    HealthStatus,
    HeartbeatRequest,
    StatusUpdateRequest,
    SystemComponent,
    SystemEvent,
    SystemState,
    SystemStateResponse,
)

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Prometheus metrics
component_status = Gauge(
    "system_component_status",
    "Status of system components",
    ["component", "status"],
)
component_heartbeat = Gauge(
    "system_component_heartbeat",
    "Last heartbeat time of system components",
    ["component"],
)
event_counter = Counter(
    "system_events_total",
    "Total number of system events",
    ["component", "event_type", "severity"],
)
api_request_duration = Histogram(
    "system_controller_api_request_duration_seconds",
    "Duration of API requests",
    ["endpoint", "method"],
)
api_request_counter = Counter(
    "system_controller_api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(status="ok")


@router.post("/components", response_model=ComponentResponse)
async def register_component_endpoint(component: ComponentRegistration):
    """Register a component with the system controller."""
    logger.info(f"Registering component: {component.name}")
    
    system_component = SystemComponent(
        name=component.name,
        status=component.status,
        last_heartbeat=datetime.now(),
        details=component.details,
    )
    
    success, message = register_component(system_component)
    
    if success:
        # Update Prometheus metrics
        component_status.labels(component=component.name, status=component.status.value).set(1)
        component_heartbeat.labels(component=component.name).set(datetime.now().timestamp())
        
        # Log event
        event = SystemEvent(
            component=component.name,
            event_type=EventType.STARTUP,
            severity=EventSeverity.INFO,
            message=f"Component {component.name} registered with status {component.status.value}",
            details=component.details,
        )
        log_event(event)
        
        return ComponentResponse(
            success=True,
            message=message,
            component=system_component,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


@router.post("/components/{component_name}/heartbeat", response_model=ComponentResponse)
async def heartbeat_endpoint(component_name: str, heartbeat: HeartbeatRequest):
    """Send a heartbeat for a component."""
    logger.debug(f"Heartbeat received from component: {component_name}")
    
    if component_name != heartbeat.component:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Component name in path does not match component name in request body",
        )
    
    success, message, component = update_component_status(
        component_name=component_name,
        status=heartbeat.status,
        details=heartbeat.details,
    )
    
    if success:
        # Update Prometheus metrics
        component_status.labels(component=component_name, status=heartbeat.status.value).set(1)
        component_heartbeat.labels(component=component_name).set(datetime.now().timestamp())
        
        # Log event
        event = SystemEvent(
            component=component_name,
            event_type=EventType.HEARTBEAT,
            severity=EventSeverity.INFO,
            message=f"Heartbeat received from {component_name} with status {heartbeat.status.value}",
            details=heartbeat.details,
        )
        log_event(event)
        
        return ComponentResponse(
            success=True,
            message=message,
            component=component,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in message else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


@router.post("/components/{component_name}/status", response_model=ComponentResponse)
async def update_status_endpoint(component_name: str, status_update: StatusUpdateRequest):
    """Update the status of a component."""
    logger.info(f"Status update for component {component_name}: {status_update.status.value}")
    
    if component_name != status_update.component:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Component name in path does not match component name in request body",
        )
    
    success, message, component = update_component_status(
        component_name=component_name,
        status=status_update.status,
        details=status_update.details,
    )
    
    if success:
        # Update Prometheus metrics
        component_status.labels(component=component_name, status=status_update.status.value).set(1)
        
        # Log event
        event = SystemEvent(
            component=component_name,
            event_type=EventType.STATUS_CHANGE,
            severity=EventSeverity.INFO,
            message=f"Status of {component_name} changed to {status_update.status.value}",
            details=status_update.details,
        )
        log_event(event)
        
        return ComponentResponse(
            success=True,
            message=message,
            component=component,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in message else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


@router.get("/components/{component_name}", response_model=ComponentResponse)
async def get_component_endpoint(component_name: str):
    """Get information about a component."""
    logger.debug(f"Getting component: {component_name}")
    
    component = get_component(component_name)
    
    if component:
        return ComponentResponse(
            success=True,
            message=f"Component {component_name} found",
            component=component,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component {component_name} not found",
        )


@router.get("/components", response_model=SystemStateResponse)
async def get_all_components_endpoint():
    """Get all components."""
    logger.debug("Getting all components")
    
    components = get_all_components()
    
    # Convert to dictionary
    components_dict = {component.name: component for component in components}
    
    system_state = SystemState(
        components=components_dict,
        last_updated=datetime.now(),
    )
    
    return SystemStateResponse(
        success=True,
        message="System state retrieved successfully",
        state=system_state,
    )


@router.post("/events", response_model=EventResponse)
async def log_event_endpoint(event_request: EventRequest):
    """Log an event."""
    logger.info(f"Logging event from {event_request.component}: {event_request.event_type.value} - {event_request.message}")
    
    event = SystemEvent(
        component=event_request.component,
        event_type=event_request.event_type,
        severity=event_request.severity,
        message=event_request.message,
        details=event_request.details,
    )
    
    success, message, logged_event = log_event(event)
    
    if success:
        # Update Prometheus metrics
        event_counter.labels(
            component=event_request.component,
            event_type=event_request.event_type.value,
            severity=event_request.severity.value,
        ).inc()
        
        return EventResponse(
            success=True,
            message=message,
            event=logged_event,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )


@router.get("/events", response_model=EventsResponse)
async def get_events_endpoint(
    component: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Get recent events, optionally filtered by component."""
    logger.debug(f"Getting events, component={component}, limit={limit}")
    
    if component:
        events = get_component_events(component, limit)
    else:
        events = get_recent_events(limit)
    
    return EventsResponse(
        success=True,
        message=f"Retrieved {len(events)} events",
        events=events,
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="System Controller API",
        description="API for the Autonomous Trading System Controller",
        version="0.1.0",
    )
    
    app.include_router(router)
    
    return app