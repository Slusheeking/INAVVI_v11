"""
FastAPI extension for service registry integration.
"""

import logging
from typing import Callable, Dict, List, Optional
from fastapi import FastAPI, Request, Response, APIRouter, Depends
import atexit

from src.utils.service_registry.registry import ServiceRegistry

logger = logging.getLogger("api.service_registry")


class ServiceRegistryExtension:
    """
    FastAPI extension for service registry integration.
    """

    def __init__(
        self,
        app: FastAPI,
        service_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize the service registry extension.

        Args:
            app: FastAPI application
            service_name: Name of this service
            host: Host this service is running on (default: auto-detect)
            port: Port this service is listening on
        """
        self.app = app
        self.service_name = service_name

        # Create router for registry-related endpoints
        self.router = APIRouter(tags=["Service Registry"])

        # Create service registry
        self.registry = ServiceRegistry(
            service_name=service_name,
            host=host,
            api_port=port,
        )

        # Register service
        self.registry.register()

        # Start heartbeat thread
        self.heartbeat_thread = self.registry.start_heartbeat()

        # Add deregister hook on shutdown
        atexit.register(self.registry.deregister)

        # Add routes
        self._add_routes()

        # Add extension to app
        app.include_router(self.router)

        logger.info(
            f"Service registry extension initialized for {service_name}")

    def _add_routes(self):
        """Add service registry routes to the router."""

        @self.router.get("/registry/status")
        async def get_registry_status():
            """Get the status of the service registry."""
            return {
                "service_name": self.service_name,
                "registry_available": self.registry.registry_available,
                "status": self.registry.status,
                "feature_flags": self.registry.feature_flags
            }

        @self.router.get("/registry/services")
        async def get_services():
            """Get information about all registered services."""
            return self.registry.get_all_services()

        @self.router.get("/registry/services/{service_name}")
        async def get_service(service_name: str):
            """Get information about a specific service."""
            service = self.registry.get_service(service_name)
            if service is None:
                return Response(status_code=404, content=f"Service {service_name} not found")
            return service
