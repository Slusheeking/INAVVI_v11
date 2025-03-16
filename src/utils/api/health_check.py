"""
Comprehensive health check API for all services.
"""

import os
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from fastapi import APIRouter, Response, HTTPException, Depends
import psutil
import requests
from datetime import datetime, timedelta

logger = logging.getLogger("health_check")


class HealthCheckComponent:
    """Component that can be health checked."""

    def __init__(
        self,
        name: str,
        description: str,
        check_function: Callable[[], Dict[str, Any]]
    ):
        """
        Initialize a health check component.

        Args:
            name: Name of the component
            description: Description of the component
            check_function: Function that checks the component health
        """
        self.name = name
        self.description = description
        self.check_function = check_function
        self.last_check_time = None
        self.last_status = None
        self.last_error = None

    def check(self) -> Dict[str, Any]:
        """
        Check the health of the component.

        Returns:
            Dict[str, Any]: Health check result
        """
        self.last_check_time = datetime.now()

        try:
            result = self.check_function()
            self.last_status = result.get("status", "unknown")
            self.last_error = result.get("error")
            return result
        except Exception as e:
            logger.error(f"Error checking component {self.name}: {e}")
            self.last_status = "unhealthy"
            self.last_error = str(e)
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class SystemHealthCheck:
    """System-wide health check."""

    def __init__(self):
        """Initialize the system health check."""
        self.components: Dict[str, HealthCheckComponent] = {}
        self.lock = threading.RLock()

    def add_component(
        self,
        name: str,
        description: str,
        check_function: Callable[[], Dict[str, Any]]
    ) -> None:
        """
        Add a component to health check.

        Args:
            name: Name of the component
            description: Description of the component
            check_function: Function that checks the component health
        """
        with self.lock:
            self.components[name] = HealthCheckComponent(
                name=name,
                description=description,
                check_function=check_function
            )

    def check_all(self) -> Dict[str, Any]:
        """
        Check the health of all components.

        Returns:
            Dict[str, Any]: Health check result for all components
        """
        with self.lock:
            results = {}
            system_status = "healthy"

            # Perform checks
            for name, component in self.components.items():
                result = component.check()
                results[name] = result

                # If any critical component is unhealthy, the system is unhealthy
                if result.get("status") == "unhealthy" and result.get("critical", False):
                    system_status = "unhealthy"
                # If any component is degraded and system isn't unhealthy, mark as degraded
                elif result.get("status") == "degraded" and system_status != "unhealthy":
                    system_status = "degraded"

            return {
                "status": system_status,
                "timestamp": datetime.now().isoformat(),
                "components": results
            }

    def check_component(self, name: str) -> Dict[str, Any]:
        """
        Check the health of a specific component.

        Args:
            name: Name of the component

        Returns:
            Dict[str, Any]: Health check result for the component

        Raises:
            KeyError: If the component is not found
        """
        with self.lock:
            if name not in self.components:
                raise KeyError(f"Component {name} not found")

            component = self.components[name]
            return component.check()


def create_health_check_router(health_check: SystemHealthCheck) -> APIRouter:
    """
    Create a FastAPI router for health checks.

    Args:
        health_check: System health check instance

    Returns:
        APIRouter: FastAPI router with health check endpoints
    """
    router = APIRouter(tags=["Health"])

    @router.get("/health")
    async def get_health():
        """Check the health of the service."""
        result = health_check.check_all()

        # Return appropriate status code based on health
        status_code = 200
        if result["status"] == "unhealthy":
            status_code = 503
        elif result["status"] == "degraded":
            status_code = 200  # Could also be 429 or another appropriate code

        return Response(
            content=result,
            status_code=status_code,
            media_type="application/json"
        )

    @router.get("/health/{component}")
    async def get_component_health(component: str):
        """Check the health of a specific component."""
        try:
            result = health_check.check_component(component)

            # Return appropriate status code based on health
            status_code = 200
            if result["status"] == "unhealthy":
                status_code = 503
            elif result["status"] == "degraded":
                status_code = 200  # Could also be 429 or another appropriate code

            return Response(
                content=result,
                status_code=status_code,
                media_type="application/json"
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Component {component} not found")

    return router

# Common health check functions


def check_database_health() -> Dict[str, Any]:
    """
    Check the health of the database.

    Returns:
        Dict[str, Any]: Health check result
    """
    try:
        import psycopg2

        # Connect to the database
        conn = psycopg2.connect(
            host=os.environ.get("TIMESCALEDB_HOST", "timescaledb"),
            port=os.environ.get("TIMESCALEDB_PORT", "5432"),
            dbname=os.environ.get("TIMESCALEDB_DATABASE", "ats_db"),
            user=os.environ.get("TIMESCALEDB_USER", "ats_user"),
            password=os.environ.get("TIMESCALEDB_PASSWORD", "ats_password")
        )

        # Check connection
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()

        # Check TimescaleDB extension
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
        timescale_installed = cur.fetchone() is not None

        # Close connection
        cur.close()
        conn.close()

        return {
            "status": "healthy" if timescale_installed else "degraded",
            "critical": True,
            "message": "Database connection successful",
            "details": {
                "timescaledb_installed": timescale_installed
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "critical": True,
            "error": str(e)
        }


def check_redis_health() -> Dict[str, Any]:
    """
    Check the health of Redis.

    Returns:
        Dict[str, Any]: Health check result
    """
    try:
        import redis

        # Connect to Redis
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD", ""),
            socket_connect_timeout=1,
            socket_timeout=1
        )

        # Check connection
        result = r.ping()

        # Check memory usage
        info = r.info()
        used_memory = info.get("used_memory", 0)
        total_memory = info.get("maxmemory", 0)
        memory_percent = used_memory / total_memory if total_memory > 0 else 0

        status = "healthy"
        message = "Redis connection successful"

        # Check if memory usage is high
        if memory_percent > 0.9:
            status = "degraded"
            message = "Redis memory usage is high"

        return {
            "status": status,
            "critical": False,
            "message": message,
            "details": {
                "used_memory": used_memory,
                "total_memory": total_memory,
                "memory_percent": memory_percent
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "critical": False,
            "error": str(e)
        }


def check_system_resources() -> Dict[str, Any]:
    """
    Check system resources.

    Returns:
        Dict[str, Any]: Health check result
    """
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get memory usage
        memory = psutil.virtual_memory()

        # Get disk usage
        disk = psutil.disk_usage("/")

        status = "healthy"
        message = "System resources are healthy"

        # Check if resources are low
        if cpu_percent > 90:
            status = "degraded"
            message = "CPU usage is high"

        if memory.percent > 90:
            status = "degraded"
            message = "Memory usage is high"

        if disk.percent > 95:
            status = "degraded"
            message = "Disk usage is high"

        return {
            "status": status,
            "critical": False,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
        }
    except Exception as e:
        return {
            "status": "unknown",
            "critical": False,
            "error": str(e)
        }


def create_service_check(service_url: str, service_name: str, is_critical: bool = False) -> Callable[[], Dict[str, Any]]:
    """
    Create a health check function for a service.

    Args:
        service_url: URL of the service health check endpoint
        service_name: Name of the service
        is_critical: Whether the service is critical

    Returns:
        Callable[[], Dict[str, Any]]: Health check function
    """
    def check_service() -> Dict[str, Any]:
        try:
            # Make a request to the service health check endpoint
            response = requests.get(service_url, timeout=5)

            if response.status_code == 200:
                try:
                    # Try to parse JSON response
                    data = response.json()

                    return {
                        "status": data.get("status", "healthy"),
                        "critical": is_critical,
                        "message": f"{service_name} is healthy",
                        "details": data
                    }
                except ValueError:
                    return {
                        "status": "degraded",
                        "critical": is_critical,
                        "message": f"{service_name} returned non-JSON response",
                        "details": {
                            "status_code": response.status_code,
                            "response": response.text[:100]
                        }
                    }
            else:
                return {
                    "status": "unhealthy",
                    "critical": is_critical,
                    "message": f"{service_name} returned status code {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response": response.text[:100]
                    }
                }
        except requests.RequestException as e:
            return {
                "status": "unhealthy",
                "critical": is_critical,
                "error": str(e)
            }

    return check_service
