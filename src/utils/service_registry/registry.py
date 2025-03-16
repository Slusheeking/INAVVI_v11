"""
Service Registry module for service discovery and health management.
"""

import os
import json
import time
import socket
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union
import redis
from datetime import datetime, timedelta

logger = logging.getLogger("service_registry")


class ServiceRegistry:
    """
    Redis-based service registry for service discovery and health tracking.
    """

    def __init__(
        self,
        service_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_port: Optional[int] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        heartbeat_interval: int = 15,
    ):
        """
        Initialize the service registry.

        Args:
            service_name: Name of this service
            host: Host this service is running on (default: auto-detect)
            port: Main port this service is listening on
            api_port: API port this service is listening on
            redis_host: Redis host (default: from environment)
            redis_port: Redis port (default: from environment)
            redis_password: Redis password (default: from environment)
            heartbeat_interval: Seconds between heartbeats (default: 15)
        """
        self.service_name = service_name
        self.host = host or socket.gethostname()
        self.port = port
        self.api_port = api_port
        self.heartbeat_interval = heartbeat_interval
        self.status = "starting"
        self.feature_flags = {}

        # Load feature flags if available
        self._load_feature_flags()

        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=redis_host or os.environ.get("REDIS_HOST", "redis"),
                port=redis_port or int(os.environ.get("REDIS_PORT", "6379")),
                password=redis_password or os.environ.get(
                    "REDIS_PASSWORD", ""),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            self.registry_available = True
            logger.info(
                f"Connected to service registry (Redis) at {redis_host or os.environ.get('REDIS_HOST', 'redis')}")
        except Exception as e:
            self.redis = None
            self.registry_available = False
            logger.warning(f"Could not connect to service registry: {e}")
            logger.warning("Service registry features will be disabled")

    def _load_feature_flags(self):
        """Load feature flags from disk if available."""
        try:
            with open("/app/feature_flags.json", "r") as f:
                self.feature_flags = json.load(f)
                logger.info(f"Loaded feature flags: {self.feature_flags}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(
                "No feature flags file found or invalid JSON. Using empty dict.")
            self.feature_flags = {}

    def register(self) -> bool:
        """
        Register this service in the registry.

        Returns:
            bool: True if registration was successful, False otherwise
        """
        if not self.registry_available:
            logger.warning(
                "Service registry not available. Skipping registration.")
            return False

        try:
            service_data = {
                "host": self.host,
                "port": self.port,
                "api_port": self.api_port,
                "last_heartbeat": datetime.utcnow().isoformat(),
                "status": self.status,
                "feature_flags": json.dumps(self.feature_flags)
            }

            # Register in Redis
            key = f"service:{self.service_name}"
            self.redis.hmset(key, service_data)
            # Auto-expire if no heartbeats
            self.redis.expire(key, self.heartbeat_interval * 4)

            logger.info(f"Service {self.service_name} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False

    def deregister(self) -> bool:
        """
        Deregister this service from the registry.

        Returns:
            bool: True if deregistration was successful, False otherwise
        """
        if not self.registry_available:
            return False

        try:
            key = f"service:{self.service_name}"
            self.redis.delete(key)
            logger.info(
                f"Service {self.service_name} deregistered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            return False

    def update_status(self, status: str) -> bool:
        """
        Update this service's status in the registry.

        Args:
            status: New status (e.g., "starting", "healthy", "degraded", "unhealthy")

        Returns:
            bool: True if status update was successful, False otherwise
        """
        self.status = status

        if not self.registry_available:
            return False

        try:
            key = f"service:{self.service_name}"
            self.redis.hset(key, "status", status)
            self.redis.hset(key, "last_heartbeat",
                            datetime.utcnow().isoformat())
            logger.info(
                f"Service {self.service_name} status updated to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update service status: {e}")
            return False

    def get_service(self, service_name: str) -> Optional[Dict]:
        """
        Get information about a service from the registry.

        Args:
            service_name: Name of the service to look up

        Returns:
            Optional[Dict]: Service information or None if not found
        """
        if not self.registry_available:
            return None

        try:
            key = f"service:{service_name}"
            data = self.redis.hgetall(key)

            if not data:
                return None

            # Parse feature flags if available
            if "feature_flags" in data:
                try:
                    data["feature_flags"] = json.loads(data["feature_flags"])
                except json.JSONDecodeError:
                    data["feature_flags"] = {}

            return data
        except Exception as e:
            logger.error(f"Failed to get service {service_name}: {e}")
            return None

    def get_all_services(self) -> Dict[str, Dict]:
        """
        Get information about all registered services.

        Returns:
            Dict[str, Dict]: Dictionary of service names to service information
        """
        if not self.registry_available:
            return {}

        try:
            services = {}
            for key in self.redis.keys("service:*"):
                service_name = key.split(":", 1)[1]
                service_data = self.get_service(service_name)
                if service_data:
                    services[service_name] = service_data

            return services
        except Exception as e:
            logger.error(f"Failed to get all services: {e}")
            return {}

    def start_heartbeat(self) -> threading.Thread:
        """
        Start sending periodic heartbeats to the registry.

        Returns:
            threading.Thread: Thread that sends heartbeats
        """
        def heartbeat_worker():
            while True:
                try:
                    if self.registry_available:
                        key = f"service:{self.service_name}"
                        self.redis.hset(key, "last_heartbeat",
                                        datetime.utcnow().isoformat())
                        self.redis.expire(key, self.heartbeat_interval * 4)
                        logger.debug(f"Sent heartbeat for {self.service_name}")
                except Exception as e:
                    logger.error(f"Failed to send heartbeat: {e}")

                time.sleep(self.heartbeat_interval)

        thread = threading.Thread(target=heartbeat_worker, daemon=True)
        thread.start()
        return thread
