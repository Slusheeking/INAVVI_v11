"""
Resilient HTTP client for inter-service communication.
"""

import os
import time
import logging
import random
import threading
from typing import Any, Dict, Optional, Union
import requests
from datetime import datetime, timedelta

from src.utils.service_registry.registry import ServiceRegistry

logger = logging.getLogger("api.resilient_client")


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures."""

    # Circuit breaker states
    CLOSED = "CLOSED"  # Normal operation, requests flow through
    OPEN = "OPEN"      # Circuit is open, requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        timeout: int = 10,
        exclude_status_codes: list = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            timeout: Request timeout in seconds
            exclude_status_codes: HTTP status codes that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.exclude_status_codes = exclude_status_codes or [404, 429]

        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()

    def execute(self, func, *args, **kwargs):
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            The function result

        Raises:
            CircuitBreakerError: If circuit is open
            Any exception raised by the function
        """
        with self.lock:
            if self.state == self.OPEN:
                if self._should_attempt_recovery():
                    self.state = self.HALF_OPEN
                    logger.info(
                        f"Circuit {self.name} entering half-open state")
                else:
                    logger.warning(
                        f"Circuit {self.name} is OPEN. Failing fast.")
                    raise CircuitBreakerError(f"Circuit {self.name} is open")

        try:
            result = func(*args, **kwargs)

            with self.lock:
                if self.state == self.HALF_OPEN:
                    self._close()
                    logger.info(
                        f"Circuit {self.name} recovered and is now CLOSED")

            return result

        except Exception as e:
            status_code = None

            # Extract status code from request exceptions if available
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, 'response'):
                status_code = e.response.status_code

            # Don't count excluded status codes as failures
            if status_code in self.exclude_status_codes:
                logger.debug(
                    f"Request failed with excluded status {status_code}, not counting as circuit failure")
                raise

            with self.lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                    self._open()
                    logger.warning(
                        f"Circuit {self.name} is now OPEN after {self.failure_count} failures")

                if self.state == self.HALF_OPEN:
                    self._open()
                    logger.warning(
                        f"Circuit {self.name} reopened after failed recovery attempt")

            logger.error(
                f"Request failed. Circuit {self.name} failure count: {self.failure_count}")
            raise

    def _open(self):
        """Open the circuit."""
        self.state = self.OPEN
        self.last_failure_time = datetime.now()

    def _close(self):
        """Close the circuit."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def _should_attempt_recovery(self):
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True

        recovery_time = self.last_failure_time + \
            timedelta(seconds=self.recovery_timeout)
        return datetime.now() >= recovery_time


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ResilientServiceClient:
    """Resilient HTTP client for service-to-service communication."""

    def __init__(
        self,
        service_name: str,
        base_url: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 30,
        use_service_registry: bool = True
    ):
        """
        Initialize the resilient service client.

        Args:
            service_name: Name of the target service
            base_url: Base URL of the service (if not using service registry)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Seconds to wait before attempting recovery
            use_service_registry: Whether to use service registry for discovery
        """
        self.service_name = service_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"client_{service_name}",
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout,
            timeout=timeout
        )

        # Use service registry for discovery if enabled
        self.use_service_registry = use_service_registry
        if use_service_registry:
            self.registry = ServiceRegistry(service_name="client")
        else:
            self.registry = None

    def _get_service_url(self) -> str:
        """
        Get the URL for the service, using service registry if available.

        Returns:
            str: Service base URL

        Raises:
            ServiceDiscoveryError: If service cannot be discovered
        """
        if not self.use_service_registry or not self.registry or not self.registry.registry_available:
            if self.base_url:
                return self.base_url
            raise ServiceDiscoveryError(
                f"Service registry not available and no base URL provided for {self.service_name}")

        service = self.registry.get_service(self.service_name)
        if not service:
            raise ServiceDiscoveryError(
                f"Service {self.service_name} not found in registry")

        host = service.get("host")
        port = service.get("api_port") or service.get("port")

        if not host or not port:
            raise ServiceDiscoveryError(
                f"Service {self.service_name} found but has no host or port information")

        return f"http://{host}:{port}"

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """
        Send an HTTP request to the service.

        Args:
            method: HTTP method
            path: Request path (will be appended to base URL)
            params: Query parameters
            data: Form data
            json: JSON body
            headers: HTTP headers
            timeout: Request timeout in seconds (overrides default)
            **kwargs: Additional arguments to pass to requests

        Returns:
            requests.Response: Response object

        Raises:
            ServiceRequestError: If the request fails
            CircuitBreakerError: If circuit breaker is open
        """
        timeout = timeout or self.timeout
        headers = headers or {}

        # Add default headers
        if "Content-Type" not in headers and json is not None:
            headers["Content-Type"] = "application/json"

        def do_request():
            base_url = self._get_service_url()
            url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"

            logger.debug(f"Sending {method} request to {url}")

            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                raise ServiceRequestError(
                    f"Error requesting {url}: {str(e)}") from e

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Execute request with circuit breaker protection
                return self.circuit_breaker.execute(do_request)
            except CircuitBreakerError:
                # Don't retry if circuit is open
                raise
            except Exception as e:
                last_error = e
                retries += 1

                if retries <= self.max_retries:
                    # Add jitter to backoff
                    backoff = 0.1 * (2 ** retries) + random.uniform(0, 0.1)
                    logger.warning(
                        f"Request failed, retrying in {backoff:.2f} seconds ({retries}/{self.max_retries})")
                    time.sleep(backoff)
                else:
                    break

        # All retries failed
        raise last_error or ServiceRequestError(
            f"Request to {self.service_name} failed after {retries} retries")

    def get(self, path: str, **kwargs) -> requests.Response:
        """Send a GET request."""
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        """Send a POST request."""
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        """Send a PUT request."""
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """Send a DELETE request."""
        return self.request("DELETE", path, **kwargs)

    def patch(self, path: str, **kwargs) -> requests.Response:
        """Send a PATCH request."""
        return self.request("PATCH", path, **kwargs)


class ServiceDiscoveryError(Exception):
    """Exception raised when service discovery fails."""
    pass


class ServiceRequestError(Exception):
    """Exception raised when a service request fails."""
    pass
