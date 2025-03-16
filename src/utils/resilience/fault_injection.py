"""
Fault injection for testing resilience.
"""

import random
import time
import logging
from typing import Callable, Optional, Dict, Any, List, Tuple

logger = logging.getLogger("fault_injection")


class FaultInjector:
    """Injects faults into system for resilience testing."""

    def __init__(
        self,
        enabled: bool = False,
        failure_rate: float = 0.1,
        latency_rate: float = 0.2,
        max_latency: float = 2.0,
        chaos_mode: bool = False
    ):
        """
        Initialize the fault injector.

        Args:
            enabled: Whether fault injection is enabled
            failure_rate: Probability of injecting a failure (0-1)
            latency_rate: Probability of injecting latency (0-1)
            max_latency: Maximum latency to inject in seconds
            chaos_mode: If True, use more aggressive fault injection
        """
        self.enabled = enabled
        self.failure_rate = failure_rate
        self.latency_rate = latency_rate
        self.max_latency = max_latency
        self.chaos_mode = chaos_mode

        if chaos_mode:
            # Double the failure and latency rates in chaos mode
            self.failure_rate = min(self.failure_rate * 2, 0.5)
            self.latency_rate = min(self.latency_rate * 2, 0.5)
            self.max_latency = self.max_latency * 3

    def inject_failure(self, operation_name: str = "unknown") -> bool:
        """
        Potentially inject a failure based on failure rate.

        Args:
            operation_name: Name of the operation for logging

        Returns:
            bool: True if a failure should be injected, False otherwise
        """
        if not self.enabled:
            return False

        if random.random() < self.failure_rate:
            logger.info(f"Injecting failure into operation: {operation_name}")
            return True

        return False

    def inject_latency(self, operation_name: str = "unknown") -> None:
        """
        Potentially inject latency based on latency rate.

        Args:
            operation_name: Name of the operation for logging
        """
        if not self.enabled:
            return

        if random.random() < self.latency_rate:
            # Generate a random latency between 0 and max_latency
            latency = random.uniform(0, self.max_latency)
            logger.info(
                f"Injecting {latency:.2f}s latency into operation: {operation_name}")
            time.sleep(latency)

    def __call__(self, func):
        """
        Decorator for injecting faults into functions.

        Args:
            func: Function to decorate

        Returns:
            Wrapped function with fault injection
        """
        def wrapper(*args, **kwargs):
            operation_name = func.__name__

            # Potentially inject latency
            self.inject_latency(operation_name)

            # Potentially inject failure
            if self.inject_failure(operation_name):
                raise RuntimeError(f"Injected failure in {operation_name}")

            # Call the actual function
            return func(*args, **kwargs)

        return wrapper


# Global fault injector instances for different contexts
HTTP_FAULT_INJECTOR = FaultInjector(
    enabled=False, failure_rate=0.05, latency_rate=0.1)
DATABASE_FAULT_INJECTOR = FaultInjector(
    enabled=False, failure_rate=0.03, latency_rate=0.15)
REDIS_FAULT_INJECTOR = FaultInjector(
    enabled=False, failure_rate=0.07, latency_rate=0.2)


def configure_fault_injectors(config: Dict[str, Any]) -> None:
    """
    Configure fault injectors based on configuration.

    Args:
        config: Configuration dictionary for fault injectors
    """
    global HTTP_FAULT_INJECTOR, DATABASE_FAULT_INJECTOR, REDIS_FAULT_INJECTOR

    fault_injection = config.get("fault_injection", {})
    enabled = fault_injection.get("enabled", False)
    chaos_mode = fault_injection.get("chaos_mode", False)

    # Configure HTTP fault injector
    http_config = fault_injection.get("http", {})
    HTTP_FAULT_INJECTOR = FaultInjector(
        enabled=enabled and http_config.get("enabled", True),
        failure_rate=http_config.get("failure_rate", 0.05),
        latency_rate=http_config.get("latency_rate", 0.1),
        max_latency=http_config.get("max_latency", 2.0),
        chaos_mode=chaos_mode
    )

    # Configure database fault injector
    db_config = fault_injection.get("database", {})
    DATABASE_FAULT_INJECTOR = FaultInjector(
        enabled=enabled and db_config.get("enabled", True),
        failure_rate=db_config.get("failure_rate", 0.03),
        latency_rate=db_config.get("latency_rate", 0.15),
        max_latency=db_config.get("max_latency", 1.5),
        chaos_mode=chaos_mode
    )

    # Configure Redis fault injector
    redis_config = fault_injection.get("redis", {})
    REDIS_FAULT_INJECTOR = FaultInjector(
        enabled=enabled and redis_config.get("enabled", True),
        failure_rate=redis_config.get("failure_rate", 0.07),
        latency_rate=redis_config.get("latency_rate", 0.2),
        max_latency=redis_config.get("max_latency", 1.0),
        chaos_mode=chaos_mode
    )

    logger.info(
        f"Fault injectors configured: enabled={enabled}, chaos_mode={chaos_mode}")
