"""
Circuit breaker configuration for all services.
"""

from typing import Dict, Any

# Default circuit breaker configuration
DEFAULT_CIRCUIT_BREAKER_CONFIG = {
    # Number of consecutive failures before opening the circuit
    "failure_threshold": 5,

    # Time in seconds to wait before attempting recovery
    "recovery_timeout": 30,

    # Request timeout in seconds
    "timeout": 10,

    # HTTP status codes that don't count as failures
    "exclude_status_codes": [404, 429],

    # Maximum number of retries for failed requests
    "max_retries": 3,

    # Whether to use exponential backoff for retries
    "use_exponential_backoff": True,

    # Base delay in seconds for exponential backoff
    "retry_base_delay": 0.1,

    # Maximum delay in seconds for exponential backoff
    "retry_max_delay": 10,

    # Whether to add jitter to retry delays
    "add_jitter": True
}

# Service-specific circuit breaker configuration
SERVICE_CIRCUIT_BREAKER_CONFIG = {
    "data-processing": {
        "failure_threshold": 3,
        "recovery_timeout": 15,
        "timeout": 30,  # Data processing may take longer
        "max_retries": 2
    },
    "model-services": {
        "failure_threshold": 3,
        "recovery_timeout": 60,  # Models may take longer to recover
        "timeout": 60,  # Model operations may take longer
        "max_retries": 2
    },
    "trading-strategy": {
        # Trading strategy needs faster failure detection and recovery
        "failure_threshold": 2,
        "recovery_timeout": 10,
        "timeout": 5,
        "max_retries": 1
    },
    "system-controller": {
        # System controller is critical, more aggressive retry
        "failure_threshold": 4,
        "recovery_timeout": 20,
        "timeout": 15,
        "max_retries": 5
    }
}


def get_circuit_breaker_config(service_name: str) -> Dict[str, Any]:
    """
    Get circuit breaker configuration for a service.

    Args:
        service_name: Name of the service

    Returns:
        Circuit breaker configuration for the service
    """
    # Make a copy of the default config
    config = DEFAULT_CIRCUIT_BREAKER_CONFIG.copy()

    # Override with service-specific config if available
    if service_name in SERVICE_CIRCUIT_BREAKER_CONFIG:
        config.update(SERVICE_CIRCUIT_BREAKER_CONFIG[service_name])

    return config
