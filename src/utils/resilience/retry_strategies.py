"""
Retry strategies for resilient communication.
"""

import random
import time
from typing import Callable, Optional, Dict, Any


class RetryStrategy:
    """Base class for retry strategies."""

    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate the delay in seconds before the next retry.

        Args:
            retry_count: Number of retries so far (starting from 1)

        Returns:
            Delay in seconds
        """
        raise NotImplementedError("Subclasses must implement calculate_delay")


class FixedDelayRetryStrategy(RetryStrategy):
    """Fixed delay retry strategy."""

    def __init__(self, delay: float = 1.0, add_jitter: bool = False, jitter_factor: float = 0.1):
        """
        Initialize the fixed delay retry strategy.

        Args:
            delay: Fixed delay in seconds
            add_jitter: Whether to add jitter to the delay
            jitter_factor: Factor for jitter (0-1)
        """
        self.delay = delay
        self.add_jitter = add_jitter
        self.jitter_factor = jitter_factor

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate the fixed delay with optional jitter."""
        if not self.add_jitter:
            return self.delay

        # Add jitter as a percentage of the delay
        jitter = self.delay * self.jitter_factor * random.random()
        return self.delay + jitter


class ExponentialBackoffRetryStrategy(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(
        self,
        base_delay: float = 0.1,
        max_delay: float = 60.0,
        add_jitter: bool = True,
        jitter_factor: float = 0.1
    ):
        """
        Initialize the exponential backoff retry strategy.

        Args:
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            add_jitter: Whether to add jitter to the delay
            jitter_factor: Factor for jitter (0-1)
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.add_jitter = add_jitter
        self.jitter_factor = jitter_factor

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate the exponential backoff delay with optional jitter."""
        # Calculate exponential backoff: base_delay * 2^retry_count
        delay = self.base_delay * (2 ** (retry_count - 1))

        # Cap at max delay
        delay = min(delay, self.max_delay)

        if not self.add_jitter:
            return delay

        # Add jitter as a percentage of the delay
        jitter = delay * self.jitter_factor * random.random()
        return delay + jitter


def create_retry_strategy(config: Dict[str, Any]) -> RetryStrategy:
    """
    Create a retry strategy based on configuration.

    Args:
        config: Configuration for the retry strategy

    Returns:
        RetryStrategy: Configured retry strategy
    """
    if config.get("use_exponential_backoff", True):
        return ExponentialBackoffRetryStrategy(
            base_delay=config.get("retry_base_delay", 0.1),
            max_delay=config.get("retry_max_delay", 60.0),
            add_jitter=config.get("add_jitter", True),
            jitter_factor=config.get("jitter_factor", 0.1)
        )
    else:
        return FixedDelayRetryStrategy(
            delay=config.get("retry_delay", 1.0),
            add_jitter=config.get("add_jitter", False),
            jitter_factor=config.get("jitter_factor", 0.1)
        )


def retry_with_strategy(
    retry_strategy: RetryStrategy,
    max_retries: int,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator for retrying functions with a specified strategy.

    Args:
        retry_strategy: Strategy for calculating retry delays
        max_retries: Maximum number of retries
        on_retry: Callback function when a retry happens

    Returns:
        Function decorator
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        raise  # Re-raise the last exception when retries exhausted

                    if on_retry:
                        on_retry(retries, e)

                    # Calculate delay based on strategy and sleep
                    delay = retry_strategy.calculate_delay(retries)
                    time.sleep(delay)
        return wrapper
    return decorator
