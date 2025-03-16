"""
API utilities for the Autonomous Trading System.

This module provides utilities for working with external APIs, including
rate limiting, error handling, response parsing, and specialized support
for financial market APIs like Alpaca.
"""

from src.utils.api.api_utils import (
    APIClient,
    AlpacaAPIClient,
    APICache,
    RateLimiter,
    RetryHandler,
    ConnectionPool,
    cached_api_call,
    create_api_client_from_env,
    create_alpaca_client_from_env,
    rate_limited,
)

__all__ = [
    "APIClient",
    "AlpacaAPIClient",
    "APICache",
    "RateLimiter",
    "RetryHandler",
    "ConnectionPool",
    "cached_api_call",
    "create_api_client_from_env",
    "create_alpaca_client_from_env",
    "rate_limited",
]