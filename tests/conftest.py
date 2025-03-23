#!/usr/bin/env python3
"""
Test configuration and fixtures for the trading system.
"""

import os
import sys
import pytest
import redis
import logging
import asyncio
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tests")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def redis_client():
    """Create a Redis client for testing."""
    try:
        # Get Redis configuration from environment variables
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6380"))
        redis_db = int(os.environ.get("REDIS_DB", "0"))
        redis_username = os.environ.get("REDIS_USERNAME", "default")
        redis_password = os.environ.get(
            "REDIS_PASSWORD", "trading_system_2025")
        redis_ssl = os.environ.get("REDIS_SSL", "false").lower() == "true"

        # Create Redis client
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            username=redis_username,
            password=redis_password,
            ssl=redis_ssl,
            ssl_cert_reqs=None,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=False,
        )

        # Test connection
        client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        yield client

        # Clean up
        client.close()

    except (redis.RedisError, ConnectionError) as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        yield None


@pytest.fixture(scope="session")
def api_keys():
    """Get API keys from environment variables."""
    return {
        "polygon": os.environ.get("POLYGON_API_KEY", ""),
        "unusual_whales": os.environ.get("UNUSUAL_WHALES_API_KEY", ""),
        "alpaca_key": os.environ.get("ALPACA_API_KEY", ""),
        "alpaca_secret": os.environ.get("ALPACA_API_SECRET", ""),
    }


@pytest.fixture(scope="session")
def test_tickers():
    """Return a list of tickers to use for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
