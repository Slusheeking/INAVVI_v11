#!/usr/bin/env python3
"""
GPU-Optimized Unusual Whales API Client

This module provides an optimized client for interacting with the Unusual Whales API,
designed specifically for high-performance trading systems using GPU acceleration.
"""

import os
import time
import json
import logging
import hashlib
import pickle
import requests
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import cupy as cp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import redis
import signal
import sys
import random
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env files
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()  # Try to load from the main .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_unusual_whales_client')

# Environment variables
UNUSUAL_WHALES_API_KEY = os.environ.get('UNUSUAL_WHALES_API_KEY', '')
# 5 minutes default TTL
CACHE_TTL = int(os.environ.get('UNUSUAL_WHALES_CACHE_TTL', 300))
MAX_RETRIES = int(os.environ.get('UNUSUAL_WHALES_MAX_RETRIES', 3))
RETRY_BACKOFF_FACTOR = float(os.environ.get(
    'UNUSUAL_WHALES_RETRY_BACKOFF_FACTOR', 0.5))
CONNECTION_TIMEOUT = int(os.environ.get(
    'UNUSUAL_WHALES_CONNECTION_TIMEOUT', 15))
MAX_POOL_SIZE = int(os.environ.get('UNUSUAL_WHALES_MAX_POOL_SIZE', 20))
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))  # Ensure we use port 6379
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_USERNAME = os.environ.get('REDIS_USERNAME', 'default')
# Added default Redis password
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'trading_system_2025')
# Fixed to false since we're running locally
REDIS_SSL = os.environ.get('REDIS_SSL', 'false').lower() == 'true'
REDIS_TIMEOUT = int(os.environ.get('REDIS_TIMEOUT', 5))
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'

# Endpoint refresh intervals (in seconds)
REFRESH_INTERVALS = {
    'unusual_activity': 30,  # Every 30 seconds
    'flow': 10,              # Every 10 seconds
    'latest_sweep': 5,       # Every 5 seconds
    'dark_pool': 60,         # Every 60 seconds
    'alerts': 60,            # Every 60 seconds
    'alert_configs': 3600,   # Every hour
    'insider': 3600          # Every hour
}


class RedisCache:
    """Redis-based cache for API responses with fallback to in-memory cache"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl=CACHE_TTL):
        """Initialize Redis cache"""
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.enabled = True
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
        self.size_limit = 10000  # Maximum number of items in memory cache

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                ssl_cert_reqs=None,
                socket_timeout=REDIS_TIMEOUT,
                socket_connect_timeout=5,
                decode_responses=False  # Keep binary data for efficient serialization
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except (redis.RedisError, ConnectionError) as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            logger.warning("Falling back to in-memory cache")
            self.enabled = False

    def _generate_key(self, key_parts):
        """Generate a consistent cache key from parts"""
        if isinstance(key_parts, str):
            key_parts = [key_parts]

        # Join all parts and create a hash
        key_str = ":".join([str(part) for part in key_parts])
        return f"unusual_whales:{hashlib.md5(key_str.encode()).hexdigest()}"

    def get(self, key_parts):
        """Get value from cache"""
        key = self._generate_key(key_parts)

        # Try memory cache first for fastest access
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry['expiry']:
                self.hits += 1
                return entry['value']
            else:
                # Entry expired
                del self.memory_cache[key]

        # Try Redis if enabled
        if self.enabled:
            try:
                data = self.redis_client.get(key)
                if data:
                    self.hits += 1
                    # Store in memory cache for faster subsequent access
                    value = pickle.loads(data)
                    self.memory_cache[key] = {
                        'value': value,
                        'expiry': time.time() + self.ttl
                    }
                    return value
            except (redis.RedisError, pickle.PickleError) as e:
                logger.warning(f"Error retrieving from Redis cache: {e}")

        self.misses += 1
        return None

    def set(self, key_parts, value, ttl=None):
        """Set value in cache"""
        if value is None:
            return False

        key = self._generate_key(key_parts)
        ttl = ttl or self.ttl
        expiry = time.time() + ttl

        # Implement LRU-like behavior by limiting memory cache size
        if len(self.memory_cache) >= self.size_limit:
            # Remove random 10% of entries when cache gets too large
            keys_to_remove = random.sample(list(self.memory_cache.keys()),
                                           int(len(self.memory_cache) * 0.1))
            for k in keys_to_remove:
                self.memory_cache.pop(k, None)

        # Store in memory cache
        self.memory_cache[key] = {
            'value': value,
            'expiry': expiry
        }

        # Store in Redis if enabled
        if self.enabled:
            try:
                serialized = pickle.dumps(value)
                return self.redis_client.setex(key, ttl, serialized)
            except (redis.RedisError, pickle.PickleError) as e:
                logger.warning(f"Error setting Redis cache: {e}")
                return False

        return True

    def store_dataframe(self, key_parts, df, ttl=None):
        """Store DataFrame in cache with optimized serialization"""
        if df is None or df.empty:
            return False

        # Use pickle for serialization instead of PyArrow due to compatibility issues
        try:
            key = self._generate_key(key_parts)
            ttl = ttl or self.ttl
            serialized = pickle.dumps(df)

            if self.enabled:
                try:
                    return self.redis_client.setex(key, ttl, serialized)
                except redis.RedisError as e:
                    logger.warning(f"Error storing DataFrame in Redis: {e}")

            # Store in memory cache
            self.memory_cache[key] = {
                'value': df,
                'expiry': time.time() + ttl
            }
            return True
        except Exception as e:
            return self.set(key_parts, df, ttl)

    def get_dataframe(self, key_parts):
        """Retrieve DataFrame from cache with optimized deserialization"""
        key = self._generate_key(key_parts)

        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry['expiry']:
                self.hits += 1
                return entry['value']
            else:
                del self.memory_cache[key]

        # Try Redis if enabled
        if self.enabled:
            try:
                data = self.redis_client.get(key)
                if data:
                    self.hits += 1

                    # Deserialize using pickle
                    try:
                        df = pickle.loads(data)

                        # Store in memory cache
                        self.memory_cache[key] = {
                            'value': df,
                            'expiry': time.time() + self.ttl
                        }
                        return df
                    except Exception as e:
                        logger.warning(f"Error deserializing DataFrame: {e}")
            except Exception as e:
                logger.warning(f"Error retrieving DataFrame from cache: {e}")

        self.misses += 1
        return None


class AsyncConnectionPool:
    """Asynchronous HTTP connection pool with retry logic"""

    def __init__(self, max_retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF_FACTOR,
                 max_pool_size=MAX_POOL_SIZE, timeout=CONNECTION_TIMEOUT):
        """Initialize connection pool"""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = None
        self.connector = None

        logger.info(
            f"Async connection pool initialized with max_pool_size={max_pool_size}, max_retries={max_retries}")

    async def initialize(self):
        """Initialize the aiohttp session"""
        if self.session is None or self.session.closed:
            # Configure TCP connector with connection pooling
            self.connector = aiohttp.TCPConnector(
                limit=MAX_POOL_SIZE,
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                use_dns_cache=True,
                ssl=False  # Unusual Whales uses HTTPS but we don't need to verify
            )

            # Create session with default headers and timeout
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                headers={
                    'User-Agent': 'UnusualWhalesGPUClient/1.0',
                    'Accept': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def get(self, url, params=None, headers=None):
        """Make GET request with retry logic"""
        if self.session is None or self.session.closed:
            await self.initialize()

        last_error = None

        for retry in range(self.max_retries + 1):
            try:
                logger.debug(f"Making request to {url} with params: {params}")
                async with self.session.get(url, params=params, headers=headers) as response:
                    # Check if response is successful
                    if response.status == 200:
                        response_json = await response.json()
                        logger.debug(
                            f"Response status: 200, body: {response_json}")
                        return response_json

                    # Handle rate limiting
                    if response.status == 429:
                        wait_time = (2 ** retry) * \
                            self.backoff_factor + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limit hit, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    # Handle other errors
                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text}"
                    logger.error(f"Request failed: {last_error}")

                    # Don't retry on client errors except rate limits
                    if 400 <= response.status < 500 and response.status != 429:
                        return {"status": "ERROR", "error": last_error}

                    # Retry on server errors
                    wait_time = (2 ** retry) * \
                        self.backoff_factor + random.uniform(0, 1)
                    logger.warning(
                        f"Server error, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"Request failed: {type(e).__name__}: {str(e)}"
                logger.error(last_error)

                # Exponential backoff with jitter
                wait_time = (2 ** retry) * self.backoff_factor + \
                    random.uniform(0, 1)
                logger.warning(
                    f"Connection error, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries}): {e}")
                await asyncio.sleep(wait_time)

        # All retries failed
        return {"status": "ERROR", "error": last_error or "Max retries exceeded"}

    async def close(self):
        """Close the session and all connections"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Async connection pool closed")


class GPUUnusualWhalesClient:
    """GPU-Optimized client for the Unusual Whales API"""

    def __init__(self, api_key=UNUSUAL_WHALES_API_KEY, redis_client=None, use_gpu=USE_GPU,
                 max_pool_size=MAX_POOL_SIZE, max_retries=MAX_RETRIES, cache_ttl=CACHE_TTL):
        """
        Initialize GPU-Optimized Unusual Whales API client

        Args:
            api_key: API key for authentication
            redis_client: Optional Redis client for caching
            use_gpu: Whether to use GPU acceleration
            max_pool_size: Maximum connection pool size
            max_retries: Maximum number of retries for failed requests
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.use_gpu = use_gpu

        # Verify API key is provided
        if not self.api_key:
            logger.warning(
                "No API key provided. Set the UNUSUAL_WHALES_API_KEY environment variable.")
        else:
            logger.info(
                f"Initialized Unusual Whales API client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}")

        # Initialize base URL
        self.base_url = "https://api.unusualwhales.com/v2"

        # Initialize cache
        if redis_client:
            self.redis_client = redis_client
            self.cache = RedisCache(ttl=cache_ttl)
        else:
            self.redis_client = None
            self.cache = RedisCache(ttl=cache_ttl)

        # Initialize connection pool
        self.connection_pool = AsyncConnectionPool(
            max_pool_size=max_pool_size, max_retries=max_retries)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Initialize GPU if available and requested
        self.gpu_initialized = False
        if self.use_gpu:
            # Initialize CuPy - no fallback to CPU
            self._initialize_gpu()

        # Flag to track if client is running
        self.running = True

        # Scheduled tasks
        self.scheduled_tasks = {}
        self.event_loop = None

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False  # Set running flag to false

        # Check if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close())
        except RuntimeError:
            # No running event loop, schedule close for later
            logger.info("No running event loop, will close resources later")

    def _initialize_gpu(self):
        """Initialize GPU for data processing"""
        try:
            import cupy as cp

            # Get device count
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                # Find GH200 device if available
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    if "GH200" in device_props["name"].decode():
                        cp.cuda.Device(i).use()
                        logger.info(f"Using GH200 GPU at index {i}")
                        break

                # Use unified memory for better performance
                cp.cuda.set_allocator(cp.cuda.MemoryPool(
                    cp.cuda.malloc_managed).malloc)
                logger.info("CuPy configured with unified memory")
                self.gpu_initialized = True
            else:
                logger.warning("No GPU devices found")
                self.use_gpu = False
        except ImportError:
            logger.warning("CuPy not available, GPU acceleration disabled")
            self.use_gpu = False
        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.use_gpu = False

    async def _make_request(self, endpoint, params=None):
        """
        Make API request with caching

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            API response as a dictionary
        """
        # Generate cache key
        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data

        # Prepare request
        url = f"{self.base_url}/{endpoint}"

        # Verify API key is set
        if not self.api_key:
            logger.error("API key not provided")
            return {"status": "ERROR", "error": "API key not provided"}

        # Set proper headers for Unusual Whales API
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Add debug logging for the request
        logger.debug(f"Making request to endpoint: {endpoint}")
        logger.debug(f"Full URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Params: {params}")

        # Make request
        try:
            data = await self.connection_pool.get(url, params=params, headers=headers)

            # Add debug logging for the response
            logger.debug(f"Response from {endpoint}: {data}")

            # Cache successful responses
            if isinstance(data, dict) and "data" in data:
                self.cache.set(cache_key, data)

            return data
        except Exception as e:
            logger.error(
                f"Unexpected error during API request: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"Unexpected error: {type(e).__name__}: {e}"}

    async def check_api_health(self):
        """Check the health of the Unusual Whales API"""
        try:
            endpoint = "status"
            logger.info(f"Checking API health using endpoint: {endpoint}")

            data = await self._make_request(endpoint)

            if data and "status" in data:
                logger.info(f"API health check result: {data['status']}")
                return data["status"] == "OK"
            else:
                logger.warning(f"Failed to check API health. Response: {data}")
                return False
        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            return False

    async def get_flow_alerts(self, ticker, is_ask_side=True, is_bid_side=True, limit=100):
        """
        Get flow alerts for a specific ticker

        Args:
            ticker: Stock ticker symbol
            is_ask_side: Boolean flag whether a transaction is ask side
            is_bid_side: Boolean flag whether a transaction is bid side
            limit: How many items to return (max: 200)

        Returns:
            dict: Flow alerts data
        """
        try:
            endpoint = f"stock/{ticker}/flow-alerts"
            params = {
                "is_ask_side": str(is_ask_side).lower(),
                "is_bid_side": str(is_bid_side).lower(),
                "limit": min(limit, 200)
            }

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(
                    f"Flow alerts retrieved for {ticker}: {len(data['data'])} items")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete(f"unusual_whales:flow:{ticker}")

                    # Add each alert to the sorted set
                    for alert in data["data"]:
                        # Use created_at timestamp as score for sorting
                        created_at = alert.get("created_at", "")
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(
                                    created_at.replace("Z", "+00:00"))
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(f"unusual_whales:flow:{ticker}", {
                                      json.dumps(alert): score})

                    # Store last update timestamp
                    pipeline.set(
                        f"unusual_whales:flow:{ticker}:last_update", datetime.now().isoformat())

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_flow_alerts_gpu, ticker, data["data"])

                return data["data"]
            else:
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"
                logger.warning(
                    f"Failed to get flow alerts for {ticker}: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"Error getting flow alerts for {ticker}: {e}")
            return []

    def _process_flow_alerts_gpu(self, ticker, alerts):
        """Process flow alerts with GPU acceleration"""
        if not alerts:
            return

        try:
            # Extract relevant data
            premiums = [float(alert.get("total_premium", 0))
                        for alert in alerts]
            volumes = [int(alert.get("total_size", 0)) for alert in alerts]
            strikes = [float(alert.get("strike", 0)) for alert in alerts]

            # Use GPU for calculations
            if self.use_gpu and self.gpu_initialized:
                # Convert to CuPy arrays
                cp_premiums = cp.array(premiums, dtype=cp.float32)
                cp_volumes = cp.array(volumes, dtype=cp.float32)
                cp_strikes = cp.array(strikes, dtype=cp.float32)

                # Calculate statistics
                total_premium = cp.sum(cp_premiums)
                total_volume = cp.sum(cp_volumes)
                avg_premium = cp.mean(cp_premiums)
                avg_volume = cp.mean(cp_volumes)
                avg_strike = cp.mean(cp_strikes)

                # Convert back to numpy
                total_premium = float(cp.asnumpy(total_premium))
                total_volume = float(cp.asnumpy(total_volume))
                avg_premium = float(cp.asnumpy(avg_premium))
                avg_volume = float(cp.asnumpy(avg_volume))
                avg_strike = float(cp.asnumpy(avg_strike))
            else:
                # CPU calculations
                total_premium = sum(premiums)
                total_volume = sum(volumes)
                avg_premium = sum(premiums) / len(premiums) if premiums else 0
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                avg_strike = sum(strikes) / len(strikes) if strikes else 0

            # Count call vs put
            call_count = sum(
                1 for alert in alerts if alert.get("type") == "call")
            put_count = sum(
                1 for alert in alerts if alert.get("type") == "put")

            # Store statistics in Redis if available
            if self.redis_client:
                self.redis_client.hmset(f"unusual_whales:flow_stats:{ticker}", {
                    "total_premium": total_premium,
                    "total_volume": total_volume,
                    "avg_premium": avg_premium,
                    "avg_volume": avg_volume,
                    "avg_strike": avg_strike,
                    "call_count": call_count,
                    "put_count": put_count,
                    "alert_count": len(alerts),
                    "last_update": datetime.now().isoformat()
                })

        except Exception as e:
            logger.error(
                f"Error processing flow alerts for {ticker} on GPU: {e}")

    async def get_alerts(self, config_ids=None, intraday_only=True, limit=100,
                         noti_types=None, page=0, ticker_symbols=None):
        """
        Get all alerts that have been triggered for the user

        Args:
            config_ids: A list of alert ids to filter by
            intraday_only: Boolean flag whether to return only intraday alerts
            limit: How many items to return (max: 200)
            noti_types: A list of notification types
            page: The page number to return
            ticker_symbols: A comma separated list of tickers

        Returns:
            list: Alerts data
        """
        try:
            endpoint = "alerts"
            params = {
                "intraday_only": str(intraday_only).lower(),
                "limit": min(limit, 200),
                "page": max(page, 0)
            }

            if config_ids:
                if isinstance(config_ids, list):
                    params["config_ids[]"] = config_ids
                else:
                    params["config_ids[]"] = [config_ids]

            if noti_types:
                if isinstance(noti_types, list):
                    params["noti_types[]"] = noti_types
                else:
                    params["noti_types[]"] = [noti_types]

            if ticker_symbols:
                params["ticker_symbols"] = ticker_symbols

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(f"Alerts retrieved: {len(data['data'])} items")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data if page is 0
                    if page == 0:
                        pipeline.delete("unusual_whales:alerts")

                    # Add each alert to the sorted set
                    for alert in data["data"]:
                        # Use tape_time timestamp as score for sorting
                        tape_time = alert.get("tape_time", "")
                        if tape_time:
                            try:
                                dt = datetime.fromisoformat(
                                    tape_time.replace("Z", "+00:00"))
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd("unusual_whales:alerts", {
                                      json.dumps(alert): score})

                    # Store last update timestamp
                    pipeline.set("unusual_whales:alerts:last_update",
                                 datetime.now().isoformat())

                    # Execute pipeline
                    pipeline.execute()

                return data["data"]
            else:
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"
                logger.warning(f"Failed to get alerts: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    async def get_alert_configurations(self):
        """
        Get all alert configurations of the user

        Returns:
            list: Alert configurations
        """
        try:
            endpoint = "alerts/configuration"

            data = await self._make_request(endpoint)

            if data and "data" in data:
                logger.info(
                    f"Alert configurations retrieved: {len(data['data'])} items")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a hash in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete("unusual_whales:alert_configs")

                    # Add each config to the hash
                    for config in data["data"]:
                        config_id = config.get("id", "")
                        if config_id:
                            pipeline.hset(
                                "unusual_whales:alert_configs", config_id, json.dumps(config))

                    # Store last update timestamp
                    pipeline.set(
                        "unusual_whales:alert_configs:last_update", datetime.now().isoformat())

                    # Execute pipeline
                    pipeline.execute()

                return data["data"]
            else:
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"
                logger.warning(
                    f"Failed to get alert configurations: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"Error getting alert configurations: {e}")
            return []

    async def get_recent_dark_pool_trades(self, date=None, limit=100, max_premium=None,
                                          max_size=None, max_volume=None, min_premium=0,
                                          min_size=0, min_volume=0):
        """
        Get the latest darkpool trades

        Args:
            date: A trading date in the format of YYYY-MM-DD
            limit: How many items to return (max: 200)
            max_premium: The maximum premium requested trades should have
            max_size: The maximum size requested trades should have
            max_volume: The maximum consolidated volume requested trades should have
            min_premium: The minimum premium requested trades should have
            min_size: The minimum size requested trades should have
            min_volume: The minimum consolidated volume requested trades should have

        Returns:
            list: Dark pool trades data
        """
        try:
            endpoint = "darkpool/recent"
            params = {
                "limit": min(limit, 200),
                "min_premium": min_premium,
                "min_size": min_size,
                "min_volume": min_volume
            }

            if date:
                params["date"] = date

            if max_premium is not None:
                params["max_premium"] = max_premium

            if max_size is not None:
                params["max_size"] = max_size

            if max_volume is not None:
                params["max_volume"] = max_volume

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(
                    f"Recent dark pool trades retrieved: {len(data['data'])} items")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete("unusual_whales:darkpool:recent")

                    # Add each trade to the sorted set
                    for trade in data["data"]:
                        # Use executed_at timestamp as score for sorting
                        executed_at = trade.get("executed_at", "")
                        if executed_at:
                            try:
                                dt = datetime.fromisoformat(
                                    executed_at.replace("Z", "+00:00"))
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd("unusual_whales:darkpool:recent", {
                                      json.dumps(trade): score})

                    # Store last update timestamp
                    pipeline.set(
                        "unusual_whales:darkpool:recent:last_update", datetime.now().isoformat())

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_dark_pool_trades_gpu, data["data"])

                return data["data"]
            else:
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"
                logger.warning(
                    f"Failed to get recent dark pool trades: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"Error getting recent dark pool trades: {e}")
            return []

    def _process_dark_pool_trades_gpu(self, trades):
        """Process dark pool trades with GPU acceleration"""
        if not trades:
            return

        try:
            # Group trades by ticker
            trades_by_ticker = {}
            for trade in trades:
                ticker = trade.get("ticker")
                if ticker:
                    if ticker not in trades_by_ticker:
                        trades_by_ticker[ticker] = []
                    trades_by_ticker[ticker].append(trade)

            # Process each ticker's trades
            for ticker, ticker_trades in trades_by_ticker.items():
                # Extract relevant data
                premiums = [float(trade.get("premium", 0))
                            for trade in ticker_trades]
                sizes = [int(trade.get("size", 0)) for trade in ticker_trades]
                prices = [float(trade.get("price", 0))
                          for trade in ticker_trades]

                # Use GPU for calculations
                if self.use_gpu and self.gpu_initialized:
                    # Convert to CuPy arrays
                    cp_premiums = cp.array(premiums, dtype=cp.float32)
                    cp_sizes = cp.array(sizes, dtype=cp.float32)
                    cp_prices = cp.array(prices, dtype=cp.float32)

                    # Calculate statistics
                    total_premium = cp.sum(cp_premiums)
                    total_size = cp.sum(cp_sizes)
                    avg_premium = cp.mean(cp_premiums)
                    avg_size = cp.mean(cp_sizes)
                    avg_price = cp.mean(cp_prices)

                    # Convert back to numpy
                    total_premium = float(cp.asnumpy(total_premium))
                    total_size = float(cp.asnumpy(total_size))
                    avg_premium = float(cp.asnumpy(avg_premium))
                    avg_size = float(cp.asnumpy(avg_size))
                    avg_price = float(cp.asnumpy(avg_price))
                else:
                    # CPU calculations
                    total_premium = sum(premiums)
                    total_size = sum(sizes)
                    avg_premium = sum(premiums) / \
                        len(premiums) if premiums else 0
                    avg_size = sum(sizes) / len(sizes) if sizes else 0
                    avg_price = sum(prices) / len(prices) if prices else 0

                # Store statistics in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(f"unusual_whales:darkpool_stats:{ticker}", {
                        "total_premium": total_premium,
                        "total_size": total_size,
                        "avg_premium": avg_premium,
                        "avg_size": avg_size,
                        "avg_price": avg_price,
                        "trade_count": len(ticker_trades),
                        "last_update": datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error processing dark pool trades on GPU: {e}")

    async def get_dark_pool_trades(self, ticker, date=None, limit=500, max_premium=None,
                                   max_size=None, max_volume=None, min_premium=0,
                                   min_size=0, min_volume=0, newer_than=None, older_than=None):
        """
        Get the darkpool trades for the given ticker on a given day

        Args:
            ticker: Stock ticker symbol
            date: A trading date in the format of YYYY-MM-DD
            limit: How many items to return (max: 500)
            max_premium: The maximum premium requested trades should have
            max_size: The maximum size requested trades should have
            max_volume: The maximum consolidated volume requested trades should have
            min_premium: The minimum premium requested trades should have
            min_size: The minimum size requested trades should have
            min_volume: The minimum consolidated volume requested trades should have
            newer_than: The unix time in milliseconds or seconds
            older_than: The unix time in milliseconds or seconds

        Returns:
            list: Dark pool trades data
        """
        try:
            endpoint = f"darkpool/{ticker}"
            params = {
                "limit": min(limit, 500),
                "min_premium": min_premium,
                "min_size": min_size,
                "min_volume": min_volume
            }

            if date:
                params["date"] = date

            if max_premium is not None:
                params["max_premium"] = max_premium

            if max_size is not None:
                params["max_size"] = max_size

            if max_volume is not None:
                params["max_volume"] = max_volume

            if newer_than is not None:
                params["newer_than"] = newer_than

            if older_than is not None:
                params["older_than"] = older_than

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(
                    f"Dark pool trades retrieved for {ticker}: {len(data['data'])} items")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete(f"unusual_whales:darkpool:{ticker}")

                    # Add each trade to the sorted set
                    for trade in data["data"]:
                        # Use executed_at timestamp as score for sorting
                        executed_at = trade.get("executed_at", "")
                        if executed_at:
                            try:
                                dt = datetime.fromisoformat(
                                    executed_at.replace("Z", "+00:00"))
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(f"unusual_whales:darkpool:{ticker}", {
                                      json.dumps(trade): score})

                    # Store last update timestamp
                    pipeline.set(
                        f"unusual_whales:darkpool:{ticker}:last_update", datetime.now().isoformat())

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_dark_pool_trades_gpu, data["data"])

                return data["data"]
            else:
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"
                logger.warning(
                    f"Failed to get dark pool trades for {ticker}: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"Error getting dark pool trades for {ticker}: {e}")
            return []

    async def start_scheduled_tasks(self, event_loop=None):
        """
        Start scheduled tasks for data collection

        Args:
            event_loop: Optional event loop to use
        """
        if event_loop:
            self.event_loop = event_loop
        else:
            self.event_loop = asyncio.get_event_loop()

        # Schedule unusual activity updates
        self.scheduled_tasks['unusual_activity'] = self.event_loop.create_task(
            self._schedule_task(
                self.get_alerts, REFRESH_INTERVALS['unusual_activity'])
        )

        # Schedule dark pool updates
        self.scheduled_tasks['dark_pool'] = self.event_loop.create_task(
            self._schedule_task(self.get_recent_dark_pool_trades,
                                REFRESH_INTERVALS['dark_pool'])
        )

        logger.info("Scheduled tasks started")

    async def _schedule_task(self, task_func, interval, *args, **kwargs):
        """
        Schedule a task to run at regular intervals

        Args:
            task_func: Function to run
            interval: Interval in seconds
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
        """
        while self.running:
            try:
                await task_func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in scheduled task {task_func.__name__}: {e}")

            await asyncio.sleep(interval)

    async def schedule_ticker_updates(self, tickers, intervals=None):
        """
        Schedule updates for specific tickers

        Args:
            tickers: List of tickers to update
            intervals: Optional dictionary of intervals for different endpoints
        """
        if not intervals:
            intervals = {
                'flow': REFRESH_INTERVALS['flow'],
                'dark_pool': REFRESH_INTERVALS['dark_pool']
            }

        # Schedule flow updates
        self.scheduled_tasks['flow'] = self.event_loop.create_task(
            self._schedule_flow_updates(tickers, intervals['flow'])
        )

        # Schedule dark pool updates
        self.scheduled_tasks['dark_pool_tickers'] = self.event_loop.create_task(
            self._schedule_dark_pool_updates(tickers, intervals['dark_pool'])
        )

        logger.info(f"Scheduled updates for {len(tickers)} tickers")

    async def _schedule_flow_updates(self, tickers, interval):
        """Schedule flow updates for specific tickers"""
        while self.running:
            for ticker in tickers:
                if not self.running:
                    break

                try:
                    await self.get_flow_alerts(ticker)
                except Exception as e:
                    logger.error(
                        f"Error updating flow alerts for {ticker}: {e}")

                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.5)

            # Wait until next update cycle
            await asyncio.sleep(interval)

    async def _schedule_dark_pool_updates(self, tickers, interval):
        """Schedule dark pool updates for specific tickers"""
        while self.running:
            for ticker in tickers:
                if not self.running:
                    break

                try:
                    await self.get_dark_pool_trades(ticker)
                except Exception as e:
                    logger.error(
                        f"Error updating dark pool trades for {ticker}: {e}")

                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.5)

            # Wait until next update cycle
            await asyncio.sleep(interval)

    async def close(self):
        """Close all connections and resources"""
        logger.info("Closing Unusual Whales API client")
        self.running = False

        # Cancel all scheduled tasks
        for name, task in self.scheduled_tasks.items():
            if not task.done():
                logger.info(f"Cancelling scheduled task: {name}")
                task.cancel()

        # Close connection pool
        await self.connection_pool.close()

        # Clean up GPU resources
        if self.use_gpu and self.gpu_initialized:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Error clearing CuPy memory pool: {e}")

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

        logger.info("Unusual Whales API client closed")
