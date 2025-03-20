#!/usr/bin/env python3
"""
GPU-Optimized Polygon.io REST API Client

This module provides an optimized client for interacting with the Polygon.io REST API,
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
import random
from functools import lru_cache
from tqdm import tqdm
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
logger = logging.getLogger('gpu_polygon_api_client')

# Environment variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
# 1 hour default TTL
CACHE_TTL = int(os.environ.get('POLYGON_CACHE_TTL', 3600))
MAX_RETRIES = int(os.environ.get('POLYGON_MAX_RETRIES', 5))
RETRY_BACKOFF_FACTOR = float(os.environ.get(
    'POLYGON_RETRY_BACKOFF_FACTOR', 0.5))
# Increase default timeout to 60 seconds
CONNECTION_TIMEOUT = int(os.environ.get('POLYGON_CONNECTION_TIMEOUT', 60))
MAX_POOL_SIZE = int(os.environ.get('POLYGON_MAX_POOL_SIZE', 30))
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_USERNAME = os.environ.get('REDIS_USERNAME', 'default')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
REDIS_SSL = os.environ.get('REDIS_SSL', 'true').lower() == 'true'
# Increase Redis timeout to 30 seconds
REDIS_TIMEOUT = int(os.environ.get('REDIS_TIMEOUT', 30))
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'

# Endpoint refresh intervals (in seconds)
REFRESH_INTERVALS = {
    'market_status': 60,  # Every minute
    'ticker_details': 86400,  # Daily
    'ticker_news': 1800,  # Every 30 minutes
    'aggregates': {
        'minute': 60,  # Every minute during market hours
        'hour': 3600,  # Every hour
        'day': 86400,  # Daily
    },
    'previous_close': 86400,  # Daily
    'last_quote': 5,  # Every 5 seconds during market hours
    'last_trade': 5,  # Every 5 seconds during market hours
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
                socket_timeout=REDIS_TIMEOUT,  # Using the increased timeout value
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
        return f"polygon:{hashlib.md5(key_str.encode()).hexdigest()}"

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

                    # Deserialize using pickle instead of PyArrow
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
                ssl=False  # Polygon.io uses HTTPS but we don't need to verify
            )

            # Create session with default headers and timeout
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                headers={
                    'User-Agent': 'PolygonGPUClient/1.0',
                    'Accept': 'application/json'
                },
                # Add more specific timeouts
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout, connect=30, sock_connect=30, sock_read=30)
            )

    async def get(self, url, params=None, headers=None):
        """Make GET request with retry logic"""
        if self.session is None or self.session.closed:
            await self.initialize()

        last_error = None

        for retry in range(self.max_retries + 1):
            try:
                async with self.session.get(url, params=params, headers=headers) as response:
                    # Check if response is successful
                    if response.status == 200:
                        return await response.json()

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


class GPUPolygonAPIClient:
    """GPU-Optimized client for the Polygon.io REST API"""

    def __init__(self, api_key=POLYGON_API_KEY, redis_client=None, use_gpu=USE_GPU,
                 max_pool_size=MAX_POOL_SIZE, max_retries=MAX_RETRIES, cache_ttl=CACHE_TTL):
        """
        Initialize GPU-Optimized Polygon API client

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
                "No API key provided. Set the POLYGON_API_KEY environment variable.")
        else:
            logger.info(
                f"Initialized Polygon API client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}")

        # Initialize base URL
        self.base_url = "https://api.polygon.io"

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
            try:
                # Initialize CuPy
                self._initialize_gpu()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU: {e}")
                logger.warning("Falling back to CPU processing")

        # Flag to track if client is running
        self.running = True

        # Scheduled tasks
        self.scheduled_tasks = {}
        self.event_loop = None

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        asyncio.create_task(self.close())

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
        # Skip cache for certain endpoints that change frequently
        skip_cache = False
        if "range" in endpoint and "minute" in endpoint:
            skip_cache = True

        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check cache first
        if not skip_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data

        # Prepare request
        url = f"{self.base_url}/{endpoint}"

        # Verify API key is set and valid format
        if not self.api_key or len(self.api_key) < 10:
            logger.error(f"Invalid API key format: {self.api_key[:4]}...")
            return {"status": "ERROR", "error": "Invalid API key format"}

        # Set proper headers for Polygon API
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Make request
        try:
            data = await self.connection_pool.get(url, params=params, headers=headers)

            # Cache successful responses
            if isinstance(data, dict):
                if data.get("status") == "OK" and not skip_cache:
                    self.cache.set(cache_key, data)
                elif "results" in data and not skip_cache:
                    # Some endpoints return results without a status field
                    self.cache.set(cache_key, data)

            return data
        except Exception as e:
            logger.error(
                f"Unexpected error during API request: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"Unexpected error: {type(e).__name__}: {e}"}

    async def get_market_status(self):
        """
        Get the current market status

        Returns:
            dict: Market status information
        """
        try:
            # Make request to market status endpoint
            endpoint = "v1/marketstatus/now"
            data = await self._make_request(endpoint)

            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Market status retrieved: {data.get('market')}")

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.set("market:status", json.dumps(data))
                    self.redis_client.set(
                        "market:status:last_update", datetime.now().isoformat())

                return data
            else:
                logger.warning(
                    f"Failed to get market status: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return None

    async def get_ticker_details(self, ticker):
        """
        Get details for a specific ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict: Ticker details
        """
        try:
            endpoint = f"v3/reference/tickers/{ticker}"
            data = await self._make_request(endpoint)

            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Ticker details retrieved for {ticker}")

                # Store in Redis if available
                if self.redis_client:
                    result = data.get("results")
                    if result:
                        self.redis_client.hset(
                            f"stock:{ticker}:metadata", mapping=result)
                        self.redis_client.set(
                            f"stock:{ticker}:metadata:last_update", datetime.now().isoformat())

                return data.get("results")
            else:
                logger.warning(
                    f"Failed to get ticker details for {ticker}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting ticker details for {ticker}: {e}")
            return None

    async def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                             from_date=None, to_date=None, limit=5000, adjusted=True):
        """
        Get aggregated data for a ticker

        Args:
            ticker: Stock ticker symbol
            multiplier: The size of the timespan multiplier
            timespan: The size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: The start date (format: YYYY-MM-DD)
            to_date: The end date (format: YYYY-MM-DD)
            limit: Maximum number of results to return
            adjusted: Whether the results are adjusted for splits

        Returns:
            DataFrame: Aggregated data
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Validate ticker
        if not ticker:
            logger.warning("Empty ticker provided")
            return pd.DataFrame()

        ticker = ticker.upper()  # Ensure ticker is uppercase for consistency

        # Check cache for DataFrame
        cache_key = [
            f"aggregates:{ticker}:{multiplier}:{timespan}:{from_date}:{to_date}:{adjusted}"]
        cached_df = self.cache.get_dataframe(cache_key)
        if cached_df is not None and not cached_df.empty:
            logger.debug(f"Cache hit for aggregates: {ticker}")
            return cached_df

        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": min(limit, 50000),
                  "adjusted": str(adjusted).lower()}

        # Make request
        data = await self._make_request(endpoint, params)

        # Process response
        if data.get("status") == "OK" and "results" in data and data["results"]:
            # Convert to DataFrame
            df = pd.DataFrame(data["results"])

            # Rename columns
            column_map = {
                "v": "volume",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "t": "timestamp",
                "n": "transactions"
            }
            df = df.rename(columns=column_map)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Set timestamp as index
            df = df.set_index("timestamp")

            # Sort by timestamp
            df = df.sort_index()

            logger.info(f"Retrieved {len(df)} aggregates for {ticker}")

            # Store in Redis if available
            if self.redis_client:
                # Store in Redis hash map
                redis_key = f"stock:{ticker}:candles:{timespan}"

                # Store the most recent candles in Redis
                recent_df = df.tail(100)  # Store last 100 candles

                # Convert to dictionary for Redis
                candles_dict = {}
                for idx, row in recent_df.iterrows():
                    timestamp = int(idx.timestamp())
                    candle_data = {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "timestamp": timestamp
                    }
                    candles_dict[timestamp] = json.dumps(candle_data)

                # Store in Redis
                if candles_dict:
                    # Use pipeline for efficiency
                    pipeline = self.redis_client.pipeline()
                    pipeline.delete(redis_key)
                    pipeline.hmset(redis_key, candles_dict)
                    pipeline.set(f"{redis_key}:last_update",
                                 datetime.now().isoformat())
                    pipeline.execute()

            # Cache the DataFrame
            self.cache.store_dataframe(cache_key, df)

            return df
        elif data.get("status") == "OK" and ("results" not in data or not data["results"]):
            # No results found, but API call was successful
            logger.warning(
                f"No data found for {ticker} in the specified date range")
            return pd.DataFrame()
        else:
            # API call failed
            error_msg = data.get("error", "Unknown error") if isinstance(
                data, dict) else "Unknown error"
            logger.warning(
                f"Failed to get aggregates for {ticker}: {error_msg}")
            return pd.DataFrame()

    async def get_last_quote(self, ticker):
        """
        Get the last quote for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict: Last quote data
        """
        try:
            # Check cache first
            cache_key = f"last_quote:{ticker}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

            # Prepare endpoint
            endpoint = f"v2/last/nbbo/{ticker}"

            # Make request
            data = await self._make_request(endpoint)

            # Process response
            if data and "status" in data and data["status"] == "OK" and "results" in data:
                # Store in Redis if available
                if self.redis_client:
                    # Store the quote data
                    quote_data = {
                        "askprice": data["results"]["a"],
                        "bidprice": data["results"]["b"],
                        "asksize": data["results"]["as"],
                        "bidsize": data["results"]["bs"],
                        "timestamp": data["results"]["t"],
                        "last_update": datetime.now().isoformat()
                    }
                    self.redis_client.hset(
                        f"stock:{ticker}:last_quote", mapping=quote_data)

                # Cache the result for a short time (5 seconds)
                self.cache.set(cache_key, data, ttl=5)

                return data
            else:
                logger.warning(f"Failed to get last quote for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error getting last quote for {ticker}: {e}")
            return None

    async def get_last_trade(self, ticker):
        """
        Get the last trade for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict: Last trade data
        """
        try:
            # Check cache first
            cache_key = f"last_trade:{ticker}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data

            # Prepare endpoint
            endpoint = f"v2/last/trade/{ticker}"

            # Make request
            data = await self._make_request(endpoint)

            # Process response
            if data and "status" in data and data["status"] == "OK" and "results" in data:
                # Store in Redis if available
                if self.redis_client:
                    # Store the trade data
                    trade_data = {
                        "price": data["results"]["p"],
                        "size": data["results"]["s"],
                        "timestamp": data["results"]["t"],
                        "last_update": datetime.now().isoformat()
                    }
                    self.redis_client.hset(
                        f"stock:{ticker}:last_trade", mapping=trade_data)

                # Cache the result for a short time (5 seconds)
                self.cache.set(cache_key, data, ttl=5)

                return data
            else:
                # Fall back to getting price from aggregates
                price = await self._get_last_price_from_aggregates(ticker)
                if price:
                    return {"price": price}
                logger.warning(f"Failed to get last trade for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error getting last trade for {ticker}: {e}")
            return None

    async def _get_last_price_from_aggregates(self, ticker):
        """Get the last price from recent aggregates as a fallback"""
        try:
            # Get today's data
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)
                         ).strftime("%Y-%m-%d")

            # Try to get the most recent minute data
            aggs = await self.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_date=yesterday,
                to_date=today,
                limit=10
            )

            if not aggs.empty:
                return float(aggs['close'].iloc[-1])

            # If no minute data, try daily data
            aggs = await self.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=yesterday,
                to_date=today,
                limit=1
            )

            if not aggs.empty:
                return float(aggs['close'].iloc[-1])

            return None
        except Exception as e:
            logger.error(
                f"Error getting last price from aggregates for {ticker}: {e}")
            return None

    async def get_aggregates_batch(self, tickers, multiplier=1, timespan="minute",
                                   from_date=None, to_date=None, limit=5000, adjusted=True):
        """
        Get aggregated data for multiple tickers in parallel

        Args:
            tickers: List of stock ticker symbols
            multiplier: The size of the timespan multiplier
            timespan: The size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: The start date (format: YYYY-MM-DD)
            to_date: The end date (format: YYYY-MM-DD)
            limit: Maximum number of results to return
            adjusted: Whether the results are adjusted for splits

        Returns:
            dict: Dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Create tasks for each ticker
        tasks = []
        for ticker in tickers:
            tasks.append(self.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
                limit=limit,
                adjusted=adjusted
            ))

        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        data = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {ticker}: {result}")
                continue

            if not result.empty:
                data[ticker] = result

        return data

    async def process_data_with_gpu(self, data):
        """
        Process data using GPU acceleration

        Args:
            data: Dictionary of DataFrames, keyed by ticker

        Returns:
            dict: Dictionary of processed DataFrames
        """
        if not data:
            return {}

        if not self.use_gpu or not self.gpu_initialized:
            logger.warning(
                "GPU processing requested but GPU is not available or initialized")
            return self._process_data_with_cpu(data)

        results = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            try:
                # Convert to CuPy arrays for GPU processing
                close_prices = cp.array(df['close'].values, dtype=cp.float32)
                volumes = cp.array(df['volume'].values, dtype=cp.float32)

                # Calculate moving averages
                window_5 = 5
                window_20 = 20

                # Use optimized algorithms for better performance
                # Simple moving averages - use cumsum for faster calculation
                padded_prices = cp.pad(
                    close_prices, (window_5-1, 0), 'constant')
                cumsum = cp.cumsum(padded_prices)
                sma_5 = (cumsum[window_5:] - cumsum[:-window_5]) / window_5

                padded_prices = cp.pad(
                    close_prices, (window_20-1, 0), 'constant')
                cumsum = cp.cumsum(padded_prices)
                sma_20 = (cumsum[window_20:] - cumsum[:-window_20]) / window_20

                # Volume weighted average price
                # Use parallel reduction for better performance
                price_volume = close_prices * volumes
                total_price_volume = cp.sum(price_volume)
                total_volume = cp.sum(volumes)
                vwap = total_price_volume / total_volume if total_volume > 0 else 0

                # Calculate RSI
                delta = cp.diff(close_prices)
                gain = cp.where(delta > 0, delta, 0)
                loss = cp.where(delta < 0, -delta, 0)

                # Use exponential moving average for RSI
                avg_gain = cp.mean(gain[:14]) if len(gain) >= 14 else 0
                avg_loss = cp.mean(loss[:14]) if len(loss) >= 14 else 0

                if len(gain) >= 14:
                    for i in range(14, len(gain)):
                        avg_gain = (avg_gain * 13 + gain[i]) / 14
                        avg_loss = (avg_loss * 13 + loss[i]) / 14

                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))

                # Convert back to numpy arrays
                sma_5_np = cp.asnumpy(sma_5)
                sma_20_np = cp.asnumpy(sma_20)
                vwap_np = float(cp.asnumpy(vwap))
                rsi_np = float(cp.asnumpy(rsi))

                # Create result DataFrame
                result_df = pd.DataFrame({
                    'ticker': ticker,
                    'last_price': df['close'].iloc[-1],
                    'sma_5': sma_5_np[-1] if len(sma_5_np) > 0 else None,
                    'sma_20': sma_20_np[-1] if len(sma_20_np) > 0 else None,
                    'vwap': vwap_np,
                    'rsi': rsi_np,
                    'volume': df['volume'].sum()
                }, index=[0])

                results[ticker] = result_df

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(f"stock:{ticker}:technical", {
                        "sma_5": float(sma_5_np[-1]) if len(sma_5_np) > 0 else 0,
                        "sma_20": float(sma_20_np[-1]) if len(sma_20_np) > 0 else 0,
                        "vwap": float(vwap_np),
                        "rsi": float(rsi_np),
                        "last_update": datetime.now().isoformat()
                    })

            except Exception as e:
                logger.error(f"Error processing data for {ticker} on GPU: {e}")

        return results

    def _process_data_with_cpu(self, data):
        """
        Process data using CPU (fallback when GPU is not available)

        Args:
            data: Dictionary of DataFrames, keyed by ticker

        Returns:
            dict: Dictionary of processed DataFrames
        """
        results = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            try:
                # Calculate indicators using pandas
                sma_5 = df['close'].rolling(
                    5).mean().iloc[-1] if len(df) >= 5 else None
                sma_20 = df['close'].rolling(
                    20).mean().iloc[-1] if len(df) >= 20 else None

                # Calculate VWAP
                vwap = (df['close'] * df['volume']).sum() / \
                    df['volume'].sum() if df['volume'].sum() > 0 else 0

                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                avg_gain = gain.rolling(14).mean(
                ).iloc[-1] if len(gain) >= 14 else 0
                avg_loss = loss.rolling(14).mean(
                ).iloc[-1] if len(loss) >= 14 else 0

                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))

                # Create result DataFrame
                result_df = pd.DataFrame({
                    'ticker': ticker,
                    'last_price': df['close'].iloc[-1],
                    'sma_5': sma_5,
                    'sma_20': sma_20,
                    'vwap': vwap,
                    'rsi': rsi,
                    'volume': df['volume'].sum()
                }, index=[0])

                results[ticker] = result_df

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(f"stock:{ticker}:technical", {
                        "sma_5": float(sma_5) if sma_5 is not None else 0,
                        "sma_20": float(sma_20) if sma_20 is not None else 0,
                        "vwap": float(vwap),
                        "rsi": float(rsi),
                        "last_update": datetime.now().isoformat()
                    })

            except Exception as e:
                logger.error(f"Error processing data for {ticker} on CPU: {e}")

        return results

    async def close(self):
        """Close all connections and resources"""
        logger.info("Closing Polygon API client")
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

        logger.info("Polygon API client closed")


# Example usage
async def main():
    # Create client
    client = GPUPolygonAPIClient()

    # Get market status
    market_status = await client.get_market_status()
    print("\nMarket Status:")
    print(market_status)

    # Get ticker details
    ticker = "SPY"  # Using SPY (ETF) instead of AAPL
    ticker_details = await client.get_ticker_details(ticker)
    print(f"\nTicker Details for {ticker}:")
    print(ticker_details)

    # Get aggregates
    aggregates = await client.get_aggregates(ticker, timespan="day", limit=5)
    print(f"\nAggregates for {ticker}:")
    print(aggregates.head())

    # Process data with GPU
    # Using ETFs instead of individual stocks
    tickers = ["SPY", "QQQ", "IWM", "DIA", "XLK"]
    data = await client.get_aggregates_batch(tickers, timespan="day")
    processed = await client.process_data_with_gpu(data)
    print("\nProcessed Results:")
    for ticker, df in processed.items():
        print(f"{ticker}:")
        print(df)

    # Close client
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
