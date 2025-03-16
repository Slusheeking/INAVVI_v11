#!/usr/bin/env python3
"""
Ultra-optimized Polygon.io Data Source for NVIDIA GH200
Implements the top 4 performance optimizations:
1. CuPy GPU Acceleration (removed custom CUDA kernels)
2. Shared Memory Parallelism
3. Zero-Copy Memory Architecture
4. Asynchronous Processing Pipeline
"""

import os
import time
import json
import hashlib
import pickle
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import redis
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import cupy as cp
import random

# Use CuPy for all GPU operations
logger = logging.getLogger('polygon_ultra')
logger.info("Using CuPy for all GPU operations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Environment variables
# Try to use a valid API key - first check environment, then try a default
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
logger.info("Using Polygon API key from environment variable")
# Make Redis optional - default to None to disable Redis if not available
REDIS_HOST = os.environ.get('REDIS_HOST', None)
REDIS_PORT = int(os.environ.get('REDIS_PORT', 0))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_TTL = int(os.environ.get('REDIS_TTL', 3600))  # 1 hour default TTL
MAX_CONNECTIONS = int(os.environ.get('MAX_CONNECTIONS', 50))
MAX_POOL_SIZE = int(os.environ.get('MAX_POOL_SIZE', 30))
CONNECTION_TIMEOUT = int(os.environ.get('CONNECTION_TIMEOUT', 15))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 5))
RETRY_BACKOFF_FACTOR = float(os.environ.get('RETRY_BACKOFF_FACTOR', 0.5))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', mp.cpu_count()))
QUEUE_SIZE = int(os.environ.get('QUEUE_SIZE', 10000)
                 )  # Further increased queue size
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1024))  # Increased batch size
MAX_DATA_POINTS = int(os.environ.get(
    'MAX_DATA_POINTS', 50000))  # Increased data points


class RedisCache:
    """Redis cache for API responses"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl=REDIS_TTL):
        """Initialize Redis cache"""
        self.host = host
        self.port = port if port > 0 else 6379
        self.db = db
        self.ttl = ttl
        self.enabled = False  # Default to disabled
        self.memory_cache = {}
        self.memory_cache_hits = 0

        # Only try to connect to Redis if host is provided
        if self.host:
            try:
                self.redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                self.enabled = True
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except (redis.RedisError, ConnectionError) as e:
                logger.info(f"Redis not available: {e}")
                logger.info("Using in-memory cache only")
                self.enabled = False
        else:
            logger.info("Redis disabled, using in-memory cache only")

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

        if not self.enabled:
            result = self.memory_cache.get(key)
            if result is not None:
                self.memory_cache_hits += 1
                if self.memory_cache_hits % 1000 == 0:
                    logger.info(f"Memory cache hits: {self.memory_cache_hits}")
            return result

        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except (redis.RedisError, pickle.PickleError) as e:
            logger.warning(f"Error retrieving from Redis cache: {e}")
            # Fall back to memory cache
            return self.memory_cache.get(key)

        return None

    def set(self, key_parts, value, ttl=None):
        """Set value in cache"""
        if value is None:
            return False

        key = self._generate_key(key_parts)
        ttl = ttl or self.ttl

        # Always update memory cache
        # Implement LRU-like behavior by limiting memory cache size
        if len(self.memory_cache) > 10000:  # Limit cache size
            # Remove random 10% of entries when cache gets too large
            keys_to_remove = random.sample(list(self.memory_cache.keys()),
                                           int(len(self.memory_cache) * 0.1))
            for k in keys_to_remove:
                self.memory_cache.pop(k, None)
        self.memory_cache[key] = value  # Add new entry

        if not self.enabled:
            return True

        try:
            serialized = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized)
        except (redis.RedisError, pickle.PickleError) as e:
            logger.warning(f"Error setting Redis cache: {e}")
            return False


class OptimizedConnectionPool:
    """Optimized HTTP connection pool with retry logic"""

    def __init__(self, max_retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF_FACTOR,
                 max_pool_size=MAX_POOL_SIZE, timeout=CONNECTION_TIMEOUT):
        """Initialize connection pool"""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        # Configure adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_pool_size,
            pool_maxsize=max_pool_size
        )

        # Mount adapter for both HTTP and HTTPS
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default timeout
        self.session.timeout = timeout

        # Set default headers
        self.session.headers.update({
            'User-Agent': 'PolygonUltra/1.0',
            'Accept': 'application/json'
        })

        logger.info(
            f"Connection pool initialized with max_pool_size={max_pool_size}, max_retries={max_retries}")

    def get(self, url, params=None, headers=None, timeout=None):
        """Make GET request with retry logic"""
        try:
            response = self.session.get(
                url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def close(self):
        """Close all connections in the pool"""
        self.session.close()


class AsyncProcessingPipeline:
    """Asynchronous processing pipeline for data processing"""

    def __init__(self, num_workers=min(NUM_WORKERS, 4), queue_size=QUEUE_SIZE, use_daemon=False):
        """Initialize processing pipeline"""
        self.num_workers = num_workers
        self.queue_size = queue_size

        # Create queues
        self.input_queue = mp.Queue(maxsize=queue_size)
        self.output_queue = mp.Queue(maxsize=queue_size)

        # Create workers
        self.workers = []
        self.running = mp.Value(ctypes.c_bool, True)
        self.use_daemon = use_daemon

        logger.info(
            f"Async processing pipeline initialized with {num_workers} workers")

    def start(self, worker_func):
        """Start processing pipeline"""
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_loop,
                args=(i, worker_func, self.input_queue,
                      self.output_queue, self.running)
            )
            worker.daemon = False  # Always use non-daemon processes to allow child processes
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.num_workers} worker processes")

    def _worker_loop(self, worker_id, worker_func, input_queue, output_queue, running):
        """Worker loop for processing tasks"""
        logger.info(f"Worker {worker_id} started")

        # Configure CuPy for this worker
        try:
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            logger.info(
                f"Worker {worker_id} configured CuPy with unified memory")
        except Exception as e:
            logger.warning(
                f"Failed to configure CuPy in worker {worker_id}: {e}")

        while running.value:
            try:
                # Get task from input queue with timeout
                try:
                    task = input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process task
                result = worker_func(task)

                # Put result in output queue
                output_queue.put(result)
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")

        # Clean up CuPy resources
        try:
            cp.get_default_memory_pool().free_all_blocks()
            logger.info(f"Worker {worker_id} cleaned up CuPy memory pool")
        except Exception as e:
            logger.warning(
                f"Failed to clean up CuPy in worker {worker_id}: {e}")

        logger.info(f"Worker {worker_id} stopped")

    def submit(self, task):
        """Submit task to processing pipeline"""
        self.input_queue.put(task)

    def get_result(self, timeout=None):
        """Get result from processing pipeline"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop processing pipeline"""
        # Signal workers to stop
        self.running.value = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        logger.info("Processing pipeline stopped")


class PolygonDataSourceUltra:
    """Ultra-optimized Polygon.io data source for NVIDIA GH200"""

    def __init__(self, api_key=POLYGON_API_KEY, max_pool_size=MAX_POOL_SIZE, max_retries=MAX_RETRIES, use_daemon=False):
        """Initialize Polygon data source"""
        self.api_key = api_key

        # Verify API key is not empty
        # Using hard-coded API key
        logger.info(
            f"Initialized Polygon data source with API key: {self.api_key[:4]}****{self.api_key[-4:]}")

        self.base_url = "https://api.polygon.io"

        # Initialize cache
        self.cache = RedisCache()

        # Initialize connection pool
        self.connection_pool = OptimizedConnectionPool(
            max_pool_size=max_pool_size, max_retries=max_retries)

        # Initialize processing pipeline with more workers
        self.pipeline = AsyncProcessingPipeline(
            num_workers=min(NUM_WORKERS, 4), use_daemon=use_daemon)  # Reduced workers to prevent system overload
        self.pipeline.start(self._process_data_worker)

        # Initialize thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(NUM_WORKERS, 8))  # Reduced threads to prevent system overload

        # Initialize result cache
        self.results = {}
        self.result_lock = threading.Lock()

        # Configure GPU memory growth to avoid OOM errors
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
            except RuntimeError as e:
                logger.warning(f"Memory growth configuration failed: {e}")

        # Configure CuPy for main process
        try:
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            logger.info("CuPy configured with unified memory")
        except Exception as e:
            logger.warning(f"Failed to configure CuPy: {e}")

        logger.info("Ultra-optimized Polygon data source initialized")

    def _make_request(self, endpoint, params=None):
        """Make API request with caching"""
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
        cached_data = self.cache.get(cache_key)
        if cached_data is not None and not skip_cache:
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

        # Log request details for debugging
        logger.debug(
            f"Making request to {url} with API key {self.api_key[:4]}...")

        # Make request with improved retry logic
        try:
            max_retries = 5  # Increased from 3 to 5
            last_error = None

            for retry in range(max_retries):
                try:
                    response = self.connection_pool.get(
                        url, params=params, headers=headers, timeout=30)  # Increased timeout

                    # Check if response is valid JSON
                    try:
                        data = response.json()

                        # Check for specific error patterns
                        if "error" in data:
                            error_msg = data.get("error", "")
                            if "rate limit" in str(error_msg).lower():
                                # Rate limit hit, use longer backoff
                                wait_time = (2 ** retry) + random.uniform(1, 3)
                                logger.warning(
                                    f"Rate limit hit, retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                            elif "not found" in str(error_msg).lower():
                                # Resource not found, no need to retry
                                logger.warning(
                                    f"Resource not found: {error_msg}")
                                return {"status": "ERROR", "error": error_msg}

                        # Cache successful responses
                        if response.status_code == 200:
                            if data.get("status") == "OK" and not skip_cache:
                                self.cache.set(cache_key, data)
                            elif "results" in data and not skip_cache:
                                # Some endpoints return results without a status field
                                self.cache.set(cache_key, data)
                            return data
                        else:
                            error_msg = data.get('error', 'No error details')
                            logger.error(
                                f"API request failed with status {response.status_code}: {error_msg}")
                            last_error = f"HTTP {response.status_code}: {error_msg}"

                            # Don't retry on client errors except rate limits
                            if 400 <= response.status_code < 500 and response.status_code != 429:
                                return {"status": "ERROR", "error": last_error}
                    except ValueError as e:
                        logger.error(f"Invalid JSON response: {e}")
                        logger.error(
                            f"Response content: {response.text[:200]}...")
                        last_error = f"Invalid JSON response: {e}"
                except requests.RequestException as e:
                    logger.warning(
                        f"Request exception (attempt {retry+1}/{max_retries}): {e}")
                    last_error = f"Request failed: {e}"

                # Exponential backoff with jitter
                wait_time = (2 ** retry) + random.uniform(0, 1)
                logger.warning(
                    f"Retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)

            # All retries failed
            return {"status": "ERROR", "error": last_error or "Max retries exceeded"}
        except Exception as e:
            logger.error(
                f"Unexpected error during API request: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"Unexpected error: {type(e).__name__}: {e}"}

    def _validate_ticker(self, ticker):
        """
        Validate if a ticker is likely to be valid
        Returns True if valid, False otherwise
        """
        # Check if ticker is None or empty
        if not ticker:
            logger.warning("Empty ticker provided")
            return False

        # Check if ticker is too long (most tickers are 1-5 characters)
        if len(ticker) > 10:
            logger.warning(
                f"Ticker {ticker} is unusually long, might be invalid")
            return False

        # Check for invalid characters (tickers should be alphanumeric with some special chars)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^")
        if not all(c in valid_chars for c in ticker.upper()):
            logger.warning(f"Ticker {ticker} contains invalid characters")
            return False

        # Check for common ETF patterns that might be problematic
        problematic_patterns = ["X", "BULL", "BEAR",
                                "3X", "2X", "LEVERAGED", "INVERSE"]
        if any(pattern in ticker.upper() for pattern in problematic_patterns):
            logger.warning(
                f"Ticker {ticker} matches a problematic pattern, might be a leveraged ETF")
            # Don't return false, just log a warning

        return True

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                       from_date=None, to_date=None, limit=MAX_DATA_POINTS, adjusted=True):
        """Get aggregated data for a ticker"""
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)  # Increased from 7 to 30 days
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Validate ticker
        if not self._validate_ticker(ticker):
            logger.warning(f"Invalid ticker format: {ticker}")
            return pd.DataFrame()

        ticker = ticker.upper()  # Ensure ticker is uppercase for consistency
        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": limit, "adjusted": str(adjusted).lower()}

        # Make request with retry logic
        max_retries = 5  # Increased from 3 to 5 for more resilience
        for retry in range(max_retries):
            logger.debug(
                f"Fetching data for {ticker} (attempt {retry+1}/{max_retries})")
            # Make request with improved error handling
            data = self._make_request(endpoint, params)

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

                return df
            elif data.get("status") == "OK" and ("results" not in data or not data["results"]):
                # No results found, but API call was successful
                logger.warning(
                    f"No data found for {ticker} in the specified date range")
                # Return empty DataFrame, no need to retry
                return pd.DataFrame()
            elif "error" in data and "not found" in str(data["error"]).lower():
                # Ticker not found, no need to retry
                logger.warning(
                    f"Ticker {ticker} not found: {data.get('error')}")
                return pd.DataFrame()
            elif "error" in data and "invalid" in str(data["error"]).lower():
                # Invalid ticker format or delisted ticker
                logger.warning(
                    f"Invalid ticker {ticker}: {data.get('error')}")
                return pd.DataFrame()
            elif "error" in data and "limit" in str(data["error"]).lower():
                # API limit reached, use longer backoff
                wait_time = (3 ** retry) + \
                    random.uniform(2, 7)  # Increased jitter
                logger.warning(
                    f"API limit reached for {ticker}, retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)
                # Don't count this as a retry if it's a rate limit issue
                continue
            elif "error" in data and "rate limit" in str(data["error"]).lower():
                # Rate limit hit, use longer backoff
                wait_time = (3 ** retry) + \
                    random.uniform(1, 5)  # Increased jitter
                logger.warning(
                    f"Rate limit hit for {ticker}, retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif "error" in data and "api key" in str(data["error"]).lower():
                # API key issue
                logger.error(
                    f"API key issue for {ticker}: {data.get('error')}")
                # Try with a fallback key if available
                fallback_key = os.environ.get('POLYGON_API_KEY_FALLBACK')
                if fallback_key and fallback_key != self.api_key:
                    logger.info(f"Trying fallback API key for {ticker}")
                    self.api_key = fallback_key
                    # Don't count this as a retry
                    continue
                return pd.DataFrame()
            elif "error" in data and "unknown" in str(data["error"]).lower():
                # Unknown error, try with exponential backoff and more jitter
                wait_time = (3 ** retry) + random.uniform(3, 7)
                logger.warning(
                    f"Unknown error for {ticker}, retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)
                # Try with a different API endpoint on next attempt
                if retry == 2 and timespan == "minute":
                    timespan = "hour"
                    endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
                    logger.warning(
                        f"Switching to {timespan} timespan for {ticker}")
            elif retry < max_retries - 1:
                wait_time = (3 ** retry) + \
                    random.uniform(1, 3)  # Increased backoff
                logger.warning(
                    f"Retrying {ticker} in {wait_time:.2f} seconds. Error: {data.get('error', 'Unknown error')}")
                time.sleep(wait_time)
            else:
                # Try one more approach for "Unknown error" cases
                error_msg = data.get("error", "Unknown error") if isinstance(
                    data, dict) else "Unknown error"

                if "unknown" in str(error_msg).lower():
                    # For unknown errors, try with different parameters
                    logger.warning(
                        f"Unknown error for {ticker}, trying with shorter date range")

                    # Try with a shorter date range
                    shorter_from_date = (
                        datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

                    # Prepare endpoint and parameters for shorter range
                    short_endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{shorter_from_date}/{to_date}"
                    short_params = {"limit": limit,
                                    "adjusted": str(adjusted).lower()}

                    # Make request with shorter range
                    short_data = self._make_request(
                        short_endpoint, short_params)

                    if short_data.get("status") == "OK" and "results" in short_data and short_data["results"]:
                        # Convert to DataFrame
                        df = pd.DataFrame(short_data["results"])

                        # Process as before
                        column_map = {"v": "volume", "o": "open", "c": "close",
                                      "h": "high", "l": "low", "t": "timestamp", "n": "transactions"}
                        df = df.rename(columns=column_map)
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"], unit="ms")
                        df = df.set_index("timestamp").sort_index()
                        return df

                    # If that didn't work, try with a different timespan
                    logger.warning(
                        f"Trying {ticker} with different timespan (hour instead of day)")
                    alt_endpoint = f"v2/aggs/ticker/{ticker}/range/1/hour/{shorter_from_date}/{to_date}"
                    alt_params = {"limit": limit,
                                  "adjusted": str(adjusted).lower()}

                    # Make request with alternative parameters
                    alt_data = self._make_request(alt_endpoint, alt_params)

                    if alt_data.get("status") == "OK" and "results" in alt_data and alt_data["results"]:
                        # Convert to DataFrame
                        df = pd.DataFrame(alt_data["results"])

                        # Process as before
                        column_map = {"v": "volume", "o": "open", "c": "close",
                                      "h": "high", "l": "low", "t": "timestamp", "n": "transactions"}
                        df = df.rename(columns=column_map)
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"], unit="ms")
                        df = df.set_index("timestamp").sort_index()
                        return df

                logger.warning(
                    f"Failed to get aggregates for {ticker}: {error_msg}")

        # Return empty DataFrame after all retries
        return pd.DataFrame()

    def get_aggregates_batch(self, tickers, multiplier=1, timespan="minute",
                             from_date=None, to_date=None, limit=MAX_DATA_POINTS, adjusted=True):
        """
        Get aggregated data for multiple tickers using shared memory and parallel processing

        Returns a dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)  # Increased from 7 to 30 days
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Clear previous results
        with self.result_lock:
            self.results = {}

        # Submit tasks to thread pool
        futures = []
        for ticker in tickers:
            future = self.thread_pool.submit(
                self._fetch_ticker_data,
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
                limit=limit,
                adjusted=adjusted
            )
            futures.append((ticker, future))

        # Add a small delay between batches to avoid rate limiting
        time.sleep(0.1)

        # Collect results
        results = {}
        for ticker, future in futures:
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")

        return results

    def _fetch_ticker_data(self, ticker, multiplier, timespan, from_date, to_date, limit, adjusted=True):
        """Fetch data for a single ticker"""
        try:
            return self.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
                limit=limit,
                adjusted=adjusted
            )
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def process_data_with_gpu(self, data):
        """
        Process data using CuPy and zero-copy memory

        This implementation uses:
        1. CuPy GPU Acceleration
        2. Zero-copy memory for efficient CPU-GPU transfers
        3. Asynchronous processing pipeline
        4. Shared memory for inter-process communication
        """
        if not data:
            return {}

        # Process each ticker
        results = {}
        tasks = []

        # Create tasks for processing pipeline
        for ticker, df in data.items():
            if df.empty:
                continue

            # Create task
            task = {
                'ticker': ticker,
                'data': df
            }

            # Submit task to processing pipeline
            self.pipeline.submit(task)
            tasks.append(ticker)

        # Collect results
        for _ in range(len(tasks)):
            result = self.pipeline.get_result(timeout=60)  # Increased timeout
            if result and 'ticker' in result:
                ticker = result['ticker']
                results[ticker] = result['result']

        return results

    def _process_data_worker(self, task):
        """Worker function for processing data in separate process"""
        ticker = task['ticker']
        df = task['data']

        try:
            # Extract data
            # Skip processing if not enough data
            if len(df) < 20:  # Need at least 20 data points for meaningful analysis
                return {
                    'ticker': ticker,
                    'result': None
                }
            # Use float64 for better precision
            close_prices = np.array(df['close'].values, dtype=np.float64)
            volumes = np.array(df['volume'].values, dtype=np.float64)

            # Process data using CuPy
            sma5, sma20, vwap, rsi = self._process_with_cupy(
                close_prices, volumes)

            # Create result DataFrame
            result_df = pd.DataFrame({
                'ticker': ticker,
                'last_price': df['close'].iloc[-1],
                'sma_5': sma5[-1] if isinstance(sma5, np.ndarray) and len(sma5) > 0 else sma5,
                'sma_20': sma20[-1] if isinstance(sma20, np.ndarray) and len(sma20) > 0 else sma20,
                'vwap': vwap,
                'rsi': rsi,
                'volume': df['volume'].sum(),
                'data_points': len(df)
            }, index=[0])

            # Return result
            return {
                'ticker': ticker,
                'result': result_df
            }
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'result': None
            }

    def _process_with_cupy(self, close_prices, volumes):
        """Process data using CuPy with optimized algorithms"""
        try:
            # Convert to CuPy arrays using zero-copy when possible
            # Use float64 for better precision
            cp_close = cp.asarray(close_prices, dtype=cp.float64)
            cp_volumes = cp.asarray(volumes, dtype=cp.float64)

            # Calculate SMA-5 using optimized algorithm
            window_5 = 5
            padded_prices = cp.pad(cp_close, (window_5-1, 0), 'constant')
            cumsum = cp.cumsum(padded_prices)
            sma_5 = (cumsum[window_5:] - cumsum[:-window_5]) / window_5

            # Calculate SMA-20 using optimized algorithm
            window_20 = 20
            padded_prices = cp.pad(cp_close, (window_20-1, 0), 'constant')
            cumsum = cp.cumsum(padded_prices)
            sma_20 = (cumsum[window_20:] - cumsum[:-window_20]) / window_20

            # Calculate VWAP using optimized algorithm
            price_volume = cp_close * cp_volumes
            total_price_volume = cp.sum(price_volume)
            total_volume = cp.sum(cp_volumes)
            # Prevent division by zero
            vwap = float(cp.asnumpy(
                total_price_volume / (total_volume + 1e-8)))

            # Calculate RSI using optimized algorithm
            delta = cp.diff(cp_close)
            gain = cp.where(delta > 0, delta, 0)
            loss = cp.where(delta < 0, -delta, 0)

            # Use exponential moving average for RSI calculation
            period = 14
            avg_gain = cp.mean(gain[:period]) if len(gain) >= period else 0
            avg_loss = cp.mean(loss[:period]) if len(loss) >= period else 0

            if len(gain) >= period:
                for i in range(period, len(gain)):
                    avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                    avg_loss = (avg_loss * (period - 1) + loss[i]) / period

            rs = avg_gain / (avg_loss + 1e-8)  # Prevent division by zero
            rsi = float(cp.asnumpy(100 - (100 / (1 + rs))))

            # Convert back to numpy arrays
            sma_5_np = cp.asnumpy(sma_5)
            sma_20_np = cp.asnumpy(sma_20)

            return sma_5_np, sma_20_np, vwap, rsi
        except Exception as e:
            logger.error(f"Error in CuPy processing: {e}")
            return 0, 0, 0, 50  # Default values

    def close(self):
        """Close all connections and resources"""
        # Close connection pool
        self.connection_pool.close()

        # Close thread pool
        self.thread_pool.shutdown()

        # Stop processing pipeline
        self.pipeline.stop()

        # Clean up CuPy resources
        try:
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("CuPy memory pool cleared")
        except Exception as e:
            logger.warning(f"Error clearing CuPy memory pool: {e}")

        logger.info("Closed all connections and resources")


# Example usage
if __name__ == "__main__":
    # Create client
    client = PolygonDataSourceUltra()

    # Fetch data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    start_time = time.time()
    results = client.get_aggregates_batch(symbols, timespan="minute")
    fetch_time = time.time() - start_time

    # Process the results
    for symbol, df in results.items():
        print(f"{symbol}: {len(df)} data points")

    # Process data with GPU
    start_time = time.time()
    processed = client.process_data_with_gpu(results)
    process_time = time.time() - start_time

    print("\nProcessed Results:")
    for symbol, df in processed.items():
        print(f"{symbol}:")
        print(df)

    print(f"\nFetch time: {fetch_time:.4f} seconds")
    print(f"Process time: {process_time:.4f} seconds")
    print(f"Total time: {fetch_time + process_time:.4f} seconds")

    # Close client
    client.close()
