#!/usr/bin/env python3
"""
Data Ingestion API Clients

This module provides direct API clients for Polygon.io and Unusual Whales APIs,
optimized for high-performance data ingestion in the TensorFlow GH200 environment.
"""

import os
import time
import json
import hashlib
import pickle
import logging
import asyncio
import psutil
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
import signal
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_ingestion_api_clients')

# Import environment variables
try:
    from load_env import (
        POLYGON_API_KEY, UNUSUAL_WHALES_API_KEY,
        REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL,
        MAX_CONNECTIONS, MAX_POOL_SIZE, CONNECTION_TIMEOUT,
        MAX_RETRIES, RETRY_BACKOFF_FACTOR, NUM_WORKERS,
        QUEUE_SIZE, BATCH_SIZE, MAX_DATA_POINTS, DEFAULT_WATCHLIST
    )
    logger.info("Loaded environment variables from load_env.py")
except ImportError:
    logger.warning("Failed to import load_env.py, using default environment variables")
    # Default values from environment variables
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf')
    UNUSUAL_WHALES_API_KEY = os.environ.get('UNUSUAL_WHALES_API_KEY', '4ad71b9e-7ace-4f24-bdfc-532ace219a18')
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_TTL = int(os.environ.get('REDIS_TTL', 3600))
    MAX_CONNECTIONS = int(os.environ.get('MAX_CONNECTIONS', 50))
    MAX_POOL_SIZE = int(os.environ.get('MAX_POOL_SIZE', 30))
    CONNECTION_TIMEOUT = int(os.environ.get('CONNECTION_TIMEOUT', 15))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 5))
    RETRY_BACKOFF_FACTOR = float(os.environ.get('RETRY_BACKOFF_FACTOR', 0.5))
    NUM_WORKERS = int(os.environ.get('NUM_WORKERS', mp.cpu_count()))
    QUEUE_SIZE = int(os.environ.get('QUEUE_SIZE', 10000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1024))
    MAX_DATA_POINTS = int(os.environ.get('MAX_DATA_POINTS', 100000))  # Increased to handle more data
    DEFAULT_WATCHLIST = os.environ.get('DEFAULT_WATCHLIST', 'AAPL,MSFT,GOOGL,AMZN,TSLA').split(',')


class RedisCache:
    """Redis cache for API responses"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl=REDIS_TTL):
        """Initialize Redis cache"""
        self.host = host
        self.port = port
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
                    socket_connect_timeout=5,
                    decode_responses=True
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
                return pickle.loads(data.encode())
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


def log_memory_usage(location_tag):
    """Log CPU and GPU memory usage"""
    try:
        # Log CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 * 1024)
        
        # Log GPU memory
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free, total = mem_info[0], mem_info[1]
            used = total - free
            
            logger.info(f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
        except Exception:
            # If CuPy fails, try TensorFlow
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Available")
            else:
                logger.info(f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Not available")
    except Exception as e:
        logger.error(f"Failed to log memory usage at {location_tag}: {e}")


def configure_cupy():
    """Configure CuPy for optimal performance on GH200"""
    try:
        if cp is not None and hasattr(cp, 'cuda'):
            # Check if CUDA is available
            try:
                cp.cuda.runtime.getDeviceCount()
            except Exception as e:
                logger.warning(f"CUDA not available: {e}")
                return False
                
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            logger.info("CuPy configured with unified memory")
            return True
        else:
            logger.warning("CuPy or CUDA not available")
            return False
    except Exception as e:
        logger.error(f"Failed to configure CuPy: {e}")
        return False


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
        self.exit_event = mp.Event()
        self.use_daemon = use_daemon

        logger.info(
            f"Async processing pipeline initialized with {num_workers} workers")

    def start(self, worker_func):
        """Start processing pipeline"""
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_loop,
                args=(i, worker_func, self.input_queue,
                      self.output_queue, self.running, self.exit_event)
            )
            worker.daemon = False  # Always use non-daemon processes to allow child processes
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.num_workers} worker processes")

    def _worker_loop(self, worker_id, worker_func, input_queue, output_queue, running, exit_event):
        """Worker loop for processing tasks"""
        logger.info(f"Worker {worker_id} started")

        # Configure CuPy for this worker
        try:
            cupy_configured = configure_cupy()
        except Exception as e:
            logger.warning(f"Failed to configure CuPy in worker {worker_id}: {e}")
            cupy_configured = False
            
        if cupy_configured:
            logger.info(f"Worker {worker_id} configured CuPy successfully")

        # Set up signal handlers for graceful shutdown
        def handle_signal(signum, frame):
            logger.info(f"Worker {worker_id} received signal {signum}, shutting down gracefully")
            exit_event.set()

        # Register signal handlers
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while running.value and not exit_event.is_set():
            try:
                # Get task from input queue with timeout
                try:
                    # Use shorter timeout to check exit_event more frequently
                    task = input_queue.get(timeout=0.5)
                except KeyboardInterrupt:
                    logger.info(f"Worker {worker_id} received KeyboardInterrupt, exiting")
                    break
                except queue.Empty:
                    continue

                # Process task
                log_memory_usage(f"worker_{worker_id}_before_processing")
                result = worker_func(task)
                log_memory_usage(f"worker_{worker_id}_after_processing")

                # Put result in output queue
                output_queue.put(result)
            except KeyboardInterrupt:
                logger.info(f"Worker {worker_id} received KeyboardInterrupt, exiting")
                break
            except Exception as e:
                if exit_event.is_set():
                    break
                logger.error(f"Error in worker {worker_id}: {e}")

        # Clean up CuPy resources
        try:
            if cp is not None and hasattr(cp, 'get_default_memory_pool'):
                try:
                    # First try to get the memory pool
                    mem_pool = cp.get_default_memory_pool()
                    if mem_pool is not None:
                        # Then try to free all blocks
                        mem_pool.free_all_blocks()
                        logger.info(f"Worker {worker_id} cleaned up CuPy memory pool")
                except Exception as e:
                    logger.warning(f"Failed to clean up CuPy memory pool in worker {worker_id}: {e}")
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
        self.exit_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        logger.info("Processing pipeline stopped")


class PolygonDataClient:
    """High-performance client for the Polygon.io API"""

    def __init__(self, api_key=POLYGON_API_KEY, max_pool_size=MAX_POOL_SIZE, max_retries=MAX_RETRIES):
        """Initialize Polygon data client"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Verify API key is not empty
        if not self.api_key:
            logger.warning("No API key provided. Set the POLYGON_API_KEY environment variable.")
        else:
            logger.info(f"Initialized Polygon client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}")

        # Initialize cache
        self.cache = RedisCache()

        # Initialize connection pool
        self.connection_pool = OptimizedConnectionPool(
            max_pool_size=max_pool_size, max_retries=max_retries)

        # Initialize processing pipeline
        self.pipeline = AsyncProcessingPipeline(num_workers=min(NUM_WORKERS, 4))
        self.pipeline.start(self._process_data_worker)

        # Initialize thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 8))

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
        configure_cupy()
        log_memory_usage("initialization")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Flag to track if client is running
        self.running = True

        logger.info("Polygon data client initialized")

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        self.close()

    def _make_request(self, endpoint, params=None):
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

        # Log request details for debugging
        logger.debug(f"Making request to {url} with API key {self.api_key[:4]}...")

        # Make request with improved retry logic
        try:
            max_retries = 5
            last_error = None

            for retry in range(max_retries):
                try:
                    response = self.connection_pool.get(
                        url, params=params, headers=headers, timeout=30)

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
                                logger.warning(f"Resource not found: {error_msg}")
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
                        logger.error(f"Response content: {response.text[:200]}...")
                        last_error = f"Invalid JSON response: {e}"
                except requests.RequestException as e:
                    logger.warning(f"Request exception (attempt {retry+1}/{max_retries}): {e}")
                    last_error = f"Request failed: {e}"

                # Exponential backoff with jitter
                wait_time = (2 ** retry) + random.uniform(0, 1)
                logger.warning(f"Retrying in {wait_time:.2f}s (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)

            # All retries failed
            return {"status": "ERROR", "error": last_error or "Max retries exceeded"}
        except Exception as e:
            logger.error(f"Unexpected error during API request: {type(e).__name__}: {e}")
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
            logger.warning(f"Ticker {ticker} is unusually long, might be invalid")
            return False

        # Check for invalid characters (tickers should be alphanumeric with some special chars)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^")
        if not all(c in valid_chars for c in ticker.upper()):
            logger.warning(f"Ticker {ticker} contains invalid characters")
            return False

        # Check for common ETF patterns that might be problematic
        problematic_patterns = ["X", "BULL", "BEAR", "3X", "2X", "LEVERAGED", "INVERSE"]
        if any(pattern in ticker.upper() for pattern in problematic_patterns):
            logger.warning(f"Ticker {ticker} matches a problematic pattern, might be a leveraged ETF")
            # Don't return false, just log a warning

        return True

    def get_market_status(self):
        """
        Get the current market status
        
        Returns:
            dict: Market status information
        """
        try:
            # Make request to market status endpoint
            endpoint = "v1/marketstatus/now"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Market status retrieved: {data.get('market')}")
                return data
            else:
                logger.warning(f"Failed to get market status: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return None

    def get_ticker_details(self, ticker):
        """
        Get details for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Ticker details
        """
        try:
            endpoint = f"v3/reference/tickers/{ticker}"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Ticker details retrieved for {ticker}")
                return data.get("results")
            else:
                logger.warning(f"Failed to get ticker details for {ticker}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting ticker details for {ticker}: {e}")
            return None

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                      from_date=None, to_date=None, limit=MAX_DATA_POINTS, adjusted=True):
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
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Validate ticker
        if not self._validate_ticker(ticker):
            logger.warning(f"Invalid ticker format: {ticker}")
            return pd.DataFrame()

        ticker = ticker.upper()  # Ensure ticker is uppercase for consistency
        
        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": min(limit, 100000), "adjusted": str(adjusted).lower()}
        
        # For large date ranges, we may need to make multiple requests to get all data
        all_results = []
        
        # Make initial request
        data = self._make_request(endpoint, params)
        
        # Process response
        if data.get("status") == "OK" and "results" in data and data["results"]:
            all_results.extend(data["results"])
            
            # Check if we need to paginate (if we got exactly the limit number of results)
            if len(data["results"]) == params["limit"]:
                logger.info(f"Retrieved maximum number of results ({params['limit']}) for {ticker}, may need pagination")
                
                # If we have next_url in the response, we could use it for pagination
                # For now, we'll just log a warning
                logger.warning(f"Data for {ticker} may be incomplete due to API limits")
        elif data.get("status") == "OK" and ("results" not in data or not data["results"]):
            # No results found, but API call was successful
            logger.warning(f"No data found for {ticker} in the specified date range")
            return pd.DataFrame()
        else:
            # API call failed
            error_msg = data.get("error", "Unknown error") if isinstance(data, dict) else "Unknown error"
            logger.warning(f"Failed to get aggregates for {ticker}: {error_msg}")
            return pd.DataFrame()

        # Convert to DataFrame
        if all_results:
            df = pd.DataFrame(all_results)

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

            logger.info(f"Retrieved {len(df)} aggregates for {ticker} from {from_date} to {to_date}")
            return df
        else:
            return pd.DataFrame()

    def get_aggregates_batch(self, tickers, multiplier=1, timespan="minute",
                            from_date=None, to_date=None, limit=MAX_DATA_POINTS, adjusted=True):
        """
        Get aggregated data for multiple tickers using shared memory and parallel processing
        
        Returns a dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

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

    def get_previous_close(self, ticker):
        """
        Get the previous day's close for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Previous close data
        """
        try:
            endpoint = f"v2/aggs/ticker/{ticker}/prev"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Previous close retrieved for {ticker}")
                return data.get("results")[0] if data.get("results") else None
            else:
                logger.warning(f"Failed to get previous close for {ticker}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting previous close for {ticker}: {e}")
            return None

    def get_last_quote(self, ticker):
        """
        Get the last quote for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Last quote data
        """
        try:
            endpoint = f"v2/last/nbbo/{ticker}"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Last quote retrieved for {ticker}")
                return data.get("results")
            else:
                logger.warning(f"Failed to get last quote for {ticker}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting last quote for {ticker}: {e}")
            return None

    def get_last_trade(self, ticker):
        """
        Get the last trade for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Last trade data
        """
        try:
            endpoint = f"v2/last/trade/{ticker}"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Last trade retrieved for {ticker}")
                return data.get("results")
            else:
                logger.warning(f"Failed to get last trade for {ticker}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting last trade for {ticker}: {e}")
            return None

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
        log_memory_usage("before_process_data_with_gpu")

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

        log_memory_usage("after_process_data_with_gpu")

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

            log_memory_usage(f"before_process_with_cupy_{ticker}")
            # Process data using CuPy
            sma5, sma20, vwap, rsi = self._process_with_cupy(
                close_prices, volumes)
            log_memory_usage(f"after_process_with_cupy_{ticker}")

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

            # Log GPU memory usage before calculations
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                free, total = mem_info[0], mem_info[1]
                used = total - free
                logger.debug(f"GPU Memory before calculations: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")

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

            # Log GPU memory usage after calculations
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                free, total = mem_info[0], mem_info[1]
                used = total - free
                logger.debug(f"GPU Memory after calculations: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")

            # Convert back to numpy arrays
            sma_5_np = cp.asnumpy(sma_5)
            sma_20_np = cp.asnumpy(sma_20)

            # Free GPU memory explicitly
            del cp_close, cp_volumes, padded_prices, cumsum, sma_5, sma_20, price_volume, total_price_volume, total_volume, delta, gain, loss
            cp.get_default_memory_pool().free_all_blocks()

            return sma_5_np, sma_20_np, vwap, rsi
        except Exception as e:
            logger.error(f"Error in CuPy processing: {e}")
            return 0, 0, 0, 50  # Default values

    def close(self):
        """Close all connections and resources"""
        log_memory_usage("before_close")

        # Close connection pool
        self.connection_pool.close()

        # Close thread pool
        self.thread_pool.shutdown()

        # Stop processing pipeline
        self.pipeline.stop()

        # Clean up CuPy resources
        try:
            if cp is not None and hasattr(cp, 'get_default_memory_pool'):
                try:
                    # First try to get the memory pool
                    mem_pool = cp.get_default_memory_pool()
                    if mem_pool is not None:
                        # Then try to free all blocks
                        mem_pool.free_all_blocks()
                        logger.info("CuPy memory pool cleared")
                except Exception as e:
                    logger.warning(f"Error clearing CuPy memory pool: {e}")
        except Exception as e:
            logger.warning(f"Error clearing CuPy memory pool: {e}")

        log_memory_usage("after_close")

        logger.info("Closed all connections and resources")


class UnusualWhalesClient:
    """Client for the Unusual Whales API"""
    
    def __init__(self, api_key=UNUSUAL_WHALES_API_KEY, max_retries=3, 
                 retry_delay=1.0, timeout=30):
        """
        Initialize the Unusual Whales API client
        
        Args:
            api_key: API key for authentication
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com/api"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Verify API key is provided
        if not self.api_key:
            logger.warning("No API key provided. Set the UNUSUAL_WHALES_API_KEY environment variable.")
        else:
            logger.info(f"Initialized Unusual Whales client with API key: {self.api_key[:4]}****{self.api_key[-4:]}")
            
        # Set up session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'UnusualWhalesClient/1.0'
        })

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Flag to track if client is running
        self.running = True

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        self.close()

    def _make_request(self, endpoint: str, method: str = 'GET', params: Dict = None) -> Dict:
        """
        Make a request to the Unusual Whales API
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=self.timeout)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=params, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    sleep_time = self.retry_delay * (2 ** attempt) + (0.1 * attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return {"error": str(e)}
        
        return {"error": "Maximum retries exceeded"}

    def get_flow_alerts(self, ticker: str, is_ask_side: bool = True, 
                        is_bid_side: bool = True, limit: int = 100) -> pd.DataFrame:
        """
        Get flow alerts for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            is_ask_side: Include ask-side transactions
            is_bid_side: Include bid-side transactions
            limit: Maximum number of results to return
            
        Returns:
            DataFrame containing flow alerts
        """
        endpoint = f"stock/{ticker}/flow-alerts"
        params = {
            "is_ask_side": str(is_ask_side).lower(),
            "is_bid_side": str(is_bid_side).lower(),
            "limit": min(limit, 200)  # API max is 200
        }
        
        response = self._make_request(endpoint, params=params)
        
        if "error" in response:
            logger.error(f"Error getting flow alerts for {ticker}: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            return pd.DataFrame(response["data"])
        
        logger.warning(f"No flow alerts found for {ticker}")
        return pd.DataFrame()

    def get_alerts(self, config_ids: List[str] = None, intraday_only: bool = True,
                  limit: int = 100, noti_types: List[str] = None, 
                  page: int = 0, ticker_symbols: str = None) -> pd.DataFrame:
        """
        Get all alerts that have been triggered
        
        Args:
            config_ids: List of alert IDs to filter by
            intraday_only: Only return intraday alerts
            limit: Maximum number of results to return
            noti_types: List of notification types to filter by
            page: Page number for pagination
            ticker_symbols: Comma-separated list of tickers
            
        Returns:
            DataFrame containing alerts
        """
        endpoint = "alerts"
        params = {
            "intraday_only": str(intraday_only).lower(),
            "limit": min(limit, 200),  # API max is 200
            "page": max(page, 0)
        }
        
        if config_ids:
            params["config_ids[]"] = config_ids
            
        if noti_types:
            params["noti_types[]"] = noti_types
            
        if ticker_symbols:
            params["ticker_symbols"] = ticker_symbols
            
        response = self._make_request(endpoint, params=params)
        
        if "error" in response:
            logger.error(f"Error getting alerts: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            return pd.DataFrame(response["data"])
        
        logger.warning("No alerts found")
        return pd.DataFrame()

    def get_alert_configurations(self) -> pd.DataFrame:
        """
        Get all alert configurations
        
        Returns:
            DataFrame containing alert configurations
        """
        endpoint = "alerts/configuration"
        
        response = self._make_request(endpoint)
        
        if "error" in response:
            logger.error(f"Error getting alert configurations: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            return pd.DataFrame(response["data"])
        
        logger.warning("No alert configurations found")
        return pd.DataFrame()

    def get_recent_dark_pool_trades(self, date: str = None, limit: int = 100,
                                   min_premium: int = 0, max_premium: int = None,
                                   min_size: int = 0, max_size: int = None,
                                   min_volume: int = 0, max_volume: int = None) -> pd.DataFrame:
        """
        Get recent dark pool trades
        
        Args:
            date: Trading date in YYYY-MM-DD format
            limit: Maximum number of results to return
            min_premium: Minimum premium
            max_premium: Maximum premium
            min_size: Minimum size
            max_size: Maximum size
            min_volume: Minimum volume
            max_volume: Maximum volume
            
        Returns:
            DataFrame containing dark pool trades
        """
        endpoint = "darkpool/recent"
        params = {
            "limit": min(limit, 200),  # API max is 200
            "min_premium": min_premium,
            "min_size": min_size,
            "min_volume": min_volume
        }
        
        if date:
            params["date"] = date
            
        if max_premium:
            params["max_premium"] = max_premium
            
        if max_size:
            params["max_size"] = max_size
            
        if max_volume:
            params["max_volume"] = max_volume
            
        response = self._make_request(endpoint, params=params)
        
        if "error" in response:
            logger.error(f"Error getting recent dark pool trades: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            df = pd.DataFrame(response["data"])
            
            # Convert timestamps to datetime
            if "executed_at" in df.columns:
                df["executed_at"] = pd.to_datetime(df["executed_at"])
                
            return df
        
        logger.warning("No recent dark pool trades found")
        return pd.DataFrame()

    def get_dark_pool_trades(self, ticker: str, date: str = None, limit: int = 500,
                            min_premium: int = 0, max_premium: int = None,
                            min_size: int = 0, max_size: int = None,
                            min_volume: int = 0, max_volume: int = None,
                            newer_than: str = None, older_than: str = None) -> pd.DataFrame:
        """
        Get dark pool trades for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            date: Trading date in YYYY-MM-DD format
            limit: Maximum number of results to return
            min_premium: Minimum premium
            max_premium: Maximum premium
            min_size: Minimum size
            max_size: Maximum size
            min_volume: Minimum volume
            max_volume: Maximum volume
            newer_than: Unix timestamp in milliseconds or seconds
            older_than: Unix timestamp in milliseconds or seconds
            
        Returns:
            DataFrame containing dark pool trades
        """
        endpoint = f"darkpool/{ticker}"
        params = {
            "limit": min(limit, 500),  # API max is 500
            "min_premium": min_premium,
            "min_size": min_size,
            "min_volume": min_volume
        }
        
        if date:
            params["date"] = date
            
        if max_premium:
            params["max_premium"] = max_premium
            
        if max_size:
            params["max_size"] = max_size
            
        if max_volume:
            params["max_volume"] = max_volume
            
        if newer_than:
            params["newer_than"] = newer_than
            
        if older_than:
            params["older_than"] = older_than
            
        response = self._make_request(endpoint, params=params)
        
        if "error" in response:
            logger.error(f"Error getting dark pool trades for {ticker}: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            df = pd.DataFrame(response["data"])
            
            # Convert timestamps to datetime
            if "executed_at" in df.columns:
                df["executed_at"] = pd.to_datetime(df["executed_at"])
                
            return df
        
        logger.warning(f"No dark pool trades found for {ticker}")
        return pd.DataFrame()

    def get_insider_transactions(self, common_stock_only: str = None, industries: str = None,
                               is_director: str = None, is_officer: str = None,
                               is_s_p_500: str = None, is_ten_percent_owner: str = None,
                               market_cap_size: str = None, min_marketcap: int = None,
                               max_marketcap: int = None, ticker_symbol: str = None,
                               transaction_codes: str = None) -> pd.DataFrame:
        """
        Get insider transactions
        
        Args:
            common_stock_only: Filter by common stock only
            industries: Filter by industries
            is_director: Filter by director status
            is_officer: Filter by officer status
            is_s_p_500: Filter by S&P 500 status
            is_ten_percent_owner: Filter by 10% owner status
            market_cap_size: Filter by market cap size
            min_marketcap: Minimum market cap
            max_marketcap: Maximum market cap
            ticker_symbol: Comma-separated list of tickers
            transaction_codes: Filter by transaction codes
            
        Returns:
            DataFrame containing insider transactions
        """
        endpoint = "insider/transactions"
        params = {}
        
        if common_stock_only:
            params["common_stock_only"] = common_stock_only
            
        if industries:
            params["industries"] = industries
            
        if is_director:
            params["is_director"] = is_director
            
        if is_officer:
            params["is_officer"] = is_officer
            
        if is_s_p_500:
            params["is_s_p_500"] = is_s_p_500
            
        if is_ten_percent_owner:
            params["is_ten_percent_owner"] = is_ten_percent_owner
            
        if market_cap_size:
            params["market_cap_size"] = market_cap_size
            
        if min_marketcap:
            params["min_marketcap"] = min_marketcap
            
        if max_marketcap:
            params["max_marketcap"] = max_marketcap
            
        if ticker_symbol:
            params["ticker_symbol"] = ticker_symbol
            
        if transaction_codes:
            params["transaction_codes"] = transaction_codes
            
        response = self._make_request(endpoint, params=params)
        
        if "error" in response:
            logger.error(f"Error getting insider transactions: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            df = pd.DataFrame(response["data"])
            
            # Convert date columns to datetime
            date_columns = ["filing_date", "transaction_date", "next_earnings_date"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
            return df
        
        logger.warning("No insider transactions found")
        return pd.DataFrame()

    def get_sector_flow(self, sector: str) -> pd.DataFrame:
        """
        Get insider flow for a specific sector
        
        Args:
            sector: Financial sector (e.g., "Technology")
            
        Returns:
            DataFrame containing sector flow
        """
        endpoint = f"insider/{sector}/sector-flow"
        
        response = self._make_request(endpoint)
        
        if "error" in response:
            logger.error(f"Error getting sector flow for {sector}: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            df = pd.DataFrame(response["data"])
            
            # Convert date columns to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                
            return df
        
        logger.warning(f"No sector flow found for {sector}")
        return pd.DataFrame()

    def get_insiders(self, ticker: str) -> pd.DataFrame:
        """
        Get insiders for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame containing insiders
        """
        endpoint = f"insider/{ticker}"
        
        response = self._make_request(endpoint)
        
        if "error" in response:
            logger.error(f"Error getting insiders for {ticker}: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            return pd.DataFrame(response["data"])
        
        logger.warning(f"No insiders found for {ticker}")
        return pd.DataFrame()

    def get_ticker_flow(self, ticker: str) -> pd.DataFrame:
        """
        Get insider flow for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame containing ticker flow
        """
        endpoint = f"insider/{ticker}/ticker-flow"
        
        response = self._make_request(endpoint)
        
        if "error" in response:
            logger.error(f"Error getting ticker flow for {ticker}: {response['error']}")
            return pd.DataFrame()
            
        if "data" in response and response["data"]:
            df = pd.DataFrame(response["data"])
            
            # Convert date columns to datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                
            return df
        
        logger.warning(f"No ticker flow found for {ticker}")
        return pd.DataFrame()

    def close(self):
        """Close the client and release resources"""
        self.session.close()
        logger.info("Unusual Whales client closed")


# Example usage
if __name__ == "__main__":
    # Create clients
    polygon_client = PolygonDataClient()
    unusual_whales_client = UnusualWhalesClient()
    
    try:
        # Test Polygon client
        print("\nTesting Polygon client...")
        market_status = polygon_client.get_market_status()
        print(f"Market status: {market_status}")
        
        # Get data for a few tickers
        tickers = ["AAPL", "MSFT", "GOOGL"]
        print(f"\nFetching data for {tickers}...")
        data = polygon_client.get_aggregates_batch(tickers, timespan="minute", limit=100)
        
        for ticker, df in data.items():
            print(f"{ticker}: {len(df)} data points")
            
        # Process data with GPU
        print("\nProcessing data with GPU...")
        processed = polygon_client.process_data_with_gpu(data)
        
        for ticker, result in processed.items():
            if result is not None:
                print(f"\n{ticker} processed results:")
                print(result)
        
        # Test Unusual Whales client
        print("\nTesting Unusual Whales client...")
        flow_alerts = unusual_whales_client.get_flow_alerts("AAPL", limit=5)
        print(f"Flow alerts for AAPL: {len(flow_alerts)} items")
        
        dark_pool = unusual_whales_client.get_recent_dark_pool_trades(limit=5)
        print(f"Recent dark pool trades: {len(dark_pool)} items")
        
    finally:
        # Close clients
        polygon_client.close()
        unusual_whales_client.close()