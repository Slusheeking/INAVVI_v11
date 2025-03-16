#!/usr/bin/env python3
"""
Turbo-charged Polygon.io Data Source with Advanced Optimizations - Fixed Version
Specifically designed for NVIDIA GH200 Grace Hopper Superchip

Features:
- Redis caching for API responses
- Optimized HTTP requests
- Sequential processing to avoid pickling errors
- Optimized CuPy operations for GH200
"""

import os
import time
import json
import hashlib
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import redis
import tensorflow as tf
import cupy as cp
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('polygon_turbo_fixed')

# Environment variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_TTL = int(os.environ.get('REDIS_TTL', 3600))  # 1 hour default TTL
MAX_CONNECTIONS = int(os.environ.get('MAX_CONNECTIONS', 50))
MAX_POOL_SIZE = int(os.environ.get('MAX_POOL_SIZE', 20))
CONNECTION_TIMEOUT = int(os.environ.get('CONNECTION_TIMEOUT', 10))
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
RETRY_BACKOFF_FACTOR = float(os.environ.get('RETRY_BACKOFF_FACTOR', 0.5))


class RedisCache:
    """Redis cache for API responses"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl=REDIS_TTL):
        """Initialize Redis cache"""
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.enabled = True
        self.memory_cache = {}

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_timeout=5,
                socket_connect_timeout=5
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

        if not self.enabled:
            return self.memory_cache.get(key)

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
        self.memory_cache[key] = value

        if not self.enabled:
            return True

        try:
            serialized = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized)
        except (redis.RedisError, pickle.PickleError) as e:
            logger.warning(f"Error setting Redis cache: {e}")
            return False

    def delete(self, key_parts):
        """Delete value from cache"""
        key = self._generate_key(key_parts)

        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]

        if not self.enabled:
            return True

        try:
            return self.redis_client.delete(key)
        except redis.RedisError as e:
            logger.warning(f"Error deleting from Redis cache: {e}")
            return False

    def clear(self, pattern="polygon:*"):
        """Clear all cache entries matching pattern"""
        # Clear memory cache
        self.memory_cache = {}

        if not self.enabled:
            return True

        try:
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, pattern, 100)
                if keys:
                    self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            return True
        except redis.RedisError as e:
            logger.warning(f"Error clearing Redis cache: {e}")
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
            'User-Agent': 'PolygonTurbo/1.0',
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


class PolygonDataSourceTurboFixed:
    """Turbo-charged Polygon.io data source with fixed implementation"""

    def __init__(self, api_key=POLYGON_API_KEY):
        """Initialize Polygon data source"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

        # Initialize cache
        self.cache = RedisCache()

        # Initialize connection pool
        self.connection_pool = OptimizedConnectionPool()

        # Configure GPU memory growth to avoid OOM errors
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
            except RuntimeError as e:
                logger.warning(f"Memory growth configuration failed: {e}")

        # Configure CuPy for GH200
        self.cupy_initialized = False
        try:
            # Add a small random delay to avoid race conditions
            time.sleep(random.uniform(0.1, 0.5))

            # Get device count
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                # Find GH200 device if available
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    if "GH200" in device_props["name"].decode():
                        cp.cuda.Device(i).use()
                        break

                # Use unified memory for better performance on GH200
                cp.cuda.set_allocator(cp.cuda.MemoryPool(
                    cp.cuda.malloc_managed).malloc)
                logger.info("CuPy configured for GH200 with unified memory")
                self.cupy_initialized = True
        except Exception as e:
            logger.warning(f"CuPy configuration failed: {e}")

        logger.info(f"Turbo-charged Polygon data source initialized")

    def _make_request(self, endpoint, params=None):
        """Make API request with caching"""
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
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Make request
        try:
            response = self.connection_pool.get(
                url, params=params, headers=headers, timeout=15)

            # Check if response is valid JSON
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")
                return {"status": "ERROR", "error": f"Invalid JSON response: {e}"}

            # Cache successful responses
            if response.status_code == 200 and data.get("status") == "OK":
                self.cache.set(cache_key, data)
            elif response.status_code == 200 and "results" in data:
                # Some endpoints return results without a status field
                self.cache.set(cache_key, data)
            elif response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {data.get('error', 'No error details')}")
                return {"status": "ERROR", "error": f"HTTP {response.status_code}: {data.get('error', 'No error details')}"}
            elif "error" in data:
                logger.error(f"API returned error: {data.get('error')}")
                return {"status": "ERROR", "error": data.get('error')}

            return data
        except requests.RequestException as e:
            logger.error(f"API request failed: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            logger.error(
                f"Unexpected error during API request: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"Unexpected error: {type(e).__name__}: {e}"}

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                       from_date=None, to_date=None, limit=10000):
        """Get aggregated data for a ticker"""
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": limit}

        # Make request
        data = self._make_request(endpoint, params)

        # Process response
        if data.get("status") == "OK" and "results" in data:
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
        else:
            error_msg = data.get("error", "Unknown error") if isinstance(
                data, dict) else "Unknown error"
            logger.warning(
                f"Failed to get aggregates for {ticker}: {error_msg}")
            return pd.DataFrame()

    def get_aggregates_batch(self, tickers, multiplier=1, timespan="minute",
                             from_date=None, to_date=None, limit=10000):
        """
        Get aggregated data for multiple tickers sequentially

        Returns a dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        results = {}

        # Process each ticker sequentially to avoid pickling errors
        for ticker in tickers:
            try:
                df = self.get_aggregates(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit
                )
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")

        return results

    def process_data_with_gpu(self, data):
        """Process data using GPU acceleration"""
        if not data:
            return {}

        results = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            try:
                # Skip GPU processing if CuPy initialization failed
                if not self.cupy_initialized:
                    logger.warning(
                        f"Skipping GPU processing for {ticker} due to CuPy initialization failure")
                    continue

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
                vwap = total_price_volume / total_volume

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

            except Exception as e:
                logger.error(f"Error processing data for {ticker} on GPU: {e}")

        return results

    def close(self):
        """Close all connections and resources"""
        self.connection_pool.close()

        # Clean up CuPy resources
        if self.cupy_initialized:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Error clearing CuPy memory pool: {e}")
        logger.info("Closed all connections and resources")


# Example usage
if __name__ == "__main__":
    # Create client
    client = PolygonDataSourceTurboFixed()

    # Fetch data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results = client.get_aggregates_batch(symbols, timespan="minute")

    # Process the results
    for symbol, df in results.items():
        print(f"{symbol}: {len(df)} data points")

    # Process data with GPU
    processed = client.process_data_with_gpu(results)
    print("\nProcessed Results:")
    for symbol, df in processed.items():
        print(f"{symbol}:")
        print(df)

    # Close client
    client.close()
