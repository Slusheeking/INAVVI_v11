#!/usr/bin/env python3
"""
Turbo-charged Polygon.io Data Source with Advanced Optimizations
Specifically designed for NVIDIA GH200 Grace Hopper Superchip

Features:
- Memory-mapped file caching for large datasets
- LZMA compression for Redis cache
- Asynchronous API requests with aiohttp
- Process pool for CPU-intensive tasks
- Thread pool for I/O-bound tasks
- Optimized CuPy operations for GH200
- Parallel reduction algorithms
"""

import os
import time
import json
import hashlib
import pickle
import lzma
import logging
import asyncio
import aiohttp
import mmap
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import redis
import tensorflow as tf
import cupy as cp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('polygon_turbo')

# Environment variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_TTL = int(os.environ.get('REDIS_TTL', 3600))  # 1 hour default TTL
MAX_CONNECTIONS = int(os.environ.get('MAX_CONNECTIONS', 100))
MAX_POOL_SIZE = int(os.environ.get('MAX_POOL_SIZE', 50))
CONNECTION_TIMEOUT = int(os.environ.get('CONNECTION_TIMEOUT', 10))
PROCESS_WORKERS = int(os.environ.get('PROCESS_WORKERS', 4))
USE_COMPRESSION = os.environ.get('USE_COMPRESSION', 'true').lower() == 'true'
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
RETRY_BACKOFF_FACTOR = float(os.environ.get('RETRY_BACKOFF_FACTOR', 0.5))
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 20))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 512))
QUEUE_SIZE = int(os.environ.get('QUEUE_SIZE', 200))
USE_MMAP_CACHE = os.environ.get('USE_MMAP_CACHE', 'true').lower() == 'true'
USE_ASYNC = os.environ.get('USE_ASYNC', 'true').lower() == 'true'


class MemoryMappedCache:
    """Memory-mapped file cache for large datasets"""

    def __init__(self, max_size_mb=1024):
        """Initialize memory-mapped cache"""
        self.max_size_mb = max_size_mb
        self.cache_dir = tempfile.mkdtemp(prefix="polygon_cache_")
        self.cache_files = {}
        self.cache_mmaps = {}
        self.cache_sizes = {}
        self.total_size_mb = 0
        logger.info(f"Initialized memory-mapped cache in {self.cache_dir}")

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

        if key not in self.cache_files:
            return None

        try:
            with open(self.cache_files[key], 'rb') as f:
                # Use memory mapping for efficient access
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                data = pickle.loads(mm.read())
                mm.close()
                return data
        except Exception as e:
            logger.warning(f"Error retrieving from memory-mapped cache: {e}")
            return None

    def set(self, key_parts, value, ttl=None):
        """Set value in cache"""
        if value is None:
            return False

        key = self._generate_key(key_parts)

        try:
            # Serialize the data
            serialized = pickle.dumps(value)
            size_mb = len(serialized) / (1024 * 1024)

            # Check if we need to make room
            if self.total_size_mb + size_mb > self.max_size_mb:
                self._evict_oldest()

            # Create a new file
            file_path = os.path.join(self.cache_dir, key)
            with open(file_path, 'wb') as f:
                f.write(serialized)

            # Update cache metadata
            self.cache_files[key] = file_path
            self.cache_sizes[key] = size_mb
            self.total_size_mb += size_mb

            return True
        except Exception as e:
            logger.warning(f"Error setting memory-mapped cache: {e}")
            return False

    def _evict_oldest(self):
        """Evict oldest entries to make room"""
        # Sort by access time
        sorted_keys = sorted(self.cache_files.keys(),
                             key=lambda k: os.path.getatime(self.cache_files[k]))

        # Remove oldest entries until we have enough space
        for key in sorted_keys:
            if self.total_size_mb <= self.max_size_mb * 0.8:  # Keep 20% free
                break

            file_path = self.cache_files[key]
            size_mb = self.cache_sizes[key]

            try:
                os.remove(file_path)
                del self.cache_files[key]
                del self.cache_sizes[key]
                self.total_size_mb -= size_mb
            except Exception as e:
                logger.warning(f"Error evicting cache entry: {e}")


class CompressedRedisCache:
    """Redis cache with LZMA compression for API responses"""

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl=REDIS_TTL):
        """Initialize Redis cache"""
        self.host = host
        self.port = port
        self.db = db
        self.compression = USE_COMPRESSION
        self.ttl = ttl
        self.enabled = True
        self.memory_cache = {}

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
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
            if data and self.compression:
                try:
                    return pickle.loads(lzma.decompress(data))
                except Exception as e:
                    logger.warning(f"Error decompressing cached data: {e}")
            elif data:
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
            if self.compression:
                try:
                    serialized = lzma.compress(pickle.dumps(
                        value), preset=1)  # Fast compression
                except Exception as e:
                    logger.warning(f"Error compressing data: {e}")
                    serialized = pickle.dumps(value)
            else:
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


class AsyncConnectionPool:
    """Asynchronous connection pool for parallel requests"""

    def __init__(self, max_connections=MAX_CONNECTIONS, timeout=CONNECTION_TIMEOUT):
        """Initialize async connection pool"""
        self.max_connections = max_connections
        self.timeout = timeout
        self.connector = None
        self.client_session = None
        logger.info(
            f"Async connection pool initialized with max_connections={max_connections}")

    async def get_session(self):
        """Get or create aiohttp client session"""
        if self.connector is None:
            # Configure TCP connector with connection limits
            self.connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                ttl_dns_cache=300,
                use_dns_cache=True
            )

        if self.client_session is None or self.client_session.closed:
            # Create session with timeout and headers
            self.client_session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'User-Agent': 'PolygonTurboAsync/1.0',
                    'Accept': 'application/json'
                }
            )

        return self.client_session

    async def close(self):
        """Close the session and connector"""
        if self.client_session and not self.client_session.closed:
            await self.client_session.close()

        if self.connector:
            await self.connector.close()

        self.client_session = None
        self.connector = None


class PolygonDataSourceTurbo:
    """Turbo-charged Polygon.io data source with advanced optimizations"""

    def __init__(self, api_key=POLYGON_API_KEY, max_workers=MAX_WORKERS, process_workers=PROCESS_WORKERS):
        """Initialize Polygon data source"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

        # Initialize caches
        self.redis_cache = CompressedRedisCache()
        if USE_MMAP_CACHE:
            self.mmap_cache = MemoryMappedCache()

        # Initialize connection pools
        self.connection_pool = OptimizedConnectionPool()
        if USE_ASYNC:
            self.async_pool = AsyncConnectionPool()
            # Create event loop in a separate thread
            self.loop = asyncio.new_event_loop()
            self.loop_thread = ThreadPoolExecutor(max_workers=1)
            self.loop_future = self.loop_thread.submit(
                self._run_event_loop, self.loop)

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=process_workers)

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
        try:
            # Use unified memory for better performance on GH200
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            # Set preferred memory bank to high-bandwidth memory
            cp.cuda.Device(0).use()
            logger.info("CuPy configured for GH200 with unified memory")
        except Exception as e:
            logger.warning(f"CuPy configuration failed: {e}")

        logger.info(
            f"Turbo-charged Polygon data source initialized with max_workers={max_workers}, process_workers={process_workers}")

    def _run_event_loop(self, loop):
        """Run the event loop in a separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _make_request(self, endpoint, params=None):
        """Make API request with caching"""
        # Generate cache key
        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check Redis cache first
        cached_data = self.redis_cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Redis cache hit for {endpoint}")
            return cached_data

        # Check memory-mapped cache if enabled
        if USE_MMAP_CACHE:
            cached_data = self.mmap_cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"MMAP cache hit for {endpoint}")
                # Also update Redis cache for future requests
                self.redis_cache.set(cache_key, cached_data)
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
                self.redis_cache.set(cache_key, data)
                if USE_MMAP_CACHE:
                    self.mmap_cache.set(cache_key, data)
            elif response.status_code == 200 and "results" in data:
                # Some endpoints return results without a status field
                self.redis_cache.set(cache_key, data)
                if USE_MMAP_CACHE:
                    self.mmap_cache.set(cache_key, data)
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

    async def _make_async_request(self, endpoint, params=None):
        """Make API request asynchronously"""
        # Generate cache key
        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check Redis cache first
        cached_data = self.redis_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Check memory-mapped cache if enabled
        if USE_MMAP_CACHE:
            cached_data = self.mmap_cache.get(cache_key)
            if cached_data is not None:
                # Also update Redis cache for future requests
                self.redis_cache.set(cache_key, cached_data)
                return cached_data

        # Prepare request
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Make async request
        try:
            session = await self.async_pool.get_session()
            async with session.get(url, params=params, headers=headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()

                    # Cache successful responses
                    if data.get("status") == "OK" or "results" in data:
                        self.redis_cache.set(cache_key, data)
                        if USE_MMAP_CACHE:
                            self.mmap_cache.set(cache_key, data)

                    return data
                else:
                    text = await response.text()
                    logger.error(
                        f"API request failed with status {response.status}: {text}")
                    return {"status": "ERROR", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Async API request failed: {type(e).__name__}: {e}")
            return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}

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
            # Submit to process pool for CPU-intensive work
            future = self.process_pool.submit(
                self._process_aggregates_data, data["results"])
            return future.result()
        else:
            error_msg = data.get("error", "Unknown error") if isinstance(
                data, dict) else "Unknown error"
            logger.warning(
                f"Failed to get aggregates for {ticker}: {error_msg}")
            return pd.DataFrame()

    def _process_aggregates_data(self, results):
        """Process aggregates data in a separate process"""
        # Convert to DataFrame
        df = pd.DataFrame(results)

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

    def get_aggregates_batch(self, tickers, multiplier=1, timespan="minute",
                             from_date=None, to_date=None, limit=10000):
        """
        Get aggregated data for multiple tickers in parallel

        Returns a dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        if USE_ASYNC:
            # Use async/await for parallel requests
            return self._get_aggregates_batch_async(
                tickers, multiplier, timespan, from_date, to_date, limit)
        else:
            # Use thread pool for parallel requests
            return self._get_aggregates_batch_threaded(
                tickers, multiplier, timespan, from_date, to_date, limit)

    def _get_aggregates_batch_threaded(self, tickers, multiplier, timespan, from_date, to_date, limit):
        """Get aggregated data using thread pool"""
        # Create a future for each ticker
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.thread_pool.submit(
                self.get_aggregates,
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date,
                limit=limit
            )
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")

        return results

    def _get_aggregates_batch_async(self, tickers, multiplier, timespan, from_date, to_date, limit):
        """Get aggregated data using async/await"""
        async def fetch_all():
            tasks = []
            for ticker in tickers:
                endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
                params = {"limit": limit}
                task = self._make_async_request(endpoint, params)
                tasks.append((ticker, asyncio.ensure_future(task)))

            results = {}
            for ticker, task in tasks:
                try:
                    data = await task
                    if data.get("status") == "OK" and "results" in data:
                        # Process in thread pool to avoid blocking event loop
                        future = self.process_pool.submit(
                            self._process_aggregates_data, data["results"])
                        df = future.result()
                        if not df.empty:
                            results[ticker] = df
                    else:
                        error_msg = data.get("error", "Unknown error") if isinstance(
                            data, dict) else "Unknown error"
                        logger.warning(
                            f"Failed to get aggregates for {ticker}: {error_msg}")
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")

            return results

        # Run async function in the event loop
        future = asyncio.run_coroutine_threadsafe(fetch_all(), self.loop)
        return future.result()

    def process_data_with_gpu(self, data):
        """Process data using GPU acceleration"""
        if not data:
            return {}

        results = {}

        # Process in batches to maximize GPU utilization
        batch_size = 10
        ticker_batches = [list(data.keys())[i:i+batch_size]
                          for i in range(0, len(data), batch_size)]

        for ticker_batch in ticker_batches:
            batch_futures = {}

            for ticker in ticker_batch:
                df = data[ticker]
                if df.empty:
                    continue

                # Submit to thread pool to parallelize GPU work
                future = self.thread_pool.submit(
                    self._process_ticker_data_gpu, ticker, df)
                batch_futures[future] = ticker

            # Collect results
            for future in as_completed(batch_futures):
                ticker = batch_futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[ticker] = result
                except Exception as e:
                    logger.error(
                        f"Error processing data for {ticker} on GPU: {e}")

        return results

    def _process_ticker_data_gpu(self, ticker, df):
        """Process ticker data on GPU"""
        try:
            # Convert to CuPy arrays for GPU processing
            close_prices = cp.array(df['close'].values, dtype=cp.float32)
            volumes = cp.array(df['volume'].values, dtype=cp.float32)

            # Calculate moving averages
            window_5 = 5
            window_20 = 20

            # Use optimized algorithms for better performance
            # Simple moving averages - use cumsum for faster calculation
            padded_prices = cp.pad(close_prices, (window_5-1, 0), 'constant')
            cumsum = cp.cumsum(padded_prices)
            sma_5 = (cumsum[window_5:] - cumsum[:-window_5]) / window_5

            padded_prices = cp.pad(close_prices, (window_20-1, 0), 'constant')
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
            avg_gain = cp.mean(gain[:14])
            avg_loss = cp.mean(loss[:14])

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

            return result_df

        except Exception as e:
            logger.error(f"Error processing data for {ticker} on GPU: {e}")
            return None

    def close(self):
        """Close all connections and resources"""
        self.connection_pool.close()
        self.thread_pool.shutdown()
        self.process_pool.shutdown()

        if USE_ASYNC:
            # Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.shutdown()

            # Close async connection pool
            async def close_async_pool():
                await self.async_pool.close()

            # Create a new event loop for cleanup
            cleanup_loop = asyncio.new_event_loop()
            cleanup_loop.run_until_complete(close_async_pool())
            cleanup_loop.close()

        logger.info("Closed all connections and resources")


# Example usage
if __name__ == "__main__":
    # Create client
    client = PolygonDataSourceTurbo()

    # Fetch data for multiple symbols in parallel
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN",
               "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]

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
