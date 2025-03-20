#!/usr/bin/env python3
"""
Data Loader Module

This module provides functionality for loading and preprocessing historical market data
from various sources including Polygon.io and Unusual Whales.
Features include:
1. Efficient data caching
2. Rate limiting for API calls
3. Automatic retry logic
4. Data normalization and preprocessing
"""

import os
import time
import logging
import datetime
import pandas as pd
from datetime import timedelta
import numpy as np
import requests
import pickle
import cupy as cp
import asyncio
from io import StringIO
from retrying import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_loader')

# Check if CUDA is available through CuPy
cuda_available = False
try:
    if cp.cuda.is_available():
        cuda_available = True
        logger.info(f"CUDA is available through CuPy version {cp.__version__}")
except Exception as e:
    logger.warning(f"Error checking CUDA availability: {e}")


class DataLoader:
    """
    Data loader for fetching and preparing historical data
    Compatible with Polygon and Unusual Whales APIs
    """

    def __init__(self, polygon_client, unusual_whales_client, redis_client, use_gh200=True, test_mode=False):
        self.polygon = polygon_client
        self.unusual_whales = unusual_whales_client
        self.redis = redis_client
        self.use_gpu = False  # Temporarily disable GPU processing to avoid length mismatch errors
        self.test_mode = test_mode  # Flag to enable test mode with mock data
        self.use_gh200 = use_gh200

        # Configuration
        self.config = {
            'cache_dir': os.environ.get('DATA_CACHE_DIR', './data/cache'),
            'cache_expiry': 86400,  # 1 day in seconds
            'rate_limit': {
                'polygon': 5,        # requests per second
                'unusual_whales': 2  # requests per second
            },
            'retry_settings': {
                'stop_max_attempt_number': 3,
                'wait_exponential_multiplier': 1000,
                'wait_exponential_max': 10000
            }
        }

        # Ensure cache directory exists
        os.makedirs(self.config['cache_dir'], exist_ok=True)

        # Create a custom cache function to avoid PyArrow serialization issues
        def save_dataframe(df, path):
            """Save DataFrame to CSV file"""
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if df is not None:
                    pickle.dump(df, open(path, 'wb'))
                    return True
                return False
            except Exception as e:
                logger.error(f"Error saving DataFrame to {path}: {e}")
                return False

        def load_dataframe(path):
            """Load DataFrame from CSV file"""
            try:
                if os.path.exists(path):
                    return pickle.load(open(path, 'rb'))
                return None
            except Exception as e:
                logger.error(f"Error loading DataFrame from {path}: {e}")
                return None

        self.cache_functions = {"save": save_dataframe, "load": load_dataframe}

        # Initialize GPU memory for data processing if available
        if self.use_gpu:
            try:
                # Enhanced GPU initialization based on polygon_data_source_turbo_fixed.py
                # Find and use GH200 if available
                device_count = cp.cuda.runtime.getDeviceCount()
                gh200_found = False

                if device_count > 0:
                    # Look for GH200 device
                    for i in range(device_count):
                        device_props = cp.cuda.runtime.getDeviceProperties(i)
                        device_name = device_props["name"].decode()

                        if "GH200" in device_name and self.use_gh200:
                            # Use the GH200 device
                            cp.cuda.Device(i).use()
                            gh200_found = True

                            # GH200-specific optimizations
                            # Use unified memory for better performance on GH200
                            self.mempool = cp.cuda.MemoryPool(
                                cp.cuda.malloc_managed)
                            cp.cuda.set_allocator(self.mempool.malloc)

                            # Get memory info
                            free, total = cp.cuda.runtime.memGetInfo()
                            logger.info(
                                f"Using GH200 device with {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total memory")
                            # Found GH200, no need to continue the loop
                            break

                if not gh200_found:
                    # Use the first available GPU if GH200 not found
                    device_id = cp.cuda.Device().id
                    device_props = cp.cuda.runtime.getDeviceProperties(
                        device_id)
                    device_name = device_props["name"].decode()

                    # Standard memory pool for non-GH200 GPUs
                    self.mempool = cp.cuda.MemoryPool()
                    cp.cuda.set_allocator(self.mempool.malloc)

                    logger.info(
                        f"Using GPU device: {device_name} (GH200 not found)")

                logger.info(
                    f"Initialized CUDA memory pool on device {cp.cuda.Device().id}")

                # Create GPU utility functions
                self.gpu_utils = {
                    'to_gpu': lambda data: cp.asarray(data) if isinstance(data, (np.ndarray, list)) else data,
                    'from_gpu': lambda data: cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
                    'process_dataframe': self._process_dataframe_with_gpu
                }
                logger.info("GPU utilities initialized for data processing")
            except Exception as e:
                logger.error(f"Error initializing GPU utilities: {e}")
                self.use_gpu = False

        self.cache_functions = {"save": save_dataframe, "load": load_dataframe}

        # Create event loop for async calls
        try:
            # Try to get the current event loop
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists in this thread, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        logger.info("Data Loader initialized")

    def _process_dataframe_with_gpu(self, df, batch_size=1000):
        """Process DataFrame with GPU acceleration using CuPy"""
        if not self.use_gpu or df is None or df.empty:
            return df

        try:
            # Enhanced GH200-specific processing based on polygon_data_source_turbo_fixed.py
            # Select numeric columns for GPU processing
            numeric_cols = df.select_dtypes(
                include=['number']).columns.tolist()
            if not numeric_cols:
                return df

            # Create a copy to avoid modifying the original
            result = df.copy()

            # Process numeric columns with GPU
            if self.use_gh200:
                # GH200-specific optimizations: process in batches for better memory management
                n_rows = len(df)

                # Pre-allocate GPU memory for all columns to avoid fragmentation
                gpu_arrays = {}
                for col in numeric_cols:
                    if col in ['close', 'open', 'high', 'low', 'volume']:
                        # Use unified memory for better performance on GH200
                        gpu_arrays[col] = cp.asarray(
                            df[col].values, dtype=cp.float32)

                # Calculate technical indicators using GPU acceleration

                # Simple Moving Averages - use cumsum for faster calculation
                if 'close' in gpu_arrays:
                    close_prices = gpu_arrays['close']

                    # SMA 5
                    window_5 = 5
                    if len(close_prices) > window_5:
                        padded_prices = cp.pad(
                            close_prices, (window_5-1, 0), 'constant')
                        cumsum = cp.cumsum(padded_prices)
                        sma_5 = (cumsum[window_5:] -
                                 cumsum[:-window_5]) / window_5
                        # Ensure array length matches by padding with NaN values
                        full_sma_5 = cp.full(len(close_prices), cp.nan)
                        full_sma_5[window_5-1:] = sma_5
                        result['sma5'] = cp.asnumpy(full_sma_5)

                    # SMA 10
                    window_10 = 10
                    if len(close_prices) > window_10:
                        padded_prices = cp.pad(
                            close_prices, (window_10-1, 0), 'constant')
                        cumsum = cp.cumsum(padded_prices)
                        sma_10 = (cumsum[window_10:] -
                                  cumsum[:-window_10]) / window_10
                        # Ensure array length matches by padding with NaN values
                        full_sma_10 = cp.full(len(close_prices), cp.nan)
                        full_sma_10[window_10-1:] = sma_10
                        result['sma10'] = cp.asnumpy(full_sma_10)

                    # SMA 20
                    window_20 = 20
                    if len(close_prices) > window_20:
                        padded_prices = cp.pad(
                            close_prices, (window_20-1, 0), 'constant')
                        cumsum = cp.cumsum(padded_prices)
                        sma_20 = (cumsum[window_20:] -
                                  cumsum[:-window_20]) / window_20
                        # Ensure array length matches by padding with NaN values
                        full_sma_20 = cp.full(len(close_prices), cp.nan)
                        full_sma_20[window_20-1:] = sma_20
                        result['sma20'] = cp.asnumpy(full_sma_20)

                # Volume weighted average price if both close and volume are available
                if 'close' in gpu_arrays and 'volume' in gpu_arrays:
                    close_prices = gpu_arrays['close']
                    volumes = gpu_arrays['volume']

                    # Use parallel reduction for better performance
                    price_volume = close_prices * volumes
                    total_price_volume = cp.sum(price_volume)
                    total_volume = cp.sum(volumes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0
                    result['vwap'] = float(cp.asnumpy(vwap))

                # Calculate RSI
                if 'close' in gpu_arrays and len(close_prices) > 1:
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
                    result['rsi'] = float(cp.asnumpy(rsi))

                # Bollinger Bands
                if 'close' in gpu_arrays and len(close_prices) > 20:
                    # Middle band is SMA 20
                    if 'sma20' in result.columns:
                        bb_middle_full = cp.asarray(
                            result['sma20'].values, dtype=cp.float32)
                    else:
                        window_20 = 20
                        padded_prices = cp.pad(
                            close_prices, (window_20-1, 0), 'constant')
                        cumsum = cp.cumsum(padded_prices)
                        bb_middle = (cumsum[window_20:] -
                                     cumsum[:-window_20]) / window_20
                        # Ensure array length matches
                        bb_middle_full = cp.full(len(close_prices), cp.nan)
                        bb_middle_full[window_20-1:] = bb_middle
                        result['bb_middle'] = cp.asnumpy(bb_middle_full)

                    # Calculate standard deviation
                    rolling_std = cp.zeros_like(close_prices)
                    for i in range(20, len(close_prices)):
                        rolling_std[i] = cp.std(close_prices[i-20:i])

                    # Use the full-length middle band for calculations
                    if 'sma20' in result.columns:
                        # Upper and lower bands (ensure same length)
                        bb_upper = bb_middle_full + 2 * rolling_std
                        bb_lower = bb_middle_full - 2 * rolling_std
                    else:
                        # Upper and lower bands (ensure same length)
                        bb_upper = bb_middle_full + 2 * rolling_std
                        bb_lower = bb_middle_full - 2 * rolling_std

                    # Store results
                    result['bb_upper'] = cp.asnumpy(bb_upper)
                    result['bb_lower'] = cp.asnumpy(bb_lower)
                    result['bb_width'] = cp.asnumpy(
                        (bb_upper - bb_lower) / cp.where(bb_middle_full != 0, bb_middle_full, 1e-9))
            else:
                # Standard GPU processing for non-GH200 GPUs
                for col in numeric_cols:
                    # Transfer to GPU
                    gpu_array = cp.asarray(df[col].values)

                    # Example processing: calculate moving averages
                    if len(gpu_array) > 5:
                        result[f'{col}_sma5'] = cp.asnumpy(
                            cp.convolve(gpu_array, cp.ones(5)/5, mode='valid'))

            # Free GPU memory explicitly
            cp.get_default_memory_pool().free_all_blocks()

            # Return processed DataFrame
            return result

        except Exception as e:
            logger.error(f"Error in GPU processing: {e}")
            return df

    def _generate_mock_price_data(self, tickers, start_date, end_date, timeframe='1d'):
        """Generate mock price data for testing purposes"""
        results = {}

        # Calculate number of days between start and end date
        days = (end_date - start_date).days + 1
        if days <= 0:
            days = 1

        for ticker in tickers:
            # Create date range
            dates = [start_date + timedelta(days=i) for i in range(days)]

            # Generate random price data
            base_price = float(os.getenv('DEFAULT_BASE_PRICE', '50.0'))

            data = []
            for i, date in enumerate(dates):
                # Generate slightly random prices
                close_price = base_price + \
                    (i * 0.5) + (np.random.random() * 2 - 1)
                open_price = close_price - (np.random.random() * 1)
                high_price = max(close_price, open_price) + \
                    (np.random.random() * 1)
                low_price = min(close_price, open_price) - \
                    (np.random.random() * 1)
                volume = int(1000000 * np.random.random())

                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

            results[ticker] = pd.DataFrame(data)
        return results

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def load_price_data(self, tickers, start_date, end_date, timeframe='1m'):
        """
        Load historical price data for specified tickers

        Args:
            tickers (list): List of ticker symbols
            start_date (datetime): Start date
            end_date (datetime): End date
            timeframe (str): Timeframe ('1m', '5m', '1h', '1d')

        Returns:
            dict: Dictionary of ticker -> DataFrame with OHLCV data
        """
        try:
            logger.info(
                f"Loading price data for {len(tickers)} tickers from {start_date} to {end_date}")

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock data for testing")
                return self._generate_mock_price_data(tickers, start_date, end_date, timeframe)

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Determine multiplier and timespan for Polygon API
            if timeframe == '1m':
                multiplier = 1
                timespan = 'minute'
            elif timeframe == '5m':
                multiplier = 5
                timespan = 'minute'
            elif timeframe == '1h':
                multiplier = 1
                timespan = 'hour'
            elif timeframe == '1d':
                multiplier = 1
                timespan = 'day'
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Load data for each ticker
            results = {}

            for ticker in tickers:
                try:
                    # Check cache first
                    cache_key = f"{ticker}_{timeframe}_{start_str}_{end_str}"
                    cache_path = os.path.join(
                        self.config['cache_dir'], f"{cache_key}.csv")

                    # Use custom cache function
                    df = self.cache_functions["load"](cache_path)
                    if df is not None and time.time() - os.path.getmtime(cache_path) < self.config['cache_expiry']:
                        # Load from cache
                        # Convert timestamp to datetime if needed
                        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        results[ticker] = df
                        logger.debug(
                            f"Loaded {ticker} data from cache: {len(df)} rows")

                    else:
                        # Fetch from API
                        # Run async method in the event loop
                        try:
                            # Use run_until_complete if the loop is not running
                            if not self.loop.is_running():
                                aggs = self.loop.run_until_complete(self.polygon.get_aggregates(
                                    ticker=ticker,
                                    multiplier=multiplier,
                                    timespan=timespan,
                                    from_date=start_str,
                                    to_date=end_str,
                                    limit=50000
                                ))
                            else:
                                # If loop is already running, use a future
                                future = asyncio.run_coroutine_threadsafe(
                                    self.polygon.get_aggregates(
                                        ticker=ticker,
                                        multiplier=multiplier,
                                        timespan=timespan,
                                        from_date=start_str,
                                        to_date=end_str,
                                        limit=50000
                                    ),
                                    self.loop
                                )
                                # 30 second timeout
                                # Increase timeout to 60 seconds
                                aggs = future.result(timeout=60)
                        except Exception as e:
                            logger.error(
                                f"Error fetching aggregates for {ticker}: {e}")
                            aggs = []

                        # Convert to DataFrame
                        if isinstance(aggs, pd.DataFrame):
                            df = aggs
                            df['timestamp'] = pd.to_datetime(
                                df['t'], unit='ms') if 't' in df.columns else pd.to_datetime(df.index)
                        else:
                            try:
                                # Try to extract results directly
                                if isinstance(aggs, dict) and "results" in aggs and aggs["results"]:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs and isinstance(aggs, dict) and "results" in aggs:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs:
                                    logger.debug(
                                        f"Converting raw data to DataFrame for {ticker}: {type(aggs)}")
                                    df = pd.DataFrame(aggs)
                                else:
                                    logger.warning(
                                        f"No data returned for {ticker}")
                                    continue
                            except Exception as e:
                                logger.error(
                                    f"Error converting data to DataFrame for {ticker}: {e}")
                                continue

                        # Debug the DataFrame structure
                        logger.debug(
                            f"DataFrame columns for {ticker}: {df.columns.tolist()}")
                        if len(df) > 0:
                            logger.debug(
                                f"First row for {ticker}: {df.iloc[0].to_dict()}")

                        # Ensure timestamp column exists
                        if 'timestamp' not in df.columns and 't' in df.columns:
                            df['timestamp'] = pd.to_datetime(
                                df['t'], unit='ms')
                            logger.info(
                                f"Created timestamp column from 't' for {ticker}")

                        # Rename columns if needed
                        column_mapping = {
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume',
                            't': 'timestamp',
                            'vw': 'vwap'
                        }

                        df = df.rename(
                            columns={k: v for k, v in column_mapping.items() if k in df.columns})
                        logger.debug(
                            f"Columns after renaming: {df.columns.tolist()}")

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            if df['timestamp'].dtype == 'int64':
                                df['timestamp'] = pd.to_datetime(
                                    df['timestamp'], unit='ms')
                            elif not pd.api.types.is_datetime64_dtype(df['timestamp']):
                                logger.debug(
                                    f"Converting timestamp column for {ticker}, type: {df['timestamp'].dtype}")
                                df['timestamp'] = pd.to_datetime(
                                    df['timestamp'])
                        else:
                            # Create timestamp from 't' column if available
                            if 't' in df.columns:
                                df['timestamp'] = pd.to_datetime(
                                    df['t'], unit='ms')
                                logger.info(
                                    f"Created timestamp column from 't' for {ticker}")
                            else:
                                logger.warning(
                                    f"No timestamp or 't' column found for {ticker}")

                        # Ensure required columns
                        required_columns = [
                            'open', 'high', 'low', 'close', 'volume', 'timestamp']
                        if not all(col in df.columns for col in required_columns):
                            logger.warning(
                                f"Missing required columns for {ticker}: {[col for col in required_columns if col not in df.columns]}")
                            continue

                        # Save to cache
                        self.cache_functions["save"](df, cache_path)

                        # Apply GPU processing if available
                        if self.use_gpu:
                            try:
                                processed_df = self.gpu_utils['process_dataframe'](
                                    df)
                                results[ticker] = processed_df
                                logger.debug(
                                    f"Processed {ticker} data with GPU")
                            except Exception as e:
                                logger.warning(
                                    f"GPU processing failed for {ticker}: {e}")
                                results[ticker] = df
                        else:
                            results[ticker] = df
                        logger.debug(
                            f"Fetched {ticker} data from API: {len(df)} rows")

                    # Rate limiting
                    time.sleep(1.0 / self.config['rate_limit']['polygon'])

                except Exception as e:
                    logger.error(
                        f"Error loading price data for {ticker}: {str(e)}")

            logger.info(f"Loaded price data for {len(results)} tickers")
            return results

        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}", exc_info=True)
            return {}

    def _generate_mock_options_data(self, tickers, start_date, end_date):
        """Generate mock options data for testing purposes"""
        results = {}

        for ticker in tickers:
            # Generate random options data
            data = []
            for i in range(5):  # Generate 5 mock options
                expiration = start_date + timedelta(days=30 * (i+1))
                strike = 100.0 + (i * 10)

                data.append({
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    'side': 'call' if i % 2 == 0 else 'put',
                    'strike': strike,
                    'expiration': expiration,
                    'volume': int(1000 * np.random.random()),
                    'open_interest': int(5000 * np.random.random()),
                    'implied_volatility': 0.2 + (np.random.random() * 0.3),
                    'delta': 0.5 + (np.random.random() * 0.4 - 0.2),
                    'gamma': 0.05 * np.random.random(),
                    'theta': -0.05 * np.random.random(),
                    'vega': 0.1 * np.random.random(),
                    'premium': 2.0 + (np.random.random() * 3)
                })

            results[ticker] = pd.DataFrame(data)
        return results

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def load_options_data(self, tickers, start_date, end_date):
        """
        Load options data for specified tickers

        Args:
            tickers (list): List of ticker symbols
            start_date (datetime): Start date
            end_date (datetime): End date

        Returns:
            dict: Dictionary of ticker -> options data
        """
        try:
            logger.info(
                f"Loading options data for {len(tickers)} tickers from {start_date} to {end_date}")

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock options data for testing")
                return self._generate_mock_options_data(tickers, start_date, end_date)

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Load data for each ticker
            results = {}

            for ticker in tickers:
                try:
                    # Check cache first
                    cache_key = f"{ticker}_options_{start_str}_{end_str}"
                    cache_path = os.path.join(
                        self.config['cache_dir'], f"{cache_key}.csv")

                    # Use custom cache function
                    df = self.cache_functions["load"](cache_path)
                    if df is not None and time.time() - os.path.getmtime(cache_path) < self.config['cache_expiry']:
                        # Load from cache
                        # Convert timestamp to datetime if needed
                        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        results[ticker] = df
                        logger.debug(
                            f"Loaded {ticker} options data from cache: {len(df)} rows")

                    else:
                        # Fetch from Unusual Whales API
                        # Run async method in the event loop
                        try:
                            # Use run_until_complete if the loop is not running
                            if not self.loop.is_running():
                                flow_data = self.loop.run_until_complete(self.unusual_whales.get_flow_alerts(
                                    ticker=ticker,
                                    limit=1000
                                ))
                            else:
                                # If loop is already running, use a future
                                future = asyncio.run_coroutine_threadsafe(
                                    self.unusual_whales.get_flow_alerts(
                                        ticker=ticker,
                                        limit=1000
                                    ),
                                    self.loop
                                )
                                # Increase timeout to 60 seconds
                                flow_data = future.result(timeout=60)
                        except Exception as e:
                            logger.error(
                                f"Error loading options data for {ticker}: {e}")
                            flow_data = []

                        # Old synchronous approach
                        """flow_data = self.loop.run_until_complete(self.unusual_whales.get_flow_alerts(
                            ticker=ticker,
                            limit=1000
                        ))"""

                        # Convert to DataFrame
                        if isinstance(flow_data, pd.DataFrame):
                            df = flow_data
                        else:
                            try:
                                # Try to extract data directly
                                if isinstance(flow_data, dict) and "data" in flow_data and flow_data["data"]:
                                    df = pd.DataFrame(flow_data["data"])
                                elif flow_data and isinstance(flow_data, list):
                                    df = pd.DataFrame(flow_data)
                                elif flow_data and isinstance(flow_data, dict) and "data" in flow_data:
                                    df = pd.DataFrame(flow_data["data"])
                                else:
                                    logger.warning(
                                        f"Unexpected options data format for {ticker}")
                                    continue
                            except Exception as e:
                                logger.error(
                                    f"Error converting options data to DataFrame for {ticker}: {e}")
                                continue

                        # Ensure required columns
                        required_columns = [
                            'ticker', 'timestamp', 'side', 'strike', 'expiration', 'volume', 'open_interest']
                        if not all(col in df.columns for col in required_columns):
                            missing_cols = [
                                col for col in required_columns if col not in df.columns]
                            logger.warning(
                                f"Missing required columns for {ticker} options data: {missing_cols}")

                            # Add missing columns with default values
                            for col in missing_cols:
                                if col == 'ticker':
                                    df['ticker'] = ticker
                                elif col == 'timestamp':
                                    df['timestamp'] = pd.Timestamp.now()
                                elif col == 'side':
                                    df['side'] = 'unknown'
                                elif col == 'strike':
                                    df['strike'] = 0.0
                                elif col == 'expiration':
                                    df['expiration'] = pd.Timestamp.now() + \
                                        pd.Timedelta(days=30)
                                elif col == 'volume':
                                    df['volume'] = 0
                                elif col == 'open_interest':
                                    df['open_interest'] = 0
                                else:
                                    df[col] = None

                        # Save to cache if not empty
                        if not df.empty:
                            self.cache_functions["save"](df, cache_path)

                        results[ticker] = df
                        logger.debug(
                            f"Fetched {ticker} options data from API: {len(df)} rows")

                    # Rate limiting
                    time.sleep(
                        1.0 / self.config['rate_limit']['unusual_whales'])

                except Exception as e:
                    logger.error(
                        f"Error loading options data for {ticker}: {str(e)}")

            logger.info(f"Loaded options data for {len(results)} tickers")
            return results

        except Exception as e:
            logger.error(
                f"Error loading options data: {str(e)}", exc_info=True)
            return {}

    def _generate_mock_market_data(self, start_date, end_date, symbols=['SPY', 'VIX']):
        """Generate mock market data for testing purposes"""
        # Calculate number of days between start and end date
        days = (end_date - start_date).days + 1
        if days <= 0:
            days = 1

        # Create date range
        dates = [start_date + timedelta(days=i) for i in range(days)]

        # Generate random market data
        data = []
        for date in dates:
            row = {'timestamp': date}

            for symbol in symbols:
                symbol_lower = symbol.lower()
                base_price = 400.0 if symbol == 'SPY' else 20.0

                # Generate random prices
                close_price = base_price + (np.random.random() * 10 - 5)
                row[f"{symbol_lower}_close"] = close_price
                row[f"{symbol_lower}_open"] = close_price - \
                    (np.random.random() * 2 - 1)
                row[f"{symbol_lower}_high"] = close_price + \
                    (np.random.random() * 2)
                row[f"{symbol_lower}_low"] = close_price - \
                    (np.random.random() * 2)
                row[f"{symbol_lower}_volume"] = int(
                    10000000 * np.random.random())

            data.append(row)
        return pd.DataFrame(data)

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def load_market_data(self, start_date, end_date, symbols=['SPY', 'VIX']):
        """
        Load market data for specified symbols

        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            symbols (list): List of market symbols to load

        Returns:
            DataFrame: Market data
        """
        try:
            logger.info(
                f"Loading market data for {symbols} from {start_date} to {end_date}")

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock market data for testing")
                return self._generate_mock_market_data(start_date, end_date, symbols)

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Check cache first
            cache_key = f"market_data_{','.join(symbols)}_{start_str}_{end_str}"
            cache_path = os.path.join(
                self.config['cache_dir'], f"{cache_key}.csv")

            # Use custom cache function
            df = self.cache_functions["load"](cache_path)
            if df is not None and time.time() - os.path.getmtime(cache_path) < self.config['cache_expiry']:
                # Load from cache
                # Convert timestamp to datetime if needed
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                logger.debug(f"Loaded market data from cache: {len(df)} rows")
                return df

            else:
                # Load data for each symbol
                dfs = []

                for symbol in symbols:
                    try:
                        # Fetch from Polygon API
                        # Run async method in the event loop
                        try:
                            # Use run_until_complete if the loop is not running
                            if not self.loop.is_running():
                                aggs = self.loop.run_until_complete(self.polygon.get_aggregates(
                                    ticker=symbol,
                                    multiplier=1,
                                    timespan='minute',
                                    from_date=start_str,
                                    to_date=end_str,
                                    limit=50000
                                ))
                            else:
                                # If loop is already running, use a future
                                future = asyncio.run_coroutine_threadsafe(
                                    self.polygon.get_aggregates(
                                        ticker=symbol,
                                        multiplier=1,
                                        timespan='minute',
                                        from_date=start_str,
                                        to_date=end_str,
                                        limit=50000
                                    ),
                                    self.loop
                                )
                                # 30 second timeout
                                # Increase timeout to 60 seconds
                                aggs = future.result(timeout=60)
                        except Exception as e:
                            logger.error(
                                f"Error fetching market data for {symbol}: {e}")
                            aggs = []

                        # Old synchronous approach
                        """aggs = self.loop.run_until_complete(self.polygon.get_aggregates(
                            ticker=symbol,
                            multiplier=1,
                            timespan='minute',
                            from_date=start_str,
                            to_date=end_str,
                            limit=50000
                        ))"""

                        # Convert to DataFrame
                        if isinstance(aggs, pd.DataFrame):
                            df = aggs
                            df['timestamp'] = pd.to_datetime(
                                df['t'], unit='ms') if 't' in df.columns else pd.to_datetime(df.index)
                        else:
                            try:
                                # Try to extract results directly
                                if isinstance(aggs, dict) and "results" in aggs and aggs["results"]:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs and isinstance(aggs, dict) and "results" in aggs:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs:
                                    df = pd.DataFrame(aggs)
                                else:
                                    logger.warning(
                                        f"No data returned for {symbol}")
                                    continue
                            except Exception as e:
                                logger.error(
                                    f"Error converting data to DataFrame for {symbol}: {e}")
                                continue

                        # Rename columns if needed
                        column_mapping = {
                            'o': f"{symbol.lower()}_open",
                            'h': f"{symbol.lower()}_high",
                            'l': f"{symbol.lower()}_low",
                            'c': f"{symbol.lower()}_close",
                            'v': f"{symbol.lower()}_volume",
                            't': 'timestamp',
                            'vw': f"{symbol.lower()}_vwap"
                        }

                        df = df.rename(
                            columns={k: v for k, v in column_mapping.items() if k in df.columns})

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            if df['timestamp'].dtype == 'int64':
                                df['timestamp'] = pd.to_datetime(
                                    df['timestamp'], unit='ms')
                            else:
                                df['timestamp'] = pd.to_datetime(
                                    df['timestamp'])

                        # Add to list
                        dfs.append(df)
                        logger.debug(
                            f"Fetched {symbol} market data: {len(df)} rows")

                        # Rate limiting
                        time.sleep(1.0 / self.config['rate_limit']['polygon'])

                    except Exception as e:
                        logger.error(
                            f"Error loading market data for {symbol}: {str(e)}")

                # Merge DataFrames
                if not dfs:
                    logger.warning("No market data fetched")
                    return pd.DataFrame()

                # Ensure all dataframes have timestamp as index
                result = dfs[0].copy()

                # Check if timestamp is already in the columns
                for df in dfs:
                    if 'timestamp' in df.columns and df.index.name != 'timestamp':
                        df.set_index('timestamp', inplace=True)

                for df in dfs[1:]:
                    # Reset index to make timestamp a column for merging
                    # Handle the case where timestamp is both in index and columns
                    result_reset = result.copy()
                    if result.index.name == 'timestamp' and 'timestamp' in result_reset.columns:
                        # Rename the timestamp column to avoid conflict
                        result_reset = result_reset.rename(
                            columns={'timestamp': 'timestamp_col'})
                        result_reset = result_reset.reset_index()
                    elif result.index.name == 'timestamp':
                        result_reset = result.reset_index()

                    df_reset = df.copy()
                    if df.index.name == 'timestamp' and 'timestamp' in df_reset.columns:
                        df_reset = df_reset.rename(
                            columns={'timestamp': 'timestamp_col'})
                        df_reset = df_reset.reset_index()
                    elif df.index.name == 'timestamp':
                        df_reset = df.reset_index()

                    # Merge dataframes
                    result = pd.merge_asof(
                        result_reset, df_reset, on='timestamp', direction='nearest')

                # Add market metrics
                for symbol in symbols:
                    symbol_lower = symbol.lower()

                    # Daily change
                    if f"{symbol_lower}_close" in result.columns:
                        result[f"{symbol_lower}_change"] = result[f"{symbol_lower}_close"].pct_change(
                        )

                        # Calculate day-over-day change
                        result[f"{symbol_lower}_daily_change"] = result[f"{symbol_lower}_close"].pct_change(
                            periods=390)  # ~1 trading day in minutes

                # Save to cache
                self.cache_functions["save"](result, cache_path)

                logger.info(f"Loaded market data: {len(result)} rows")
                return result

        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}", exc_info=True)
            return pd.DataFrame()
