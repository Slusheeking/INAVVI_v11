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
    
    def __init__(self, polygon_client, unusual_whales_client, redis_client, use_gh200=True):
        self.polygon = polygon_client
        self.unusual_whales = unusual_whales_client
        self.redis = redis_client
        self.use_gpu = cuda_available
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
                # Initialize CUDA memory pool for better performance
                if self.use_gh200:
                    # GH200-specific optimizations for CuPy
                    # Use unified memory for GH200
                    self.mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
                    cp.cuda.set_allocator(self.mempool.malloc)
                    
                    # Get device properties to check for GH200
                    device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                    device_name = device_props["name"].decode()
                    if "GH200" in device_name:
                        logger.info(f"Using GH200-specific optimizations for {device_name}")
                    else:
                        logger.info(f"Using GPU device: {device_name}")
                else:
                    self.mempool = cp.cuda.MemoryPool()
                    cp.cuda.set_allocator(self.mempool.malloc)
                logger.info(f"Initialized CUDA memory pool on device {cp.cuda.Device().id}")
                
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
        self.loop = asyncio.new_event_loop()
        
        logger.info("Data Loader initialized")
    
    def _process_dataframe_with_gpu(self, df, batch_size=1000):
        """Process DataFrame with GPU acceleration using CuPy"""
        if not self.use_gpu or df is None or df.empty:
            return df
            
        try:
            # Select numeric columns for GPU processing
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return df
                
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Process numeric columns with GPU
            if self.use_gh200:
                # GH200-specific optimizations: process in batches for better memory management
                n_rows = len(df)
                for col in numeric_cols:
                    # Process in batches
                    for i in range(0, n_rows, batch_size):
                        end_idx = min(i + batch_size, n_rows)
                        # Transfer batch to GPU
                        gpu_array = cp.asarray(df[col].values[i:end_idx])
                        
                        # Example processing: calculate moving averages
                        if len(gpu_array) > 5:
                            # Use unified memory for better performance on GH200
                            sma5 = cp.convolve(gpu_array, cp.ones(5)/5, mode='valid')
                            result.loc[i+4:end_idx-1, f'{col}_sma5'] = cp.asnumpy(sma5)
            else:
                # Standard GPU processing
                for col in numeric_cols:
                    # Transfer to GPU
                    gpu_array = cp.asarray(df[col].values)
                    
                    # Example processing: calculate moving averages
                    if len(gpu_array) > 5:
                        result[f'{col}_sma5'] = cp.asnumpy(cp.convolve(gpu_array, cp.ones(5)/5, mode='valid'))
            
            # Free GPU memory explicitly
            cp.get_default_memory_pool().free_all_blocks()
                
            # Return processed DataFrame
            return result
            
        except Exception as e:
            logger.error(f"Error in GPU processing: {e}")
            return df
    
    
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
            logger.info(f"Loading price data for {len(tickers)} tickers from {start_date} to {end_date}")
            
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
                    cache_path = os.path.join(self.config['cache_dir'], f"{cache_key}.csv")

                    # Use custom cache function
                    df = self.cache_functions["load"](cache_path)
                    if df is not None and time.time() - os.path.getmtime(cache_path) < self.config['cache_expiry']:
                        # Load from cache
                        # Convert timestamp to datetime if needed
                        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            
                        results[ticker] = df
                        logger.debug(f"Loaded {ticker} data from cache: {len(df)} rows")
                        
                    else:
                        # Fetch from API
                        # Run async method in the event loop
                        aggs = self.loop.run_until_complete(self.polygon.get_aggregates(
                            ticker=ticker,
                            multiplier=multiplier,
                            timespan=timespan,
                            from_date=start_str,
                            to_date=end_str,
                            limit=50000
                        ))
                        
                        # Convert to DataFrame
                        if isinstance(aggs, pd.DataFrame):
                            df = aggs
                            df['timestamp'] = pd.to_datetime(df['t'], unit='ms') if 't' in df.columns else pd.to_datetime(df.index)
                        else:
                            try:
                                # Try to extract results directly
                                if isinstance(aggs, dict) and "results" in aggs and aggs["results"]:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs and isinstance(aggs, dict) and "results" in aggs:
                                    df = pd.DataFrame(aggs["results"])
                                elif aggs:
                                    logger.debug(f"Converting raw data to DataFrame for {ticker}: {type(aggs)}")
                                    df = pd.DataFrame(aggs)
                                else:
                                    logger.warning(f"No data returned for {ticker}")
                                    continue
                            except Exception as e:
                                logger.error(f"Error converting data to DataFrame for {ticker}: {e}")
                                continue
                        
                        # Debug the DataFrame structure
                        logger.debug(f"DataFrame columns for {ticker}: {df.columns.tolist()}")
                        if len(df) > 0:
                            logger.debug(f"First row for {ticker}: {df.iloc[0].to_dict()}")
                        
                        
                        # Ensure timestamp column exists
                        if 'timestamp' not in df.columns and 't' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                            logger.info(f"Created timestamp column from 't' for {ticker}")
                            
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
                        
                        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                        logger.debug(f"Columns after renaming: {df.columns.tolist()}")
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            if df['timestamp'].dtype == 'int64':
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            elif not pd.api.types.is_datetime64_dtype(df['timestamp']):
                                logger.debug(f"Converting timestamp column for {ticker}, type: {df['timestamp'].dtype}")
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                        else:
                            # Create timestamp from 't' column if available
                            if 't' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                                logger.info(f"Created timestamp column from 't' for {ticker}")
                            else:
                                logger.warning(f"No timestamp or 't' column found for {ticker}")
                        
                        # Ensure required columns
                        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                        if not all(col in df.columns for col in required_columns):
                            logger.warning(f"Missing required columns for {ticker}: {[col for col in required_columns if col not in df.columns]}")
                            continue
                        
                        # Save to cache
                        self.cache_functions["save"](df, cache_path)
                        
                        # Apply GPU processing if available
                        if self.use_gpu:
                            try:
                                processed_df = self.gpu_utils['process_dataframe'](df)
                                results[ticker] = processed_df
                                logger.debug(f"Processed {ticker} data with GPU")
                            except Exception as e:
                                logger.warning(f"GPU processing failed for {ticker}: {e}")
                                results[ticker] = df
                        else:
                            results[ticker] = df
                        logger.debug(f"Fetched {ticker} data from API: {len(df)} rows")
                    
                    # Rate limiting
                    time.sleep(1.0 / self.config['rate_limit']['polygon'])
                    
                except Exception as e:
                    logger.error(f"Error loading price data for {ticker}: {str(e)}")
            
            logger.info(f"Loaded price data for {len(results)} tickers")
            return results
            
        except Exception as e:
            logger.error(f"Error loading price data: {str(e)}", exc_info=True)
            return {}
    
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
            logger.info(f"Loading options data for {len(tickers)} tickers from {start_date} to {end_date}")
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Load data for each ticker
            results = {}
            
            for ticker in tickers:
                try:
                    # Check cache first
                    cache_key = f"{ticker}_options_{start_str}_{end_str}"
                    cache_path = os.path.join(self.config['cache_dir'], f"{cache_key}.csv")

                    # Use custom cache function
                    df = self.cache_functions["load"](cache_path)
                    if df is not None and time.time() - os.path.getmtime(cache_path) < self.config['cache_expiry']:
                        # Load from cache
                        # Convert timestamp to datetime if needed
                        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            
                        results[ticker] = df
                        logger.debug(f"Loaded {ticker} options data from cache: {len(df)} rows")
                        
                    else:
                        # Fetch from Unusual Whales API
                        # Run async method in the event loop
                        flow_data = self.loop.run_until_complete(self.unusual_whales.get_flow_alerts(
                            ticker=ticker,
                            limit=1000
                        ))
                        
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
                                    logger.warning(f"Unexpected options data format for {ticker}")
                                    continue
                            except Exception as e:
                                logger.error(f"Error converting options data to DataFrame for {ticker}: {e}")
                                continue
                            
                        # Ensure required columns
                        required_columns = ['ticker', 'timestamp', 'side', 'strike', 'expiration', 'volume', 'open_interest']
                        if not all(col in df.columns for col in required_columns):
                            missing_cols = [col for col in required_columns if col not in df.columns]
                            logger.warning(f"Missing required columns for {ticker} options data: {missing_cols}")
                            
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
                                    df['expiration'] = pd.Timestamp.now() + pd.Timedelta(days=30)
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
                        logger.debug(f"Fetched {ticker} options data from API: {len(df)} rows")
                    
                    # Rate limiting
                    time.sleep(1.0 / self.config['rate_limit']['unusual_whales'])
                    
                except Exception as e:
                    logger.error(f"Error loading options data for {ticker}: {str(e)}")
            
            logger.info(f"Loaded options data for {len(results)} tickers")
            return results
            
        except Exception as e:
            logger.error(f"Error loading options data: {str(e)}", exc_info=True)
            return {}
    
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
            logger.info(f"Loading market data for {symbols} from {start_date} to {end_date}")
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Check cache first
            cache_key = f"market_data_{','.join(symbols)}_{start_str}_{end_str}"
            cache_path = os.path.join(self.config['cache_dir'], f"{cache_key}.csv")

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
                        aggs = self.loop.run_until_complete(self.polygon.get_aggregates(
                            ticker=symbol,
                            multiplier=1,
                            timespan='minute',
                            from_date=start_str,
                            to_date=end_str,
                            limit=50000
                        ))
                        
                        # Convert to DataFrame
                        if isinstance(aggs, pd.DataFrame):
                            df = aggs
                            df['timestamp'] = pd.to_datetime(df['t'], unit='ms') if 't' in df.columns else pd.to_datetime(df.index)
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
                                    logger.warning(f"No data returned for {symbol}")
                                    continue
                            except Exception as e:
                                logger.error(f"Error converting data to DataFrame for {symbol}: {e}")
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
                        
                        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            if df['timestamp'].dtype == 'int64':
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            else:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Add to list
                        dfs.append(df)
                        logger.debug(f"Fetched {symbol} market data: {len(df)} rows")
                        
                        # Rate limiting
                        time.sleep(1.0 / self.config['rate_limit']['polygon'])
                        
                    except Exception as e:
                        logger.error(f"Error loading market data for {symbol}: {str(e)}")
                
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
                        result_reset = result_reset.rename(columns={'timestamp': 'timestamp_col'})
                        result_reset = result_reset.reset_index()
                    elif result.index.name == 'timestamp':
                        result_reset = result.reset_index()
                    
                    df_reset = df.copy()
                    if df.index.name == 'timestamp' and 'timestamp' in df_reset.columns:
                        df_reset = df_reset.rename(columns={'timestamp': 'timestamp_col'})
                        df_reset = df_reset.reset_index()
                    elif df.index.name == 'timestamp':
                        df_reset = df.reset_index()
                    
                    # Merge dataframes
                    result = pd.merge_asof(result_reset, df_reset, on='timestamp', direction='nearest')
                    
                
                # Add market metrics
                for symbol in symbols:
                    symbol_lower = symbol.lower()
                    
                    # Daily change
                    if f"{symbol_lower}_close" in result.columns:
                        result[f"{symbol_lower}_change"] = result[f"{symbol_lower}_close"].pct_change()
                        
                        # Calculate day-over-day change
                        result[f"{symbol_lower}_daily_change"] = result[f"{symbol_lower}_close"].pct_change(periods=390)  # ~1 trading day in minutes
                
                # Save to cache
                self.cache_functions["save"](result, cache_path)
                
                logger.info(f"Loaded market data: {len(result)} rows")
                return result
                
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}", exc_info=True)
            return pd.DataFrame()