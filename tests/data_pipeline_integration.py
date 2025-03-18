#!/usr/bin/env python3
"""
Data Pipeline Integration

This module provides integration between the ML model trainer and data sources.
It acts as a bridge between the ML models and various data providers.
"""

import os
import logging
import redis
import json
import pandas as pd
from datetime import datetime, timedelta
import sys

# Import data loader
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_pipeline_integration')

class DataPipelineIntegration:
    """
    Data Pipeline Integration for ML Model Trainer
    
    This class integrates various data sources and provides a unified interface
    for the ML model trainer to access market data, options data, and other
    information needed for training models.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6380, redis_db=0,
                 polygon_api_key='', unusual_whales_api_key='', use_gpu=True, use_gh200=True):
        """
        Initialize the data pipeline integration
        
        Args:
            redis_host (str): Redis host
            redis_port (int): Redis port
            redis_db (int): Redis database
            polygon_api_key (str): Polygon.io API key
            unusual_whales_api_key (str): Unusual Whales API key
            use_gpu (bool): Whether to use GPU acceleration
            use_gh200 (bool): Whether to use GH200-specific optimizations
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.polygon_api_key = polygon_api_key
        self.unusual_whales_api_key = unusual_whales_api_key
        self.use_gpu = use_gpu
        self.use_gh200 = use_gh200

        # Enforce GPU usage - exit if GPU is disabled
        if not self.use_gpu:
            logger.error("GPU acceleration is required for this application.")
            logger.error("Exiting as GPU is mandatory.")
            sys.exit(1)
        
        if self.use_gh200:
            logger.info("GH200-specific optimizations enabled")
        
        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            logger.warning("Creating a mock Redis client")
            
            # Create a mock Redis client
            class MockRedis:
                def __init__(self):
                    self.data = {}
                
                def set(self, key, value):
                    self.data[key] = value
                    return True
                
                def get(self, key):
                    return self.data.get(key)
                
                def hset(self, name, key, value):
                    if name not in self.data:
                        self.data[name] = {}
                    self.data[name][key] = value
                    return True
                
                def hget(self, name, key):
                    if name in self.data and key in self.data[name]:
                        return self.data[name][key]
                    return None
                
                def hgetall(self, name):
                    return self.data.get(name, {})
                
                def sadd(self, name, *values):
                    if name not in self.data:
                        self.data[name] = set()
                    for value in values:
                        self.data[name].add(value)
                    return len(values)
                
                def smembers(self, name):
                    return self.data.get(name, set())
                
                def zrange(self, name, start, end, withscores=False):
                    return []
                
                def publish(self, channel, message):
                    return 0
                
                def ping(self):
                    return True
            
            self.redis = MockRedis()
        
        # Initialize API clients
        try:
            from gpu_optimized_polygon_api_client import GPUPolygonAPIClient
            # Initialize with only the parameters that are supported
            self.polygon_api = GPUPolygonAPIClient(
                api_key=self.polygon_api_key,
                redis_client=self.redis,
                use_gpu=True  # Force GPU usage
            )
            logger.info("Initialized Polygon API client")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon API client: {e}")
            self.polygon_api = None
        
        try:
            from gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
            # Initialize with only the parameters that are supported
            self.unusual_whales = GPUUnusualWhalesClient(
                api_key=self.unusual_whales_api_key,
                redis_client=self.redis,
                use_gpu=True  # Force GPU usage
            )
            logger.info("Initialized Unusual Whales API client")
        except Exception as e:
            logger.error(f"Failed to initialize Unusual Whales API client: {e}")
            self.unusual_whales = None
        
        # Initialize data loader
        self.data_loader = DataLoader(
            polygon_client=self.polygon_api,
            unusual_whales_client=self.unusual_whales,
            redis_client=self.redis,
            use_gh200=self.use_gh200
        )
        
        logger.info("Data Pipeline Integration initialized")
    
    # Delegate methods to the data_loader
    
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
        return self.data_loader.load_price_data(tickers, start_date, end_date, timeframe)
    
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
        return self.data_loader.load_options_data(tickers, start_date, end_date)
    
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
        return self.data_loader.load_market_data(start_date, end_date, symbols)


# Example usage
if __name__ == "__main__":
    # Create data pipeline
    pipeline = DataPipelineIntegration(
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=int(os.environ.get('REDIS_PORT', 6380)),
        redis_db=int(os.environ.get('REDIS_DB', 0)),
        polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
        unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
        use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true',
        use_gh200=os.environ.get('USE_GH200', 'true').lower() == 'true'
    )
    
    # Test loading data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Load price data
    price_data = pipeline.load_price_data(
        tickers=['SPY', 'AAPL', 'MSFT'],
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    print(f"Loaded price data for {len(price_data)} tickers")