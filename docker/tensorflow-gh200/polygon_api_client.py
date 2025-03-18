#!/usr/bin/env python3
"""
Polygon.io REST API Client

This module provides a client for interacting with the Polygon.io REST API,
which offers market data for stocks, options, forex, and crypto.
"""

import os
import time
import json
import logging
import hashlib
import pickle
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import random
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('polygon_api_client')

# Environment variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
CACHE_TTL = int(os.environ.get('POLYGON_CACHE_TTL', 3600))  # 1 hour default TTL
MAX_RETRIES = int(os.environ.get('POLYGON_MAX_RETRIES', 5))
RETRY_BACKOFF_FACTOR = float(os.environ.get('POLYGON_RETRY_BACKOFF_FACTOR', 0.5))
CONNECTION_TIMEOUT = int(os.environ.get('POLYGON_CONNECTION_TIMEOUT', 15))
MAX_POOL_SIZE = int(os.environ.get('POLYGON_MAX_POOL_SIZE', 30))


class MemoryCache:
    """In-memory cache for API responses"""

    def __init__(self, ttl=CACHE_TTL):
        """Initialize memory cache"""
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.size_limit = 10000  # Maximum number of items in cache

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
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expiry']:
                self.hits += 1
                if self.hits % 1000 == 0:
                    logger.info(f"Cache hits: {self.hits}, misses: {self.misses}")
                return entry['value']
            else:
                # Entry expired
                del self.cache[key]
        
        self.misses += 1
        return None

    def set(self, key_parts, value, ttl=None):
        """Set value in cache"""
        if value is None:
            return False

        key = self._generate_key(key_parts)
        ttl = ttl or self.ttl
        expiry = time.time() + ttl

        # Implement LRU-like behavior by limiting cache size
        if len(self.cache) >= self.size_limit:
            # Remove random 10% of entries when cache gets too large
            keys_to_remove = random.sample(list(self.cache.keys()),
                                          int(len(self.cache) * 0.1))
            for k in keys_to_remove:
                self.cache.pop(k, None)
        
        self.cache[key] = {
            'value': value,
            'expiry': expiry
        }
        
        return True

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("Cache cleared")


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
            'User-Agent': 'PolygonClient/1.0',
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


class PolygonAPIClient:
    """Client for the Polygon.io REST API"""

    def __init__(self, api_key=POLYGON_API_KEY, max_pool_size=MAX_POOL_SIZE, 
                 max_retries=MAX_RETRIES, cache_ttl=CACHE_TTL):
        """
        Initialize Polygon API client
        
        Args:
            api_key: API key for authentication
            max_pool_size: Maximum connection pool size
            max_retries: Maximum number of retries for failed requests
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        
        # Verify API key is provided
        if not self.api_key:
            logger.warning("No API key provided. Set the POLYGON_API_KEY environment variable.")
        else:
            logger.info(f"Initialized Polygon API client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}")
            
        # Initialize base URL
        self.base_url = "https://api.polygon.io"
        
        # Initialize cache
        self.cache = MemoryCache(ttl=cache_ttl)
        
        # Initialize connection pool
        self.connection_pool = OptimizedConnectionPool(
            max_pool_size=max_pool_size, max_retries=max_retries)
            
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

    def get_ticker_news(self, ticker, limit=10):
        """
        Get news for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news items to return
            
        Returns:
            list: News items
        """
        try:
            endpoint = f"v2/reference/news"
            params = {
                "ticker": ticker,
                "limit": min(limit, 1000)  # API max is 1000
            }
            
            data = self._make_request(endpoint, params)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"News retrieved for {ticker}: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get news for {ticker}: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
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
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Validate ticker
        if not ticker:
            logger.warning("Empty ticker provided")
            return pd.DataFrame()

        ticker = ticker.upper()  # Ensure ticker is uppercase for consistency
        
        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": min(limit, 50000), "adjusted": str(adjusted).lower()}

        # Make request
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

            logger.info(f"Retrieved {len(df)} aggregates for {ticker}")
            return df
        elif data.get("status") == "OK" and ("results" not in data or not data["results"]):
            # No results found, but API call was successful
            logger.warning(f"No data found for {ticker} in the specified date range")
            return pd.DataFrame()
        else:
            # API call failed
            error_msg = data.get("error", "Unknown error") if isinstance(data, dict) else "Unknown error"
            logger.warning(f"Failed to get aggregates for {ticker}: {error_msg}")
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

    def get_daily_open_close(self, ticker, date):
        """
        Get the open, close, high, and low for a ticker on a specific date
        
        Args:
            ticker: Stock ticker symbol
            date: The date in format YYYY-MM-DD
            
        Returns:
            dict: Daily open/close data
        """
        try:
            endpoint = f"v1/open-close/{ticker}/{date}"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Daily open/close retrieved for {ticker} on {date}")
                return data
            else:
                logger.warning(f"Failed to get daily open/close for {ticker} on {date}: {data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            logger.error(f"Error getting daily open/close for {ticker} on {date}: {e}")
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

    def get_ticker_types(self):
        """
        Get all ticker types
        
        Returns:
            list: Ticker types
        """
        try:
            endpoint = "v3/reference/tickers/types"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info("Ticker types retrieved")
                return data.get("results", {}).get("types", [])
            else:
                logger.warning(f"Failed to get ticker types: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting ticker types: {e}")
            return []

    def get_stock_splits(self, ticker):
        """
        Get stock splits for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            list: Stock splits
        """
        try:
            endpoint = f"v3/reference/splits"
            params = {"ticker": ticker}
            
            data = self._make_request(endpoint, params)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Stock splits retrieved for {ticker}: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get stock splits for {ticker}: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting stock splits for {ticker}: {e}")
            return []

    def get_stock_dividends(self, ticker):
        """
        Get stock dividends for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            list: Stock dividends
        """
        try:
            endpoint = f"v3/reference/dividends"
            params = {"ticker": ticker}
            
            data = self._make_request(endpoint, params)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Stock dividends retrieved for {ticker}: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get stock dividends for {ticker}: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting stock dividends for {ticker}: {e}")
            return []

    def get_stock_financials(self, ticker, limit=5):
        """
        Get stock financials for a ticker
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return
            
        Returns:
            list: Stock financials
        """
        try:
            endpoint = f"v2/reference/financials/{ticker}"
            params = {"limit": min(limit, 100)}  # API max is 100
            
            data = self._make_request(endpoint, params)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Stock financials retrieved for {ticker}: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get stock financials for {ticker}: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting stock financials for {ticker}: {e}")
            return []

    def get_market_holidays(self):
        """
        Get market holidays
        
        Returns:
            list: Market holidays
        """
        try:
            endpoint = "v1/marketstatus/upcoming"
            data = self._make_request(endpoint)
            
            if data:
                logger.info(f"Market holidays retrieved: {len(data)} items")
                return data
            else:
                logger.warning("Failed to get market holidays")
                return []
        except Exception as e:
            logger.error(f"Error getting market holidays: {e}")
            return []

    def get_stock_exchanges(self):
        """
        Get stock exchanges
        
        Returns:
            list: Stock exchanges
        """
        try:
            endpoint = "v3/reference/exchanges"
            data = self._make_request(endpoint)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Stock exchanges retrieved: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get stock exchanges: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting stock exchanges: {e}")
            return []

    def get_conditions(self, asset_class="stocks"):
        """
        Get conditions for a specific asset class
        
        Args:
            asset_class: Asset class (stocks, options, forex, crypto)
            
        Returns:
            dict: Conditions
        """
        try:
            endpoint = f"v3/reference/conditions"
            params = {"asset_class": asset_class}
            
            data = self._make_request(endpoint, params)
            
            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Conditions retrieved for {asset_class}: {len(data.get('results', []))} items")
                return data.get("results", [])
            else:
                logger.warning(f"Failed to get conditions for {asset_class}: {data.get('error', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error getting conditions for {asset_class}: {e}")
            return []

    def close(self):
        """Close all connections and resources"""
        logger.info("Closing Polygon API client")
        self.running = False
        self.connection_pool.close()
        logger.info("Polygon API client closed")


# Example usage
if __name__ == "__main__":
    # Create client
    client = PolygonAPIClient()
    
    # Get market status
    market_status = client.get_market_status()
    print("\nMarket Status:")
    print(market_status)
    
    # Get ticker details
    ticker = "AAPL"
    ticker_details = client.get_ticker_details(ticker)
    print(f"\nTicker Details for {ticker}:")
    print(ticker_details)
    
    # Get aggregates
    aggregates = client.get_aggregates(ticker, timespan="day", limit=5)
    print(f"\nAggregates for {ticker}:")
    print(aggregates.head())
    
    # Close client
    client.close()