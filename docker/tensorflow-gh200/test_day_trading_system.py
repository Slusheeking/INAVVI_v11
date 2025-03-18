#!/usr/bin/env python3
"""
Comprehensive test script for AI Day Trading System

This script tests the integration of multiple data sources:
1. Polygon REST API
2. Polygon WebSocket
3. Unusual Whales API

It simulates the data flow for a day trading system with position limits
and provides performance metrics for each data source.
"""

import os
import sys
import time
import json
import random
import logging
import asyncio
import argparse
import pandas as pd
import numpy as np
import websockets
from datetime import datetime, timedelta, date
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our enhanced clients
from polygon_data_source_ultra import PolygonDataSourceUltra, log_memory_usage
# We'll define UnusualWhalesClient directly in this file instead of importing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('day_trading_system_test')

# API Keys
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf')
UNUSUAL_WHALES_API_KEY = '4ad71b9e-7ace-4f24-bdfc-532ace219a18'

# Trading parameters
MAX_POSITION_VALUE = 2500.0  # Maximum $2500 per stock
MAX_DAILY_VALUE = 5000.0     # Maximum $5000 per day
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

class UnusualWhalesClient:
    """
    Simplified Unusual Whales API client for testing purposes.
    
    This is a streamlined version of the full client that includes only
    the methods needed for the day trading system test.
    """
    
    def __init__(
        self,
        api_key: str = UNUSUAL_WHALES_API_KEY,
        base_url: str = "https://api.unusualwhales.com/api",
        max_position_value: float = MAX_POSITION_VALUE,
        max_daily_value: float = MAX_DAILY_VALUE,
    ):
        """
        Initialize the Unusual Whales API client with dollar-based position limits.
        
        Args:
            api_key: Unusual Whales API key
            base_url: Base URL for API requests
            max_position_value: Maximum position value per stock in dollars
            max_daily_value: Maximum total position value per day in dollars
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Unusual Whales API key is required")
            
        self.BASE_URL = base_url
        
        # Position tracking for dollar-based limits
        self.max_position_value = max_position_value
        self.max_daily_value = max_daily_value
        self.current_daily_value = 0.0
        self.position_values = {}
        self.position_count = 0
        self.last_reset_date = datetime.now().date()
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}", 
            "Accept": "application/json, text/plain", 
            "User-Agent": "DayTradingSystemTest/0.1"
        })
        
        logger.info(f"Initialized UnusualWhalesClient with position limits: ${max_position_value} per stock, ${max_daily_value} daily")
    
    def _make_request(self, endpoint, params=None, method="GET"):
        """
        Make a request to the Unusual Whales API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters or JSON body (for POST requests)
            method: HTTP method (GET or POST)
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        
        logger.debug(f"Making request to {method} {url} with params: {params}")
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}, must be GET or POST")
            
            # Handle HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for API-level errors
            if data.get("status") == "error" or data.get("error"):
                error_msg = data.get("message", "Unknown API error")
                if "error_code" in data:
                    error_msg = f"{data.get('error_code')}: {data.get('error_message', 'Unknown error')}"
                raise ValueError(f"Unusual Whales API error: {error_msg}")
            
            logger.debug(f"Successful response from {url}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in request to {url}: {e}")
            # For testing, return empty data rather than raising an exception
            if method.upper() == "GET":
                return {"data": []}
            else:
                return {"data": {}}
    
    def _validate_date_param(self, date_param):
        """
        Validate and convert date parameter to string format.
        
        Args:
            date_param: Date parameter (string, datetime, or date)
            
        Returns:
            Validated date string or None
        """
        if isinstance(date_param, (datetime, date)):
            return date_param.strftime("%Y-%m-%d")
        return date_param
    
    def get_options_flow(self, limit=100, ticker=None, **additional_params):
        """
        Get options flow data.
        
        Args:
            limit: Number of results to return (default: 100)
            ticker: Filter by ticker symbol
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of options flow data
        """
        params = {"limit": min(limit, 200)}
        
        # Add ticker filter if provided
        if ticker:
            params["ticker_symbols"] = ticker
        
        # Add any additional parameters
        params.update(additional_params)
        
        # For testing, we'll generate mock data if the API request fails
        try:
            result = self._make_request("/alerts", params)
            data = result.get("data", [])
            
            if not data and ticker:
                # Try using the options/trades endpoint directly
                options_params = {"limit": min(limit, 200)}
                if ticker:
                    options_params["symbol"] = ticker
                options_result = self._make_request("/option-trades/flow-alerts", options_params)
                data = options_result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_options_flow(limit, ticker)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching options flow: {e}")
            # Generate mock data for testing
            return self._generate_mock_options_flow(limit, ticker)
    
    def get_flow_alerts(self, ticker, limit=10, **additional_params):
        """
        Get flow alerts for a specific ticker.
        
        Args:
            ticker: Stock symbol
            limit: Number of results to return (default: 10)
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of flow alerts
        """
        params = {"limit": min(limit, 200)}
        
        # Add any additional parameters
        params.update(additional_params)
        
        try:
            result = self._make_request(f"/stock/{ticker}/flow-alerts", params)
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_flow_alerts(ticker, limit)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching flow alerts for {ticker}: {e}")
            # Generate mock data for testing
            return self._generate_mock_flow_alerts(ticker, limit)
    
    def get_darkpool_recent(self, limit=10, **additional_params):
        """
        Get recent dark pool trades.
        
        Args:
            limit: Number of results to return (default: 10)
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of dark pool trades
        """
        params = {"limit": min(limit, 200)}
        
        # Add any additional parameters
        params.update(additional_params)
        
        try:
            result = self._make_request("/darkpool/recent", params)
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_darkpool_data(limit)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching recent dark pool trades: {e}")
            # Generate mock data for testing
            return self._generate_mock_darkpool_data(limit)
    
    def get_darkpool_ticker(self, ticker, limit=10, **additional_params):
        """
        Get dark pool trades for a specific ticker.
        
        Args:
            ticker: Stock symbol
            limit: Number of results to return (default: 10)
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of dark pool trades for the ticker
        """
        params = {"limit": min(limit, 200)}
        
        # Add any additional parameters
        params.update(additional_params)
        
        try:
            result = self._make_request(f"/darkpool/{ticker}", params)
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_darkpool_data(limit, ticker)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching dark pool trades for {ticker}: {e}")
            # Generate mock data for testing
            return self._generate_mock_darkpool_data(limit, ticker)
    
    def get_insider_transactions(self, limit=10, **additional_params):
        """
        Get insider transactions.
        
        Args:
            limit: Number of results to return (default: 10)
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of insider transactions
        """
        params = {"limit": min(limit, 200)}
        
        # Add any additional parameters
        params.update(additional_params)
        
        try:
            result = self._make_request("/insider/transactions", params)
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_insider_transactions(limit)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching insider transactions: {e}")
            # Generate mock data for testing
            return self._generate_mock_insider_transactions(limit)
    
    def get_ticker_flow(self, ticker):
        """
        Get an aggregated view of the insider flow for the given ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            List of insider flow data for the ticker
        """
        try:
            result = self._make_request(f"/insider/{ticker}/ticker-flow")
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_ticker_flow(ticker)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching ticker flow for {ticker}: {e}")
            # Generate mock data for testing
            return self._generate_mock_ticker_flow(ticker)
    
    def get_alerts(self, limit=10, **additional_params):
        """
        Get all alerts that have been triggered.
        
        Args:
            limit: Number of results to return (default: 10)
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            List of alerts
        """
        params = {"limit": min(limit, 200)}
        
        # Add any additional parameters
        params.update(additional_params)
        
        try:
            result = self._make_request("/alerts", params)
            data = result.get("data", [])
            
            if not data:
                # Generate mock data for testing
                data = self._generate_mock_alerts(limit)
                
            return data
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            # Generate mock data for testing
            return self._generate_mock_alerts(limit)
    
    def can_take_position(self, ticker, value):
        """
        Check if a position can be taken based on dollar-based limits.
        
        Args:
            ticker: Ticker symbol
            value: Position value in dollars
            
        Returns:
            Whether the position can be taken
        """
        # Reset daily tracking if it's a new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            logger.info(f"Resetting daily position tracking (new day: {current_date})")
            self.current_daily_value = 0.0
            self.position_values = {}
            self.position_count = 0
            self.last_reset_date = current_date
        
        # Check if adding this position would exceed the daily limit
        if self.current_daily_value + value > self.max_daily_value:
            logger.warning(f"Position for {ticker} (${value:.2f}) would exceed daily limit of ${self.max_daily_value:.2f} (current: ${self.current_daily_value:.2f})")
            return False
        
        # Check if adding this position would exceed the per-stock limit
        current_ticker_value = self.position_values.get(ticker, 0.0)
        if current_ticker_value + value > self.max_position_value:
            logger.warning(f"Position for {ticker} (${value:.2f}) would exceed per-stock limit of ${self.max_position_value:.2f} (current: ${current_ticker_value:.2f})")
            return False
        
        return True
    
    def update_position_tracking(self, ticker, value):
        """
        Update position tracking after taking a position.
        
        Args:
            ticker: Ticker symbol
            value: Position value in dollars
        """
        # Update daily tracking
        self.current_daily_value += value
        
        # Update per-stock tracking
        current_ticker_value = self.position_values.get(ticker, 0.0)
        self.position_values[ticker] = current_ticker_value + value
        
        # Update position count
        self.position_count += 1
        
        logger.info(f"Position taken for {ticker}: ${value:.2f} (daily total: ${self.current_daily_value:.2f}/{self.max_daily_value:.2f}, ticker total: ${self.position_values[ticker]:.2f}/{self.max_position_value:.2f})")
    
    def close(self):
        """
        Close the session and release resources.
        """
        if self.session:
            logger.debug("Closing session and releasing resources")
            self.session.close()
    
    # Mock data generation methods for testing
    def _generate_mock_options_flow(self, limit=10, ticker=None):
        """Generate mock options flow data for testing"""
        symbols = [ticker] if ticker else DEFAULT_SYMBOLS
        result = []
        
        for _ in range(limit):
            symbol = random.choice(symbols)
            strike = round(random.uniform(50, 500), 2)
            premium = round(random.uniform(0.5, 10), 2)
            
            option = {
                "symbol": symbol,
                "strike": strike,
                "premium": premium,
                "type": random.choice(["call", "put"]),
                "expiration": (datetime.now() + timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                "volume": random.randint(100, 10000),
                "open_interest": random.randint(100, 50000),
                "unusual_score": round(random.uniform(50, 100), 2)
            }
            result.append(option)
            
        return result
    
    def _generate_mock_flow_alerts(self, ticker, limit=10):
        """Generate mock flow alerts for testing"""
        result = []
        
        for _ in range(limit):
            alert = {
                "symbol": ticker,
                "alert_type": random.choice(["unusual_volume", "large_block", "sweep"]),
                "timestamp": datetime.now().isoformat(),
                "price": round(random.uniform(50, 500), 2),
                "volume": random.randint(100, 10000),
                "sentiment": random.choice(["bullish", "bearish", "neutral"])
            }
            result.append(alert)
            
        return result
    
    def _generate_mock_darkpool_data(self, limit=10, ticker=None):
        """Generate mock dark pool data for testing"""
        symbols = [ticker] if ticker else DEFAULT_SYMBOLS
        result = []
        
        for _ in range(limit):
            symbol = random.choice(symbols)
            
            trade = {
                "symbol": symbol,
                "price": round(random.uniform(50, 500), 2),
                "volume": random.randint(1000, 100000),
                "timestamp": datetime.now().isoformat(),
                "exchange": random.choice(["XDARK", "XADF", "XPST"]),
                "trade_id": str(random.randint(1000000, 9999999))
            }
            result.append(trade)
            
        return result
    
    def _generate_mock_insider_transactions(self, limit=10):
        """Generate mock insider transactions for testing"""
        result = []
        
        for _ in range(limit):
            symbol = random.choice(DEFAULT_SYMBOLS)
            
            transaction = {
                "symbol": symbol,
                "insider_name": f"John Doe {random.randint(1, 100)}",
                "title": random.choice(["CEO", "CFO", "CTO", "Director"]),
                "transaction_type": random.choice(["buy", "sell"]),
                "shares": random.randint(100, 10000),
                "price": round(random.uniform(50, 500), 2),
                "value": round(random.uniform(10000, 1000000), 2),
                "filing_date": datetime.now().isoformat()
            }
            result.append(transaction)
            
        return result
    
    def _generate_mock_ticker_flow(self, ticker):
        """Generate mock ticker flow data for testing"""
        return [{
            "symbol": ticker,
            "buy_count": random.randint(1, 20),
            "sell_count": random.randint(1, 20),
            "net_count": random.randint(-10, 10),
            "buy_value": round(random.uniform(100000, 10000000), 2),
            "sell_value": round(random.uniform(100000, 10000000), 2),
            "net_value": round(random.uniform(-5000000, 5000000), 2),
            "period": "3m"
        }]
    
    def _generate_mock_alerts(self, limit=10):
        """Generate mock alerts for testing"""
        result = []
        
        for _ in range(limit):
            symbol = random.choice(DEFAULT_SYMBOLS)
            
            alert = {
                "symbol": symbol,
                "alert_type": random.choice(["price_target", "earnings", "insider_trade", "unusual_options"]),
                "timestamp": datetime.now().isoformat(),
                "message": f"Alert for {symbol}: {random.choice(['Price target increased', 'Earnings beat', 'Insider buying', 'Unusual call activity'])}",
                "source": random.choice(["analyst", "company", "sec", "market"])
            }
            result.append(alert)
            
        return result

class PolygonRESTClient:
    """Polygon REST API client for market data"""
    
    def __init__(self, api_key=POLYGON_API_KEY):
        """Initialize Polygon REST client"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PolygonDayTradingSystem/1.0",
            "Connection": "keep-alive"
        })
        
    def get_aggregates(self, symbol, multiplier=1, timespan="minute", from_date=None, to_date=None, limit=1000):
        """
        Get aggregated data for a symbol
        
        Args:
            symbol: Stock symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            limit: Number of results to return
            
        Returns:
            DataFrame with aggregated data
        """
        # Format dates
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        elif isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")
            
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")
            
        # Build URL
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        # Make request
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                df = pd.DataFrame(data["results"])
                df["symbol"] = symbol
                
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                
                # Rename columns
                df = df.rename(columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "n": "transactions"
                })
                
                return df
            else:
                logger.warning(f"No results for {symbol}: {data.get('error', 'Unknown error')}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching aggregates for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_ticker_details(self, symbol):
        """
        Get details for a ticker
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with ticker details
        """
        url = f"{self.base_url}/v3/reference/tickers/{symbol}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No details for {symbol}: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {e}")
            return {}
            
    def get_market_snapshot(self, tickers=None):
        """
        Get current market snapshot for tickers
        
        Args:
            tickers: List of stock symbols (optional)
            
        Returns:
            Dictionary with market snapshots
        """
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        
        params = {}
        if tickers:
            params["tickers"] = ",".join(tickers)
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "tickers" in data:
                return data["tickers"]
            else:
                logger.warning(f"No snapshot data: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching market snapshot: {e}")
            return []
            
    def get_open_close(self, symbol, date=None):
        """
        Get open/close data for a symbol on a specific date
        
        Args:
            symbol: Stock symbol
            date: Date (YYYY-MM-DD or datetime)
            
        Returns:
            Dictionary with open/close data
        """
        # Format date
        if date is None:
            # Use previous trading day
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
            
        url = f"{self.base_url}/v1/open-close/{symbol}/{date}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK":
                return data
            else:
                logger.warning(f"No open/close data for {symbol} on {date}: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching open/close data for {symbol} on {date}: {e}")
            return {}
            
    def get_options_contracts(self, underlying_ticker, expiration_date=None, contract_type=None, limit=100):
        """
        Get options contracts for an underlying ticker
        
        Args:
            underlying_ticker: Stock symbol
            expiration_date: Expiration date (YYYY-MM-DD or datetime)
            contract_type: Contract type (call or put)
            limit: Number of results to return
            
        Returns:
            List of options contracts
        """
        url = f"{self.base_url}/v3/reference/options/contracts"
        
        params = {
            "underlying_ticker": underlying_ticker,
            "limit": limit
        }
        
        if expiration_date:
            if isinstance(expiration_date, datetime):
                params["expiration_date"] = expiration_date.strftime("%Y-%m-%d")
            else:
                params["expiration_date"] = expiration_date
                
        if contract_type:
            params["contract_type"] = contract_type
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No options contracts for {underlying_ticker}: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching options contracts for {underlying_ticker}: {e}")
            return []
            
    def get_ticker_news(self, ticker=None, limit=10, order="desc", sort="published_utc", published_utc_gte=None, published_utc_lte=None):
        """
        Get news articles for a ticker
        
        Args:
            ticker: Stock symbol (optional)
            limit: Number of results to return (default: 10)
            order: Order of results (asc or desc)
            sort: Field to sort by
            published_utc_gte: Published date greater than or equal to (YYYY-MM-DD)
            published_utc_lte: Published date less than or equal to (YYYY-MM-DD)
            
        Returns:
            List of news articles
        """
        url = f"{self.base_url}/v2/reference/news"
        
        params = {
            "limit": limit,
            "order": order,
            "sort": sort
        }
        
        if ticker:
            params["ticker"] = ticker
            
        if published_utc_gte:
            params["published_utc.gte"] = published_utc_gte
            
        if published_utc_lte:
            params["published_utc.lte"] = published_utc_lte
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No news for {ticker}: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
            
    def get_ticker_financials(self, ticker, limit=5, type="Q", sort="filing_date", order="desc"):
        """
        Get financial data for a ticker
        
        Args:
            ticker: Stock symbol
            limit: Number of results to return (default: 5)
            type: Report type (Q for quarterly, A for annual)
            sort: Field to sort by
            order: Order of results (asc or desc)
            
        Returns:
            List of financial reports
        """
        url = f"{self.base_url}/v2/reference/financials/{ticker}"
        
        params = {
            "limit": limit,
            "type": type,
            "sort": sort,
            "order": order
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No financials for {ticker}: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return []
            
    def get_previous_close(self, ticker):
        """
        Get previous day's close data for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with previous close data
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"][0] if data["results"] else {}
            else:
                logger.warning(f"No previous close data for {ticker}: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching previous close for {ticker}: {e}")
            return {}
            
    def get_ticker_types(self):
        """
        Get all ticker types
        
        Returns:
            Dictionary with ticker types
        """
        url = f"{self.base_url}/v3/reference/tickers/types"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No ticker types: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching ticker types: {e}")
            return {}
            
    def get_exchanges(self):
        """
        Get all exchanges
        
        Returns:
            List of exchanges
        """
        url = f"{self.base_url}/v3/reference/exchanges"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No exchanges: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching exchanges: {e}")
            return []
            
    def get_market_holidays(self):
        """
        Get market holidays
        
        Returns:
            List of market holidays
        """
        url = f"{self.base_url}/v1/marketstatus/upcoming"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            return data
                
        except Exception as e:
            logger.error(f"Error fetching market holidays: {e}")
            return []
            
    def get_market_status(self):
        """
        Get current market status
        
        Returns:
            Dictionary with market status
        """
        url = f"{self.base_url}/v1/marketstatus/now"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            return data
                
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {}
            
    def get_ticker_events(self, ticker, event_type="dividend", limit=10):
        """
        Get events for a ticker (dividends, splits, etc.)
        
        Args:
            ticker: Stock symbol
            event_type: Event type (dividend, split)
            limit: Number of results to return
            
        Returns:
            List of events
        """
        if event_type == "dividend":
            url = f"{self.base_url}/v3/reference/dividends"
            params = {"ticker": ticker, "limit": limit}
        elif event_type == "split":
            url = f"{self.base_url}/v3/reference/splits"
            params = {"ticker": ticker, "limit": limit}
        else:
            logger.warning(f"Invalid event type: {event_type}")
            return []
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No {event_type} events for {ticker}: {data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching {event_type} events for {ticker}: {e}")
            return []
            
    def get_technical_indicators(self, ticker, indicator_type="rsi", timespan="day", window=14, series_type="close", limit=1):
        """
        Get technical indicators for a ticker
        
        Args:
            ticker: Stock symbol
            indicator_type: Indicator type (rsi, ema, sma, macd)
            timespan: Timespan (minute, hour, day, week, month, quarter, year)
            window: Window size for the indicator
            series_type: Series type (close, open, high, low)
            limit: Number of results to return
            
        Returns:
            Dictionary with technical indicator data
        """
        url = f"{self.base_url}/v1/indicators/{indicator_type}/{ticker}"
        
        params = {
            "timespan": timespan,
            "window": window,
            "series_type": series_type,
            "limit": limit
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "OK" and "results" in data:
                return data["results"]
            else:
                logger.warning(f"No {indicator_type} data for {ticker}: {data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching {indicator_type} for {ticker}: {e}")
            return {}
            
    def close(self):
        """Close the session"""
        self.session.close()

class PolygonWebSocketClient:
    """Polygon WebSocket client for real-time market data"""
    
    def __init__(self, api_key=POLYGON_API_KEY):
        """Initialize WebSocket client"""
        self.api_key = api_key
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.subscribed_channels = set()
        self.messages = []
        self.running = True
        
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            self.ws = await websockets.connect("wss://socket.polygon.io/stocks")
            self.connected = True
            logger.info("Connected to Polygon WebSocket")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
            
    async def authenticate(self):
        """Authenticate with API key"""
        if not self.connected:
            logger.error("Cannot authenticate: not connected")
            return False
            
        auth_message = {"action": "auth", "params": self.api_key}
        await self.ws.send(json.dumps(auth_message))
        
        # Wait for auth response
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=5)
            data = json.loads(response)
            
            if isinstance(data, list) and len(data) > 0 and "status" in data[0] and data[0]["status"] == "auth_success":
                self.authenticated = True
                logger.info("Authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {data}")
                return False
        except asyncio.TimeoutError:
            logger.error("Authentication timed out")
            return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
        
    async def subscribe(self, channels):
        """Subscribe to channels"""
        if not self.authenticated:
            logger.error("Cannot subscribe: not authenticated")
            return False
            
        sub_message = {"action": "subscribe", "params": ",".join(channels)}
        await self.ws.send(json.dumps(sub_message))
        self.subscribed_channels.update(channels)
        logger.info(f"Subscribed to channels: {channels}")
        return True
        
    async def listen(self, duration=10):
        """Listen for messages for a specified duration"""
        if not self.authenticated:
            logger.error("Cannot listen: not authenticated")
            return []
            
        logger.info(f"Listening for messages for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1)
                data = json.loads(message)
                self.messages.append(data)
                logger.debug(f"Received message: {data}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
                
        logger.info(f"Received {len(self.messages)} messages")
        return self.messages
        
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            self.authenticated = False
            logger.info("WebSocket connection closed")
            
    def generate_mock_data(self, symbols, num_messages=50):
        """Generate mock WebSocket data for testing during non-market hours"""
        logger.info(f"Generating {num_messages} mock messages for {len(symbols)} symbols")
        mock_messages = []
        
        for _ in range(num_messages):
            symbol = np.random.choice(symbols)
            
            # Generate a random message type
            message_type = np.random.choice(["T", "Q", "A"])
            
            if message_type == "T":  # Trade
                message = {
                    "ev": "T",
                    "sym": symbol,
                    "p": round(np.random.uniform(50, 500), 2),  # Price
                    "s": np.random.randint(1, 1000),  # Size
                    "t": int(time.time() * 1000),  # Timestamp
                    "c": [0],  # Conditions
                    "i": str(np.random.randint(1000000, 9999999))  # Trade ID
                }
            elif message_type == "Q":  # Quote
                price = round(np.random.uniform(50, 500), 2)
                message = {
                    "ev": "Q",
                    "sym": symbol,
                    "bp": price - np.random.uniform(0.01, 0.1),  # Bid price
                    "bs": np.random.randint(1, 1000),  # Bid size
                    "ap": price + np.random.uniform(0.01, 0.1),  # Ask price
                    "as": np.random.randint(1, 1000),  # Ask size
                    "t": int(time.time() * 1000),  # Timestamp
                    "c": [0],  # Conditions
                    "i": str(np.random.randint(1000000, 9999999))  # Quote ID
                }
            else:  # Aggregate
                price = round(np.random.uniform(50, 500), 2)
                message = {
                    "ev": "A",
                    "sym": symbol,
                    "v": np.random.randint(1, 10000),  # Volume
                    "o": price - np.random.uniform(0.5, 2),  # Open
                    "c": price,  # Close
                    "h": price + np.random.uniform(0.1, 1),  # High
                    "l": price - np.random.uniform(0.1, 1),  # Low
                    "a": np.random.randint(1, 100),  # Average
                    "s": int(time.time() * 1000),  # Start timestamp
                    "e": int(time.time() * 1000) + 60000  # End timestamp
                }
                
            mock_messages.append(message)
            
        self.messages = mock_messages
        return mock_messages

class DayTradingSystem:
    """
    Day Trading System that integrates multiple data sources
    
    This class simulates a day trading system that:
    1. Uses Polygon REST API for historical data and reference data
    2. Uses Polygon WebSocket for real-time market data
    3. Uses Unusual Whales API for options flow, dark pool, and insider trading data
    4. Implements position limits and risk management
    """
    
    def __init__(self, 
                 polygon_api_key=POLYGON_API_KEY, 
                 unusual_whales_api_key=UNUSUAL_WHALES_API_KEY,
                 max_position_value=MAX_POSITION_VALUE,
                 max_daily_value=MAX_DAILY_VALUE):
        """Initialize Day Trading System"""
        self.polygon_rest = PolygonRESTClient(api_key=polygon_api_key)
        self.polygon_ultra = PolygonDataSourceUltra(api_key=polygon_api_key)
        self.unusual_whales = UnusualWhalesClient(api_key=unusual_whales_api_key,
                                                 max_position_value=max_position_value,
                                                 max_daily_value=max_daily_value)
        
        # Position tracking
        self.max_position_value = max_position_value
        self.max_daily_value = max_daily_value
        self.positions = {}
        self.daily_value = 0.0
        
        # Data storage
        self.historical_data = {}
        self.real_time_data = {}
        self.options_data = {}
        self.dark_pool_data = {}
        self.insider_data = {}
        
        logger.info(f"Initialized Day Trading System with position limits: ${max_position_value} per stock, ${max_daily_value} daily")
        
    def fetch_historical_data(self, symbols, timespan="minute", days=5):
        """
        Fetch historical data for symbols
        
        Args:
            symbols: List of stock symbols
            timespan: Timespan for aggregates (minute, hour, day)
            days: Number of days of historical data
            
        Returns:
            Dictionary with historical data by symbol
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            futures = {executor.submit(self.polygon_rest.get_aggregates, symbol, 1, timespan, from_date, to_date): symbol for symbol in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[symbol] = data
                        logger.info(f"Fetched {len(data)} {timespan} bars for {symbol}")
                    else:
                        logger.warning(f"No historical data for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        self.historical_data = results
        return results
        
    def fetch_market_snapshots(self, symbols):
        """
        Fetch current market snapshots for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with market snapshots by symbol
        """
        logger.info(f"Fetching market snapshots for {len(symbols)} symbols")
        
        snapshots = self.polygon_rest.get_market_snapshot(symbols)
        
        results = {}
        for snapshot in snapshots:
            symbol = snapshot.get("ticker")
            if symbol:
                results[symbol] = snapshot
                
        logger.info(f"Fetched market snapshots for {len(results)} symbols")
        
        return results
        
    def fetch_options_data(self, symbols):
        """
        Fetch options data for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with options data by symbol
        """
        logger.info(f"Fetching options data for {len(symbols)} symbols")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            futures = {executor.submit(self.polygon_rest.get_options_contracts, symbol, limit=50): symbol for symbol in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data:
                        results[symbol] = data
                        logger.info(f"Fetched {len(data)} options contracts for {symbol}")
                    else:
                        logger.warning(f"No options contracts for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching options contracts for {symbol}: {e}")
        
        self.options_data = results
        return results
        
    def fetch_unusual_options_flow(self, symbols=None, limit=50):
        """
        Fetch unusual options flow data
        
        Args:
            symbols: List of stock symbols (optional)
            limit: Number of results to return
            
        Returns:
            List of unusual options flow data
        """
        logger.info(f"Fetching unusual options flow data")
        
        if symbols:
            # Fetch for each symbol
            results = {}
            
            with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                futures = {executor.submit(self.unusual_whales.get_options_flow, ticker=symbol, limit=limit): symbol for symbol in symbols}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            results[symbol] = data
                            logger.info(f"Fetched {len(data)} unusual options flow entries for {symbol}")
                        else:
                            logger.warning(f"No unusual options flow data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching unusual options flow for {symbol}: {e}")
            
            return results
        else:
            # Fetch all unusual options flow
            try:
                data = self.unusual_whales.get_options_flow(limit=limit)
                logger.info(f"Fetched {len(data)} unusual options flow entries")
                return data
            except Exception as e:
                logger.error(f"Error fetching unusual options flow: {e}")
                return []
                
    def fetch_dark_pool_data(self, symbols, limit=20):
        """
        Fetch dark pool data for symbols
        
        Args:
            symbols: List of stock symbols
            limit: Number of results to return per symbol
            
        Returns:
            Dictionary with dark pool data by symbol
        """
        logger.info(f"Fetching dark pool data for {len(symbols)} symbols")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            futures = {executor.submit(self.unusual_whales.get_darkpool_ticker, ticker=symbol, limit=limit): symbol for symbol in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data:
                        results[symbol] = data
                        logger.info(f"Fetched {len(data)} dark pool trades for {symbol}")
                    else:
                        logger.warning(f"No dark pool data for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching dark pool data for {symbol}: {e}")
        
        self.dark_pool_data = results
        return results
        
    def fetch_insider_data(self, symbols=None, limit=50):
        """
        Fetch insider trading data
        
        Args:
            symbols: List of stock symbols (optional)
            limit: Number of results to return
            
        Returns:
            List of insider trading data
        """
        logger.info(f"Fetching insider trading data")
        
        if symbols:
            # Fetch for each symbol
            results = {}
            
            with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                futures = {executor.submit(self.unusual_whales.get_ticker_flow, ticker=symbol): symbol for symbol in symbols}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            results[symbol] = data
                            logger.info(f"Fetched insider trading data for {symbol}")
                        else:
                            logger.warning(f"No insider trading data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching insider trading data for {symbol}: {e}")
            
            return results
        else:
            # Fetch all insider transactions
            try:
                data = self.unusual_whales.get_insider_transactions(limit=limit)
                logger.info(f"Fetched {len(data)} insider transactions")
                return data
            except Exception as e:
                logger.error(f"Error fetching insider transactions: {e}")
                return []
                
    async def fetch_real_time_data(self, symbols, duration=10, use_mock=True):
        """
        Fetch real-time data for symbols using WebSocket
        
        Args:
            symbols: List of stock symbols
            duration: Duration to listen for messages (seconds)
            use_mock: Whether to use mock data for testing
            
        Returns:
            Dictionary with real-time data by symbol and type
        """
        logger.info(f"Fetching real-time data for {len(symbols)} symbols")
        
        ws_client = PolygonWebSocketClient()
        
        if use_mock:
            # Generate mock data for testing
            mock_data = ws_client.generate_mock_data(symbols, num_messages=duration * 5)
            
            # Process mock data
            results = self._process_websocket_messages(mock_data)
            
            self.real_time_data = results
            return results
        else:
            # Connect to WebSocket
            connected = await ws_client.connect()
            if not connected:
                logger.error("Failed to connect to WebSocket")
                return {}
            
            try:
                # Authenticate
                authenticated = await ws_client.authenticate()
                if not authenticated:
                    logger.error("Failed to authenticate with WebSocket")
                    return {}
                
                # Subscribe to channels
                channels = []
                for symbol in symbols:
                    channels.extend([f"T.{symbol}", f"Q.{symbol}", f"A.{symbol}"])
                
                subscribed = await ws_client.subscribe(channels)
                if not subscribed:
                    logger.error("Failed to subscribe to channels")
                    return {}
                
                # Listen for messages
                messages = await ws_client.listen(duration)
                
                # Process messages
                results = self._process_websocket_messages(messages)
                
                self.real_time_data = results
                return results
            finally:
                # Close connection
                await ws_client.close()
                
    def _process_websocket_messages(self, messages):
        """
        Process WebSocket messages
        
        Args:
            messages: List of WebSocket messages
            
        Returns:
            Dictionary with processed messages by symbol and type
        """
        results = {}
        
        for message in messages:
            if isinstance(message, list):
                # Process each message in the list
                for msg in message:
                    self._process_single_message(msg, results)
            else:
                # Process single message
                self._process_single_message(message, results)
                
        return results
        
    def _process_single_message(self, message, results):
        """
        Process a single WebSocket message
        
        Args:
            message: WebSocket message
            results: Dictionary to store results
            
        Returns:
            Updated results dictionary
        """
        if not isinstance(message, dict) or "ev" not in message:
            return
            
        event_type = message["ev"]
        symbol = message.get("sym")
        
        if not symbol:
            return
            
        # Initialize symbol in results if not exists
        if symbol not in results:
            results[symbol] = {
                "trades": [],
                "quotes": [],
                "aggregates": []
            }
            
        # Process by event type
        if event_type == "T":  # Trade
            results[symbol]["trades"].append({
                "price": message.get("p"),
                "size": message.get("s"),
                "timestamp": message.get("t"),
                "conditions": message.get("c", []),
                "id": message.get("i")
            })
        elif event_type == "Q":  # Quote
            results[symbol]["quotes"].append({
                "bid_price": message.get("bp"),
                "bid_size": message.get("bs"),
                "ask_price": message.get("ap"),
                "ask_size": message.get("as"),
                "timestamp": message.get("t"),
                "conditions": message.get("c", []),
                "id": message.get("i")
            })
        elif event_type == "A" or event_type == "AM":  # Aggregate
            results[symbol]["aggregates"].append({
                "open": message.get("o"),
                "high": message.get("h"),
                "low": message.get("l"),
                "close": message.get("c"),
                "volume": message.get("v"),
                "start_timestamp": message.get("s"),
                "end_timestamp": message.get("e")
            })
            
        return results
        
    def analyze_data(self, symbols):
        """
        Analyze all collected data to generate trading signals
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with trading signals by symbol
        """
        logger.info(f"Analyzing data for {len(symbols)} symbols")
        
        signals = {}
        
        for symbol in symbols:
            # Skip if no historical data
            if symbol not in self.historical_data:
                continue
                
            # Get historical data
            hist_data = self.historical_data[symbol]
            
            # Get real-time data if available
            real_time = self.real_time_data.get(symbol, {})
            
            # Get options data if available
            options = self.options_data.get(symbol, [])
            
            # Get dark pool data if available
            dark_pool = self.dark_pool_data.get(symbol, [])
            
            # Get insider data if available
            insider = self.insider_data.get(symbol, [])
            
            # Generate signal
            signal = self._generate_signal(symbol, hist_data, real_time, options, dark_pool, insider)
            
            if signal:
                signals[symbol] = signal
                
        logger.info(f"Generated signals for {len(signals)} symbols")
        
        return signals
        
    def _generate_signal(self, symbol, hist_data, real_time, options, dark_pool, insider):
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Stock symbol
            hist_data: Historical data
            real_time: Real-time data
            options: Options data
            dark_pool: Dark pool data
            insider: Insider data
            
        Returns:
            Dictionary with trading signal
        """
        # Simple signal generation for testing
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "target_price": None,
            "stop_loss": None,
            "position_size": 0.0,
            "data_sources": []
        }
        
        # Count data sources
        data_sources = []
        
        if not hist_data.empty:
            data_sources.append("historical")
            
        if real_time and any(real_time.values()):
            data_sources.append("real_time")
            
        if options:
            data_sources.append("options")
            
        if dark_pool:
            data_sources.append("dark_pool")
            
        if insider:
            data_sources.append("insider")
            
        signal["data_sources"] = data_sources
        
        # Generate random signal for testing
        if len(data_sources) >= 3:  # Require at least 3 data sources
            # Random direction
            direction = np.random.choice(["BUY", "SELL", "NEUTRAL"], p=[0.4, 0.4, 0.2])
            signal["direction"] = direction
            
            # Random confidence
            if direction != "NEUTRAL":
                signal["confidence"] = round(np.random.uniform(0.5, 1.0), 2)
                
                # Last price
                last_price = hist_data["close"].iloc[-1] if not hist_data.empty else 100.0
                
                # Target and stop loss
                if direction == "BUY":
                    signal["target_price"] = round(last_price * (1 + np.random.uniform(0.01, 0.05)), 2)
                    signal["stop_loss"] = round(last_price * (1 - np.random.uniform(0.01, 0.03)), 2)
                else:
                    signal["target_price"] = round(last_price * (1 - np.random.uniform(0.01, 0.05)), 2)
                    signal["stop_loss"] = round(last_price * (1 + np.random.uniform(0.01, 0.03)), 2)
                    
                # Position size
                max_position = min(self.max_position_value, self.max_daily_value - self.daily_value)
                signal["position_size"] = round(max_position * signal["confidence"], 2)
                
        return signal
        
    def execute_signals(self, signals):
        """
        Execute trading signals
        
        Args:
            signals: Dictionary with trading signals by symbol
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing signals for {len(signals)} symbols")
        
        results = {}
        
        for symbol, signal in signals.items():
            # Skip neutral signals
            if signal["direction"] == "NEUTRAL":
                continue
                
            # Skip if position size is too small
            if signal["position_size"] < 100.0:
                continue
                
            # Check if we can take the position
            position_value = signal["position_size"]
            can_take = self.unusual_whales.can_take_position(symbol, position_value)
            
            if can_take:
                # Execute trade (simulated)
                execution = {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "position_value": position_value,
                    "timestamp": datetime.now().isoformat(),
                    "status": "EXECUTED"
                }
                
                # Update position tracking
                self.unusual_whales.update_position_tracking(symbol, position_value)
                self.daily_value += position_value
                
                if symbol not in self.positions:
                    self.positions[symbol] = 0.0
                    
                self.positions[symbol] += position_value
                
                results[symbol] = execution
                logger.info(f"Executed {signal['direction']} signal for {symbol} with ${position_value:.2f}")
            else:
                # Rejected due to position limits
                execution = {
                    "symbol": symbol,
                    "direction": signal["direction"],
                    "position_value": position_value,
                    "timestamp": datetime.now().isoformat(),
                    "status": "REJECTED",
                    "reason": "Position limits exceeded"
                }
                
                results[symbol] = execution
                logger.warning(f"Rejected {signal['direction']} signal for {symbol} due to position limits")
                
        logger.info(f"Executed signals for {len(results)} symbols")
        
        return results
        
    def run_simulation(self, symbols=DEFAULT_SYMBOLS, use_mock=True):
        """
        Run a complete trading simulation
        
        Args:
            symbols: List of stock symbols
            use_mock: Whether to use mock data for testing
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running trading simulation for {len(symbols)} symbols")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Fetch historical data
        historical_start = time.time()
        self.fetch_historical_data(symbols)
        historical_time = time.time() - historical_start
        
        # Step 2: Fetch options data
        options_start = time.time()
        self.fetch_options_data(symbols)
        options_time = time.time() - options_start
        
        # Step 3: Fetch unusual options flow
        flow_start = time.time()
        unusual_flow = self.fetch_unusual_options_flow(symbols)
        flow_time = time.time() - flow_start
        
        # Step 4: Fetch dark pool data
        dark_pool_start = time.time()
        self.fetch_dark_pool_data(symbols)
        dark_pool_time = time.time() - dark_pool_start
        
        # Step 5: Fetch insider data
        insider_start = time.time()
        self.fetch_insider_data(symbols)
        insider_time = time.time() - insider_start
        
        # Step 6: Fetch real-time data
        real_time_start = time.time()
        asyncio.run(self.fetch_real_time_data(symbols, duration=5, use_mock=use_mock))
        real_time_time = time.time() - real_time_start
        
        # Step 7: Analyze data and generate signals
        analysis_start = time.time()
        signals = self.analyze_data(symbols)
        analysis_time = time.time() - analysis_start
        
        # Step 8: Execute signals
        execution_start = time.time()
        executions = self.execute_signals(signals)
        execution_time = time.time() - execution_start
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "symbols": symbols,
            "signals": signals,
            "executions": executions,
            "position_limits": {
                "max_position_value": self.max_position_value,
                "max_daily_value": self.max_daily_value,
                "current_daily_value": self.daily_value,
                "positions": self.positions
            },
            "timing": {
                "historical_data": historical_time,
                "options_data": options_time,
                "unusual_flow": flow_time,
                "dark_pool": dark_pool_time,
                "insider_data": insider_time,
                "real_time": real_time_time,
                "analysis": analysis_time,
                "execution": execution_time,
                "total": total_time
            },
            "data_counts": {
                "historical": sum(len(df) for df in self.historical_data.values()),
                "options": sum(len(options) for options in self.options_data.values()),
                "dark_pool": sum(len(trades) for trades in self.dark_pool_data.values()),
                "real_time": sum(len(data.get("trades", [])) + len(data.get("quotes", [])) + len(data.get("aggregates", [])) 
                               for data in self.real_time_data.values())
            }
        }
        
        logger.info(f"Simulation completed in {total_time:.2f} seconds")
        logger.info(f"Generated {len(signals)} signals, executed {len(executions)} trades")
        logger.info(f"Current daily value: ${self.daily_value:.2f} / ${self.max_daily_value:.2f}")
        
        return results
        
    def close(self):
        """Close all clients"""
        self.polygon_rest.close()
        self.polygon_ultra.close()
        self.unusual_whales.close()
        logger.info("Closed all clients")

def run_day_trading_test(symbols=None, num_symbols=5, use_mock=True):
    """
    Run a day trading system test
    
    Args:
        symbols: List of stock symbols (optional)
        num_symbols: Number of symbols to test if symbols not provided
        use_mock: Whether to use mock data for testing
        
    Returns:
        Dictionary with test results
    """
    # Use provided symbols or select from default list
    if symbols is None:
        symbols = DEFAULT_SYMBOLS[:min(num_symbols, len(DEFAULT_SYMBOLS))]
        
    logger.info(f"Running day trading system test with {len(symbols)} symbols: {symbols}")
    
    # Create day trading system
    system = DayTradingSystem()
    
    try:
        # Run simulation
        results = system.run_simulation(symbols, use_mock=use_mock)
        
        # Print timing results
        timing = results["timing"]
        logger.info("\nTiming Results:")
        logger.info(f"Historical Data: {timing['historical_data']:.2f} seconds")
        logger.info(f"Options Data: {timing['options_data']:.2f} seconds")
        logger.info(f"Unusual Flow: {timing['unusual_flow']:.2f} seconds")
        logger.info(f"Dark Pool: {timing['dark_pool']:.2f} seconds")
        logger.info(f"Insider Data: {timing['insider_data']:.2f} seconds")
        logger.info(f"Real-time Data: {timing['real_time']:.2f} seconds")
        logger.info(f"Analysis: {timing['analysis']:.2f} seconds")
        logger.info(f"Execution: {timing['execution']:.2f} seconds")
        logger.info(f"Total: {timing['total']:.2f} seconds")
        
        # Print data counts
        counts = results["data_counts"]
        logger.info("\nData Counts:")
        logger.info(f"Historical Data: {counts['historical']} records")
        logger.info(f"Options Data: {counts['options']} contracts")
        logger.info(f"Dark Pool Data: {counts['dark_pool']} trades")
        logger.info(f"Real-time Data: {counts['real_time']} messages")
        
        return results
    finally:
        # Close system
        system.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Day Trading System")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of symbols to test")
    parser.add_argument("--num-symbols", type=int, default=5,
                        help="Number of symbols to test if symbols not provided")
    parser.add_argument("--use-mock", action="store_true", default=True,
                        help="Use mock data for testing during non-market hours")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else None
    
    # Run test
    run_day_trading_test(symbols, args.num_symbols, args.use_mock)
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()
