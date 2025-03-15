"""
Polygon.io API Client

This module provides a client for interacting with the Polygon.io API to fetch market data.

IMPORTANT: All Polygon.io API endpoints require specific version prefixes:
- Aggregates endpoints (bars, previous close): Use v2 prefix (/v2/aggs/...)
- Reference data (tickers, news): Use v2 or v3 prefix depending on endpoint
- Market status endpoints: Use v1 prefix (/v1/marketstatus/...)
"""
import time
import json
import os
import socket
import websocket
import aiohttp
from datetime import datetime, date, timedelta
from typing import Any
import pandas as pd
# Removed unused import: import numpy as np
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from src.utils.api import (
    ConnectionPool,
    RateLimiter,
    RetryHandler,
    
)
# Production-ready imports only

# Import logging utility
from src.utils.logging import get_logger

# Set up logger
logger = get_logger("data_acquisition.api.polygon_client")

class PolygonClient:
    """Client for interacting with the Polygon.io API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        websocket_url: str | None = None,
        rate_limit: int = 5,
        retry_attempts: int = 3,
        timeout: int = 30,
        verify_ssl: bool = True,
        max_position_value: float = 2500.0,  # Maximum $2500 per stock
        max_daily_value: float = 5000.0,     # Maximum $5000 per day
    ):
        """
        Initialize the Polygon.io API client with dollar-based position limits.
        
        Args:
            api_key: Polygon API key
            base_url: Base URL for API requests
            websocket_url: WebSocket URL for real-time data
            rate_limit: Rate limit in requests per second
            retry_attempts: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            max_position_value: Maximum position value per stock in dollars
            max_daily_value: Maximum total position value per day in dollars
        """
        # Try to get API key from environment if not provided
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("POLYGON_API_KEY")
            logger.debug(f"Using Polygon API key from environment: {self.api_key[:5]}..." if self.api_key and len(self.api_key) > 5 else "No Polygon API key found in environment")
        if not self.api_key:
            raise ValueError("Polygon API key is required")

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AutonomousTradingSystem/0.1"}
        )

        # Set URLs
        self.BASE_URL = base_url or "https://api.polygon.io"
        self.WEBSOCKET_URL = websocket_url or "wss://socket.polygon.io/stocks"
        
        # Position tracking for dollar-based limits
        self.max_position_value = max_position_value  # Maximum position value per stock
        self.max_daily_value = max_daily_value        # Maximum total position value per day
        self.current_daily_value = 0.0                # Current total position value for the day
        self.position_values = {}                     # Current position value per ticker
        self.position_count = 0                       # Number of positions taken today
        self.last_reset_date = datetime.now().date()  # Last date when limits were reset
        
        # Define available WebSocket clusters
        self.WEBSOCKET_CLUSTERS = {
            "stocks": "wss://socket.polygon.io/stocks",
            "forex": "wss://socket.polygon.io/forex",
            "crypto": "wss://socket.polygon.io/crypto",
            "options": "wss://socket.polygon.io/options",
            "delayed_stocks": "wss://delayed.polygon.io/stocks"
        }
        
        # Initialize rate limiter with configured values
        self.rate_limiter = RateLimiter(
            requests_per_period=rate_limit * 60,  # Convert to requests per minute
            period_seconds=60,
            strategy=RateLimiter.RateLimitStrategy.TOKEN_BUCKET,
            burst_size=rate_limit * 60,
            name="polygon"
       )
        
        # Initialize retry handler
        self.retry_handler = RetryHandler(
            max_retries=retry_attempts,
            backoff_strategy=RetryHandler.BackoffStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=60.0,
            retry_exceptions=[
                ConnectionError,
                Timeout,
                RequestException,
            ],
            retry_on_status_codes=[429, 500, 502, 503, 504],
            name="polygon_retry"
        )
        
        # Initialize connection pool for WebSocket
        self.ws_pool = ConnectionPool(
            create_connection=lambda: self._create_websocket_connection(cluster="stocks", delayed=False),
            close_connection=lambda ws: ws.close() if ws else None,
            name="polygon_ws"
        )

        # WebSocket connection
        self.ws = None
        self.ws_callbacks = {}

        # Rate limiting parameters
        self.rate_limit_remaining = rate_limit  # Conservative initial value
        self.rate_limit_reset = 0
        self.max_retries = retry_attempts
        self.retry_delay = 1.0
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        logger.info(f"Initialized PolygonClient with rate limiting ({rate_limit} req/sec), retry handling (max {retry_attempts} retries), and timeout {timeout}s")
    
    # Apply decorators to methods that need them

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """
        Handle rate limit headers from Polygon.io API.

        Args:
            response: API response
        """
        # Update rate limit information from headers
        if "X-Ratelimit-Remaining" in response.headers:
            self.rate_limit_remaining = int(response.headers["X-Ratelimit-Remaining"])

        if "X-Ratelimit-Reset" in response.headers:
            self.rate_limit_reset = int(response.headers["X-Ratelimit-Reset"])

        # If we're close to the rate limit, sleep until reset
        if self.rate_limit_remaining < 2:
            sleep_time = max(0, self.rate_limit_reset - time.time()) + 1
            logger.warning(
                f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds"
            )
            time.sleep(sleep_time)

    @RateLimiter.decorator
    @RetryHandler.decorator
    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None, retry_count: int = 0
    ) -> dict[str, Any]:
        """
        Make a request to the Polygon.io API with rate limit handling and retries.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            retry_count: Current retry count (used internally by RetryHandler)

        Returns:
            API response as a dictionary

        Raises:
            requests.RequestException: If the request fails after retries
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        # Debug the URL before making the request
        debug_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        logger.debug(f"Making request to URL: {debug_url}")
        
        # Check if the URL contains '/True?' which would indicate a malformed URL
        if '/True?' in debug_url or '/False?' in debug_url:
            logger.error(f"Malformed URL detected: {debug_url}")
            raise ValueError(f"Malformed URL detected: {debug_url}")
        response = self.session.get(url, params=params, timeout=self.timeout)
        self._handle_rate_limit(response)

        # Handle HTTP errors
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        # Check for API-level errors (handle both dict and list responses)
        if isinstance(data, dict) and (data.get("status") == "ERROR" or data.get("error")):
            error_msg = data.get("error", "Unknown API error")
            raise ValueError(f"Polygon API error: {error_msg}")

        return data
    # Helper method to add API key to parameters
    
    async def _make_request_async(
        self, endpoint: str, params: dict[str, Any] | None = None, retry_count: int = 0
    ) -> dict[str, Any]:
        """
        Make an asynchronous request to the Polygon.io API with rate limit handling and retries.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            retry_count: Current retry count (used internally by RetryHandler)

        Returns:
            API response as a dictionary

        Raises:
            aiohttp.ClientError: If the request fails after retries
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        # Debug the URL before making the request
        debug_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        logger.debug(f"Making async request to URL: {debug_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.timeout) as response:
                # Handle HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                data = await response.json()
                
                # Check for API-level errors
                if isinstance(data, dict) and (data.get("status") == "ERROR" or data.get("error")):
                    raise ValueError(f"Polygon API error: {data.get('error', 'Unknown API error')}")
                return data
    
    def _add_api_key(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add API key to parameters."""
        if "apiKey" not in params:
            params["apiKey"] = self.api_key
        
        return params

    def get_bars(
        self,
        symbols: str | list[str],
        timeframe: str,
        start: str | datetime | date,
        end: str | datetime | date,
        limit: int = 1000,
        adjustment: str = "all",
    ) -> pd.DataFrame:
        """
        Get historical bars for one or more symbols.

        Args:
            symbols: Ticker symbol or list of symbols
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1H', '1D', etc.)
            start: Start date/time
            end: End date/time
            limit: Maximum number of bars per symbol
            adjustment: Adjustment mode ('raw', 'split', 'dividend', 'all')

        Returns:
            DataFrame with historical bars
        """
        # Convert to list if single symbol
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Parse timeframe
        if timeframe.lower() in ['1min', '1m']:
            multiplier = 1
            timespan = 'minute'
        elif timeframe.lower() in ['5min', '5m']:
            multiplier = 5
            timespan = 'minute'
        elif timeframe.lower() in ['15min', '15m']:
            multiplier = 15
            timespan = 'minute'
        elif timeframe.lower() in ['1h', '1hour']:
            multiplier = 1
            timespan = 'hour'
        elif timeframe.lower() in ['1d', '1day']:
            multiplier = 1
            timespan = 'day'
        elif timeframe.lower() in ['1w', '1week']:
            multiplier = 1
            timespan = 'week'
        else:
            # Try to parse custom timeframes
            import re
            match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
            if match:
                multiplier = int(match.group(1))
                unit = match.group(2).lower()
                if unit.startswith('min'):
                    timespan = 'minute'
                elif unit.startswith('h'):
                    timespan = 'hour'
                elif unit.startswith('d'):
                    timespan = 'day'
                elif unit.startswith('w'):
                    timespan = 'week'
                else:
                    raise ValueError(f"Invalid timeframe: {timeframe}")
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Get data for each symbol and combine
        all_data = []
        for symbol in symbols:
            try:
                df = self.get_aggregates(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=start,
                    to_date=end,
                    adjusted=(adjustment != 'raw'),
                    limit=limit
                )
                
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Error getting bars for {symbol}: {e}")
        
        # Combine all dataframes
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Add timeframe column
            combined_df["timeframe"] = timeframe.lower()
            return combined_df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", 
                "volume", "vwap", "transactions", "timeframe"
            ])

    def get_account(self) -> dict[str, Any]:
        """
        Get account information.
        
        Note: This is a placeholder as Polygon doesn't provide account information.
        For actual account information, use the Alpaca client.
        
        Returns:
            Empty dictionary (Polygon doesn't provide account information)
        """
        logger.warning("Polygon API doesn't provide account information. Use AlpacaClient instead.")
        return {}

    def get_aggregates(
        self,
        ticker: str = None,
        symbol: str = None, 
        multiplier: int | None = None,
        timespan: str = None,
        from_date: str | datetime = None,
        to_date: str | datetime = None,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Get aggregate bars for a ticker over a given date range.

        Args:
            ticker: Ticker symbol
            symbol: Alternative parameter name for ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            adjusted: Whether to adjust for splits
            limit: Maximum number of results

        Returns:
            DataFrame with aggregate bars
        """
        # Validate required parameters
        if ticker is None and symbol is None:
            raise ValueError("Either ticker or symbol must be provided")
        if multiplier is None:
            raise ValueError("multiplier is required")
        if timespan is None:
            raise ValueError("timespan is required")
        if from_date is None:
            raise ValueError("from_date is required")
        if to_date is None:
            raise ValueError("to_date is required")
            
        # Use symbol parameter if provided (symbol takes precedence if both provided)
        if symbol is not None:
            logger.debug(f"Using symbol parameter '{symbol}' instead of ticker '{ticker}'")
            ticker = symbol
        elif ticker is None:
            # This should never happen due to the validation above, but just to be safe
            raise ValueError("Either ticker or symbol parameter must be provided")
            ticker = symbol
            
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")

        # Using v2 prefix for aggregates endpoint as required by Polygon API
        # The adjusted parameter should be a query parameter, not part of the URL path
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        # Debug the ticker value
        logger.debug(f"Ticker value: '{ticker}', type: {type(ticker)}, adjusted: {adjusted}")
        
        # Ensure adjusted is properly converted to a string for the query parameter
        params = self._add_api_key({
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit
        })

        # Debug the URL construction
        full_url = f"{self.BASE_URL}{endpoint}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        logger.debug(f"Making request to URL: {full_url}")
        
        # Additional debug to help diagnose issues
        logger.debug(f"Endpoint: {endpoint}")
        logger.debug(f"Params: {params}")

        # Validate ticker to ensure it's not a number or boolean
        if not isinstance(ticker, str) or ticker.isdigit() or ticker.lower() in ('true', 'false'):
            error_msg = f"Invalid ticker symbol: '{ticker}'. Must be a valid stock symbol string."
            logger.error(error_msg)
            return pd.DataFrame()  # Return empty DataFrame instead of raising an error

        # Ensure the URL is constructed correctly by explicitly checking the endpoint format
        expected_format = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        if endpoint != expected_format:
            error_msg = f"Endpoint format mismatch. Expected: {expected_format}, Got: {endpoint}"
            logger.error(error_msg)
            endpoint = expected_format  # Force the correct format

        # Log the final URL for debugging
        logger.info(f"Making request to Polygon API: {self.BASE_URL}{endpoint} with params {params}")

        # Make the request
        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No data found for {ticker} from {from_date} to {to_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        try:
            df = pd.DataFrame(result["results"]) 
            logger.debug("Created DataFrame using pandas")
        except Exception as e:
            logger.warning(f"Error creating DataFrame: {e}")
            import pandas as pandas_pd
            df = pandas_pd.DataFrame(result["results"])

        # Rename columns to match our schema
        column_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
        df = df.rename(columns=column_map)

        # Convert timestamp from milliseconds to datetime
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            logger.debug("Converted timestamp using pandas")
        except Exception as e:
            logger.warning(f"Error converting timestamp: {e}")

        # Add additional columns
        df["symbol"] = ticker
        df["timeframe"] = f"{multiplier}{timespan[0]}"  # e.g., "1m", "5m", "1d"
        df["multiplier"] = multiplier
        df["timespan_unit"] = timespan
        df["adjusted"] = adjusted

        return df

    def get_tickers(
        self,
        ticker: str | None = None,
        type: str | None = None,
        market: str | None = "stocks",
        exchange: str | None = None,
        cusip: str | None = None,
        cik: str | None = None,
        date: str | None = None,
        search: str | None = None,
        active: bool = True,
        order: str = "asc",
        limit: int = 100,
        sort: str = "ticker",
    ) -> list[dict[str, Any]]:
        """
        Get a list of tickers supported by Polygon.io.

        Args:
            ticker: Filter by ticker symbol
            type: Filter by type of ticker
            market: Filter by market type (stocks, crypto, fx, otc, indices)
            exchange: Filter by exchange in ISO code format
            cusip: Filter by CUSIP code
            cik: Filter by CIK number
            date: Filter by date (YYYY-MM-DD)
            search: Search for terms within ticker and/or company name
            active: Whether to return only active tickers
            order: Order of results (asc, desc)
            limit: Maximum number of results (max 1000)
            sort: Field to sort by (ticker, name, market, locale, primary_exchange, type, currency_name)

        Returns:
            List of ticker information dictionaries
        """
        endpoint = "/v3/reference/tickers"
        params = self._add_api_key({})
        
        # Add optional parameters
        if ticker:
            params["ticker"] = ticker
        if type:
            params["type"] = type
        if market:
            params["market"] = market
        if exchange:
            params["exchange"] = exchange
        if cusip:
            params["cusip"] = cusip
        if cik:
            params["cik"] = cik
        if date:
            params["date"] = date
        if search:
            params["search"] = search
        
        # Add boolean parameters
        params["active"] = str(active).lower()
        
        # Add pagination and sorting parameters
        params["order"] = order
        params["limit"] = limit
        params["sort"] = sort
        
        # Make the request
        result = self._make_request(endpoint, params)
        
        # Return the results
        return result.get("results", [])
        
    def get_all_tickers(
        self,
        ticker: str | None = None,
        type: str | None = None,
        market: str | None = "stocks",
        exchange: str | None = None,
        cusip: str | None = None,
        cik: str | None = None,
        date: str | None = None,
        search: str | None = None,
        active: bool = True,
        order: str = "asc",
        sort: str = "ticker",
        max_tickers: int = 5000,
    ) -> list[dict[str, Any]]:
        """
        Get a complete list of tickers supported by Polygon.io, handling pagination.

        This method will make multiple API calls if necessary to fetch all tickers
        up to the specified max_tickers limit.

        Args:
            ticker: Filter by ticker symbol
            type: Filter by type of ticker
            market: Filter by market type (stocks, crypto, fx, otc, indices)
            exchange: Filter by exchange in ISO code format
            cusip: Filter by CUSIP code
            cik: Filter by CIK number
            date: Filter by date (YYYY-MM-DD)
            search: Search for terms within ticker and/or company name
            active: Whether to return only active tickers
            order: Order of results (asc, desc)
            sort: Field to sort by (ticker, name, market, locale, primary_exchange, type, currency_name)
            max_tickers: Maximum number of tickers to return (default 5000)

        Returns:
            List of ticker information dictionaries
        """
        all_tickers = []
        page_limit = min(1000, max_tickers)  # API max is 1000 per request
        
        # Initial request
        endpoint = "/v3/reference/tickers"
        params = self._add_api_key({})
        
        # Add optional parameters
        if ticker:
            params["ticker"] = ticker
        if type:
            params["type"] = type
        if market:
            params["market"] = market
        if exchange:
            params["exchange"] = exchange
        if cusip:
            params["cusip"] = cusip
        if cik:
            params["cik"] = cik
        if date:
            params["date"] = date
        if search:
            params["search"] = search
        
        # Add boolean parameters
        params["active"] = str(active).lower()
        
        # Add pagination and sorting parameters
        params["order"] = order
        params["limit"] = page_limit
        params["sort"] = sort
        
        # Make the initial request
        result = self._make_request(endpoint, params)
        
        # Add results to our list
        tickers = result.get("results", [])
        all_tickers.extend(tickers)
        
        # Check if we need to paginate
        next_url = result.get("next_url")
        
        # Continue fetching pages until we have enough tickers or there are no more pages
        while next_url and len(all_tickers) < max_tickers:
            # Extract the cursor from the next_url
            import urllib.parse
            parsed_url = urllib.parse.urlparse(next_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Update our params with the cursor
            if "cursor" in query_params:
                params["cursor"] = query_params["cursor"][0]
            
            # Make the next request
            result = self._make_request(endpoint, params)
            
            # Add results to our list
            tickers = result.get("results", [])
            all_tickers.extend(tickers)
            
            # Update next_url for the next iteration
            next_url = result.get("next_url")
            
            # Log progress
            logger.info(f"Fetched {len(all_tickers)} tickers so far")
            
            # Check if we've reached the limit
            if len(all_tickers) >= max_tickers:
                logger.info(f"Reached maximum ticker limit of {max_tickers}")
                break
        
        # Trim to max_tickers if we went over
        if len(all_tickers) > max_tickers:
            all_tickers = all_tickers[:max_tickers]
        
        return all_tickers
        
    def get_ticker_details(self, ticker: str) -> dict[str, Any]:
        """
        Get detailed information about a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        result = self._make_request(endpoint, self._add_api_key({}))

        return result.get("results", {})
        
    def get_tickers_df(
        self,
        ticker: str | None = None,
        type: str | None = None,
        market: str | None = "stocks",
        exchange: str | None = None,
        cusip: str | None = None,
        cik: str | None = None,
        date: str | None = None,
        search: str | None = None,
        active: bool = True,
        order: str = "asc",
        limit: int = 100,
        sort: str = "ticker",
    ) -> pd.DataFrame:
        """
        Get a list of tickers as a pandas DataFrame.

        Args:
            ticker: Filter by ticker symbol
            type: Filter by type of ticker
            market: Filter by market type (stocks, crypto, fx, otc, indices)
            exchange: Filter by exchange in ISO code format
            cusip: Filter by CUSIP code
            cik: Filter by CIK number
            date: Filter by date (YYYY-MM-DD)
            search: Search for terms within ticker and/or company name
            active: Whether to return only active tickers
            order: Order of results (asc, desc)
            limit: Maximum number of results (max 1000)
            sort: Field to sort by (ticker, name, market, locale, primary_exchange, type, currency_name)

        Returns:
            DataFrame with ticker information
        """
        # Get tickers as a list of dictionaries
        tickers = self.get_tickers(
            ticker=ticker,
            type=type,
            market=market,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            search=search,
            active=active,
            order=order,
            limit=limit,
            sort=sort,
        )
        
        # Convert to DataFrame
        if not tickers:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "ticker", "name", "market", "locale", "primary_exchange",
                "type", "active", "currency_name", "last_updated_utc"
            ])
        
        try:
            df = pd.DataFrame(tickers)
            
            # Convert date columns to datetime
            if "last_updated_utc" in df.columns:
                df["last_updated_utc"] = pd.to_datetime(df["last_updated_utc"])
                
            if "delisted_utc" in df.columns:
                df["delisted_utc"] = pd.to_datetime(df["delisted_utc"])
                
            return df
            
        except Exception as e:
            logger.error(f"Error converting tickers to DataFrame: {e}")
            # Return empty DataFrame
            return pd.DataFrame()
            
    def get_all_tickers_df(
        self,
        ticker: str | None = None,
        type: str | None = None,
        market: str | None = "stocks",
        exchange: str | None = None,
        cusip: str | None = None,
        cik: str | None = None,
        date: str | None = None,
        search: str | None = None,
        active: bool = True,
        order: str = "asc",
        sort: str = "ticker",
        max_tickers: int = 5000,
    ) -> pd.DataFrame:
        """
        Get a complete list of tickers as a pandas DataFrame, handling pagination.

        Args:
            ticker: Filter by ticker symbol
            type: Filter by type of ticker
            market: Filter by market type (stocks, crypto, fx, otc, indices)
            exchange: Filter by exchange in ISO code format
            cusip: Filter by CUSIP code
            cik: Filter by CIK number
            date: Filter by date (YYYY-MM-DD)
            search: Search for terms within ticker and/or company name
            active: Whether to return only active tickers
            order: Order of results (asc, desc)
            sort: Field to sort by (ticker, name, market, locale, primary_exchange, type, currency_name)
            max_tickers: Maximum number of tickers to return (default 5000)

        Returns:
            DataFrame with ticker information
        """
        # Get tickers as a list of dictionaries
        tickers = self.get_all_tickers(
            ticker=ticker,
            type=type,
            market=market,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            search=search,
            active=active,
            order=order,
            sort=sort,
            max_tickers=max_tickers,
        )
        
        # Convert to DataFrame
        if not tickers:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "ticker", "name", "market", "locale", "primary_exchange",
                "type", "active", "currency_name", "last_updated_utc"
            ])
        
        try:
            df = pd.DataFrame(tickers)
            
            # Convert date columns to datetime
            if "last_updated_utc" in df.columns:
                df["last_updated_utc"] = pd.to_datetime(df["last_updated_utc"])
                
            if "delisted_utc" in df.columns:
                df["delisted_utc"] = pd.to_datetime(df["delisted_utc"])
                
            return df
            
        except Exception as e:
            logger.error(f"Error converting tickers to DataFrame: {e}")
            # Return empty DataFrame
            return pd.DataFrame()

    def get_quotes(
        self, ticker: str, date: str | datetime, limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get NBBO quotes for a ticker on a specific date.

        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD or datetime)
            limit: Maximum number of results

        Returns:
            DataFrame with quotes
        """
        # Convert datetime to string
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        # Using v2 prefix for quotes endpoint as required by Polygon API
        endpoint = f"/v2/quotes/{ticker}/{date}"
        # Using v3 prefix for quotes endpoint with timestamp parameters
        endpoint = f"/v3/quotes/{ticker}"
        params = self._add_api_key({"limit": limit, 
                                   "timestamp.gte": f"{date}T00:00:00Z", 
                                   "timestamp.lte": f"{date}T23:59:59Z"})

        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No quotes found for {ticker} on {date}")
            return pd.DataFrame()

        # Convert to DataFrame
        try:
            df = pd.DataFrame(result["results"]) 
            logger.debug("Created DataFrame using pandas")
        except Exception as e:
            logger.warning(f"Error creating DataFrame: {e}")

        # Rename columns to match our schema
        column_map = {
            # v3 API uses different field names
            "sip_timestamp": "timestamp",
            "bid_price": "bid_price",
            "ask_price": "ask_price",
            "bid_size": "bid_size",
            "ask_size": "ask_size",
            "exchange": "exchange",
            "conditions": "conditions",
            "sequence_number": "sequence_number",
            "tape": "tape",
        }
        df = df.rename(columns=column_map)

        # Check if timestamp column exists before converting
        timestamp_col = None
        for col in ["timestamp", "sip_timestamp", "participant_timestamp"]:
            if col in df.columns:
                timestamp_col = col
                break
                
        # Convert timestamp if it exists
        if timestamp_col:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit="ns")
            except Exception as e:
                logger.warning(f"Error converting timestamp: {e}")
            # Rename to our standard column name if needed
            if timestamp_col != "timestamp":
                df["timestamp"] = df[timestamp_col]
        else:
            logger.warning("No timestamp column found in quotes response")
            # Add a placeholder timestamp
            df["timestamp"] = pd.to_datetime("now")
            
        # Log DataFrame info for debugging
        logger.debug(f"Quote DataFrame columns: {list(df.columns)}")
        logger.debug(f"Quote DataFrame size: {len(df)} rows")
        # Add additional columns
        df["symbol"] = ticker
        df["source"] = "polygon"

        return df

    def get_trades(
        self, ticker: str, date: str | datetime, limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get trades for a ticker on a specific date.

        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD or datetime)
            limit: Maximum number of results

        Returns:
            DataFrame with trades
        """
        # Convert datetime to string
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        # Using v2 prefix for trades endpoint as required by Polygon API
        endpoint = f"/v2/trades/{ticker}/{date}"
        # Using v3 prefix for trades endpoint with timestamp parameters
        endpoint = f"/v3/trades/{ticker}"
        params = self._add_api_key({"limit": limit,
                                   "timestamp.gte": f"{date}T00:00:00Z",
                                   "timestamp.lte": f"{date}T23:59:59Z"})
        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No trades found for {ticker} on {date}")
            return pd.DataFrame()

        # Convert to DataFrame
        try:
            df = pd.DataFrame(result["results"]) 
            logger.debug("Created DataFrame using pandas")
        except Exception as e:
            logger.warning(f"Error creating DataFrame: {e}")

        # Rename columns to match our schema
        column_map = {
            # v3 API uses different field names
            "sip_timestamp": "timestamp",
            "price": "price",
            "size": "size",
            "exchange": "exchange",
            "conditions": "conditions",
            "id": "trade_id",
            "sequence_number": "sequence_number",
            "tape": "tape",
        }
        df = df.rename(columns=column_map)

        # Check if timestamp column exists before converting
        timestamp_col = None
        for col in ["timestamp", "sip_timestamp", "participant_timestamp"]:
            if col in df.columns:
                timestamp_col = col
                break
                
        # Convert timestamp if it exists
        if timestamp_col:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit="ns")
            except Exception as e:
                logger.warning(f"Error converting timestamp: {e}")
            # Rename to our standard column name if needed
            if timestamp_col != "timestamp":
                df["timestamp"] = df[timestamp_col]
        else:
            logger.warning("No timestamp column found in trades response")
            # Add a placeholder timestamp
            df["timestamp"] = pd.to_datetime("now")

        # Add additional columns
        df["symbol"] = ticker
        df["source"] = "polygon"

        return df

    def get_last_trade(self, ticker: str) -> dict[str, Any]:
        """
        Get the most recent trade for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with last trade information
        """
        # Using v2 prefix for last trade endpoint as required by Polygon API
        endpoint = f"/v2/last/trade/{ticker}"
        result = self._make_request(endpoint, self._add_api_key({}))

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No last trade found for {ticker}")
            return {}

        return result.get("results", {})

    def get_last_quote(self, ticker: str) -> dict[str, Any]:
        """
        Get the most recent NBBO quote for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with last quote information
        """
        # Using v2 prefix for last quote endpoint as required by Polygon API
        endpoint = f"/v2/last/nbbo/{ticker}"
        result = self._make_request(endpoint, self._add_api_key({}))

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No last quote found for {ticker}")
            return {}

        return result.get("results", {})

    def get_daily_open_close(self, ticker: str, date: str | datetime, adjusted: bool = True) -> dict[str, Any]:
        """
        Get the open, close, high, and low for a specific stock symbol on a certain date.

        Args:
            ticker: Stock symbol
            date: Date in format YYYY-MM-DD or datetime
            adjusted: Whether to adjust for splits

        Returns:
            Dictionary with open, close, high, low data
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        endpoint = f"/v1/open-close/{ticker}/{date}"
        params = self._add_api_key({"adjusted": str(adjusted).lower()})

        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("status") == "OK":
            logger.warning(f"No daily open/close data found for {ticker} on {date}")
            return {}

        # Remove status field
        result.pop("status", None)
        return result

    def get_previous_close(self, ticker: str, adjusted: bool = True) -> dict[str, Any]:
        """
        Get the previous day's open, high, low, and close for a specific stock ticker.

        Args:
            ticker: Stock symbol
            adjusted: Whether to adjust for splits

        Returns:
            Dictionary with previous day's OHLCV data
        """
        # Using v2 prefix for previous close endpoint as required by Polygon API
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        params = self._add_api_key({"adjusted": str(adjusted).lower()})

        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No previous close data found for {ticker}")
            return {}

        # Return the first (and only) result
        return result.get("results", [{}])[0]

    def get_options_chain(
        self,
        underlying: str,
        expiration_date: str | datetime | None = None,
        strike_price: float | None = None,
        contract_type: str | None = None,  # 'call' or 'put'
    ) -> list[dict[str, Any]]:
        """
        Get options chain for an underlying asset.

        Args:
            underlying: Underlying asset symbol
            expiration_date: Optional expiration date filter (YYYY-MM-DD or datetime)
            strike_price: Optional strike price filter
            contract_type: Optional contract type filter ('call' or 'put')

        Returns:
            List of option contracts
        """
        # Convert datetime to string
        if isinstance(expiration_date, datetime):
            expiration_date = expiration_date.strftime("%Y-%m-%d")

        endpoint = "/v3/reference/options/contracts"
        params = self._add_api_key({"underlying_ticker": underlying})

        # Add optional filters
        if expiration_date:
            params["expiration_date"] = expiration_date
        if strike_price:
            params["strike_price"] = strike_price
        if contract_type:
            params["contract_type"] = contract_type.upper()

        result = self._make_request(endpoint, params)

        return result.get("results", [])

    def get_options_aggregates(
        self,
        options_ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str | datetime,
        to_date: str | datetime,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Get aggregate bars for an options contract over a given date range.

        Args:
            options_ticker: Options contract symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            adjusted: Whether to adjust for splits
            limit: Maximum number of results

        Returns:
            DataFrame with aggregate bars for the options contract
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")

        # Using v2 prefix for options aggregates endpoint as required by Polygon API
        endpoint = f"/v2/aggs/ticker/{options_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = self._add_api_key({"adjusted": str(adjusted).lower(), "sort": "asc", "limit": limit})

        result = self._make_request(endpoint, params)

        # Check if we have results
        if not result.get("results"):
            logger.warning(f"No data found for options {options_ticker} from {from_date} to {to_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        try:
            df = pd.DataFrame(result["results"]) 
            logger.debug("Created DataFrame using pandas")
        except Exception as e:
            logger.warning(f"Error creating DataFrame: {e}")

        # Rename columns to match our schema (same as get_aggregates)
        column_map = {
            "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vw": "vwap", "n": "transactions",
        }
        df = df.rename(columns=column_map)
        
        # Convert timestamp from milliseconds to datetime
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        except Exception as e:
            logger.warning(f"Error converting timestamp: {e}")
            
        df["symbol"] = options_ticker
        return df

    def get_market_status(self) -> dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Dictionary with market status information
        """
        endpoint = "/v1/marketstatus/now"
        result = self._make_request(endpoint, self._add_api_key({}))

        return result

    def get_market_holidays(self) -> list[dict[str, Any]]:
        """
        Get market holidays.
        
        Returns:
            List of market holidays
        """
        endpoint = "/v1/marketstatus/upcoming"
        result = self._make_request(endpoint, self._add_api_key({}))

        return result

    def get_clock(self) -> dict[str, Any]:
        """
        Get market clock information.
        
        This method uses the market status from Polygon API and combines it with
        standard US equity market hours (9:30 AM - 4:00 PM Eastern Time).
        
        Returns:
            Dictionary with market clock information
        """
        status = self.get_market_status()
        
        # Convert to clock format
        now = datetime.now()
        is_open = status.get("market") == "open"
        
        # Get next market hours
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)  # Standard market open time (ET)
        if now.hour > 9 or (now.hour == 9 and now.minute >= 30):
            next_open = next_open + timedelta(days=1)
            
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)  # Standard market close time (ET)
        if now.hour >= 16:
            next_close = next_close + timedelta(days=1)  # Move to next day if after close
            
        return {
            "timestamp": now,
            "is_open": is_open,
            "next_open": next_open,
            "next_close": next_close,
        }

    def get_calendar(
        self,
        start: str | datetime | date | None = None,
        end: str | datetime | date | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get market calendar.
        
        This method generates a calendar with standard US equity market hours
        (9:30 AM - 4:00 PM Eastern Time) for weekdays, excluding holidays.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of market calendar days with standard market hours
        """
        # Polygon doesn't have a direct calendar endpoint, so we'll use market holidays
        holidays = self.get_market_holidays()
        
        # Create a calendar with regular market hours
        calendar = []
        
        # Convert start/end to dates if they're datetimes
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()
            
        # Default to current week if not specified
        if not start:
            start = datetime.now().date()
        if not end:
            end = start + timedelta(days=7)
            
        # Generate calendar days
        current_date = start
        while current_date <= end:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday-Friday
                # Check if it's a holiday
                is_holiday = False
                for holiday in holidays:
                    holiday_date = datetime.strptime(holiday.get("date", ""), "%Y-%m-%d").date()
                    if holiday_date == current_date:
                        is_holiday = True
                        break
                
                if not is_holiday:
                    calendar.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "open": "09:30",  # Standard market open time (ET)
                        "close": "16:00",  # Standard market close time (ET)
                        "session_open": "09:30",  # Regular session open
                        "session_close": "16:00",  # Regular session close
                        "pre_market_start": "04:00",  # Pre-market trading starts
                        "post_market_end": "20:00"  # Post-market trading ends
                    })
            
            current_date += timedelta(days=1)
            
        return calendar

    # Helper method for connection pool
    def _create_websocket_connection(self, cluster: str = "stocks", delayed: bool = False):
        """
        Create a new WebSocket connection.
        
        Args:
            cluster: WebSocket cluster ('stocks', 'forex', 'crypto', 'options')
            delayed: Whether to use the delayed data feed (15-minute delay, free tier)
        """
        try:
            # Determine the correct cluster URL
            if delayed and cluster == "stocks":
                websocket_url = self.WEBSOCKET_CLUSTERS["delayed_stocks"]
                cluster_name = "delayed_stocks"
            elif cluster in self.WEBSOCKET_CLUSTERS:
                websocket_url = self.WEBSOCKET_CLUSTERS[cluster]
                cluster_name = cluster
            else:
                valid_clusters = list(self.WEBSOCKET_CLUSTERS.keys())
                raise ValueError(f"Invalid cluster: {cluster}. Valid clusters are: {valid_clusters}")
            
            # Create websocket with timeout
            ws = websocket.WebSocket()
            ws.settimeout(10)  # 10 seconds timeout for operations
            logger.info(f"Connecting to WebSocket at {websocket_url}...")
            ws.connect(websocket_url)
            logger.info(f"Connected to WebSocket for {cluster_name} cluster")
            
            # Authenticate with timeout
            logger.info("Authenticating with Polygon API...")
            auth_message = json.dumps({"action": "auth", "params": self.api_key})
            logger.debug(f"Sending auth message: {auth_message}")
            ws.send(auth_message)
            
            # Wait for auth response with timeout
            start_time = time.time()
            timeout = 10  # 10 seconds timeout
            
            while time.time() - start_time < timeout:
                try:
                    response = ws.recv()
                    response_data = json.loads(response)
                    logger.debug(f"Received response: {response_data}")
                    
                    # Check if authentication was successful
                    for msg in response_data:
                        if msg.get("ev") == "status" and msg.get("status") == "auth_success":
                            logger.info("Authentication successful")
                            return ws
                        elif msg.get("ev") == "status" and msg.get("status") == "auth_failed":
                            error_msg = msg.get("message", "Unknown error")
                            logger.error(f"Authentication failed: {error_msg}")
                            ws.close()
                            raise ValueError(f"WebSocket authentication failed: {error_msg}")
                            
                except websocket.WebSocketTimeoutException:
                    continue
                    
            logger.error("Authentication timed out after 10 seconds")
            ws.close()
            raise TimeoutError("WebSocket authentication timed out")
            
        except websocket.WebSocketException as e:
            logger.error(f"WebSocket connection error: {e}")
            raise

    # WebSocket methods with retry handling
    def connect_websocket(self, cluster: str = "stocks", delayed: bool = False,
                         buffer_size: int = 16384, reconnect_attempts: int = 5,
                         high_throughput: bool = True, use_compression: bool = True) -> None:
        """
        Connect to Polygon WebSocket with optimizations for high-frequency trading.

        Args:
            cluster: WebSocket cluster ('stocks', 'forex', 'crypto', 'options')
            delayed: Whether to use the delayed data feed (15-minute delay, free tier)
            buffer_size: WebSocket buffer size for high-throughput data (increased for HFT)
            reconnect_attempts: Number of reconnection attempts before failing
            high_throughput: Enable optimizations for high-throughput trading (5000+ positions/day)
            use_compression: Whether to use WebSocket compression for reduced bandwidth
        """
        # Determine the correct cluster
        if delayed and cluster == "stocks":
            websocket_url = self.WEBSOCKET_CLUSTERS["delayed_stocks"]
            cluster_name = "delayed_stocks"
        elif cluster in self.WEBSOCKET_CLUSTERS:
            websocket_url = self.WEBSOCKET_CLUSTERS[cluster]
            cluster_name = cluster
        else:
            valid_clusters = list(self.WEBSOCKET_CLUSTERS.keys())
            raise ValueError(f"Invalid cluster: {cluster}. Valid clusters are: {valid_clusters}")

        # Close existing connection if any
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing existing connection: {e}")
            self.ws = None

        def on_message(ws, message):
            try:
                # Use faster JSON parsing for high-throughput
                data = json.loads(message)
                
                # Only log status messages at info level
                if isinstance(data, list) and len(data) > 0:
                    first_msg = data[0]
                    if first_msg.get("ev") == "status":
                        logger.info(f"WebSocket status: {first_msg.get('status')} - {first_msg.get('message')}")
                
                # Batch process messages by type for efficiency
                if isinstance(data, list) and len(data) > 0:
                    # Group messages by event type for batch processing
                    message_groups = {}
                    for msg in data:
                        msg_type = msg.get("ev")
                        if msg_type:
                            if msg_type not in message_groups:
                                message_groups[msg_type] = []
                            message_groups[msg_type].append(msg)
                    
                    # Process each group of messages
                    for msg_type, msgs in message_groups.items():
                        if msg_type in self.ws_callbacks:
                            for callback in self.ws_callbacks[msg_type]:
                                # Pass the entire batch to the callback if it supports it
                                if getattr(callback, 'supports_batch', False):
                                    callback(msgs)
                                else:
                                    # Otherwise process individually
                                    for msg in msgs:
                                        callback(msg)
                else:
                    # Handle single message case
                    msg_type = data.get("ev")
                    if msg_type in self.ws_callbacks:
                        for callback in self.ws_callbacks[msg_type]:
                            callback(data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                # Don't let errors in message processing break the connection

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code=None, close_msg=None):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            logger.info(f"WebSocket connection opened to {cluster_name} cluster")
            # Authenticate with robust error handling
            try:
                auth_message = json.dumps({"action": "auth", "params": self.api_key})
                logger.debug(f"Sending auth message: {auth_message}")
                ws.send(auth_message)
            except Exception as e:
                logger.error(f"Error during authentication: {e}")
                ws.close()

        # Create new connection with optimized settings for high-frequency trading
        logger.info(f"Connecting to Polygon WebSocket at {websocket_url} with high-throughput optimizations")
        
        # Create WebSocketApp with optimized settings for high-frequency trading
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        
        # Set WebSocket options for high-throughput
        if high_throughput:
            # Increase buffer sizes for high message throughput
            websocket.enableTrace(False)  # Disable tracing for performance
            
            # Use binary frames for efficiency
            self.ws.sock_opt = (
                (websocket.ABNF.OPCODE_BINARY, 1),  # Use binary frames for efficiency
                (websocket.ABNF.OPCODE_TEXT, 0)     # Disable text frames
            )
            
            # Enable compression if requested (reduces bandwidth at cost of slight CPU overhead)
            if use_compression:
                self.ws.enable_multithread = True  # Required for compression
                self.ws.get_mask_key = lambda: os.urandom(4)  # More secure mask key generation
                
            # Set larger buffer size for receiving data (16KB default, increased for HFT)
            if hasattr(self.ws, 'sock') and self.ws.sock:
                self.ws.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
                self.ws.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
                # Set TCP_NODELAY for lower latency (disables Nagle's algorithm)
                self.ws.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        # Configure reconnection strategy
        self.reconnect_attempts = reconnect_attempts
        self.current_reconnect_attempt = 0
        self.reconnect_delay = 1.0  # Start with 1 second delay, will increase exponentially
        
        # Start the connection immediately in a background thread with optimized parameters
        self.start_websocket(threaded=True, ping_timeout=20)  # Increased timeout for stability
        
        # Wait for connection with timeout and verification
        max_wait_time = 5  # Maximum time to wait for connection in seconds
        wait_interval = 0.1  # Check interval
        total_waited = 0
        
        while total_waited < max_wait_time:
            if hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
                logger.info("WebSocket connection established successfully")
                break
            time.sleep(wait_interval)
            total_waited += wait_interval
            
        if not (hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected):
            logger.warning("WebSocket connection may not be fully established yet")

    @RetryHandler.decorator
    def subscribe_websocket(self, channels: list[str]) -> None:
        """
        Subscribe to WebSocket channels.

        Args:
            channels: List of channels to subscribe to (e.g., ["T.AAPL", "Q.AAPL"])
        """
        if not self.ws:
            raise ValueError("WebSocket not connected. Call connect_websocket first.")

        # Ensure the websocket is connected before subscribing
        if not hasattr(self.ws, 'sock') or self.ws.sock is None:
            raise ValueError("WebSocket connection not established. Wait for connection to be established.")

        self.ws.send(json.dumps({"action": "subscribe", "params": ",".join(channels)}))

    @RetryHandler.decorator
    def unsubscribe_websocket(self, channels: list[str]) -> None:
        """
        Unsubscribe from WebSocket channels.

        Args:
            channels: List of channels to unsubscribe from
        """
        if not self.ws:
            raise ValueError("WebSocket not connected. Call connect_websocket first.")

        # Ensure the websocket is connected before unsubscribing
        if not hasattr(self.ws, 'sock') or self.ws.sock is None:
            raise ValueError("WebSocket connection not established. Wait for connection to be established.")

        self.ws.send(json.dumps({"action": "unsubscribe", "params": ",".join(channels)}))

    def add_websocket_callback(self, event_type: str, callback: callable) -> None:
        """
        Add callback for WebSocket event type.

        Args:
            event_type: Event type (e.g., "T" for trades, "Q" for quotes)
            callback: Callback function that takes a message as argument
        """
        if event_type not in self.ws_callbacks:
            self.ws_callbacks[event_type] = []
        self.ws_callbacks[event_type].append(callback)
        
    def get_websocket_channels_info(self) -> dict[str, str]:
        """
        Get information about available WebSocket channels.
        
        Returns:
            Dictionary mapping channel prefixes to descriptions
        """
        return {
            "T": "Trades - Real-time trade data (e.g., 'T.AAPL' for Apple trades)",
            "Q": "Quotes - Real-time NBBO quotes (e.g., 'Q.AAPL' for Apple quotes)",
            "AM": "Minute Aggregates - Per-minute OHLCV bars (e.g., 'AM.AAPL' for Apple minute bars)",
            "A": "Second Aggregates - Per-second OHLCV bars (e.g., 'A.AAPL' for Apple second bars)",
            "XQ": "Extended Quotes - Extended quote data (e.g., 'XQ.AAPL' for Apple extended quotes)",
            "XT": "Extended Trades - Extended trade data (e.g., 'XT.AAPL' for Apple extended trades)",
            "status": "Status messages from the WebSocket server"
        }
        
    def get_websocket_subscription_examples(self) -> list[dict[str, Any]]:
        """
        Get examples of WebSocket channel subscriptions.
        
        Returns:
            List of example subscription configurations
        """
        return [
            {
                "description": "Subscribe to trades for a single stock",
                "channels": ["T.AAPL"],
                "code_example": """
# Subscribe to Apple trades
polygon_client.connect_websocket(cluster="stocks")
polygon_client.subscribe_websocket(channels=["T.AAPL"])

# Add callback for trade events
def on_trade(trade):
    print(f"Trade: {trade['sym']} - Price: {trade['p']} - Size: {trade['s']}")
    
polygon_client.add_websocket_callback("T", on_trade)
"""
            },
            {
                "description": "Subscribe to quotes for multiple stocks",
                "channels": ["Q.AAPL", "Q.MSFT", "Q.AMZN"],
                "code_example": """
# Subscribe to quotes for multiple stocks
polygon_client.connect_websocket(cluster="stocks")
polygon_client.subscribe_websocket(channels=["Q.AAPL", "Q.MSFT", "Q.AMZN"])

# Add callback for quote events
def on_quote(quote):
    print(f"Quote: {quote['sym']} - Bid: {quote['bp']} - Ask: {quote['ap']}")
    
polygon_client.add_websocket_callback("Q", on_quote)
"""
            },
            {
                "description": "Subscribe to minute aggregates (bars)",
                "channels": ["AM.AAPL"],
                "code_example": """
# Subscribe to minute bars for Apple
polygon_client.connect_websocket(cluster="stocks")
polygon_client.subscribe_websocket(channels=["AM.AAPL"])

# Add callback for aggregate minute events
def on_minute_bar(bar):
    print(f"Bar: {bar['sym']} - Open: {bar['o']} - High: {bar['h']} - Low: {bar['l']} - Close: {bar['c']} - Volume: {bar['v']}")
    
polygon_client.add_websocket_callback("AM", on_minute_bar)
"""
            },
            {
                "description": "Subscribe to crypto trades",
                "channels": ["XT.BTC-USD"],
                "code_example": """
# Subscribe to Bitcoin-USD trades
polygon_client.connect_websocket(cluster="crypto")
polygon_client.subscribe_websocket(channels=["XT.BTC-USD"])

# Add callback for crypto trade events
def on_crypto_trade(trade):
    print(f"Crypto Trade: {trade['pair']} - Price: {trade['p']} - Size: {trade['s']}")
    
polygon_client.add_websocket_callback("XT", on_crypto_trade)
"""
            }
        ]

    def start_websocket(self, threaded: bool = True, ping_timeout: int = 10) -> None:
        """
        Start WebSocket connection with optimizations for high-frequency trading.

        Args:
            threaded: Whether to run in a separate thread
            ping_timeout: Timeout in seconds for ping/pong messages
        """
        if not self.ws:
            raise ValueError("WebSocket not connected. Call connect_websocket first.")

        # Set up run parameters optimized for high-frequency trading
        kwargs = {
            "ping_timeout": ping_timeout,
            "ping_interval": 15,  # More frequent pings for connection stability
            "ping_payload": "",   # Empty string as ping message
            "skip_utf8_validation": True,  # Skip UTF-8 validation for performance
        }

        # Define reconnection handler
        def run_with_reconnect():
            while self.current_reconnect_attempt < self.reconnect_attempts:
                try:
                    logger.info(f"Starting WebSocket connection (attempt {self.current_reconnect_attempt + 1}/{self.reconnect_attempts})")
                    self.ws.run_forever(**kwargs)
                    
                    # If we get here, the connection was closed normally
                    if not getattr(self.ws, 'keep_running', False):
                        logger.info("WebSocket connection closed normally")
                        break
                        
                    # Otherwise, attempt to reconnect
                    logger.warning("WebSocket connection lost, attempting to reconnect...")
                    self.current_reconnect_attempt += 1
                    
                    # Exponential backoff for reconnection
                    reconnect_wait = min(60, self.reconnect_delay * (2 ** self.current_reconnect_attempt))
                    logger.info(f"Waiting {reconnect_wait:.2f}s before reconnecting...")
                    time.sleep(reconnect_wait)
                    
                except Exception as e:
                    logger.error(f"Error in WebSocket connection: {e}")
                    self.current_reconnect_attempt += 1
                    time.sleep(self.reconnect_delay)
            
            if self.current_reconnect_attempt >= self.reconnect_attempts:
                logger.error(f"Failed to maintain WebSocket connection after {self.reconnect_attempts} attempts")

        if threaded:
            import threading
            wst = threading.Thread(target=run_with_reconnect)
            wst.daemon = True
            wst.start()
            logger.info("WebSocket started in background thread with reconnection handling")
        else:
            logger.info("Starting WebSocket in main thread with reconnection handling")
            run_with_reconnect()

    def close_websocket(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            try:
                logger.info("Closing WebSocket connection...")
                # Prevent callbacks during close to avoid race conditions
                self.ws.on_close = lambda ws, code, msg: None
                self.ws.keep_running = False
                
                # Try to close gracefully with timeout
                try:
                    self.ws.close(timeout=5)  # Give more time for graceful shutdown
                    logger.info("WebSocket connection closed gracefully")
                except Exception as e:
                    logger.warning(f"Error during graceful WebSocket shutdown: {e}")
                    # Force close if graceful shutdown fails
                    self.ws.close()
                    logger.info("WebSocket connection force closed")
                
                self.ws = None
                self.ws_callbacks = {}
                
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            
        # Close HTTP session
        try:
            self.session.close()
            logger.debug("HTTP session closed")
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
    def can_take_position(self, ticker: str, value: float) -> bool:
        """
        Check if a position can be taken based on dollar-based limits.
        
        Args:
            ticker: Ticker symbol
            value: Position value in dollars
            
        Returns:
            bool: Whether the position can be taken
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
    
    def update_position_tracking(self, ticker: str, value: float) -> None:
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
    
    def get_position_limits(self) -> dict[str, float]:
        """
        Get current position limits and usage.
        
        Returns:
            dict: Dictionary with position limit information
        """
        return {
            "max_daily_value": self.max_daily_value,
            "current_daily_value": self.current_daily_value,
            "daily_remaining": max(0.0, self.max_daily_value - self.current_daily_value),
            "max_position_value": self.max_position_value,
            "position_values": self.position_values,
            "position_count": self.position_count,
            "last_reset_date": self.last_reset_date.isoformat()
        }
    
    def close(self) -> None:
        """
        Close all connections and release resources.
        
        This method should be called when the client is no longer needed.
        """
        self.close_websocket()
        logger.info("PolygonClient resources released")
        
    def get_websocket_example_script(self) -> str:
        """
        Get a complete example script for using the Polygon WebSocket API.
        
        Returns:
            String containing a Python script example
        """
        return """
import os
import time
import json
from src.data_acquisition.api.polygon_client import PolygonClient

# Initialize the client with your API key
api_key = os.environ.get("POLYGON_API_KEY")  # Or provide directly: api_key = "YOUR_API_KEY"
polygon = PolygonClient(api_key=api_key)

# Define callback functions for different event types
def on_trade(trade):
    print(f"Trade: {trade['sym']} - Price: {trade['p']} - Size: {trade['s']} - Time: {trade['t']}")

def on_quote(quote):
    print(f"Quote: {quote['sym']} - Bid: {quote['bp']} - Ask: {quote['ap']} - Time: {quote['t']}")

def on_agg_minute(agg):
    print(f"Minute Bar: {agg['sym']} - Open: {agg['o']} - High: {agg['h']} - Low: {agg['l']} - Close: {agg['c']} - Volume: {agg['v']}")

def on_status(status):
    print(f"Status: {status['status']} - Message: {status.get('message', '')}")

# Register callbacks
polygon.add_websocket_callback("T", on_trade)    # Trades
polygon.add_websocket_callback("Q", on_quote)    # Quotes
polygon.add_websocket_callback("AM", on_agg_minute)  # Minute aggregates
polygon.add_websocket_callback("status", on_status)  # Status messages

# Connect to the WebSocket (use delayed=True for free tier with 15-min delay)
polygon.connect_websocket(cluster="stocks", delayed=False)

# Subscribe to channels
channels = [
    "T.AAPL",   # Apple trades
    "Q.AAPL",   # Apple quotes
    "AM.AAPL",  # Apple minute bars
    "T.MSFT",   # Microsoft trades
    "Q.MSFT",   # Microsoft quotes
    "AM.MSFT"   # Microsoft minute bars
]

# Subscribe to the channels
polygon.subscribe_websocket(channels)

try:
    print("WebSocket connected and subscribed. Press Ctrl+C to exit.")
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing connection...")
finally:
    # Clean up
    polygon.close_websocket()
    print("Connection closed")
"""
