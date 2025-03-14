"""
Unusual Whales API Client

This module provides a client for interacting with the Unusual Whales API to fetch options flow data,
unusual options activity, and other market insights.
"""
import os

import time
import logging
import warnings
from datetime import datetime, date
from typing import Any

import pandas as pd
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout
from src.utils.api import (
    RateLimiter,
    RetryHandler,
)

# Set up logger
logger = logging.getLogger(__name__)


class UnusualWhalesClient:
    """Client for interacting with the Unusual Whales API."""
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limit: int = 2,
        retry_attempts: int = 3,
        timeout: int = 30,
        verify_ssl: bool = True,
        max_position_value: float = 2500.0,  # Maximum $2500 per stock
        max_daily_value: float = 5000.0,     # Maximum $5000 per day
    ):
        """
        Initialize the Unusual Whales API client with dollar-based position limits.
        
        Args:
            api_key: Unusual Whales API key
            base_url: Base URL for API requests
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
            self.api_key = os.environ.get("UNUSUAL_WHALES_API_KEY")
            logger.debug(f"Using API key from environment: {self.api_key[:5]}..." if self.api_key and len(self.api_key) > 5 else "No API key found in environment")
            
        if not self.api_key:
            raise ValueError("Unusual Whales API key is required")

        # Set base URL
        self.BASE_URL = base_url or "https://api.unusualwhales.com"
        
        # Position tracking for dollar-based limits
        self.max_position_value = max_position_value  # Maximum position value per stock
        self.max_daily_value = max_daily_value        # Maximum total position value per day
        self.current_daily_value = 0.0                # Current total position value for the day
        self.position_values = {}                     # Current position value per ticker
        self.position_count = 0                       # Number of positions taken today
        self.last_reset_date = datetime.now().date()  # Last date when limits were reset
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}", 
            "Accept": "application/json, text/plain", 
            "User-Agent": "AutonomousTradingSystem/0.1"
        })
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_period=rate_limit * 60,
            period_seconds=60,
            strategy=RateLimiter.RateLimitStrategy.TOKEN_BUCKET,
            burst_size=rate_limit * 60,
            name="unusual_whales"
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
            name="unusual_whales"
        )
        
        # Rate limiting parameters
        self.rate_limit_remaining = rate_limit * 60  # Requests per minute
        self.rate_limit_reset = 0
        self.max_retries = retry_attempts
        self.retry_delay = 1.0
        self.timeout = timeout
        self.last_request_time = 0
        self.request_interval = 1.0 / rate_limit  # Minimum time between requests
        self.verify_ssl = verify_ssl
        
        logger.info(f"Initialized UnusualWhalesClient with rate limiting ({rate_limit} req/sec), retry handling (max {retry_attempts} retries), and timeout {timeout}s")

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """
        Handle rate limit headers from Unusual Whales API.

        Args:
            response: API response
        """
        # Update rate limit information from headers if available
        if "X-RateLimit-Remaining" in response.headers:
            self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])

        if "X-RateLimit-Reset" in response.headers:
            self.rate_limit_reset = int(response.headers["X-RateLimit-Reset"])

        # If we're close to the rate limit, sleep until reset
        if self.rate_limit_remaining < 5:
            sleep_time = max(0, self.rate_limit_reset - time.time()) + 1
            logger.warning(
                f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds"
            )
            time.sleep(sleep_time)

    @property
    def rate_limit(self):
        """
        Compatibility property for old code that accesses rate_limit directly.
        Returns the remaining rate limit value.
        
        Returns:
            int: The remaining rate limit value
        """
        return self.rate_limit_remaining

    def _ensure_rate_limit(self) -> None:
        """
        Ensure we don't exceed rate limits by enforcing minimum time between requests
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than request_interval seconds have passed since the last request
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.4f}s to maintain {self.rate_limit} req/sec")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    @RateLimiter.decorator
    @RetryHandler.decorator
    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
        v2_base_url: str | None = None,
        timeout: int | None = None,
        high_priority: bool = False
    ) -> dict[str, Any]:
        """
        Make a request to the Unusual Whales API with optimizations for high-frequency trading.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters or JSON body (for POST requests)
            method: HTTP method (GET or POST)
            v2_base_url: Optional base URL override for v2 API
            timeout: Request timeout in seconds (defaults to self.timeout)
            high_priority: Whether this request is high priority (for real-time trading signals)

        Returns:
            API response as a dictionary

        Raises:
            requests.RequestException: If the request fails after retries
        """
        # Use provided base URL or default
        base_url = v2_base_url or self.BASE_URL
        url = f"{base_url}{endpoint}"
        
        # Adjust timeout for high-priority requests
        if high_priority:
            request_timeout = (timeout or self.timeout) // 2  # Shorter timeout for critical requests
        else:
            request_timeout = timeout or self.timeout
            
        params = params or {}
        
        # Only log detailed debug info for non-high-priority requests to reduce overhead
        if not high_priority:
            logger.debug(f"Making request to {method} {url} with params: {params}")
        
        # Make the request with connection pooling and keep-alive
        try:
            if method.upper() == "GET":
                # Use stream=False for better performance with small responses
                response = self.session.get(
                    url,
                    params=params,
                    timeout=request_timeout,
                    stream=False,
                    headers={
                        **self.session.headers,
                        "Connection": "keep-alive"  # Ensure connection reuse
                    }
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url,
                    json=params,
                    timeout=request_timeout,
                    headers={
                        **self.session.headers,
                        "Connection": "keep-alive"  # Ensure connection reuse
                    }
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}, must be GET or POST")
            
            self._handle_rate_limit(response)
            
            # Handle HTTP errors
            response.raise_for_status()
            
            # Parse JSON response efficiently
            data = response.json()
            
            # Check for API-level errors
            if data.get("status") == "error" or data.get("error"):
                error_msg = data.get("message", "Unknown API error")
                if "error_code" in data:
                    error_msg = f"{data.get('error_code')}: {data.get('error_message', 'Unknown error')}"
                raise ValueError(f"Unusual Whales API error: {error_msg}")
            
            # Only log detailed success info for non-high-priority requests
            if not high_priority and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Successful response from {url}: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            
            return data
            
        except (ConnectionError, Timeout) as e:
            # For connection errors, try to reset the session
            logger.warning(f"Connection error in request to {url}: {e}. Resetting session.")
            self.session.close()
            self.session = requests.Session()
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json, text/plain",
                "User-Agent": "AutonomousTradingSystem/0.1"
            })
            raise  # Re-raise for retry handler

    def _validate_date_param(self, date_param: str | datetime | date | None) -> str | None:
        """
        Validate and convert date parameter to string format.
        
        Args:
            date_param: Date parameter (string, datetime, or date)
            
        Returns:
            str | None: Validated date string or None
            Validated date string or None
        """
        if isinstance(date_param, (datetime, date)):
            return date_param.strftime("%Y-%m-%d")
        return date_param

    def get_options_flow(
        self,
        limit: int = 100,
        page: int = 0,
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
        ticker: str | None = None,
        symbol: str | None = None,         # Alternative parameter name for ticker
        issue_type: str | None = None,     # 'call' or 'put'
        alert_rule: str | None = None,     # Alert rule (unusual_volume, large_block, etc.)
        high_priority: bool = False,       # Whether this is a high-priority request for real-time trading
        min_premium: float | None = None,  # Minimum premium value
        max_premium: float | None = None,  # Maximum premium value
        min_size: int | None = None,       # Minimum contract size
        max_size: int | None = None,       # Maximum contract size
        sentiment: str | None = None,      # Filter by sentiment (bullish, bearish, neutral)
        cache_results: bool = True,        # Whether to cache results for repeated queries
        **additional_params
    ) -> list[dict[str, Any]]:
        """
        Get options flow data optimized for high-frequency trading with enhanced filtering.

        Args:
            limit: Number of results to return (default: 100, max: 200)
            page: Page number for pagination (default: 0)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            ticker: Filter by ticker symbol (now called 'symbol' in the API)
            symbol: Alternative parameter name for ticker
            issue_type: Filter by contract type ('call' or 'put')
            alert_rule: Filter by alert rule ('unusual_volume', 'large_block', 'sweep', etc.)
            high_priority: Whether this is a high-priority request for real-time trading
            min_premium: Minimum premium value for filtering results
            max_premium: Maximum premium value for filtering results
            min_size: Minimum contract size for filtering results
            max_size: Maximum contract size for filtering results
            sentiment: Filter by sentiment (bullish, bearish, neutral)
            cache_results: Whether to cache results for repeated queries
            **additional_params: Additional parameters to pass to the API

        Returns:
            list[dict[str, Any]]: List of options flow data
            List of options flow data
        """
        # Static cache for repeated queries (class-level cache)
        if not hasattr(self.__class__, '_options_flow_cache'):
            self.__class__._options_flow_cache = {}
            self.__class__._cache_timestamps = {}
        
        # Generate cache key based on request parameters
        cache_key = None
        if cache_results:
            cache_params = {
                'ticker': ticker or symbol,
                'issue_type': issue_type,
                'high_priority': high_priority,
                'limit': limit,
                'page': page
            }
            cache_key = str(hash(frozenset(cache_params.items())))
            
            # Check if we have a cached result that's less than 5 seconds old for high-priority
            # or 30 seconds for non-high-priority
            cache_ttl = 5 if high_priority else 30  # seconds
            if (cache_key in self.__class__._options_flow_cache and
                cache_key in self.__class__._cache_timestamps and
                time.time() - self.__class__._cache_timestamps[cache_key] < cache_ttl):
                logger.debug(f"Using cached options flow data for {ticker or symbol}")
                return self.__class__._options_flow_cache[cache_key]
        
        # For high-frequency trading, we need to optimize the request
        # Increase limit for high-priority requests to reduce number of API calls
        if high_priority:
            actual_limit = min(200, limit)  # Maximum allowed by API
        else:
            actual_limit = min(limit, 200)
            
        params = {"limit": actual_limit, "page": page}
        
        # The new API uses ticker_symbols parameter, comma-separated
        if symbol is not None:
            params["ticker_symbols"] = symbol
        elif ticker is not None:
            params["ticker_symbols"] = ticker
        
        # For high-frequency trading, focus on intraday data when in high priority mode
        if high_priority:
            params["intraday_only"] = "true"  # Focus on intraday data for real-time trading
            
            # For high-priority requests, focus only on the most relevant notification types
            params["noti_types[]"] = [
                "option_contract",
                "flow_alerts"
            ]
        else:
            # For non-high-priority requests, get more comprehensive data
            params["intraday_only"] = "false"
            
            # Include a wide range of notification types
            params["noti_types[]"] = [
                "option_contract",
                "option_contract_interval",
                "flow_alerts",
                "chain_oi_change",
                "dividends",
                "earnings",
                "insider_trades",
                "trading_state",
                "analyst_rating",
                "price_target"
            ]

        # Add issue_type if provided
        if issue_type:
            params["issue_type"] = issue_type
            
        # Add premium filters if provided
        if min_premium is not None:
            params["min_premium"] = min_premium
        if max_premium is not None:
            params["max_premium"] = max_premium
            
        # Add size filters if provided
        if min_size is not None:
            params["min_size"] = min_size
        if max_size is not None:
            params["max_size"] = max_size
            
        # Add sentiment filter if provided
        if sentiment:
            params["sentiment"] = sentiment
            
        # Add any custom parameters
        if additional_params:
            params.update(additional_params)
        
        # For high-frequency trading, go directly to the most efficient endpoint
        if high_priority and (symbol or ticker):
            # Use the most direct endpoint for real-time trading
            options_params = {
                "limit": actual_limit,
                "page": page,
                "symbol": symbol or ticker,
                "intraday_only": "true"
            }
            
            if issue_type:
                options_params["issue_type"] = issue_type
                
            if alert_rule:
                options_params["alert_rule"] = alert_rule
                
            # Make high-priority request
            options_result = self._make_request(
                "/api/option-trades/flow-alerts",
                options_params,
                high_priority=True
            )
            return options_result.get("data", [])
        
        # For non-high-priority requests, use the standard approach with fallbacks
        result = self._make_request("/api/alerts", params)
        data = result.get("data", [])
        
        if not data and (symbol or ticker):
            # Try using the options/trades endpoint directly if alerts returns no data
            logger.debug("Trying options/trades endpoint after empty alerts response")
            options_params = {"limit": actual_limit, "page": page}
            
            if symbol:
                options_params["symbol"] = symbol
            elif ticker:
                options_params["symbol"] = ticker
                
            # Add date range parameters
            # Use proper parameters for the options flow endpoint
            options_params["intraday_only"] = "false"
            options_params["noti_types[]"] = [
                "option_contract",
                "option_contract_interval",
                "flow_alerts",
                "chain_oi_change"
            ]
            
            if issue_type:
                options_params["issue_type"] = issue_type
                
            if alert_rule:
                options_params["alert_rule"] = alert_rule
                
            options_result = self._make_request("/api/option-trades/flow-alerts", options_params)
            data = options_result.get("data", [])
            
            if not data:
                # If still no data, try screener/option-contracts
                logger.debug("Trying screener/option-contracts endpoint")
                screener_params = {"limit": actual_limit, "page": page}
            
                if symbol:
                    screener_params["ticker_symbol"] = symbol
                elif ticker:
                    screener_params["ticker_symbol"] = ticker
                
                # Add date range parameters
                screener_params["limit"] = 100  # Increase limit to get more data
                    
                if issue_type:
                    screener_params["issue_types"] = [issue_type]
                    
                screener_result = self._make_request("/api/screener/option-contracts", screener_params)
                data = screener_result.get("data", [])

        if data:
            if not high_priority:  # Only log for non-high-priority to reduce overhead
                logger.info(f"Retrieved {len(data)} options flow entries")
                
            # Cache the results if caching is enabled
            if cache_results and cache_key:
                self.__class__._options_flow_cache[cache_key] = data
                self.__class__._cache_timestamps[cache_key] = time.time()
                
                # Limit cache size to prevent memory issues (keep last 100 queries)
                if len(self.__class__._options_flow_cache) > 100:
                    # Remove oldest cache entries
                    oldest_keys = sorted(
                        self.__class__._cache_timestamps.keys(),
                        key=lambda k: self.__class__._cache_timestamps[k]
                    )[:len(self.__class__._cache_timestamps) - 100]
                    
                    for key in oldest_keys:
                        self.__class__._options_flow_cache.pop(key, None)
                        self.__class__._cache_timestamps.pop(key, None)
        else:
            logger.warning("No options flow data returned from API after trying multiple endpoints")
            
        return data

    def get_option_trades(self, 
                         symbol: str,
                         issue_type: str | None = None,
                         alert_rule: str | None = None,
                         contract_type: str | None = None,
                         min_strike: float | None = None,
                         max_strike: float | None = None,
                         min_expiry: str | None = None,
                         max_expiry: str | None = None,
                         **additional_params) -> list[dict[str, Any]]:
        """
        Get option trades for a symbol.
        
        Args:
            symbol: Stock symbol
            issue_type: Filter by contract type ('call' or 'put')
            alert_rule: Filter by alert rule ('unusual_volume', 'large_block', 'sweep', etc.)
            contract_type: Filter by contract type
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            min_expiry: Minimum expiration date
            max_expiry: Maximum expiration date
            **additional_params: Additional parameters to pass to the API
        """
        params = {"symbol": symbol}
        
        # Add optional parameters if provided
        if issue_type:
            params["issue_type"] = issue_type
        if alert_rule:
            params["alert_rule"] = alert_rule
        if contract_type:
            params["contract_type"] = contract_type
        if min_strike is not None:
            params["min_strike"] = min_strike
        if max_strike is not None:
            params["max_strike"] = max_strike
        if min_expiry:
            params["min_expiry"] = min_expiry
        if max_expiry:
            params["max_expiry"] = max_expiry
        
        # Add any additional parameters
        params.update(additional_params)
        
        # Add date range parameters if not provided
        if "from_date" not in params:
            params["limit"] = 100  # Increase limit to get more data
        
        result = self._make_request("/api/option-trades/flow-alerts", params)
        return result.get("data", [])

    def get_historical_flow(
        self,
        limit: int = 100,
        page: int = 1,
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
        ticker: str | None = None,
        min_premium: float | None = None,
        max_premium: float | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        min_oi: int | None = None,
        sentiment: str | None = None,
        contract_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get historical options flow data.

        Args:
            limit: Number of results to return (default: 100)
            page: Page number for pagination (default: 1)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            ticker: Filter by ticker symbol
            min_premium: Minimum premium
            max_premium: Maximum premium
            min_size: Minimum size
            max_size: Maximum size
            min_oi: Minimum open interest
            sentiment: Filter by sentiment
            contract_type: Filter by contract type

        Returns:
            list[dict[str, Any]]: List of historical unusual options activity
            List of historical unusual options activity
        """
        # Validate date parameters
        from_date = self._validate_date_param(from_date)
        to_date = self._validate_date_param(to_date)

        params = {"limit": limit, "page": page}

        # Add optional parameters
        if ticker:
            params["ticker"] = ticker
        if min_premium is not None:
            params["min_premium"] = min_premium
        if max_premium is not None:
            params["max_premium"] = max_premium
        if min_size is not None:
            params["min_size"] = min_size
        if max_size is not None:
            params["max_size"] = max_size
        if min_oi is not None:
            params["min_oi"] = min_oi
        if sentiment:
            params["sentiment"] = sentiment
        if contract_type:
            params["contract_type"] = contract_type

        # Add date parameters if provided
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        result = self._make_request("/api/flow/historical", params)

        return result.get("data", [])

    def get_alert_details(self, alert_id: str) -> dict[str, Any]:
        """
        Get detailed information about a specific alert.

        Args:
            alert_id: ID of the alert

        Returns:
            dict[str, Any]: Detailed information about the alert
            Detailed information about the alert
        """
        result = self._make_request(f"/api/alerts/{alert_id}")

        return result.get("data", {})

    def get_options_chain(
        self,
        ticker: str,  # Required parameter
        expiration: str | datetime | None = None,  # Now called expiry_dates
        strike: float | None = None,  # Can use min_strike and max_strike
        contract_type: str | None = None,  # Now called issue_types
        limit: int = 100,
        page: int = 1,
        **additional_params
    ) -> list[dict[str, Any]]:
        """
        Get the options chain for a specific ticker.

        Args:
            ticker: Stock symbol
            expiration: Filter by expiration date (YYYY-MM-DD or datetime) - now called expiry_dates
            strike: Filter by strike price - can use min_strike and max_strike
            contract_type: Filter by contract type (call, put) - now called issue_types
            limit: Number of results to return (default: 100)
            page: Page number for pagination
            **additional_params: Additional parameters to pass to the API

        Returns:
            list[dict[str, Any]]: Options chain with strikes, expirations, and pricing
            Options chain with strikes, expirations, and pricing
        """
        params = {
            "ticker_symbol": ticker,
            "limit": limit,
            "page": page
        }
        
        # Handle optional parameters
        if expiration:
            expiry = self._validate_date_param(expiration)
            params["expiry_dates"] = [expiry]
        if strike is not None:
            # For exact strike, set min and max to the same value
            params["min_strike"] = strike
            params["max_strike"] = strike
        if contract_type:
            params["issue_types"] = [contract_type.lower()]  # API expects lowercase
            
        # Add any additional parameters
        params.update(additional_params)
        
        # Use the new endpoint
        result = self._make_request("/api/screener/option-contracts", params)
        
        return result.get("data", [])

    def get_unusual_score(self, ticker: str) -> dict[str, Any]:
        """
        Get the unusual score for a ticker. 

        Note: This endpoint may have changed in the API. Using stock screener as a fallback.

        Args:
            ticker: Stock symbol

        Returns:
            dict[str, Any]: Stock data including volatility metrics or empty dict if data not available
            Stock data including volatility metrics or empty dict if data not available
        """
        # Use the stock screener endpoint with the ticker filter
        params = {
            "ticker": ticker,
            "limit": 1  # We only need one result
        }
        
        try:
            result = self._make_request("/api/screener/stocks", params)
            data = result.get("data", [])

            # If we got data, return the first item
            if data:
                result = data[0]
                
                # Only use actual data from the API, no synthetic values
                # If 'score' is not in the response, we don't add it
                return result

            # If no data was returned, log a warning
            if not data:
                logger.warning(f"No unusual score data available for ticker {ticker} from /api/screener/stocks endpoint")
        except Exception as e:
            logger.error(f"Error getting unusual score for {ticker} from /api/screener/stocks endpoint: {e}")
        
        # Return empty dict if no data or error
        return {}


    def get_top_tickers(
        self,
        limit: int = 100,
        order: str = "volume",  # Default to sorting by volume
        order_direction: str = "desc",  # Default to descending order
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
        **additional_params
    ) -> list[dict[str, Any]]:
        """
        Get the top tickers by unusual activity.

        Args:
            limit: Number of results to return (default: 100)
            order: Field to sort by (default: "volume")
            order_direction: Sort direction (default: "desc")
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            **additional_params: Additional parameters to pass to the API

        Returns:
            list[dict[str, Any]]: List of top tickers with unusual activity metrics
            List of top tickers with unusual activity metrics
        """
        # Validate date parameters
        params = {
            "limit": limit,
            "order": order,
            "order_direction": order_direction
        }
        
        # Handle date parameters
        if from_date:
            # For stock screener, we don't have direct date filters
            # Could potentially use next_earnings_date as a proxy
            pass
        if to_date:
            pass
            
        # Add any additional parameters
        params.update(additional_params)
            
        # Use the stock screener endpoint
        result = self._make_request("/api/screener/stocks", params)

        return result.get("data", [])

    def get_sector_analysis(
        self,
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get unusual activity by sector.

        Args:
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)

        Returns:
            dict[str, Any]: Unusual activity metrics by sector
            Unusual activity metrics by sector
        """
        # Validate date parameters
        from_date = self._validate_date_param(from_date)
        to_date = self._validate_date_param(to_date)

        params = {}

        # Add optional parameters if provided
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        result = self._make_request("/api/flow/sectors", params)

        return result.get("data", {})
    
    def get_unusual_options(self, 
                            limit: int = 100,
                           page: int = 1,
                           from_date: str | datetime | date | None = None,
                           to_date: str | datetime | date | None = None,
                           ticker: str | None = None,
                           **additional_params) -> dict[str, Any]:
        """
        Get unusual options activity.
        
        This method is deprecated and will be removed in a future version.
        Use get_options_flow instead.
        
        Args:
            limit: Number of results to return
            page: Page number for pagination
            from_date: Start date
            to_date: End date
            ticker: Filter by ticker symbol
            **additional_params: Additional parameters to pass to the API
            
        Returns:
            list[dict[str, Any]]: Unusual options activity data
            Unusual options activity data
        """
        warnings.warn("get_unusual_options is deprecated and will be removed in version 2.0. "
                     "Use get_options_flow instead.", DeprecationWarning, stacklevel=2)
        
        # Validate date parameters
        from_date = self._validate_date_param(from_date)
        to_date = self._validate_date_param(to_date)
        
        params = {"limit": limit, "page": page}
        
        # Add date parameters if provided
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if ticker:
            params["ticker"] = ticker
            
        # Add any additional parameters
        params.update(additional_params)
        
        # Use the new endpoint with the old parameters
        return self.get_options_flow(**params)
    
    # Added alias for backward compatibility
    def get_unusual_options_activity(self, **kwargs) -> list[dict[str, Any]]:
        warnings.warn("get_unusual_options_activity is deprecated and will be removed in version 2.0. "
                     "Use get_options_flow instead.", DeprecationWarning, stacklevel=2)
        return self.get_unusual_options(**kwargs)
    
    def get_dark_pool(self, 
                      ticker: str | None = None,
                     limit: int = 100,
                     page: int = 1,
                     from_date: str | datetime | date | None = None,
                     to_date: str | datetime | date | None = None,
                     min_volume: int | None = None,
                     max_volume: int | None = None) -> dict[str, Any]:
        """
        Get dark pool data.
        
        Args:
            ticker: Filter by ticker symbol
            limit: Number of results to return
            page: Page number for pagination
            from_date: Start date
            to_date: End date
            min_volume: Minimum volume
            max_volume: Maximum volume
            
        Returns:
            dict[str, Any]: Dark pool data containing dark pool trades
        """
        params = {"limit": limit, "page": page}
        if ticker:
            params["ticker"] = ticker
            
        # Add date range parameters
        params["limit"] = 100  # Increase limit to get more data
            
        result = self._make_request("/api/darkpool/recent", params)
        return result.get("data", {})

    def flow_to_dataframe(self, flow_data: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Convert options flow data to a pandas DataFrame.

        Args:
            flow_data: Options flow data from get_options_flow or get_historical_flow

        Returns:
            pd.DataFrame: DataFrame with options flow data
            DataFrame with options flow data
        """
        if not flow_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(flow_data)
            logger.debug("Created flow DataFrame using pandas")
        except Exception as e:
            logger.warning(f"Error creating DataFrame: {e}. Retrying.")
            df = pd.DataFrame(flow_data)

        # Convert timestamp columns to datetime
        datetime_columns = ["timestamp", "expiration_date", "trade_time"]
        for col in datetime_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.debug(f"Converted {col} using pandas")
                except Exception as e:
                    logger.warning(f"Error converting {col} to datetime: {e}")
                    df[col] = pd.to_datetime(df[col]) 

        # Convert numeric columns
        numeric_columns = [
            "strike",
            "premium",
            "size",
            "open_interest",
            "implied_volatility",
            "delta",
            "gamma",
            "theta",
            "vega",
        ]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logger.debug(f"Converted numeric column {col} using pandas")
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {e}")
                    df[col] = pd.to_numeric(df[col], errors="coerce") 

        return df
    
    def dark_pool_to_dataframe(self, dark_pool_data: dict[str, Any]) -> pd.DataFrame:
        """
        Convert dark pool data to a pandas DataFrame.
        
        Args:
            dark_pool_data: Dark pool data from get_dark_pool
            
        Returns:
            pd.DataFrame: DataFrame with dark pool data
            DataFrame with dark pool data
        """
        if not dark_pool_data or "data" not in dark_pool_data:
            # If dark_pool_data is already the data array, use it directly
            try:
                df = pd.DataFrame(dark_pool_data if isinstance(dark_pool_data, list) else [])
                logger.debug("Created dark pool DataFrame using pandas")
            except Exception as e:
                logger.warning(f"Error creating DataFrame: {e}")
                df = pd.DataFrame(dark_pool_data if isinstance(dark_pool_data, list) else [])
        else:
            try:
                df = pd.DataFrame(dark_pool_data["data"])
                logger.debug("Created dark pool DataFrame from data using pandas")
            except Exception as e:
                logger.warning(f"Error creating DataFrame: {e}")
                df = pd.DataFrame(dark_pool_data["data"])
            
        # Convert timestamp columns to datetime
        datetime_columns = ["timestamp", "trade_time"]
        for col in datetime_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.debug(f"Converted dark pool {col} using pandas")
                except Exception as e:
                    logger.warning(f"Error converting {col} to datetime: {e}")
                    df[col] = pd.to_datetime(df[col]) 
                
        # Convert numeric columns
        numeric_columns = ["volume", "price", "market_cap"]
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logger.debug(f"Converted dark pool numeric column {col} using pandas")
                except Exception as e:
                    logger.warning(f"Error converting {col} to numeric: {e}")
                    df[col] = pd.to_numeric(df[col], errors="coerce") 
                
        return df
    
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
        Close the session and release resources.
        
        This method should be called when the client is no longer needed.
        """
        if self.session:
            logger.debug("Closing session and releasing resources")
            self.session.close()
    
    def __enter__(self):
        """
        Support for context manager protocol
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol
        """
        self.close()
        return False  # Don't suppress exceptions

