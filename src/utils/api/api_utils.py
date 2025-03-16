"""
API utilities for the Autonomous Trading System.

This module provides utilities for working with external APIs, including
rate limiting, error handling, response parsing, and specialized support
for financial market APIs like Alpaca.
"""

import os
import time
import json
import logging
import functools
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum, auto
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logger = logging.getLogger(__name__)

# Forward declarations for class decorators
def rate_limiter_decorator(func: Callable) -> Callable:
    """Placeholder declaration for RateLimiter.decorator - implemented below."""
    return func

def retry_handler_decorator(func: Callable) -> Callable:
    """Placeholder declaration for RetryHandler.decorator - implemented below."""
    return func

# These will be replaced by the actual class methods when they're defined
RateLimiter = type('RateLimiter', (), {'decorator': staticmethod(rate_limiter_decorator)})
RetryHandler = type('RetryHandler', (), {'decorator': staticmethod(retry_handler_decorator)})


class APIClient:
    """Base API client with rate limiting and error handling."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        rate_limit: int = 5,
        retry_attempts: int = 3,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for API requests
            api_key: API key
            api_secret: API secret
            rate_limit: Rate limit in requests per second
            retry_attempts: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            headers: Additional headers to include in all requests
        """
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Create a session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "AutonomousTradingSystem/0.1",
            "Content-Type": "application/json",
            **(headers or {})
        })
        
        # Add API key to headers if provided
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
            
        # Initialize rate limiter and retry handler
        self.rate_limiter = RateLimiter(
            requests_per_period=rate_limit,
            period_seconds=1.0,
            strategy=RateLimiter.RateLimitStrategy.TOKEN_BUCKET,
            name="api_client"
        )
        
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
            name="api_client"
        )
        
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            API response (parsed JSON)
        """
        url = f"{self.base_url}{endpoint}"
        return self._make_request("GET", url, params=params)
        
    def post(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            API response (parsed JSON)
        """
        url = f"{self.base_url}{endpoint}"
        return self._make_request("POST", url, json=data)
        
    def put(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint (without base URL)
            data: Request body data
            
        Returns:
            API response (parsed JSON)
        """
        url = f"{self.base_url}{endpoint}"
        return self._make_request("PUT", url, json=data)
        
    def delete(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            API response (parsed JSON)
        """
        url = f"{self.base_url}{endpoint}"
        return self._make_request("DELETE", url, params=params)
        
    @RateLimiter.decorator
    @RetryHandler.decorator
    def _make_request(self, method: str, url: str, **kwargs) -> Any:
        """
        Make a request to the API with rate limiting and retry handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Full URL for the request
            **kwargs: Additional keyword arguments for requests.request
            
        Returns:
            API response (parsed JSON)
            
        Raises:
            RequestException: If the request fails after retries
        """
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
        
    def close(self) -> None:
        """Close the session."""
        self.session.close()


class AlpacaAPIClient(APIClient):
    """Alpaca API client for trading stocks and crypto."""
    
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper: bool = True,
        rate_limit: int = 5,
        retry_attempts: int = 3,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Alpaca API client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Whether to use paper trading
            rate_limit: Rate limit in requests per second
            retry_attempts: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        # Try to get API credentials from environment if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # Set base URL based on paper/live trading
        if paper:
            base_url = "https://paper-api.alpaca.markets/v2"
        else:
            base_url = "https://api.alpaca.markets/v2"
            
        # Set headers for Alpaca API
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        
        super().__init__(
            base_url=base_url,
            api_key=None,  # API key is sent in headers instead
            api_secret=None,
            rate_limit=rate_limit,
            retry_attempts=retry_attempts,
            timeout=timeout,
            verify_ssl=verify_ssl,
            headers=headers,
        )
        
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        return self.get("/account")
        
    def list_positions(self) -> List[Dict[str, Any]]:
        """
        List open positions.
        
        Returns:
            List of open positions
        """
        return self.get("/positions")
        
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position information
        """
        return self.get(f"/positions/{symbol}")
        
    def list_orders(
        self,
        status: str | None = None,
        limit: int | None = None,
        after: str | None = None,
        until: str | None = None,
        direction: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        List orders.
        
        Args:
            status: Filter by order status
            limit: Maximum number of orders to return
            after: Only return orders after this timestamp
            until: Only return orders until this timestamp
            direction: Order direction (asc or desc)
            
        Returns:
            List of orders
        """
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        if direction:
            params["direction"] = direction
            
        return self.get("/orders", params=params)
        
    def submit_order(
        self,
        symbol: str,
        qty: int | None = None,
        notional: float | None = None,
        side: str = "buy",
        type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Submit an order.
        
        Args:
            symbol: Symbol to trade
            qty: Quantity to trade (either qty or notional must be provided)
            notional: Notional value to trade (either qty or notional must be provided)
            side: Order side (buy or sell)
            type: Order type (market, limit, stop, stop_limit)
            time_in_force: Time in force (day, gtc, opg, cls, ioc, fok)
            limit_price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            client_order_id: Client order ID
            
        Returns:
            Order information
        """
        if (qty is None and notional is None) or (qty is not None and notional is not None):
            raise ValueError("Either qty or notional must be provided, but not both")
            
        order = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        
        if qty is not None:
            order["qty"] = str(qty)
        else:
            order["notional"] = str(notional)
            
        if limit_price is not None:
            order["limit_price"] = str(limit_price)
            
        if stop_price is not None:
            order["stop_price"] = str(stop_price)
            
        if client_order_id is not None:
            order["client_order_id"] = client_order_id
            
        return self.post("/orders", data=order)
        
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order information
        """
        return self.get(f"/orders/{order_id}")
        
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Success message
        """
        return self.delete(f"/orders/{order_id}")
        
    def list_assets(self, status: str | None = None, asset_class: str | None = None) -> List[Dict[str, Any]]:
        """
        List assets.
        
        Args:
            status: Filter by asset status
            asset_class: Filter by asset class
            
        Returns:
            List of assets
        """
        params = {}
        if status:
            params["status"] = status
        if asset_class:
            params["asset_class"] = asset_class
            
        return self.get("/assets", params=params)
        
    def get_asset(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset by symbol.
        
        Args:
            symbol: Symbol to get asset for
            
        Returns:
            Asset information
        """
        return self.get(f"/assets/{symbol}")
        
    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock.
        
        Returns:
            Market clock information
        """
        return self.get("/clock")
        
    def get_calendar(self, start: str | None = None, end: str | None = None) -> List[Dict[str, Any]]:
        """
        Get market calendar.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Market calendar
        """
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return self.get("/calendar", params=params)

    def get_bars(
        self,
        symbols: List[str] | str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        page_token: str | None = None,
        adjustment: str = "raw",
    ) -> Dict[str, Any]:
        """
        Get historical bars (OHLCV) for one or more symbols.
        
        Args:
            symbols: Symbol(s) to get bars for
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1H, 1D, etc.)
            start: Start date/time (RFC 3339/ISO 8601)
            end: End date/time (RFC 3339/ISO 8601)
            limit: Maximum number of bars per symbol
            page_token: Pagination token
            adjustment: Price adjustment (raw, split, dividend, all)
            
        Returns:
            Bars data
        """
        # Ensure symbols is a list for the query parameter
        if isinstance(symbols, str):
            symbols = [symbols]
            
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "adjustment": adjustment,
        }
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = limit
        if page_token:
            params["page_token"] = page_token
            
        return self.get("/bars", params=params)


class RetryHandler:
    """
    Handler for retrying operations with different backoff strategies.
    
    This class handles retry logic for failed operations, applying different backoff
    strategies such as constant, linear, or exponential backoff.
    """
    
    class BackoffStrategy(Enum):
        """Backoff strategies for retries."""
        CONSTANT = auto()
        LINEAR = auto()
        EXPONENTIAL = auto()
        
    def __init__(
        self,
        max_retries: int = 3,
        backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_exceptions: List[Type[Exception]] | None = None,
        retry_on_status_codes: List[int] | None = None,
        name: str | None = None,
    ):
        """
        Initialize the retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_strategy: Backoff strategy to use for retries
            initial_delay: Initial delay in seconds between retries
            max_delay: Maximum delay in seconds between retries
            retry_exceptions: List of exception types to retry on
            retry_on_status_codes: List of HTTP status codes to retry on
            name: Optional name for identifying this handler in logs
        """
        self.max_retries = max_retries
        self.backoff_strategy = backoff_strategy
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_exceptions = retry_exceptions or [RequestException]
        self.retry_on_status_codes = retry_on_status_codes or [429, 500, 502, 503, 504]
        self.name = name or "retry_handler"
        
    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate the delay for a retry attempt.
        
        Args:
            retry_count: Current retry attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == self.BackoffStrategy.CONSTANT:
            delay = self.initial_delay
        elif self.backoff_strategy == self.BackoffStrategy.LINEAR:
            delay = self.initial_delay * (retry_count + 1)
        elif self.backoff_strategy == self.BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (2 ** retry_count)
        else:
            delay = self.initial_delay
            
        return min(delay, self.max_delay)
        
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an operation should be retried based on the exception.
        
        Args:
            exception: Exception that occurred
            
        Returns:
            Whether to retry the operation
        """
        # Check if it's a RequestException with a response
        if isinstance(exception, RequestException) and hasattr(exception, "response") and exception.response:
            return exception.response.status_code in self.retry_on_status_codes
            
        # Check if it's a type of exception to retry
        return any(isinstance(exception, exc_type) for exc_type in self.retry_exceptions)
        
    @classmethod
    def decorator(cls, func: Callable) -> Callable:
        """
        Decorator for retrying a function with the handler's settings.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function with retry handling
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract retry count from kwargs if present
            retry_count = kwargs.pop("retry_count", 0)
            
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Check if we have a retry handler attribute
                retry_handler = getattr(self, "retry_handler", None)
                if retry_handler is None or not isinstance(retry_handler, RetryHandler):
                    # No retry handler, just re-raise
                    raise
                    
                # Check if we should retry
                if retry_count >= retry_handler.max_retries or not retry_handler.should_retry(e):
                    raise
                    
                # Calculate delay
                delay = retry_handler.calculate_delay(retry_count)
                
                # Log retry
                logger.warning(
                    f"{retry_handler.name}: Retry {retry_count + 1}/{retry_handler.max_retries} after {delay:.2f}s ({str(e)})"
                )
                
                # Sleep and retry
                time.sleep(delay)
                return wrapper(self, *args, **kwargs, retry_count=retry_count + 1)
                
        return wrapper


class RateLimiter:
    """
    Rate limiter for controlling the rate of operations.
    
    This class implements rate limiting for operations, ensuring that they don't
    exceed a specified rate. It supports different rate limiting strategies, such
    as fixed window and token bucket.
    """
    
    class RateLimitStrategy(Enum):
        """Rate limiting strategies."""
        FIXED_WINDOW = auto()
        TOKEN_BUCKET = auto()
        
    def __init__(
        self,
        requests_per_period: int = 10,
        period_seconds: float = 1.0,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        burst_size: int | None = None,
        name: str | None = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_period: Maximum number of requests allowed per period
            period_seconds: Length of the period in seconds
            strategy: Rate limiting strategy to use
            burst_size: Maximum burst size (for token bucket)
            name: Optional name for identifying this rate limiter in logs
        """
        self.requests_per_period = requests_per_period
        self.period_seconds = period_seconds
        self.strategy = strategy
        self.burst_size = burst_size or requests_per_period
        self.name = name or "rate_limiter"
        
        # Initialize based on strategy
        if strategy == self.RateLimitStrategy.FIXED_WINDOW:
            self.window_start = time.time()
            self.request_count = 0
            self.lock = threading.Lock()
        elif strategy == self.RateLimitStrategy.TOKEN_BUCKET:
            self.tokens = self.burst_size
            self.last_refill = time.time()
            self.lock = threading.Lock()
            
    def acquire(self) -> bool:
        """
        Acquire a rate limit token.
        
        Returns:
            Whether the token was acquired
        """
        with self.lock:
            if self.strategy == self.RateLimitStrategy.FIXED_WINDOW:
                current_time = time.time()
                
                # Check if we're in a new window
                if current_time - self.window_start > self.period_seconds:
                    self.window_start = current_time
                    self.request_count = 0
                    
                # Check if we can make a request
                if self.request_count < self.requests_per_period:
                    self.request_count += 1
                    return True
                else:
                    return False
                    
            elif self.strategy == self.RateLimitStrategy.TOKEN_BUCKET:
                current_time = time.time()
                
                # Refill tokens based on elapsed time
                elapsed = current_time - self.last_refill
                new_tokens = elapsed * (self.requests_per_period / self.period_seconds)
                self.tokens = min(self.burst_size, self.tokens + new_tokens)
                self.last_refill = current_time
                
                # Check if we can make a request
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                else:
                    return False
                    
    def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while True:
            with self.lock:
                if self.strategy == self.RateLimitStrategy.FIXED_WINDOW:
                    current_time = time.time()
                    
                    # Check if we're in a new window
                    if current_time - self.window_start > self.period_seconds:
                        self.window_start = current_time
                        self.request_count = 0
                        
                    # Check if we can make a request
                    if self.request_count < self.requests_per_period:
                        self.request_count += 1
                        return
                        
                elif self.strategy == self.RateLimitStrategy.TOKEN_BUCKET:
                    current_time = time.time()
                    
                    # Refill tokens based on elapsed time
                    elapsed = current_time - self.last_refill
                    new_tokens = elapsed * (self.requests_per_period / self.period_seconds)
                    self.tokens = min(self.burst_size, self.tokens + new_tokens)
                    self.last_refill = current_time
                    
                    # Check if we can make a request
                    if self.tokens >= 1:
                        self.tokens -= 1
                        return
                        
                    # Calculate time until next token
                    time_to_next_token = (1 - self.tokens) * (self.period_seconds / self.requests_per_period)
                    
            # Sleep until next token is available
            time.sleep(min(time_to_next_token, 0.1))
            
    @classmethod
    def decorator(cls, func: Callable) -> Callable:
        """
        Decorator for rate limiting a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function with rate limiting
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if we have a rate limiter attribute
            rate_limiter = getattr(self, "rate_limiter", None)
            if rate_limiter is None or not isinstance(rate_limiter, RateLimiter):
                # No rate limiter, just call the function
                return func(self, *args, **kwargs)
                
            # Wait for a token
            rate_limiter.wait_for_token()
            
            # Call the function
            return func(self, *args, **kwargs)
            
        return wrapper


class APICache:
    """
    Cache for API responses.
    
    This class implements a cache for API responses, reducing the need to make
    repeated requests for the same data.
    """
    
    @dataclass
    class CacheEntry:
        """Cache entry with value and expiration time."""
        value: Any
        expires_at: float
        
    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_size: int = 1000,
        name: str | None = None,
    ):
        """
        Initialize the API cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of entries in the cache
            name: Optional name for identifying this cache in logs
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.name = name or "api_cache"
        self.cache = {}
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found or expired
        """
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                return None
                
            # Check if expired
            if time.time() > entry.expires_at:
                del self.cache[key]
                return None
                
            return entry.value
            
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Check cache size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (simple LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                
            # Set new entry
            self.cache[key] = self.CacheEntry(
                value=value,
                expires_at=time.time() + self.ttl_seconds,
            )
            
    def invalidate(self, key: str) -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                
    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()


class ConnectionPool:
    """
    Pool of connections that can be reused.
    
    This class manages a pool of connections, allowing them to be reused instead
    of creating new ones for each operation. This reduces connection overhead and
    improves performance.
    """
    
    def __init__(
        self,
        create_connection: Callable[[], Any],
        close_connection: Callable[[Any], None],
        max_size: int = 10,
        name: str | None = None,
    ):
        """
        Initialize the connection pool.
        
        Args:
            create_connection: Function to create a new connection
            close_connection: Function to close a connection
            max_size: Maximum number of connections in the pool
            name: Optional name for identifying this pool in logs
        """
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.max_size = max_size
        self.name = name or "connection_pool"
        self.pool = queue.Queue(maxsize=max_size)
        self.active_connections = 0
        self.lock = threading.Lock()
        
    def get_connection(self) -> Any:
        """
        Get a connection from the pool, creating a new one if necessary.
        
        Returns:
            Connection
        """
        # Try to get a connection from the pool
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            pass
            
        # Create a new connection if we haven't reached max_size
        with self.lock:
            if self.active_connections < self.max_size:
                try:
                    connection = self.create_connection()
                    self.active_connections += 1
                    return connection
                except Exception as e:
                    logger.error(f"{self.name}: Error creating connection: {e}")
                    raise
                    
        # Wait for a connection to become available
        return self.pool.get()
        
    def return_connection(self, connection: Any) -> None:
        """
        Return a connection to the pool.
        
        Args:
            connection: Connection to return
        """
        # Put the connection back in the pool
        try:
            self.pool.put_nowait(connection)
        except queue.Full:
            # Pool is full, close the connection
            with self.lock:
                self.close_connection(connection)
                self.active_connections -= 1
                
    def close_all(self) -> None:
        """Close all connections in the pool."""
        # Close all connections in the pool
        with self.lock:
            while not self.pool.empty():
                try:
                    connection = self.pool.get_nowait()
                    self.close_connection(connection)
                    self.active_connections -= 1
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"{self.name}: Error closing connection: {e}")


def rate_limited(
    requests_per_period: int = 10,
    period_seconds: float = 1.0,
    strategy: RateLimiter.RateLimitStrategy = RateLimiter.RateLimitStrategy.TOKEN_BUCKET,
    burst_size: int | None = None,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator for rate limiting a function outside of a class.
    
    Args:
        requests_per_period: Maximum number of requests allowed per period
        period_seconds: Length of the period in seconds
        strategy: Rate limiting strategy to use
        burst_size: Maximum burst size (for token bucket)
        name: Optional name for identifying this rate limiter in logs
        
    Returns:
        Decorator function
    """
    # Create a rate limiter
    limiter = RateLimiter(
        requests_per_period=requests_per_period,
        period_seconds=period_seconds,
        strategy=strategy,
        burst_size=burst_size,
        name=name,
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Wait for a token
            limiter.wait_for_token()
            
            # Call the function
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator


def cached_api_call(
    ttl_seconds: float = 60.0,
    max_size: int = 1000,
    key_func: Callable[..., str] | None = None,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator for caching API calls outside of a class.
    
    Args:
        ttl_seconds: Time-to-live for cache entries in seconds
        max_size: Maximum number of entries in the cache
        key_func: Function to generate cache keys from arguments
        name: Optional name for identifying this cache in logs
        
    Returns:
        Decorator function
    """
    # Create a cache
    cache = APICache(
        ttl_seconds=ttl_seconds,
        max_size=max_size,
        name=name,
    )
    
    def default_key_func(*args, **kwargs) -> str:
        """
        Default function to generate cache keys from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
        
    key_generator = key_func or default_key_func
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = key_generator(*args, **kwargs)
            
            # Check cache
            value = cache.get(key)
            if value is not None:
                return value
                
            # Call function
            value = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, value)
            
            return value
            
        return wrapper
        
    return decorator


def create_api_client_from_env(
    base_url_env: str = "API_BASE_URL",
    api_key_env: str = "API_KEY",
    api_secret_env: str = "API_SECRET",
    **kwargs
) -> APIClient:
    """
    Create an API client from environment variables.
    
    Args:
        base_url_env: Environment variable name for base URL
        api_key_env: Environment variable name for API key
        api_secret_env: Environment variable name for API secret
        **kwargs: Additional keyword arguments for APIClient
        
    Returns:
        Initialized API client
        
    Raises:
        ValueError: If required environment variables are not set
    """
    base_url = os.environ.get(base_url_env)
    if not base_url:
        raise ValueError(f"Environment variable {base_url_env} is required")
        
    api_key = os.environ.get(api_key_env)
    api_secret = os.environ.get(api_secret_env)
    
    return APIClient(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
        **kwargs
    )


def create_alpaca_client_from_env(
    api_key_env: str = "ALPACA_API_KEY",
    api_secret_env: str = "ALPACA_API_SECRET",
    **kwargs
) -> AlpacaAPIClient:
    """
    Create an Alpaca API client from environment variables.
    
    Args:
        api_key_env: Environment variable name for API key
        api_secret_env: Environment variable name for API secret
        **kwargs: Additional keyword arguments for AlpacaAPIClient
        
    Returns:
        Initialized Alpaca API client
        
    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = os.environ.get(api_key_env)
    api_secret = os.environ.get(api_secret_env)
    
    return AlpacaAPIClient(
        api_key=api_key,
        api_secret=api_secret,
        **kwargs
    )


# Update the decorators to use the actual class methods
RateLimiter.decorator = RateLimiter.decorator
RetryHandler.decorator = RetryHandler.decorator