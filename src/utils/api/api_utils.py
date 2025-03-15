"""
API utilities for the Autonomous Trading System.

This module provides utilities for working with external APIs, including
rate limiting, error handling, response parsing, and specialized support
for financial market APIs like Alpaca.
"""

import time
from datetime import datetime

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import logging utility
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger("utils.api.api_utils")

class RateLimiter:
    """
    Rate limiter for API requests.
    
    This class implements a token bucket algorithm for rate limiting,
    which allows for bursts of requests up to a maximum rate.
    """
    
    # Rate limiting strategies
    class RateLimitStrategy:
        """Rate limiting strategies."""
        TOKEN_BUCKET = "token_bucket"
        FIXED_WINDOW = "fixed_window"
        SLIDING_WINDOW = "sliding_window"
        LEAKY_BUCKET = "leaky_bucket"
    
    def __init__(
        self,
        requests_per_second: float = None,
        requests_per_period: int = None,
        period_seconds: int = 60,
        burst_size: int = 1,
        initial_tokens: Optional[int] = None,
        strategy: str = RateLimitStrategy.TOKEN_BUCKET,
        name: str = "default"
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second (deprecated, use requests_per_period)
            requests_per_period: Maximum requests per period
            period_seconds: Period length in seconds
            burst_size: Maximum burst size (number of tokens)
            initial_tokens: Initial number of tokens (default: burst_size)
            strategy: Rate limiting strategy
            name: Name for this rate limiter (for logging)
        """
        # Handle both requests_per_second and requests_per_period
        if requests_per_second is not None:
            self.requests_per_second = requests_per_second
            self.requests_per_period = int(requests_per_second * period_seconds)
        elif requests_per_period is not None:
            self.requests_per_period = requests_per_period
            self.requests_per_second = requests_per_period / period_seconds
        else:
            self.requests_per_second = 1.0
            self.requests_per_period = period_seconds
            
        self.period_seconds = period_seconds
        self.burst_size = burst_size
        self.tokens = initial_tokens if initial_tokens is not None else burst_size
        self.last_refill_time = time.time()
        self.strategy = strategy
        self.name = name
        
        # For fixed/sliding window strategies
        self.window_start_time = time.time()
        self.request_count = 0
        
        logger.debug(f"Initialized {self.name} rate limiter with {self.requests_per_period} "
                    f"requests per {period_seconds}s ({self.requests_per_second:.2f} req/sec)")
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.requests_per_second
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst_size)
            self.last_refill_time = now
    
    def acquire(self, block: bool = True) -> bool:
        """
        Acquire a token from the rate limiter.
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if a token was acquired, False otherwise
        """
        if self.strategy == self.RateLimitStrategy.TOKEN_BUCKET:
            return self._acquire_token_bucket(block)
        elif self.strategy == self.RateLimitStrategy.FIXED_WINDOW:
            return self._acquire_fixed_window(block)
        elif self.strategy == self.RateLimitStrategy.SLIDING_WINDOW:
            return self._acquire_sliding_window(block)
        elif self.strategy == self.RateLimitStrategy.LEAKY_BUCKET:
            return self._acquire_leaky_bucket(block)
        else:
            # Default to token bucket
            return self._acquire_token_bucket(block)
    
    def _acquire_token_bucket(self, block: bool = True) -> bool:
        """
        Acquire a token using the token bucket algorithm.
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if a token was acquired, False otherwise
        """
        self._refill_tokens()
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        if not block:
            return False
        
        # Calculate time to wait for next token
        wait_time = (1 - self.tokens) / self.requests_per_second
        logger.debug(f"{self.name} rate limit reached, waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        
        # Try again after waiting
        self._refill_tokens()
        self.tokens -= 1
        return True
    
    def _acquire_fixed_window(self, block: bool = True) -> bool:
        """
        Acquire a token using the fixed window algorithm.
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if a token was acquired, False otherwise
        """
        now = time.time()
        
        # Check if we need to reset the window
        if now - self.window_start_time >= self.period_seconds:
            self.window_start_time = now
            self.request_count = 0
        
        # Check if we've exceeded the rate limit
        if self.request_count >= self.requests_per_period:
            if not block:
                return False
            
            # Calculate time to wait until the next window
            wait_time = self.window_start_time + self.period_seconds - now
            logger.debug(f"{self.name} rate limit reached, waiting {wait_time:.2f} seconds for next window")
            time.sleep(wait_time)
            
            # Reset the window
            self.window_start_time = time.time()
            self.request_count = 0
        
        # Increment the request count and return
        self.request_count += 1
        return True
    
    def _acquire_sliding_window(self, block: bool = True) -> bool:
        """
        Acquire a token using the sliding window algorithm.
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if a token was acquired, False otherwise
        """
        # Simplified sliding window implementation
        # In a real implementation, we would track timestamps of all requests
        now = time.time()
        
        # Check if we need to reset the window
        if now - self.window_start_time >= self.period_seconds:
            # Slide the window
            overlap_ratio = 1.0 - ((now - self.window_start_time) / self.period_seconds)
            overlap_ratio = max(0.0, min(1.0, overlap_ratio))
            
            # Adjust request count based on overlap
            self.request_count = int(self.request_count * overlap_ratio)
            self.window_start_time = now
        
        # Check if we've exceeded the rate limit
        if self.request_count >= self.requests_per_period:
            if not block:
                return False
            
            # Calculate time to wait
            wait_time = self.period_seconds / self.requests_per_period
            logger.debug(f"{self.name} rate limit reached, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Increment the request count and return
        self.request_count += 1
        return True
    
    def _acquire_leaky_bucket(self, block: bool = True) -> bool:
        """
        Acquire a token using the leaky bucket algorithm.
        
        Args:
            block: Whether to block until a token is available
            
        Returns:
            True if a token was acquired, False otherwise
        """
        # Leaky bucket is similar to token bucket, but with a constant refill rate
        now = time.time()
        elapsed = now - self.last_refill_time
        
        # Calculate leaked tokens
        leaked_tokens = elapsed * self.requests_per_second
        
        # Update tokens
        self.tokens = min(self.burst_size, self.tokens + leaked_tokens)
        self.last_refill_time = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        if not block:
            return False
        
        # Calculate time to wait for next token
        wait_time = 1.0 / self.requests_per_second
        logger.debug(f"{self.name} rate limit reached, waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        
        # Try again after waiting
        self.tokens += wait_time * self.requests_per_second
        self.tokens -= 1
        self.last_refill_time = time.time()
        return True

def rate_limited(
    requests_per_second: float,
    burst_size: int = 1
) -> Callable:
    """
    Decorator for rate-limiting API calls.
    
    Args:
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size
        
    Returns:
        Decorated function
    """
    limiter = RateLimiter(requests_per_second, burst_size)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        return wrapper
    
    return decorator

class APIClient:
    """
    Base API client with common functionality.
    
    This class provides common functionality for API clients, including
    request handling, error handling, and response parsing.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retry_status_codes: List[int] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for API requests
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            retry_status_codes: Status codes to retry on
            headers: Additional headers to include in requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Set up retry strategy
        if retry_status_codes is None:
            retry_status_codes = [429, 500, 502, 503, 504]
            
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=retry_status_codes,
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        
        # Create session with retry adapter
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.headers = headers or {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _build_url(self, endpoint: str) -> str:
        """
        Build a full URL from an endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Full URL
        """
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle an API response.
        
        Args:
            response: Response object
            
        Returns:
            Parsed response data
            
        Raises:
            requests.HTTPError: If the response status code indicates an error
        """
        # Raise an exception for 4xx and 5xx status codes
        response.raise_for_status()
        
        # Parse JSON response
        try:
            return response.json()
        except ValueError:
            logger.warning(f"Response is not valid JSON: {response.text}")
            return {"text": response.text}
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional arguments to pass to requests.get
            
        Returns:
            Parsed response data
        """
        url = self._build_url(endpoint)
        request_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"GET {url} with params: {params}")
        response = self.session.get(
            url,
            params=params,
            headers=request_headers,
            timeout=self.timeout,
            **kwargs
        )
        
        return self._handle_response(response)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            **kwargs: Additional arguments to pass to requests.post
            
        Returns:
            Parsed response data
        """
        url = self._build_url(endpoint)
        request_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"POST {url} with data: {data or json_data}")
        response = self.session.post(
            url,
            data=data,
            json=json_data,
            headers=request_headers,
            timeout=self.timeout,
            **kwargs
        )
        
        return self._handle_response(response)
    
    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            **kwargs: Additional arguments to pass to requests.put
            
        Returns:
            Parsed response data
        """
        url = self._build_url(endpoint)
        request_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"PUT {url} with data: {data or json_data}")
        response = self.session.put(
            url,
            data=data,
            json=json_data,
            headers=request_headers,
            timeout=self.timeout,
            **kwargs
        )
        
        return self._handle_response(response)
    
    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a DELETE request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional arguments to pass to requests.delete
            
        Returns:
            Parsed response data
        """
        url = self._build_url(endpoint)
        request_headers = {**self.headers, **(headers or {})}
        
        logger.debug(f"DELETE {url} with params: {params}")
        response = self.session.delete(
            url,
            params=params,
            headers=request_headers,
            timeout=self.timeout,
            **kwargs
        )
        
        return self._handle_response(response)
    
    def close(self):
        """Close the session."""
        self.session.close()

class APICache:
    """
    Cache for API responses.
    
    This class provides a simple cache for API responses to reduce
    the number of API calls and improve performance.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the API cache.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Time-to-live for cache items in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        # Check if the item has expired
        timestamp = self.timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl:
            # Remove expired item
            self.remove(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If the cache is full, remove the oldest item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self.remove(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def remove(self, key: str) -> None:
        """
        Remove a value from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
        
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()

def cached_api_call(
    cache: APICache,
    key_prefix: str = "",
    ttl: Optional[int] = None
) -> Callable:
    """
    Decorator for caching API calls.
    
    Args:
        cache: API cache instance
        key_prefix: Prefix for cache keys
        ttl: Time-to-live for cache items in seconds (overrides cache default)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key from the function name, args, and kwargs
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                key_parts.append(str(arg))
            
            # Add kwargs to key (sorted for consistency)
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get the value from the cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value
            
            # Call the function and cache the result
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            
            # Override TTL if specified
            if ttl is not None:
                old_ttl = cache.ttl
                cache.ttl = ttl
                cache.set(cache_key, result)
                cache.ttl = old_ttl
            else:
                cache.set(cache_key, result)
            
            return result
        
        return wrapper
    
    return decorator
def create_api_client_from_env(
    api_name: str,
    base_url_env_var: Optional[str] = None,
    api_key_env_var: Optional[str] = None,
    default_base_url: Optional[str] = None
) -> APIClient:
    """
    Create an API client from environment variables.
    
    Args:
        api_name: Name of the API (for logging)
        base_url_env_var: Environment variable for base URL
        api_key_env_var: Environment variable for API key
        default_base_url: Default base URL if not found in environment
        
    Returns:
        APIClient instance
    """
    # Determine environment variable names if not provided
    if base_url_env_var is None:
        base_url_env_var = f"{api_name.upper()}_API_URL"
    
    if api_key_env_var is None:
        api_key_env_var = f"{api_name.upper()}_API_KEY"
    
    # Get values from environment
    base_url = os.environ.get(base_url_env_var, default_base_url)
    api_key = os.environ.get(api_key_env_var)
    
    if base_url is None:
        raise ValueError(
            f"Base URL not found in environment variable {base_url_env_var} "
            f"and no default provided"
        )
    
    logger.info(f"Creating API client for {api_name} with base URL {base_url}")
    return APIClient(base_url=base_url, api_key=api_key)


class AlpacaAPIClient(APIClient):
    """
    Specialized API client for Alpaca trading API.
    
    This class extends the base APIClient with Alpaca-specific functionality,
    including authentication, rate limiting, and error handling tailored for
    the Alpaca API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        data_url: Optional[str] = None,
        api_version: str = "v2",
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit_requests_per_minute: int = 200
    ):
        """
        Initialize the Alpaca API client.
        
        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY environment variable)
            api_secret: Alpaca API secret (defaults to ALPACA_API_SECRET environment variable)
            base_url: Alpaca API base URL (defaults to ALPACA_API_BASE_URL environment variable)
            data_url: Alpaca Data API URL (defaults to ALPACA_DATA_URL environment variable)
            api_version: API version
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            rate_limit_requests_per_minute: Maximum requests per minute
        """
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        base_url = base_url or os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
        self.data_url = data_url or os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # Initialize base API client
        super().__init__(
            base_url=base_url,
            api_key=self.api_key,  # Used for header construction in base class
            timeout=timeout,
            max_retries=max_retries,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Content-Type": "application/json"
            }
        )
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_period=rate_limit_requests_per_minute,
            period_seconds=60,
            name="alpaca"
        )
        
        # Set up retry handler
        self.retry_handler = RetryHandler(
            max_retries=max_retries,
            retry_on_status_codes=[429, 500, 502, 503, 504],
            name="alpaca"
        )
        
        self.api_version = api_version
        logger.info(f"Initialized Alpaca API client with base URL {base_url}")
    
    def _build_url(self, endpoint: str, is_data_api: bool = False) -> str:
        """
        Build a full URL from an endpoint.
        
        Args:
            endpoint: API endpoint
            is_data_api: Whether to use the data API URL
            
        Returns:
            Full URL
        """
        base = self.data_url if is_data_api else self.base_url
        endpoint = endpoint.lstrip('/')
        
        # Add API version if not already in the endpoint
        if self.api_version and not endpoint.startswith(f"{self.api_version}/"):
            endpoint = f"{self.api_version}/{endpoint}"
        
        return f"{base}/{endpoint}"
    
    def _retry_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]  # Get the instance
            retry_count = 0
            max_retries = self.retry_handler.max_retries
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Extract status code if it's a requests.HTTPError
                    status_code = None
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                    
                    # Check if we should retry
                    if retry_count >= max_retries or not self.retry_handler.should_retry(e, status_code):
                        raise
                    
                    # Calculate delay
                    delay = self.retry_handler.calculate_delay(retry_count)
                    
                    # Log retry attempt
                    logger.warning(
                        f"Retrying {func.__name__} after {delay:.2f}s "
                        f"(attempt {retry_count + 1}/{max_retries}) "
                        f"due to {e.__class__.__name__}: {str(e)}"
                    )
                    
                    # Wait before retrying
                    time.sleep(delay)
                    
                    # Increment retry count
                    retry_count += 1
        
        return wrapper
    
    @_retry_decorator
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        is_data_api: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a GET request to the Alpaca API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            is_data_api: Whether to use the data API URL
            **kwargs: Additional arguments to pass to requests.get
            
        Returns:
            Parsed response data
        """
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        # Build URL with appropriate base
        url = self._build_url(endpoint, is_data_api)
        
        return super().get(url, params, headers, **kwargs)
    
    @_retry_decorator
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        is_data_api: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a POST request to the Alpaca API.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            headers: Additional headers
            is_data_api: Whether to use the data API URL
            **kwargs: Additional arguments to pass to requests.post
            
        Returns:
            Parsed response data
        """
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        # Build URL with appropriate base
        url = self._build_url(endpoint, is_data_api)
        
        return super().post(url, data, json_data, headers, **kwargs)
    
    @_retry_decorator
    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        is_data_api: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a DELETE request to the Alpaca API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            is_data_api: Whether to use the data API URL
            **kwargs: Additional arguments to pass to requests.delete
            
        Returns:
            Parsed response data
        """
        # Apply rate limiting
        self.rate_limiter.acquire()
        
        # Build URL with appropriate base
        url = self._build_url(endpoint, is_data_api)
        
        return super().delete(url, params, headers, **kwargs)
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        return self.get("account")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions.
        
        Returns:
            List of positions
        """
        return self.get("positions")
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Position information
        """
        return self.get(f"positions/{symbol}")
    
    def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[Union[str, datetime]] = None,
        until: Optional[Union[str, datetime]] = None,
        direction: str = "desc",
        nested: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status: Order status filter ('open', 'closed', 'all')
            limit: Maximum number of orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp
            direction: Sort direction ('asc' or 'desc')
            nested: Whether to include nested orders
            
        Returns:
            List of orders
        """
        params = {
            "limit": limit,
            "direction": direction,
            "nested": nested
        }
        
        if status:
            params["status"] = status
        
        if after:
            if isinstance(after, datetime):
                after = after.isoformat()
            params["after"] = after
        
        if until:
            if isinstance(until, datetime):
                until = until.isoformat()
            params["until"] = until
        
        return self.get("orders", params=params)
    
    def submit_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str = "buy",
        type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
        order_class: Optional[str] = None,
        take_profit: Optional[Dict[str, Any]] = None,
        stop_loss: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit an order.
        
        Args:
            symbol: Ticker symbol
            qty: Order quantity
            notional: Order notional value (used instead of qty for fractional shares)
            side: Order side ('buy' or 'sell')
            type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop_price: Stop price (required for 'stop' and 'stop_limit' orders)
            client_order_id: Client order ID
            extended_hours: Whether to allow trading during extended hours
            order_class: Order class ('simple', 'bracket', 'oco', 'oto')
            take_profit: Take profit parameters
            stop_loss: Stop loss parameters
            
        Returns:
            Order information
        """
        # Build order payload
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
            "extended_hours": extended_hours
        }
        
        # Set quantity or notional (one is required)
        if qty is not None:
            order_data["qty"] = str(qty)
        elif notional is not None:
            order_data["notional"] = str(notional)
        else:
            raise ValueError("Either qty or notional must be provided")
        
        # Add optional parameters
        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)
        
        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)
        
        if client_order_id is not None:
            order_data["client_order_id"] = client_order_id
        
        if order_class is not None:
            order_data["order_class"] = order_class
        
        if take_profit is not None:
            order_data["take_profit"] = take_profit
        
        if stop_loss is not None:
            order_data["stop_loss"] = stop_loss
        
        return self.post("orders", json_data=order_data)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Response data
        """
        return self.delete(f"orders/{order_id}")
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock information.
        
        Returns:
            Market clock information
        """
        return self.get("clock")
    
    def get_calendar(
        self,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get market calendar.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of market calendar days
        """
        params = {}
        
        if start:
            if isinstance(start, datetime):
                start = start.date().isoformat()
            params["start"] = start
        
        if end:
            if isinstance(end, datetime):
                end = end.date().isoformat()
            params["end"] = end
        
        return self.get("calendar", params=params)
    
    def get_assets(
        self,
        status: str = "active",
        asset_class: str = "us_equity"
    ) -> List[Dict[str, Any]]:
        """
        Get assets.
        
        Args:
            status: Asset status ('active', 'inactive')
            asset_class: Asset class ('us_equity', 'crypto')
            
        Returns:
            List of assets
        """
        params = {
            "status": status,
            "asset_class": asset_class
        }
        
        return self.get("assets", params=params)
    
    def get_bars(
        self,
        symbols: Union[str, List[str]],
        timeframe: str = "1D",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        limit: int = 1000,
        adjustment: str = "raw"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get bars (OHLCV) data.
        
        Args:
            symbols: Symbol or list of symbols
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1H', '1D', '1W', '1M')
            start: Start time
            end: End time
            limit: Maximum number of bars to return
            adjustment: Adjustment mode ('raw', 'split', 'dividend', 'all')
            
        Returns:
            Dictionary mapping symbols to lists of bars
        """
        # Convert symbols to comma-separated string if it's a list
        if isinstance(symbols, list):
            symbols = ",".join(symbols)
        
        params = {
            "symbols": symbols,
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": adjustment
        }
        
        if start:
            if isinstance(start, datetime):
                start = start.isoformat()
            params["start"] = start
        
        if end:
            if isinstance(end, datetime):
                end = end.isoformat()
            params["end"] = end
        
        return self.get("bars", params=params, is_data_api=True)


def create_alpaca_client_from_env() -> AlpacaAPIClient:
    """
    Create an Alpaca API client from environment variables.
    
    Returns:
        AlpacaAPIClient instance
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
    data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API key and secret are required")
    
    logger.info(f"Creating Alpaca API client with base URL {base_url}")
    return AlpacaAPIClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        data_url=data_url
    )



class RetryHandler:
    """
    Retry handler for API requests.
    
    This class provides a way to retry failed API requests with configurable
    backoff strategies and retry conditions.
    """
    
    class BackoffStrategy:
        """Backoff strategies for retries."""
        CONSTANT = "constant"
        LINEAR = "linear"
        EXPONENTIAL = "exponential"
        FIBONACCI = "fibonacci"
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_strategy: str = BackoffStrategy.EXPONENTIAL,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_exceptions: List[type] = None,
        retry_on_status_codes: List[int] = None,
        name: str = "default"
    ):
        """
        Initialize the retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_strategy: Backoff strategy for retries
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            retry_exceptions: List of exception types to retry on
            retry_on_status_codes: List of HTTP status codes to retry on
            name: Name for this retry handler (for logging)
        """
        self.max_retries = max_retries
        self.backoff_strategy = backoff_strategy
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_exceptions = retry_exceptions or []
        self.retry_on_status_codes = retry_on_status_codes or [429, 500, 502, 503, 504]
        self.name = name
        
        logger.debug(f"Initialized {self.name} retry handler with max_retries={max_retries}, "
                    f"backoff_strategy={backoff_strategy}")
    
    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate the delay for a retry attempt.
        
        Args:
            retry_count: Current retry attempt (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == self.BackoffStrategy.CONSTANT:
            delay = self.initial_delay
        elif self.backoff_strategy == self.BackoffStrategy.LINEAR:
            delay = self.initial_delay * (retry_count + 1)
        elif self.backoff_strategy == self.BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (2 ** retry_count)
        elif self.backoff_strategy == self.BackoffStrategy.FIBONACCI:
            # Calculate Fibonacci number (1, 1, 2, 3, 5, 8, ...)
            a, b = 1, 1
            for _ in range(retry_count):
                a, b = b, a + b
            delay = self.initial_delay * a
        else:
            # Default to exponential
            delay = self.initial_delay * (2 ** retry_count)
        
        # Apply jitter (±10%) to avoid thundering herd problem
        import random
        jitter = 0.1 * delay * (2 * random.random() - 1)
        delay += jitter
        
        # Cap at max_delay
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception, status_code: Optional[int] = None) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            exception: Exception that occurred
            status_code: HTTP status code (if available)
            
        Returns:
            True if the request should be retried, False otherwise
        """
        # Check if the exception is in the retry_exceptions list
        for exception_type in self.retry_exceptions:
            if isinstance(exception, exception_type):
                return True
        
        # Check if the status code is in the retry_on_status_codes list
        if status_code is not None and status_code in self.retry_on_status_codes:
            return True
        
        return False
    
    @staticmethod
    def decorator(func):
        """
        Decorator for retrying functions.
        
        This decorator will retry the decorated function if it raises an exception
        that matches the retry conditions.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get retry handler from self
            retry_handler = getattr(self, 'retry_handler', None)
            if retry_handler is None:
                # If no retry handler is found, create a default one
                retry_handler = RetryHandler()
            
            retry_count = 0
            while True:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    # Extract status code if it's a requests.HTTPError
                    status_code = None
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                    
                    # Check if we should retry
                    if retry_count >= retry_handler.max_retries or not retry_handler.should_retry(e, status_code):
                        raise
                    
                    # Calculate delay
                    delay = retry_handler.calculate_delay(retry_count)
                    
                    # Log retry attempt
                    logger.warning(
                        f"{retry_handler.name} retry handler: Retrying {func.__name__} "
                        f"after {delay:.2f}s (attempt {retry_count + 1}/{retry_handler.max_retries}) "
                        f"due to {e.__class__.__name__}: {str(e)}"
                    )
                    
                    # Wait before retrying
                    time.sleep(delay)
                    
                    # Increment retry count
                    retry_count += 1
        
        return wrapper


class ConnectionPool:
    """
    Connection pool for managing connections to external services.
    
    This class provides a way to manage a pool of connections to external services,
    with support for connection creation, validation, and cleanup.
    """
    
    def __init__(
        self,
        create_connection: Callable[[], Any],
        close_connection: Callable[[Any], None],
        validate_connection: Optional[Callable[[Any], bool]] = None,
        min_connections: int = 1,
        max_connections: int = 10,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        name: str = "default"
    ):
        """
        Initialize the connection pool.
        
        Args:
            create_connection: Function to create a new connection
            close_connection: Function to close a connection
            validate_connection: Function to validate a connection (returns True if valid)
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections to allow
            connection_timeout: Timeout for connection creation in seconds
            idle_timeout: Timeout for idle connections in seconds
            name: Name for this connection pool (for logging)
        """
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.validate_connection = validate_connection or (lambda _: True)
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.name = name
        
        # Connection pool
        self.pool = []
        self.in_use = set()
        self.last_used = {}
        
        # Lock for thread safety
        import threading
        self.lock = threading.RLock()
        
        # Initialize pool with min_connections
        self._initialize_pool()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.debug(f"Initialized {self.name} connection pool with min={min_connections}, "
                    f"max={max_connections} connections")
    
    def _initialize_pool(self):
        """Initialize the pool with min_connections connections."""
        with self.lock:
            for _ in range(self.min_connections):
                try:
                    conn = self.create_connection()
                    self.pool.append(conn)
                    self.last_used[conn] = time.time()
                except Exception as e:
                    logger.error(f"Error creating connection for {self.name} pool: {e}")
    
    def _start_cleanup_thread(self):
        """Start a thread to clean up idle connections."""
        import threading
        
        def cleanup():
            while True:
                try:
                    self._cleanup_idle_connections()
                except Exception as e:
                    logger.error(f"Error in {self.name} pool cleanup thread: {e}")
                
                # Sleep for idle_timeout / 2 seconds
                time.sleep(self.idle_timeout / 2)
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        with self.lock:
            now = time.time()
            
            # Keep track of connections to remove
            to_remove = []
            
            # Check each connection in the pool
            for conn in self.pool:
                # Skip connections that are in use
                if conn in self.in_use:
                    continue
                
                # Check if the connection has been idle for too long
                last_used = self.last_used.get(conn, 0)
                if now - last_used > self.idle_timeout:
                    # Only remove connections if we have more than min_connections
                    if len(self.pool) - len(to_remove) > self.min_connections:
                        to_remove.append(conn)
            
            # Remove idle connections
            for conn in to_remove:
                try:
                    self.close_connection(conn)
                    self.pool.remove(conn)
                    if conn in self.last_used:
                        del self.last_used[conn]
                    logger.debug(f"Removed idle connection from {self.name} pool")
                except Exception as e:
                    logger.error(f"Error closing connection in {self.name} pool: {e}")
    
    def get_connection(self) -> Any:
        """
        Get a connection from the pool.
        
        Returns:
            Connection object
            
        Raises:
            TimeoutError: If a connection could not be obtained within the timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.connection_timeout:
            with self.lock:
                # Try to find an available connection in the pool
                for conn in self.pool:
                    if conn not in self.in_use:
                        # Validate the connection
                        try:
                            if self.validate_connection(conn):
                                self.in_use.add(conn)
                                self.last_used[conn] = time.time()
                                return conn
                            else:
                                # Connection is invalid, remove it
                                try:
                                    self.close_connection(conn)
                                except Exception as e:
                                    logger.error(f"Error closing invalid connection in {self.name} pool: {e}")
                                self.pool.remove(conn)
                                if conn in self.last_used:
                                    del self.last_used[conn]
                        except Exception as e:
                            logger.error(f"Error validating connection in {self.name} pool: {e}")
                            # Consider the connection invalid
                            try:
                                self.close_connection(conn)
                            except Exception as e2:
                                logger.error(f"Error closing invalid connection in {self.name} pool: {e2}")
                            self.pool.remove(conn)
                            if conn in self.last_used:
                                del self.last_used[conn]
                
                # If we have fewer than max_connections, create a new one
                if len(self.pool) < self.max_connections:
                    try:
                        conn = self.create_connection()
                        self.pool.append(conn)
                        self.in_use.add(conn)
                        self.last_used[conn] = time.time()
                        return conn
                    except Exception as e:
                        logger.error(f"Error creating connection for {self.name} pool: {e}")
            
            # Wait a bit before trying again
            time.sleep(0.1)
        
        # If we get here, we timed out
        raise TimeoutError(f"Timed out waiting for a connection from {self.name} pool")
    
    def release_connection(self, conn: Any):
        """
        Release a connection back to the pool.
        
        Args:
            conn: Connection object
        """
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.last_used[conn] = time.time()
            else:
                logger.warning(f"Attempted to release a connection not in use in {self.name} pool")
    
    def close_all(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                try:
                    self.close_connection(conn)
                except Exception as e:
                    logger.error(f"Error closing connection in {self.name} pool: {e}")
            
            self.pool = []
            self.in_use = set()
            self.last_used = {}
            
            logger.info(f"Closed all connections in {self.name} pool")