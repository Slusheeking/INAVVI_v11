"""
Concurrency utilities for the Autonomous Trading System.

This module provides utilities for concurrent processing, including
thread pools, process pools, and asynchronous execution.
"""

import asyncio
import concurrent.futures
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.utils.logging import get_logger

logger = get_logger("utils.concurrency.concurrency_utils")

# Type variable for generic function
T = TypeVar('T')
R = TypeVar('R')


def run_in_thread(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Decorator to run a function in a separate thread.
    
    Args:
        func: Function to run in a thread
        
    Returns:
        Decorated function that returns a Future
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> concurrent.futures.Future[T]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(func, *args, **kwargs)
    
    return wrapper


def run_in_process(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Decorator to run a function in a separate process.
    
    Args:
        func: Function to run in a process
        
    Returns:
        Decorated function that returns a Future
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> concurrent.futures.Future[T]:
        with ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(func, *args, **kwargs)
    
    return wrapper


def run_with_timeout(
    func: Callable[..., T],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    timeout: float = 10.0,
    default: Optional[T] = None,
) -> T:
    """
    Run a function with a timeout.
    
    Args:
        func: Function to run
        args: Function arguments
        kwargs: Function keyword arguments
        timeout: Timeout in seconds
        default: Default value to return if the function times out
        
    Returns:
        Function result or default value if the function times out
    """
    if kwargs is None:
        kwargs = {}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
            return default


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    timeout: Optional[float] = None,
    chunksize: int = 1,
) -> List[R]:
    """
    Apply a function to each item in a list in parallel.
    
    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        timeout: Timeout in seconds
        chunksize: Size of chunks for process-based parallelism
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_cls(max_workers=max_workers) as executor:
        if use_processes:
            # Process-based parallelism works better with map for large datasets
            return list(executor.map(func, items, chunksize=chunksize, timeout=timeout))
        else:
            # Thread-based parallelism works better with submit for flexibility
            futures = [executor.submit(func, item) for item in items]
            return [future.result(timeout=timeout) for future in concurrent.futures.as_completed(futures)]


def parallel_execute(
    functions: List[Callable[[], T]],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    timeout: Optional[float] = None,
) -> List[T]:
    """
    Execute a list of functions in parallel.
    
    Args:
        functions: List of functions to execute
        max_workers: Maximum number of workers
        use_processes: Whether to use processes instead of threads
        timeout: Timeout in seconds
        
    Returns:
        List of results
    """
    if not functions:
        return []
    
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(func) for func in functions]
        return [future.result(timeout=timeout) for future in concurrent.futures.as_completed(futures)]


def throttle(
    func: Callable[..., T],
    wait: float,
    leading: bool = True,
    trailing: bool = False,
) -> Callable[..., Optional[T]]:
    """
    Throttle a function to limit the rate at which it can be called.
    
    Args:
        func: Function to throttle
        wait: Minimum time between function calls in seconds
        leading: Whether to call the function on the leading edge
        trailing: Whether to call the function on the trailing edge
        
    Returns:
        Throttled function
    """
    last_called = [0.0]
    timer = [None]
    last_args = [None]
    last_kwargs = [None]
    result = [None]
    
    def call_func(*args, **kwargs):
        result[0] = func(*args, **kwargs)
        last_called[0] = time.time()
        timer[0] = None
        
        # Call the trailing edge if enabled
        if trailing and last_args[0] is not None:
            args_to_use = last_args[0]
            kwargs_to_use = last_kwargs[0]
            last_args[0] = None
            last_kwargs[0] = None
            call_func(*args_to_use, **kwargs_to_use)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        now = time.time()
        elapsed = now - last_called[0]
        remaining = wait - elapsed
        
        if remaining <= 0 or not last_called[0]:
            if leading:
                call_func(*args, **kwargs)
            else:
                last_args[0] = args
                last_kwargs[0] = kwargs
                timer[0] = threading.Timer(0, call_func, args=args, kwargs=kwargs)
                timer[0].start()
        elif trailing:
            # Store the latest arguments for the trailing edge
            last_args[0] = args
            last_kwargs[0] = kwargs
            
            # Cancel the existing timer
            if timer[0]:
                timer[0].cancel()
            
            # Set a new timer for the remaining time
            timer[0] = threading.Timer(remaining, call_func, args=args, kwargs=kwargs)
            timer[0].start()
        
        return result[0]
    
    return wrapper


def debounce(
    func: Callable[..., T],
    wait: float,
    leading: bool = False,
) -> Callable[..., Optional[T]]:
    """
    Debounce a function to limit the rate at which it can be called.
    
    Args:
        func: Function to debounce
        wait: Time to wait after the last call before executing the function
        leading: Whether to call the function on the leading edge
        
    Returns:
        Debounced function
    """
    timer = [None]
    last_called = [0.0]
    result = [None]
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        now = time.time()
        
        # Cancel the existing timer
        if timer[0]:
            timer[0].cancel()
            timer[0] = None
        
        # Call the function on the leading edge if enabled
        if leading and (now - last_called[0] > wait or not last_called[0]):
            last_called[0] = now
            result[0] = func(*args, **kwargs)
            return result[0]
        
        # Set a new timer for the trailing edge
        def call_func():
            last_called[0] = time.time()
            result[0] = func(*args, **kwargs)
            timer[0] = None
        
        timer[0] = threading.Timer(wait, call_func)
        timer[0].start()
        
        return result[0]
    
    return wrapper


def retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase the delay by after each retry
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = retry_delay
            
            for retry_count in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if retry_count < max_retries:
                        logger.warning(
                            f"Retry {retry_count + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to {e.__class__.__name__}: {str(e)}"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"Failed after {max_retries} retries for {func.__name__}: "
                            f"{e.__class__.__name__}: {str(e)}"
                        )
            
            # Re-raise the last exception
            raise last_exception
        
        return wrapper
    
    return decorator


def periodic(
    interval: float,
    max_executions: Optional[int] = None,
    start_immediately: bool = True,
) -> Callable[[Callable[..., None]], Callable[[], threading.Thread]]:
    """
    Decorator to run a function periodically in a separate thread.
    
    Args:
        interval: Interval between executions in seconds
        max_executions: Maximum number of executions (None for unlimited)
        start_immediately: Whether to start the function immediately
        
    Returns:
        Decorated function that returns a Thread
    """
    def decorator(func: Callable[..., None]) -> Callable[[], threading.Thread]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> threading.Thread:
            stop_event = threading.Event()
            
            def periodic_func():
                execution_count = 0
                
                # Execute immediately if requested
                if start_immediately:
                    try:
                        func(*args, **kwargs)
                        execution_count += 1
                    except Exception as e:
                        logger.error(f"Error in periodic function {func.__name__}: {e}")
                
                # Main loop
                while not stop_event.is_set():
                    # Wait for the interval
                    if stop_event.wait(interval):
                        break
                    
                    # Check if we've reached the maximum number of executions
                    if max_executions is not None and execution_count >= max_executions:
                        break
                    
                    try:
                        func(*args, **kwargs)
                        execution_count += 1
                    except Exception as e:
                        logger.error(f"Error in periodic function {func.__name__}: {e}")
            
            # Create and start the thread
            thread = threading.Thread(target=periodic_func, daemon=True)
            thread.stop = stop_event.set  # Add a stop method to the thread
            thread.start()
            
            return thread
        
        return wrapper
    
    return decorator


def rate_limited(
    calls_per_second: float,
    burst_size: int = 1,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to rate limit a function.
    
    Args:
        calls_per_second: Maximum calls per second
        burst_size: Maximum burst size
        
    Returns:
        Decorated function
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0] * burst_size
    lock = threading.Lock()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with lock:
                now = time.time()
                
                # Find the oldest call
                oldest_idx = 0
                oldest_time = last_called[0]
                for i in range(1, burst_size):
                    if last_called[i] < oldest_time:
                        oldest_idx = i
                        oldest_time = last_called[i]
                
                # Calculate time to wait
                elapsed = now - oldest_time
                wait_time = max(0, min_interval - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                
                # Update the last called time
                last_called[oldest_idx] = now
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


async def gather_with_concurrency(
    n: int,
    *tasks: asyncio.Task,
) -> List[Any]:
    """
    Run tasks with a concurrency limit.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))


def run_async(
    coroutine: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Run an async function in a synchronous context.
    
    Args:
        coroutine: Async function to run
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine(*args, **kwargs))
    finally:
        loop.close()


class ThreadSafeCounter:
    """Thread-safe counter."""
    
    def __init__(self, initial_value: int = 0):
        """
        Initialize the counter.
        
        Args:
            initial_value: Initial counter value
        """
        self.value = initial_value
        self.lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """
        Increment the counter.
        
        Args:
            amount: Amount to increment by
            
        Returns:
            New counter value
        """
        with self.lock:
            self.value += amount
            return self.value
    
    def decrement(self, amount: int = 1) -> int:
        """
        Decrement the counter.
        
        Args:
            amount: Amount to decrement by
            
        Returns:
            New counter value
        """
        with self.lock:
            self.value -= amount
            return self.value
    
    def get(self) -> int:
        """
        Get the counter value.
        
        Returns:
            Counter value
        """
        with self.lock:
            return self.value
    
    def set(self, value: int) -> None:
        """
        Set the counter value.
        
        Args:
            value: New counter value
        """
        with self.lock:
            self.value = value


class ThreadSafeDict(dict):
    """Thread-safe dictionary."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dictionary."""
        self.lock = threading.Lock()
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key):
        """Get an item from the dictionary."""
        with self.lock:
            return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        """Set an item in the dictionary."""
        with self.lock:
            super().__setitem__(key, value)
    
    def __delitem__(self, key):
        """Delete an item from the dictionary."""
        with self.lock:
            super().__delitem__(key)
    
    def get(self, key, default=None):
        """Get an item from the dictionary with a default value."""
        with self.lock:
            return super().get(key, default)
    
    def pop(self, key, default=None):
        """Remove and return an item from the dictionary."""
        with self.lock:
            return super().pop(key, default)
    
    def update(self, *args, **kwargs):
        """Update the dictionary."""
        with self.lock:
            super().update(*args, **kwargs)
    
    def clear(self):
        """Clear the dictionary."""
        with self.lock:
            super().clear()
    
    def setdefault(self, key, default=None):
        """Set a default value for a key."""
        with self.lock:
            return super().setdefault(key, default)
    
    def items(self):
        """Get a copy of the dictionary items."""
        with self.lock:
            return list(super().items())
    
    def keys(self):
        """Get a copy of the dictionary keys."""
        with self.lock:
            return list(super().keys())
    
    def values(self):
        """Get a copy of the dictionary values."""
        with self.lock:
            return list(super().values())


class ThreadSafeList(list):
    """Thread-safe list."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the list."""
        self.lock = threading.Lock()
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        """Get an item from the list."""
        with self.lock:
            return super().__getitem__(index)
    
    def __setitem__(self, index, value):
        """Set an item in the list."""
        with self.lock:
            super().__setitem__(index, value)
    
    def __delitem__(self, index):
        """Delete an item from the list."""
        with self.lock:
            super().__delitem__(index)
    
    def append(self, value):
        """Append an item to the list."""
        with self.lock:
            super().append(value)
    
    def extend(self, values):
        """Extend the list with values."""
        with self.lock:
            super().extend(values)
    
    def insert(self, index, value):
        """Insert an item into the list."""
        with self.lock:
            super().insert(index, value)
    
    def remove(self, value):
        """Remove an item from the list."""
        with self.lock:
            super().remove(value)
    
    def pop(self, index=-1):
        """Remove and return an item from the list."""
        with self.lock:
            return super().pop(index)
    
    def clear(self):
        """Clear the list."""
        with self.lock:
            super().clear()
    
    def index(self, value, *args):
        """Get the index of an item in the list."""
        with self.lock:
            return super().index(value, *args)
    
    def count(self, value):
        """Count occurrences of an item in the list."""
        with self.lock:
            return super().count(value)
    
    def sort(self, *args, **kwargs):
        """Sort the list."""
        with self.lock:
            super().sort(*args, **kwargs)
    
    def reverse(self):
        """Reverse the list."""
        with self.lock:
            super().reverse()
    
    def copy(self):
        """Get a copy of the list."""
        with self.lock:
            return super().copy()