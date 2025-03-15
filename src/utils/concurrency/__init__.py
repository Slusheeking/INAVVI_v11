"""
Concurrency utilities for the trading system.

This module provides utilities for concurrent processing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def parallel_map(func: Callable[[T], R], items: List[T], max_workers: int = 4) -> List[R]:
    """
    Apply a function to each item in a list in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of worker threads
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    # Adjust max_workers based on number of items
    effective_workers = min(max_workers, len(items))
    
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(func, item): item for item in items}
            
            # Process results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    # Append None for failed items to maintain order
                    results.append(None)
    
    except Exception as e:
        logger.error(f"Error in parallel_map: {e}")
    
    return results

def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: int = 4,
    timeout: float = None,
    show_progress: bool = False
) -> List[R]:
    """
    Process items in parallel with more control options.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of worker threads
        timeout: Maximum time to wait for all tasks to complete (in seconds)
        show_progress: Whether to show progress information
        
    Returns:
        List of results in the same order as the input items
    """
    if not items:
        return []
    
    # Adjust max_workers based on number of items
    effective_workers = min(max_workers, len(items))
    
    # Initialize results with None values to maintain order
    results = [None] * len(items)
    
    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
            
            # Process results as they complete
            completed = 0
            total = len(items)
            
            for future in as_completed(futures, timeout=timeout):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Error processing item at index {idx}: {e}")
                
                completed += 1
                if show_progress and completed % max(1, total // 10) == 0:
                    logger.info(f"Progress: {completed}/{total} ({completed/total:.1%})")
    
    except Exception as e:
        logger.error(f"Error in parallel_process: {e}")
    
    return results

class ParallelTaskManager:
    """
    Manager for parallel task execution with more advanced features.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> bool:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            True if task was submitted, False otherwise
        """
        if task_id in self.futures:
            self.logger.warning(f"Task {task_id} already exists")
            return False
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            self.futures[task_id] = future
            return True
        except Exception as e:
            self.logger.error(f"Error submitting task {task_id}: {e}")
            return False
    
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """
        Get the result of a task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for the result (in seconds)
            
        Returns:
            Task result or None if task failed or doesn't exist
        """
        if task_id in self.results:
            return self.results[task_id]
        
        if task_id not in self.futures:
            self.logger.warning(f"Task {task_id} not found")
            return None
        
        try:
            future = self.futures[task_id]
            result = future.result(timeout=timeout)
            self.results[task_id] = result
            return result
        except Exception as e:
            self.logger.error(f"Error getting result for task {task_id}: {e}")
            return None
    
    def is_done(self, task_id: str) -> bool:
        """
        Check if a task is done.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task is done, False otherwise
        """
        if task_id not in self.futures:
            return False
        
        return self.futures[task_id].done()
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.futures:
            return False
        
        return self.futures[task_id].cancel()
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)