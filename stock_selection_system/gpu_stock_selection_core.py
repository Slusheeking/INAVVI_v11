#!/usr/bin/env python3
"""
GPU-Optimized Stock Selection System Core
This module contains the core functionality of the GPU-accelerated stock selection system.
"""

from gpu_system.gpu_utils import log_memory_usage, configure_gpu
import ml_system.market_data_helpers as mdh
from gpu_system.gh200_accelerator import GH200Accelerator, optimize_for_gh200
from concurrent.futures import ThreadPoolExecutor
from asyncio import Lock
from typing import Dict, List, Set, Optional, Any
import cupy as cp
import pandas as pd
import numpy as np
import pytz
import datetime
import logging
import json
import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_stock_selection')


class GPUStockSelectionSystem:
    """GPU-Optimized Stock Selection System for NVIDIA GH200"""

    def __init__(self, redis_client: Any, polygon_api_client: Any,
                 polygon_websocket_client: Optional[Any] = None,
                 unusual_whales_client: Optional[Any] = None):
        self.redis = redis_client
        self.polygon_api = polygon_api_client
        self.polygon_ws = polygon_websocket_client
        self.unusual_whales = unusual_whales_client

        # Apply GH200-specific optimizations
        logger.info("Applying GH200-specific optimizations...")
        optimize_for_gh200()

        # Initialize GH200 Accelerator
        logger.info("Initializing GH200 Accelerator...")
        self.gh200_accelerator = GH200Accelerator()

        # Check if GH200 GPU is available
        self.gpu_available = self.gh200_accelerator.has_tensorflow_gpu or self.gh200_accelerator.has_cupy_gpu

        if not self.gpu_available:
            # This should not happen as configure_gpu will raise an exception if no GPU is found
            logger.error(
                "NVIDIA GH200 GPU acceleration not available. This system requires GH200 GPU acceleration.")
            logger.error(f"TensorFlow GPU: {self.gh200_accelerator.has_tensorflow_gpu}, "
                         f"CuPy GPU: {self.gh200_accelerator.has_cupy_gpu}")
            raise RuntimeError(
                "GPU acceleration not available. This system requires GPU acceleration.")

        # Assign logger as instance attribute for access by other components
        self.logger = logger

        # Performance optimization with thread pools
        logger.info(
            f"Initializing thread pool with {min(os.cpu_count(), 20)} workers")
        self.executor = ThreadPoolExecutor(max_workers=20)

        # Market data cache
        self.cache = {
            'market_data': {},
            'options_data': {},
            'technical_data': {},
            'last_refresh': {}
        }

        # Configuration
        self.config = {
            'universe_size': 2000,
            'watchlist_size': 100,
            'focused_list_size': 30,
            'min_price': 5.0,        # Lowered minimum price for more opportunities
            'max_price': 200.0,
            'min_volume': 500000,
            'min_relative_volume': 1.5,
            'min_atr_percent': 1.0,
            'refresh_interval': 900,    # 15 minutes
            'cache_expiry': 300,        # 5 minutes
            'weights': {
                'volume': 0.30,
                'volatility': 0.25,
                'momentum': 0.25,
                'options': 0.20
            },
            'batch_size': 1024,         # Batch size for GPU processing
            # Limit workers to avoid overloading
            'max_workers': min(os.cpu_count(), 8),

            # Day trading specific settings
            'day_trading': {
                'enabled': True,
                'max_total_position': 5000,  # $5000 total position limit
                'max_positions': 5,          # Maximum number of concurrent positions
                'target_profit_percent': 1.0,  # Target 1% profit per trade
                'stop_loss_percent': 0.5,    # 0.5% stop loss
                'no_overnight_positions': True,
                'min_liquidity_score': 70    # Minimum liquidity score
            }
        }

        # Internal state
        self.full_universe: Set[str] = set()
        self.active_watchlist: Set[str] = set()
        self.focused_list: Set[str] = set()
        self.running: bool = False
        self.tasks: Dict[str, asyncio.Task] = {}
        self.day_trading_candidates: Set[str] = set()

        # Locks for thread safety
        self._universe_lock = Lock()
        self._watchlist_lock = Lock()
        self._focused_lock = Lock()

        # Shared memory for inter-process communication on GPU
        self.shared_data = {}

        # Log GPU device information
        logger.info(f"Using GPU device: {self.gh200_accelerator.device_name}")

        # Log initial memory usage
        log_memory_usage("initialization")

        logger.info(
            "GPU-Optimized Stock Selection System initialized with day trading capabilities")

    async def start(self):
        """Start the stock selection system"""
        if self.running:
            logger.warning("Stock selection system already running")
            return

        self.running = True
        logger.info(
            "Starting GPU-optimized stock selection system with day trading capabilities")

        # Start websocket client if available
        if self.polygon_ws:
            self.polygon_ws.start()
            # Subscribe to relevant channels for watchlist stocks
            mdh.subscribe_to_watchlist_channels(
                self.polygon_ws, self.redis, logger)

        # Initialize universe
        await self.build_initial_universe()

        # Start periodic tasks
        self.tasks['universe_refresh'] = asyncio.create_task(
            self._universe_refresh_task())
        self.tasks['watchlist_update'] = asyncio.create_task(
            self._watchlist_update_task())
        self.tasks['focus_update'] = asyncio.create_task(
            self._focus_update_task())
        self.tasks['memory_monitor'] = asyncio.create_task(
            self._memory_monitor_task())

        # Add day trading specific tasks
        if self.config['day_trading']['enabled']:
            # Initial update of day trading candidates
            await self.update_day_trading_candidates()

            # Start periodic update task
            self.tasks['day_trading_update'] = asyncio.create_task(
                self._day_trading_update_task())

            # Start market close monitor to ensure no overnight positions
            self.tasks['market_close_monitor'] = asyncio.create_task(
                self._market_close_monitor())

            logger.info(
                "Day trading tasks started - $5,000 position limit active")

        logger.info("GPU-optimized stock selection system started")

    async def stop(self):
        """Stop the stock selection system"""
        if not self.running:
            return

        logger.info("Stopping GPU-optimized stock selection system")
        self.running = False

        # Close all day trading positions first if enabled
        if self.config['day_trading']['enabled']:
            logger.info("Closing all day trading positions before shutdown")
            await self.close_all_day_trading_positions()

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling task: {name}")
                task.cancel()

        # Stop websocket client if available
        if self.polygon_ws:
            self.polygon_ws.stop()

        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        # Clean up GPU resources
        if self.gpu_available:
            try:
                # Use GH200 accelerator to clear GPU memory
                logger.info("Clearing GPU memory using GH200 accelerator...")
                self.gh200_accelerator.clear_gpu_memory()
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Error clearing CuPy memory pool: {e}")

        logger.info("GPU-optimized stock selection system stopped")

    async def _memory_monitor_task(self):
        """Task to monitor memory usage"""
        logger.info("Starting memory monitor task")

        while self.running:
            try:
                # Log memory usage
                log_memory_usage("periodic_check")

                # Clean up GPU memory if usage is high
                if self.gpu_available:
                    try:
                        # Use GH200 accelerator to monitor and clear GPU memory
                        self.gh200_accelerator.clear_gpu_memory()

                        mem_info = cp.cuda.runtime.memGetInfo()
                        free, total = mem_info[0], mem_info[1]
                        used_percent = (total - free) / total * 100

                        if used_percent > 80:
                            logger.warning(
                                f"GPU memory usage high ({used_percent:.2f}%), cleaning up")
                            cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        logger.error(f"Error cleaning up GPU memory: {e}")

                # Wait 5 minutes
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                logger.info("Memory monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in memory monitor task: {str(e)}")
                await asyncio.sleep(60)

    # Import core functionality
    from .stock_selection_core import StockSelectionCore

    # Delegate methods to StockSelectionCore
    build_initial_universe = StockSelectionCore.build_initial_universe
    _apply_tradable_filters_gpu = StockSelectionCore._apply_tradable_filters_gpu
    _check_batch_eligibility_gpu = StockSelectionCore._check_batch_eligibility_gpu
    _check_ticker_eligibility = StockSelectionCore._check_ticker_eligibility
    _universe_refresh_task = StockSelectionCore._universe_refresh_task

    update_watchlist = StockSelectionCore.update_watchlist
    update_focused_list = StockSelectionCore.update_focused_list
    _watchlist_update_task = StockSelectionCore._watchlist_update_task
    _focus_update_task = StockSelectionCore._focus_update_task

    _calculate_batch_scores_gpu = StockSelectionCore._calculate_batch_scores_gpu
    _calculate_ticker_score = StockSelectionCore._calculate_ticker_score
    _calculate_volume_factor = StockSelectionCore._calculate_volume_factor
    _calculate_volatility_factor = StockSelectionCore._calculate_volatility_factor
    _calculate_momentum_factor = StockSelectionCore._calculate_momentum_factor
    _calculate_options_factor = StockSelectionCore._calculate_options_factor

    from .day_trading_system import (
        calculate_intraday_profit_potential,
        update_day_trading_candidates,
        _calculate_day_trading_score,
        _day_trading_update_task,
        close_all_day_trading_positions,
        _market_close_monitor
    )
