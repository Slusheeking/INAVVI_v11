#!/usr/bin/env python3
"""
WebSocket-Enhanced GPU-Optimized Stock Selection System
This implementation enhances the stock selection system with real-time data from Polygon WebSocket API
for more responsive and accurate trading decisions.
"""

from trading_system.opportunity_detector import OpportunityDetector
from stock_selection_system.websocket_core import WebSocketCore
from ml_system.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_adx,
    calculate_obv
)
from trading_system.execution_system import ExecutionSystem
from stock_selection_system.stock_selection_core import StockSelectionCore
from gpu_system.gpu_utils import log_memory_usage
from stock_selection_system.gpu_stock_selection_core import GPUStockSelectionSystem
from gpu_system.gh200_accelerator import GH200Accelerator, optimize_for_gh200
from stock_selection_system.gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
from stock_selection_system.gpu_optimized_polygon_websocket_client import (
    GPUPolygonWebSocketClient,
    subscribe_to_trades,
    subscribe_to_quotes,
    subscribe_to_minute_aggs,
    subscribe_to_second_aggs
)
from stock_selection_system.gpu_optimized_polygon_api_client import GPUPolygonAPIClient
import logging
import json
import asyncio
import datetime
import pytz
import numpy as np
import pandas as pd
import cupy as cp
import tensorflow as tf
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import time
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
from asyncio import Lock
from collections import deque

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules

# Import our modularized components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('websocket_stock_selection')


class WebSocketEnhancedStockSelection(GPUStockSelectionSystem):
    """WebSocket-Enhanced GPU-Optimized Stock Selection System with real-time data processing"""

    def __init__(self, redis_client: Any, polygon_api_client: Any,
                 polygon_websocket_client: Any,
                 unusual_whales_client: Optional[Any] = None):
        """Initialize the WebSocket-enhanced stock selection system"""
        # Call parent class initializer
        super().__init__(redis_client, polygon_api_client,
                         polygon_websocket_client, unusual_whales_client)

        # Real-time data buffers
        self.real_time_data = {
            'trades': {},
            'quotes': {},
            'minute_aggs': {},
            'second_aggs': {}
        }

        # Real-time metrics
        self.real_time_metrics = {
            'volume_spikes': set(),
            'price_jumps': set(),
            'spread_changes': set(),
            'momentum_shifts': set()
        }

        # WebSocket subscription management
        self.active_subscriptions = {
            'trades': set(),
            'quotes': set(),
            'minute_aggs': set(),
            'second_aggs': set()
        }

        # Real-time data processing locks
        self.trade_lock = Lock()
        self.quote_lock = Lock()
        self.agg_lock = Lock()

        # Real-time data windows (for calculating metrics)
        self.data_windows = {
            'trades': {},
            'quotes': {},
            'minute_aggs': {},
            'second_aggs': {}
        }

        # Window sizes for different metrics
        self.window_sizes = {
            'volume_spike': 20,  # Last 20 trades
            'price_jump': 10,    # Last 10 trades
            'spread_change': 20,  # Last 20 quotes
            'momentum_shift': 5  # Last 5 minute candles
        }

        # Thresholds for real-time alerts
        self.thresholds = {
            'volume_spike': 3.0,   # 3x normal volume
            'price_jump': 0.5,     # 0.5% in a single trade
            'spread_change': 50.0,  # 50% change in spread
            'momentum_shift': 1.0  # 1% change in a short period
        }

        # Add WebSocket message handlers
        if self.polygon_ws:
            self.polygon_ws.add_message_handler(
                'T', self._handle_trade_message)
            self.polygon_ws.add_message_handler(
                'Q', self._handle_quote_message)
            self.polygon_ws.add_message_handler(
                'AM', self._handle_minute_agg_message)
            self.polygon_ws.add_message_handler(
                'A', self._handle_second_agg_message)
            logger.info("WebSocket message handlers registered")

        # Additional tasks for real-time processing
        self.real_time_tasks = {}

        logger.info("WebSocket-Enhanced Stock Selection System initialized")

    async def start(self):
        """Start the WebSocket-enhanced stock selection system"""
        # Call parent class start method
        await super().start()

        # Start additional real-time processing tasks
        self.real_time_tasks['metric_calculator'] = asyncio.create_task(
            self._real_time_metric_calculator())
        self.real_time_tasks['subscription_manager'] = asyncio.create_task(
            self._subscription_manager())
        self.real_time_tasks['real_time_selector'] = asyncio.create_task(
            self._real_time_stock_selector())

        logger.info("WebSocket-Enhanced Stock Selection System started")

    async def stop(self):
        """Stop the WebSocket-enhanced stock selection system"""
        logger.info("Stopping WebSocket-Enhanced Stock Selection System")

        # Cancel real-time tasks
        for name, task in self.real_time_tasks.items():
            if not task.done():
                logger.info(f"Cancelling real-time task: {name}")
                task.cancel()

        # Call parent class stop method
        await super().stop()

        logger.info("WebSocket-Enhanced Stock Selection System stopped")

    # WebSocket Message Handlers - delegate to the WebSocketHandlers class
    async def _handle_trade_message(self, message):
        """Handle trade messages from WebSocket"""
        await WebSocketCore.handle_trade_message(self, message)

    async def _handle_quote_message(self, message):
        """Handle quote messages from WebSocket"""
        await WebSocketCore.handle_quote_message(self, message)

    async def _handle_minute_agg_message(self, message):
        """Handle minute aggregate messages from WebSocket"""
        await WebSocketCore.handle_minute_agg_message(self, message)

    async def _handle_second_agg_message(self, message):
        """Handle second aggregate messages from WebSocket"""
        await WebSocketCore.handle_second_agg_message(self, message)

    # Real-time metric calculations - delegate to the WebSocketCore class
    async def _check_volume_spike(self, ticker):
        """Check for volume spikes in real-time trade data"""
        await WebSocketCore.check_volume_spike(self, ticker)

    async def _check_price_jump(self, ticker, current_price):
        """Check for significant price jumps in real-time trade data"""
        await WebSocketCore.check_price_jump(self, ticker, current_price)

    async def _check_spread_change(self, ticker):
        """Check for significant spread changes in real-time quote data"""
        await WebSocketCore.check_spread_change(self, ticker)

    async def _check_momentum_shift(self, ticker):
        """Check for momentum shifts in real-time minute aggregate data"""
        await WebSocketCore.check_momentum_shift(self, ticker)

    async def _calculate_real_time_metrics(self, ticker):
        """Calculate real-time metrics for a ticker"""
        await WebSocketCore.calculate_real_time_metrics(self, ticker)

    # Real-time position management - delegate to the OpportunityDetector class
    async def _update_day_trading_position(self, ticker, current_price):
        """Update day trading position based on real-time price data"""
        await OpportunityDetector.update_day_trading_position(self, ticker, current_price)

    def _send_entry_signal(self, ticker, shares, entry_price, stop_price, target_price):
        """Send entry signal to execution system via Redis"""
        return OpportunityDetector.send_entry_signal(self, ticker, shares, entry_price, stop_price, target_price)

    # Background tasks - delegate to the WebSocketCore class
    async def _real_time_metric_calculator(self):
        """Task to periodically calculate real-time metrics"""
        await WebSocketCore.real_time_metric_calculator(self)

    async def _subscription_manager(self):
        """Task to manage WebSocket subscriptions based on active watchlist"""
        await WebSocketCore.subscription_manager(self)

    async def _real_time_stock_selector(self):
        """Task to select stocks based on real-time data"""
        await WebSocketCore.real_time_stock_selector(self)

    # Trading opportunity detection - delegate to the OpportunityDetector class
    async def _check_real_time_opportunities(self, ranked_tickers):
        """Check for new day trading opportunities based on real-time data"""
        await OpportunityDetector.check_real_time_opportunities(self, ranked_tickers)


# Example usage
if __name__ == "__main__":
    import redis
    import asyncio
    import os

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        # Create clients
        # Load environment variables from .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        env_values = {}
        if os.path.exists(env_path):
            env_values = dotenv_values(env_path)
        else:
            # Try to load from the main .env file
            env_values = dotenv_values(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), '.env'))

        # Get Redis connection details and API keys
        redis_host = env_values.get('REDIS_HOST')
        redis_port = int(env_values.get('REDIS_PORT', 6380))
        redis_db = int(env_values.get('REDIS_DB', '0'))
        redis_password = env_values.get('REDIS_PASSWORD')
        polygon_api_key = env_values.get('POLYGON_API_KEY')
        unusual_whales_api_key = env_values.get('UNUSUAL_WHALES_API_KEY')

        # Validate required environment variables
        missing_vars = []
        if not redis_host:
            missing_vars.append('REDIS_HOST')
        if not redis_port:
            missing_vars.append('REDIS_PORT')
        if not polygon_api_key:
            missing_vars.append('POLYGON_API_KEY')
        if not unusual_whales_api_key:
            missing_vars.append('UNUSUAL_WHALES_API_KEY')

        if missing_vars:
            logger.error(
                f"Missing required variables in .env file: {', '.join(missing_vars)}")
            logger.error(
                "Please ensure all required variables are set in the .env file")
            return

        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db
        )

        # Test Redis connection
        try:
            logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
            redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            logger.warning("Using in-memory cache instead")

        # Configure logging level based on environment
        log_level = env_values.get('LOG_LEVEL', 'INFO')
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
            logger.info(f"Log level set to {log_level}")

        # Initialize Polygon API client
        polygon_api_client = GPUPolygonAPIClient(
            api_key=polygon_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Initialize Polygon WebSocket client
        polygon_websocket_client = GPUPolygonWebSocketClient(
            api_key=polygon_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Initialize Unusual Whales client
        unusual_whales_client = GPUUnusualWhalesClient(
            api_key=unusual_whales_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Create WebSocket-enhanced stock selection system
        system = WebSocketEnhancedStockSelection(
            redis_client, polygon_api_client, polygon_websocket_client, unusual_whales_client)

        # Verify GPU is being used
        if not system.gpu_available:
            logger.error(
                "GPU is not available. This system requires either TensorFlow or CuPy GPU acceleration.")
            logger.error(f"TensorFlow GPU: {system.gh200_accelerator.has_tensorflow_gpu}, "
                         f"CuPy GPU: {system.gh200_accelerator.has_cupy_gpu}")
            logger.error(
                "Please ensure you are running in the GPU container with NVIDIA GH200 support. At least one GPU framework must detect the GPU.")
            return

        # Enable day trading
        system.config['day_trading']['enabled'] = True

        # Start system
        await system.start()

        try:
            # Run for a while
            await asyncio.sleep(3600)  # 1 hour
        finally:
            # Stop system
            await system.stop()

            # Close clients
            await polygon_api_client.close()
            if unusual_whales_client:
                await unusual_whales_client.close()

    # Run the main function
    asyncio.run(main())
