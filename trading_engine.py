#!/usr/bin/env python3
"""
Unified Trading Engine
======================

This module consolidates all trading execution components into a unified system:

1. Trading System: Overall system coordination and lifecycle management
2. Execution System: Order placement, position management, and risk control
3. Opportunity Detector: Real-time trading opportunity identification
4. Peak Detection: Price pattern recognition for entry/exit signals
5. Stock Selection: Algorithms for selecting tradable securities
6. Day Trading: High-frequency intraday trading capabilities

The engine is optimized for NVIDIA GH200 Grace Hopper Superchips with:
- GPU-accelerated signal processing
- High-performance order execution
- Real-time pattern recognition
- Integrated risk management
"""

import os
import time
import json
import logging
import argparse
import threading
import redis
import signal
import asyncio
import queue
import datetime
import pytz
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from asyncio import Lock
from typing import Dict, List, Tuple, Any, Optional, Union

# Import GPU utilities
from gpu_utils import (
    initialize_gpu,
    detect_gpus,
    select_best_gpu,
    clear_gpu_memory,
    optimize_tensorflow_model,
    gpu_array,
    to_numpy,
    gpu_apply,
    gpu_matrix_multiply,
    gpu_rolling_mean,
    gpu_batch_process,
    GPUMemoryManager
)

# Import TensorFlow with fallback
try:
    import tensorflow as tf
    TF_AVAILABLE = True

    # Check for TensorRT support
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        TENSORRT_AVAILABLE = True
    except ImportError:
        TENSORRT_AVAILABLE = False
        trt = None
except ImportError:
    TF_AVAILABLE = False
    TENSORRT_AVAILABLE = False
    tf = None
    trt = None
    print("TensorFlow not available. Some GPU functionality will be limited.")

# Import CuPy with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    print("CuPy not available. Some GPU functionality will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log')
    ]
)
logger = logging.getLogger('trading_engine')

# Constants for peak detection
MIN_PEAK_HEIGHT = 0.01  # Minimum height for a peak/trough as percentage
MIN_PEAK_DISTANCE = 5   # Minimum distance between peaks in data points
MIN_PEAK_PROMINENCE = 0.005  # Minimum prominence for peak detection


class TradingEngine:
    """
    Unified Trading Engine for the INAVVI trading system.

    This class integrates functionality from multiple subsystems:
    - Trading System: Overall control and lifecycle management
    - Execution System: Order management and position tracking
    - Opportunity Detector: Market opportunity identification
    - Peak Detection: Pattern recognition for signal generation
    - Stock Selection: Security filtering and scoring
    - Day Trading: Intraday trading capabilities

    The engine is optimized for high-performance execution on NVIDIA GH200 hardware.
    """

    def __init__(self, redis_client=None, data_pipeline=None, ml_engine=None,
                 continual_learning=None, monitoring=None, config_path=None):
        """
        Initialize the unified trading engine.

        Args:
            redis_client: Redis client for data caching and messaging
            data_pipeline: Data pipeline for market data access
            ml_engine: ML engine for predictive analytics
            continual_learning: Continual learning system for model adaptation
            monitoring: Monitoring system for system health checks
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize GPU acceleration
        self.use_gpu = self.config['system'].get('use_gpu', True)
        self.gpu_initialized = False
        self.gpu_memory_manager = None

        if self.use_gpu:
            self._initialize_gpu()

        # Store components
        self.redis = redis_client
        self.data_pipeline = data_pipeline
        self.ml_engine = ml_engine
        self.continual_learning = continual_learning
        self.monitoring = monitoring

        # Initialize Redis if not provided
        if not self.redis and 'redis' in self.config:
            # Ensure password is properly set
            password = self.config['redis']['password'] if 'password' in self.config['redis'] else None
            if not password and 'REDIS_PASSWORD' in os.environ:
                password = os.environ['REDIS_PASSWORD']

            self.redis = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                password=password,
                username=self.config['redis'].get('username', 'default'),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            logger.info(
                f"Initialized Redis connection to {self.config['redis']['host']}:{self.config['redis']['port']}")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Initialize API clients
        self.polygon_client = self._initialize_polygon_client()
        self.unusual_whales_client = self._initialize_unusual_whales_client()
        self.alpaca_client = self._initialize_alpaca_client()

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 5))

        # Processing queue for trading signals
        self.signal_queue = queue.Queue(maxsize=100)

        # Internal state
        self.running = False
        self.tasks = {}
        self.threads = []
        self.active_positions = {}
        self.pending_orders = {}
        self.day_trading_candidates = set()
        self.peak_cache = {}

        # Daily stats
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_exposure': 0.0,
            'start_time': time.time(),
            'last_update': time.time()
        }

        # Locks for thread safety
        self._universe_lock = Lock()
        self._watchlist_lock = Lock()
        self._focused_lock = Lock()

        # Real-time data storage
        self.real_time_data = {
            'trades': {},
            'quotes': {},
            'minute_aggs': {}
        }

        self.real_time_metrics = {
            'volume_spikes': set(),
            'price_jumps': set(),
            'momentum_shifts': set()
        }

        # TensorFlow models cache
        self.tf_models = {}

        # Initialize GPU memory manager if GPU is available
        if self.gpu_initialized:
            self.gpu_memory_manager = GPUMemoryManager(
                cleanup_interval=300,  # Clean up every 5 minutes
                memory_threshold=0.8   # Clean up when GPU memory usage exceeds 80%
            )
            self.gpu_memory_manager.start()
            logger.info("GPU memory manager started")

        logger.info(
            "Trading Engine initialized with GPU acceleration" if self.gpu_initialized else "Trading Engine initialized (GPU not available)")

    def _initialize_gpu(self):
        """Initialize GPU for trading engine operations"""
        try:
            # Detect available GPUs
            gpus = detect_gpus()
            if not gpus or all(gpu['device_type'] == 'CPU' for gpu in gpus):
                logger.warning(
                    "No GPU devices detected, running in CPU-only mode")
                self.use_gpu = False
                return False

            # Log detected GPUs
            for gpu in gpus:
                if gpu['device_type'] == 'GPU':
                    logger.info(
                        f"Detected GPU: {gpu['name']} (index: {gpu['index']})")

            # Initialize GPU
            success = initialize_gpu()
            if success:
                self.gpu_initialized = True
                logger.info("GPU initialization successful")

                # Log GPU capabilities
                if TF_AVAILABLE:
                    logger.info("TensorFlow is available for GPU acceleration")
                    if TENSORRT_AVAILABLE:
                        logger.info(
                            "TensorRT is available for model optimization")
                if CUPY_AVAILABLE:
                    logger.info(
                        "CuPy is available for GPU-accelerated array operations")

                return True
            else:
                logger.warning(
                    "GPU initialization failed, running in CPU-only mode")
                self.use_gpu = False
                return False

        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.use_gpu = False
            return False

    def _load_config(self, config_path):
        """Load configuration from file"""
        default_config = {
            'redis': {
                'host': 'localhost',
                'port': 6380,
                'db': 0,
                'username': 'default',
                'password': 'trading_system_2025'
            },
            'api_keys': {
                'polygon': os.environ.get('POLYGON_API_KEY', ''),
                'unusual_whales': os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
                'alpaca': {
                    'api_key': os.environ.get('ALPACA_API_KEY', ''),
                    'api_secret': os.environ.get('ALPACA_API_SECRET', ''),
                    'base_url': os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                }
            },
            'system': {
                'max_exposure': 5000.0,
                'market_hours_only': True,
                'data_dir': './data',
                'models_dir': './models',
                'use_gpu': os.environ.get('USE_GPU', 'true').lower() == 'true',
                'continual_learning': {
                    'enabled': True,
                    'daily_update_time': '23:30',
                    'full_retrain_time': '00:30'
                }
            },
            'execution': {
                'max_positions': 5,
                'max_loss_per_trade': 0.005,  # 0.5% of account
                'take_profit_default': 0.03,   # 3% target
                'stop_loss_default': 0.01,     # 1% stop
                'position_timeout': 14400,     # 4 hours max hold time
                'market_hours_only': True,
                'default_order_type': 'limit',
                'limit_price_offset': 0.001,   # 0.1% from current price
                'enable_trailing_stops': True,
                'trailing_stop_percent': 0.005,  # 0.5% trailing stop
                'slippage_tolerance': 0.002    # 0.2% max slippage
            },
            'stock_selection': {
                'universe_size': 2000,
                'watchlist_size': 100,
                'focused_list_size': 30,
                'min_price': 3.0,
                'max_price': 100.0,
                'min_volume': 500000,
                'min_relative_volume': 1.5,
                'min_atr_percent': 1.0,
                'refresh_interval': 900,
                'cache_expiry': 300,
                'weights': {
                    'volume': 0.30,
                    'volatility': 0.25,
                    'momentum': 0.25,
                    'options': 0.20
                },
                'batch_size': 1024
            },
            'day_trading': {
                'enabled': True,
                'max_total_position': 5000,
                'max_positions': 5,
                'max_position_percent': 0.25,  # Max 25% of total in one position
                'target_profit_percent': 1.0,
                'stop_loss_percent': 0.5,
                'no_overnight_positions': True,
                'min_liquidity_score': 70
            },
            'max_workers': min(os.cpu_count(), 8) if os.cpu_count() else 5,
            'batch_size': 1024
        }

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)

            # Merge configs
            self._merge_configs(default_config, file_config)

        # Ensure directories exist
        os.makedirs(default_config['system']['data_dir'], exist_ok=True)
        os.makedirs(default_config['system']['models_dir'], exist_ok=True)

        logger.info("Configuration loaded")
        return default_config

    def _merge_configs(self, base, override):
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _initialize_polygon_client(self):
        """Initialize Polygon client"""
        try:
            # Import client from api_clients module
            from api_clients import PolygonRESTClient

            client = PolygonRESTClient(
                api_key=self.config['api_keys']['polygon'],
                use_gpu=self.config['system']['use_gpu']
            )

            logger.info(
                f"Initialized Polygon client with GPU acceleration: {self.config['system']['use_gpu']}")
            return client

        except Exception as e:
            logger.error(
                f"Error initializing Polygon client: {str(e)}", exc_info=True)
            return None

    def _initialize_unusual_whales_client(self):
        """Initialize Unusual Whales client"""
        try:
            from api_clients import UnusualWhalesClient

            client = UnusualWhalesClient(
                api_key=self.config['api_keys']['unusual_whales'],
                use_gpu=self.config['system']['use_gpu']
            )

            logger.info(
                f"Initialized Unusual Whales client with GPU acceleration: {self.config['system']['use_gpu']}")
            return client

        except Exception as e:
            logger.error(
                f"Error initializing Unusual Whales client: {str(e)}", exc_info=True)
            return None

    def _initialize_alpaca_client(self):
        """Initialize Alpaca client"""
        try:
            import alpaca_trade_api as tradeapi

            alpaca = tradeapi.REST(
                key_id=self.config['api_keys']['alpaca']['api_key'],
                secret_key=self.config['api_keys']['alpaca']['api_secret'],
                base_url=self.config['api_keys']['alpaca']['base_url']
            )

            # Test connection
            account = alpaca.get_account()
            logger.info(
                f"Connected to Alpaca - Account ID: {account.id}, Status: {account.status}")

            return alpaca

        except Exception as e:
            logger.error(
                f"Error initializing Alpaca client: {str(e)}", exc_info=True)
            return None

    def _handle_signal(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def start(self):
        """Start the trading engine"""
        if self.running:
            logger.warning("Trading engine is already running")
            return

        self.running = True
        logger.info("Starting trading engine...")

        # Initialize state
        self._initialize_state()

        # Start worker threads
        self.threads.append(threading.Thread(
            target=self._signal_listener_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._execution_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._position_monitor_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._order_status_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._market_data_worker, daemon=True))

        for thread in self.threads:
            thread.start()

        # Start async tasks
        loop = asyncio.get_event_loop()
        self.tasks['peak_detection'] = asyncio.ensure_future(
            self._run_peak_detection())
        self.tasks['opportunity_detection'] = asyncio.ensure_future(
            self._run_opportunity_detection())
        self.tasks['day_trading'] = asyncio.ensure_future(
            self._run_day_trading())
        self.tasks['market_close_monitor'] = asyncio.ensure_future(
            self._market_close_monitor())

        logger.info("Trading engine started")

    def stop(self):
        """Stop the trading engine"""
        if not self.running:
            logger.warning("Trading engine not running")
            return

        logger.info("Stopping trading engine")
        self.running = False

        # Cancel async tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled {task_name} task")

        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)

        # Close positions if configured to do so
        if self.config['day_trading'].get('no_overnight_positions', True):
            # Use a new event loop for the close operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.close_all_day_trading_positions())
            finally:
                loop.close()

        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        logger.info("Trading engine stopped")

    def status(self):
        """Get the current status of the trading engine"""
        try:
            # Get account info
            account = self._get_account_info()

            # Collect status information
            status_info = {
                "running": self.running,
                "active_positions": len(self.active_positions),
                "pending_orders": len(self.pending_orders),
                "day_trading_candidates": len(self.day_trading_candidates),
                "daily_stats": self.daily_stats,
                "account": account if account else {},
                "system_info": {
                    "threads_active": len([t for t in self.threads if t.is_alive()]),
                    "tasks_active": len([t for t in self.tasks.values() if not t.done()]),
                    "uptime_seconds": time.time() - self.daily_stats.get('start_time', time.time())
                }
            }

            return status_info
        except Exception as e:
            logger.error(f"Error getting trading engine status: {str(e)}")
            return {
                "running": self.running,
                "error": str(e)
            }

    def _initialize_state(self):
        """Initialize system state"""
        try:
            logger.info("Initializing trading engine state")

            # Clear Redis state if Redis is available
            if self.redis:
                try:
                    self.redis.delete("execution:daily_stats")
                except redis.RedisError as e:
                    logger.warning(f"Error clearing Redis state: {str(e)}")

            # Get current positions from Alpaca if client is available
            if self.alpaca_client:
                try:
                    positions = self.alpaca_client.list_positions()

                    if positions:
                        logger.warning(
                            f"Found {len(positions)} existing positions")

                        # If configured to close positions on startup
                        if self.config['day_trading'].get('no_overnight_positions', True):
                            logger.info("Closing all existing positions")
                            self.alpaca_client.close_all_positions()
                            # Wait for positions to close
                            time.sleep(5.0)
                except Exception as e:
                    logger.error(
                        f"Error getting positions from Alpaca: {str(e)}")
                    positions = []
            else:
                logger.warning(
                    "Alpaca client not available, skipping position check")
                positions = []

            # Reset daily stats
            self._reset_daily_stats()

            # Reset active positions and pending orders
            self.active_positions = {}
            self.pending_orders = {}

            # Store in Redis if available
            if self.redis:
                try:
                    self.redis.delete("positions:active")
                    self.redis.delete("orders:pending")
                except redis.RedisError as e:
                    logger.warning(
                        f"Error clearing Redis positions/orders: {str(e)}")

            logger.info("Trading engine state initialized")

        except Exception as e:
            logger.error(
                f"Error initializing trading engine state: {str(e)}", exc_info=True)

    def _reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_exposure': 0.0,
            'start_time': time.time(),
            'last_update': time.time()
        }

        # Store in Redis if available
        if self.redis:
            try:
                self.redis.hmset("execution:daily_stats", self.daily_stats)
            except redis.RedisError as e:
                logger.warning(f"Error storing daily stats in Redis: {str(e)}")

    #
    # Worker threads for execution system
    #

    def _signal_listener_worker(self):
        """Worker thread to listen for new trading signals"""
        logger.info("Starting signal listener worker")

        # Check if Redis is available
        if not self.redis:
            logger.error(
                "Cannot start signal listener: Redis connection not available")
            return

        # Get Redis connection parameters
        host = self.config['redis']['host']
        port = self.config['redis']['port']
        db = self.config['redis']['db']
        password = self.config['redis']['password'] if 'password' in self.config['redis'] else None

        if not password and 'REDIS_PASSWORD' in os.environ:
            password = os.environ['REDIS_PASSWORD']

        # Track connection state
        pubsub_redis = None
        pubsub = None
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        reconnect_delay = 5  # Initial delay in seconds

        while self.running:
            try:
                # Create a dedicated Redis connection for PubSub if needed
                if pubsub_redis is None or pubsub is None:
                    pubsub_redis = redis.Redis(
                        host=host,
                        port=port,
                        db=db,
                        password=password,
                        username=self.config['redis'].get(
                            'username', 'default'),
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True
                    )

                    pubsub = pubsub_redis.pubsub()
                    pubsub.subscribe("execution:new_signal")
                    logger.info(
                        f"Subscribed to execution:new_signal channel on Redis {host}:{port}")

                    # Reset reconnect attempts on successful connection
                    reconnect_attempts = 0
                    reconnect_delay = 5

                # Get new message with timeout
                message = pubsub.get_message(timeout=1.0)

                if message and message['type'] == 'message':
                    # Extract data
                    data = message['data']

                    if isinstance(data, bytes):
                        data = data.decode('utf-8')

                    # Parse and validate signal
                    try:
                        signal = json.loads(data)
                        if not self._validate_signal(signal):
                            logger.warning(
                                f"Received invalid signal: {signal}")
                            continue

                        # Try to add to signal queue with retries
                        retries = 3
                        while retries > 0:
                            try:
                                self.signal_queue.put(
                                    signal, block=True, timeout=1.0)
                                logger.info(
                                    f"Received valid signal for {signal['ticker']}")
                                break
                            except queue.Full:
                                retries -= 1
                                if retries == 0:
                                    logger.error(
                                        f"Signal queue full after retries, dropping signal for {signal['ticker']}")
                                else:
                                    logger.warning(
                                        f"Signal queue full, retrying... ({retries} attempts left)")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in signal: {data}")
                    except Exception as e:
                        logger.error(f"Error processing signal: {str(e)}")

                # Small sleep to prevent CPU spinning when no messages
                if not message:
                    time.sleep(0.01)

            except redis.exceptions.ConnectionError as e:
                logger.error(
                    f"Redis connection error in signal listener: {str(e)}")

                # Clean up existing connection
                try:
                    if pubsub:
                        pubsub.close()
                    if pubsub_redis:
                        pubsub_redis.close()
                except Exception:
                    pass

                pubsub = None
                pubsub_redis = None

                # Implement exponential backoff for reconnection
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    logger.error(
                        f"Maximum reconnection attempts ({max_reconnect_attempts}) reached")
                    time.sleep(60)  # Wait longer before trying again
                    reconnect_attempts = 0
                else:
                    # Exponential backoff with cap
                    reconnect_delay = min(reconnect_delay * 2, 60)
                    logger.warning(
                        f"Will attempt to reconnect in {reconnect_delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                    time.sleep(reconnect_delay)

            except Exception as e:
                logger.error(
                    f"Error in signal listener: {str(e)}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on persistent errors

    def _execution_worker(self):
        """Worker thread for signal execution"""
        logger.info("Starting execution worker")

        while self.running:
            signal = None
            try:
                # Try to get signal from the queue without blocking
                try:
                    # Short timeout instead of non-blocking
                    signal = self.signal_queue.get(timeout=0.1)
                except queue.Empty:
                    # Just continue if no signals - this is expected behavior
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    continue

                # Process the signal
                if not self._check_market_status(signal):
                    logger.info(
                        f"Market closed, skipping signal for {signal['ticker']}")
                    self.signal_queue.task_done()
                    continue

                if not self._check_position_limits(signal):
                    logger.info(
                        f"Position limits reached, skipping signal for {signal['ticker']}")
                    self.signal_queue.task_done()
                    continue

                try:
                    # Execute signal
                    self._execute_signal(signal)
                finally:
                    # Always mark task as done, even if execution failed
                    self.signal_queue.task_done()

            except Exception as e:
                logger.error(
                    f"Error in execution worker: {str(e)}", exc_info=True)
                # If we got a signal but failed to process it, make sure to mark it as done
                if signal is not None:
                    try:
                        self.signal_queue.task_done()
                    except ValueError:
                        # task_done() called more times than there were items
                        pass
                time.sleep(1.0)

    def _position_monitor_worker(self):
        """Worker thread for position monitoring"""
        logger.info("Starting position monitor worker")

        while self.running:
            try:
                # Get active positions
                positions = self.active_positions.copy()

                # Check each position
                for ticker, position in positions.items():
                    # Skip if no longer active
                    if ticker not in self.active_positions:
                        continue

                    # Get current price
                    current_price = self._get_current_price(ticker)
                    if not current_price:
                        continue

                    # Update position stats
                    self._update_position_stats(ticker, current_price)

                    # Check for exits
                    if self._check_exit_conditions(ticker, current_price):
                        # Exit position
                        self._exit_position(ticker, "signal", current_price)

                # Update daily stats
                self._update_daily_stats()

                # Sleep for a bit
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in position monitor: {str(e)}")
                time.sleep(1.0)

    def _order_status_worker(self):
        """Worker thread for order status monitoring"""
        logger.info("Starting order status worker")

        while self.running:
            try:
                # Get pending orders
                orders = self.pending_orders.copy()

                # Check each order
                for order_id, order in orders.items():
                    # Skip if no longer pending
                    if order_id not in self.pending_orders:
                        continue

                    # Check order status
                    status = self._check_order_status(order_id)

                    if status == "filled":
                        # Handle filled order
                        self._handle_filled_order(order_id, order)
                    elif status == "canceled" or status == "expired":
                        # Handle canceled order
                        self._handle_canceled_order(order_id, order)
                    elif status == "rejected":
                        # Handle rejected order
                        self._handle_rejected_order(order_id, order)

                # Sleep for a bit
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in order status worker: {str(e)}")
                time.sleep(1.0)

    def _market_data_worker(self):
        """Worker thread for market data monitoring"""
        logger.info("Starting market data worker")

        while self.running:
            try:
                # Check market hours
                is_market_open = self._is_market_open()

                # Store in Redis
                self.redis.set("market:is_open",
                               "1" if is_market_open else "0")

                # If market is about to close, exit all positions
                if not is_market_open and self.active_positions and self._is_near_market_close():
                    logger.info("Market closing soon, exiting all positions")
                    self._exit_all_positions("market_close")

                # Sleep for a bit (check every minute)
                time.sleep(60.0)

            except Exception as e:
                logger.error(f"Error in market data worker: {str(e)}")
                time.sleep(60.0)

    #
    # Async tasks
    #

    async def _run_peak_detection(self):
        """Run peak detection monitors"""
        logger.info("Starting peak detection")

        try:
            # Get list of monitored tickers
            monitored_tickers = await self._get_monitored_tickers()

            # Start monitoring tasks
            monitoring_tasks = []
            for ticker in monitored_tickers:
                task = asyncio.create_task(self._monitor_ticker_peaks(ticker))
                monitoring_tasks.append(task)

            logger.info(
                f"Monitoring peaks for {len(monitored_tickers)} tickers")

            # Keep running until cancelled
            while self.running:
                await asyncio.sleep(10)

        except asyncio.CancelledError:
            logger.info("Peak detection task cancelled")
            # Cancel all monitoring tasks
            for task in monitoring_tasks:
                task.cancel()
            # Wait for tasks to complete
            if monitoring_tasks:
                await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in peak detection: {e}")

    async def _run_opportunity_detection(self):
        """Run opportunity detection"""
        logger.info("Starting opportunity detection")

        try:
            while self.running:
                # Get current watchlist
                watchlist_data = self.redis.zrevrange(
                    "watchlist:focused", 0, -1, withscores=True)

                if watchlist_data:
                    # Convert to list of ticker-score tuples
                    ranked_tickers = [(item[0].decode('utf-8') if isinstance(item[0], bytes)
                                      else item[0], item[1]) for item in watchlist_data]

                    # Check for new opportunities
                    await self._check_real_time_opportunities(ranked_tickers)

                # Wait before next check
                await asyncio.sleep(15)

        except asyncio.CancelledError:
            logger.info("Opportunity detection task cancelled")
        except Exception as e:
            logger.error(f"Error in opportunity detection: {e}")

    async def _run_day_trading(self):
        """Run day trading system"""
        logger.info("Starting day trading system")

        try:
            while self.running:
                # Get current time (Eastern)
                now = datetime.datetime.now(pytz.timezone('US/Eastern'))

                # Check if market is open
                market_open = now.replace(
                    hour=9, minute=30, second=0, microsecond=0)
                market_close = now.replace(
                    hour=16, minute=0, second=0, microsecond=0)

                if market_open <= now < market_close and now.weekday() < 5:  # Weekdays only
                    # Update frequency varies with time of day
                    if now < market_open.replace(hour=10):  # First 30 minutes
                        update_interval = 300  # 5 minutes
                    # Last 30 minutes
                    elif now > market_close.replace(hour=15, minute=30):
                        update_interval = 300  # 5 minutes
                    else:
                        update_interval = 900  # 15 minutes

                    # Update day trading candidates
                    await self._update_day_trading_candidates()

                    # Wait for next update
                    await asyncio.sleep(update_interval)
                else:
                    # Outside market hours, check less frequently
                    await asyncio.sleep(1800)  # 30 minutes

        except asyncio.CancelledError:
            logger.info("Day trading task cancelled")
        except Exception as e:
            logger.error(f"Error in day trading system: {e}")

    async def _market_close_monitor(self):
        """Monitor for market close to exit positions"""
        logger.info("Starting market close monitor")

        try:
            while self.running:
                # Check if market is closing soon
                is_near_close = self._is_near_market_close()
                is_market_open = self._is_market_open()

                # If market is about to close and we have day trading positions, close them
                if is_near_close and is_market_open and self.config['day_trading'].get('no_overnight_positions', True):
                    logger.info(
                        "Market closing soon, closing all day trading positions")
                    await self.close_all_day_trading_positions()

                # Check every minute near close, otherwise every 5 minutes
                if datetime.datetime.now(pytz.timezone('US/Eastern')).hour >= 15:
                    await asyncio.sleep(60)  # Check every minute after 3 PM
                else:
                    await asyncio.sleep(300)  # Check every 5 minutes

        except asyncio.CancelledError:
            logger.info("Market close monitor task cancelled")
        except Exception as e:
            logger.error(f"Error in market close monitor: {e}")

    #
    # Execution system methods
    #

    def _validate_signal(self, signal):
        """Validate trading signal"""
        required_fields = ['ticker', 'direction', 'signal_score', 'confidence']
        return all(field in signal for field in required_fields)

    def _check_market_status(self, signal):
        """Check if market is open for trading"""
        if not self.config['execution']['market_hours_only']:
            return True

        return self._is_market_open()

    def _is_market_open(self):
        """Check if market is currently open"""
        try:
            # Get Alpaca clock
            clock = self.alpaca_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")

            # Fallback to time-based check
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            is_weekday = now.weekday() < 5
            is_market_hours = 9 <= now.hour < 16 or (
                now.hour == 16 and now.minute == 0)

            return is_weekday and is_market_hours

    def _is_near_market_close(self):
        """Check if market is about to close"""
        try:
            # Get Alpaca clock
            clock = self.alpaca_client.get_clock()
            if not clock.is_open:
                return False

            # Check if within 5 minutes of close
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            close_time = clock.next_close.astimezone(
                pytz.timezone('US/Eastern'))

            time_to_close = (close_time - now).total_seconds()
            return 0 < time_to_close <= 300  # Within 5 minutes

        except Exception as e:
            logger.error(f"Error checking market close: {str(e)}")

            # Fallback to time-based check
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            is_near_close = now.hour == 15 and now.minute >= 55

            return is_near_close

    def _check_position_limits(self, signal):
        """Check if we can take new positions"""
        # Check number of positions
        if len(self.active_positions) >= self.config['execution']['max_positions']:
            return False

        # Check if already in this position
        if signal['ticker'] in self.active_positions:
            return False

        # Check total exposure
        account = self._get_account_info()
        if not account:
            return False

        current_exposure = sum(pos['current_value']
                               for pos in self.active_positions.values())
        max_exposure = min(
            self.config['system']['max_exposure'], float(account['equity']))

        if current_exposure >= max_exposure:
            return False

        return True

    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            logger.info(f"Executing signal for {signal['ticker']}")

            # Get current price
            ticker = signal['ticker']
            current_price = self._get_current_price(ticker)

            if not current_price:
                logger.error(f"Failed to get current price for {ticker}")
                return

            # Calculate position size
            position_size = self._calculate_position_size(
                signal, current_price)

            if position_size <= 0:
                logger.warning(f"Invalid position size for {ticker}")
                return

            # Calculate order parameters
            params = self._calculate_order_parameters(
                signal, current_price, position_size)

            # Submit order
            order_id = self._submit_order(params)

            if not order_id:
                logger.error(f"Failed to submit order for {ticker}")
                return

            # Track pending order
            self.pending_orders[order_id] = {
                'ticker': ticker,
                'direction': signal['direction'],
                'quantity': position_size,
                'limit_price': params['limit_price'],
                'submitted_at': time.time(),
                'signal': signal
            }

            # Store in Redis
            self.redis.hset("orders:pending", order_id,
                            json.dumps(self.pending_orders[order_id]))

            logger.info(
                f"Submitted order {order_id} for {ticker}: {position_size} shares at {params['limit_price']}")

        except Exception as e:
            logger.error(
                f"Error executing signal for {signal['ticker']}: {str(e)}")

    def _calculate_position_size(self, signal, current_price):
        """Calculate position size based on risk parameters"""
        try:
            # Get account info
            account = self._get_account_info()
            if not account:
                return 0

            # Calculate max position value
            max_exposure = min(
                self.config['system']['max_exposure'], float(account['equity']))
            current_exposure = sum(pos['current_value']
                                   for pos in self.active_positions.values())
            available_capital = max_exposure - current_exposure

            # Limit per position (25% of available capital or specified maximum)
            max_position_value = min(
                available_capital * 0.25,
                self.config['day_trading'].get('max_total_position', 5000) *
                self.config['day_trading'].get('max_position_percent', 0.25)
            )

            # Check if signal has specific position size
            if 'position_size' in signal and signal['position_size'] > 0:
                # Use specified position size
                position_size = signal['position_size']
                position_value = position_size * current_price

                # Cap at max position value
                if position_value > max_position_value:
                    position_size = int(max_position_value / current_price)
            else:
                # Calculate based on available capital
                position_size = int(max_position_value / current_price)

            # Ensure minimum position size
            if position_size < 1:
                return 0

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def _calculate_order_parameters(self, signal, current_price, position_size):
        """Calculate order parameters based on signal and price"""
        try:
            # Base parameters
            params = {
                'ticker': signal['ticker'],
                'quantity': position_size,
                'side': 'buy' if signal['direction'] == 'long' else 'sell',
                'order_type': self.config['execution']['default_order_type'],
                'time_in_force': 'day'
            }

            # Calculate limit price with offset
            if params['order_type'] == 'limit':
                offset = self.config['execution']['limit_price_offset']
                if params['side'] == 'buy':
                    # Buy slightly above current price to ensure execution
                    limit_price = current_price * (1 + offset)
                else:
                    # Sell slightly below current price to ensure execution
                    limit_price = current_price * (1 - offset)

                params['limit_price'] = round(limit_price, 2)

            # Set take profit and stop loss if specified in signal
            if 'price_target' in signal and signal['price_target'] > 0:
                params['take_profit'] = signal['price_target']
            else:
                # Use default take profit
                take_profit_pct = self.config['execution']['take_profit_default']
                if params['side'] == 'buy':
                    params['take_profit'] = current_price * \
                        (1 + take_profit_pct)
                else:
                    params['take_profit'] = current_price * \
                        (1 - take_profit_pct)

            if 'stop_loss' in signal and signal['stop_loss'] > 0:
                params['stop_loss'] = signal['stop_loss']
            else:
                # Use default stop loss
                stop_loss_pct = self.config['execution']['stop_loss_default']
                if params['side'] == 'buy':
                    params['stop_loss'] = current_price * (1 - stop_loss_pct)
                else:
                    params['stop_loss'] = current_price * (1 + stop_loss_pct)

            # Add trailing stop if enabled
            if self.config['execution']['enable_trailing_stops']:
                trailing_stop_pct = self.config['execution']['trailing_stop_percent']
                params['trailing_stop'] = trailing_stop_pct

            return params

        except Exception as e:
            logger.error(f"Error calculating order parameters: {str(e)}")
            return {}

    def _submit_order(self, params):
        """Submit order to broker"""
        try:
            # Prepare order parameters
            ticker = params['ticker']
            qty = params['quantity']
            side = params['side']
            order_type = params['order_type']
            time_in_force = params['time_in_force']

            if order_type == 'limit':
                # Submit limit order
                limit_price = params['limit_price']
                order = self.alpaca_client.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price
                )
            else:
                # Submit market order
                order = self.alpaca_client.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force
                )

            logger.info(f"Order submitted: {order.id} for {ticker}")
            return order.id

        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return None

    def _check_order_status(self, order_id):
        """Check the status of an order"""
        try:
            order = self.alpaca_client.get_order(order_id)
            return order.status
        except Exception as e:
            logger.error(f"Error checking order status: {str(e)}")
            return None

    def _handle_filled_order(self, order_id, order_info):
        """Handle a filled order"""
        try:
            logger.info(f"Order {order_id} for {order_info['ticker']} filled")

            # Get order details from Alpaca
            order = self.alpaca_client.get_order(order_id)
            ticker = order_info['ticker']
            filled_price = float(order.filled_avg_price)
            filled_qty = int(order.filled_qty)
            side = order.side

            # Add to active positions
            if ticker not in self.active_positions:
                self.active_positions[ticker] = {
                    'ticker': ticker,
                    'quantity': filled_qty if side == 'buy' else -filled_qty,
                    'entry_price': filled_price,
                    'entry_time': time.time(),
                    'current_price': filled_price,
                    'current_value': filled_price * filled_qty,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'max_unrealized_pnl': 0.0,
                    'min_unrealized_pnl': 0.0,
                    'signal': order_info.get('signal', {})
                }

                # Set take profit and stop loss
                if 'signal' in order_info:
                    signal = order_info['signal']
                    if 'price_target' in signal:
                        self.active_positions[ticker]['take_profit'] = signal['price_target']
                    if 'stop_loss' in signal:
                        self.active_positions[ticker]['stop_loss'] = signal['stop_loss']

                # Store in Redis
                self.redis.hset("positions:active", ticker,
                                json.dumps(self.active_positions[ticker]))

                # Set up trailing stop if enabled
                if self.config['execution']['enable_trailing_stops']:
                    self._submit_trailing_stop(
                        ticker, self.active_positions[ticker])

                # Update daily stats
                self.daily_stats['trades_executed'] += 1
                self.daily_stats['current_exposure'] += filled_price * filled_qty
                self.redis.hmset("execution:daily_stats", self.daily_stats)

                # Publish position update
                self.redis.publish("position_updates", json.dumps({
                    "type": "new_position",
                    "ticker": ticker,
                    "quantity": filled_qty,
                    "side": side,
                    "entry_price": filled_price,
                    "timestamp": time.time()
                }))

            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                self.redis.hdel("orders:pending", order_id)

        except Exception as e:
            logger.error(f"Error handling filled order {order_id}: {str(e)}")

    def _handle_canceled_order(self, order_id, order_info):
        """Handle a canceled order"""
        try:
            logger.info(
                f"Order {order_id} for {order_info['ticker']} canceled")

            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                self.redis.hdel("orders:pending", order_id)

        except Exception as e:
            logger.error(f"Error handling canceled order {order_id}: {str(e)}")

    def _handle_rejected_order(self, order_id, order_info):
        """Handle a rejected order"""
        try:
            logger.warning(
                f"Order {order_id} for {order_info['ticker']} rejected")

            # Get rejection reason
            order = self.alpaca_client.get_order(order_id)
            reject_reason = getattr(order, 'rejected_reason', 'Unknown reason')

            logger.warning(f"Rejection reason: {reject_reason}")

            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                self.redis.hdel("orders:pending", order_id)

        except Exception as e:
            logger.error(f"Error handling rejected order {order_id}: {str(e)}")

    def _get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            # Try to get from real-time data first
            if ticker in self.real_time_data['trades'] and self.real_time_data['trades'][ticker]:
                return self.real_time_data['trades'][ticker][-1]['price']

            # Fall back to API
            last_trade = self.alpaca_client.get_latest_trade(ticker)
            return float(last_trade.price)
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return None

    def _update_position_stats(self, ticker, current_price):
        """Update position statistics"""
        try:
            if ticker not in self.active_positions:
                return

            position = self.active_positions[ticker]
            entry_price = position['entry_price']
            quantity = position['quantity']

            # Update current price and value
            position['current_price'] = current_price
            position['current_value'] = current_price * abs(quantity)

            # Calculate unrealized P&L
            if quantity > 0:  # Long position
                unrealized_pnl = (current_price - entry_price) * quantity
                unrealized_pnl_pct = (current_price / entry_price - 1) * 100
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * abs(quantity)
                unrealized_pnl_pct = (entry_price / current_price - 1) * 100

            position['unrealized_pnl'] = unrealized_pnl
            position['unrealized_pnl_pct'] = unrealized_pnl_pct

            # Update max and min P&L
            if unrealized_pnl > position.get('max_unrealized_pnl', 0):
                position['max_unrealized_pnl'] = unrealized_pnl

            if unrealized_pnl < position.get('min_unrealized_pnl', 0) or 'min_unrealized_pnl' not in position:
                position['min_unrealized_pnl'] = unrealized_pnl

            # Update trailing stop if enabled
            if self.config['execution']['enable_trailing_stops'] and 'trailing_stop' in position:
                self._update_trailing_stop(ticker, position)

            # Store in Redis
            self.redis.hset("positions:active", ticker,
                            json.dumps(position))

        except Exception as e:
            logger.error(
                f"Error updating position stats for {ticker}: {str(e)}")

    def _check_exit_conditions(self, ticker, current_price):
        """Check if position should be exited"""
        try:
            if ticker not in self.active_positions:
                return False

            position = self.active_positions[ticker]
            quantity = position['quantity']
            entry_price = position['entry_price']
            entry_time = position['entry_time']

            # Take profit
            take_profit = position.get('take_profit')
            if take_profit:
                if quantity > 0 and current_price >= take_profit:  # Long position
                    logger.info(
                        f"Take profit triggered for {ticker}: {current_price} >= {take_profit}")
                    return True
                elif quantity < 0 and current_price <= take_profit:  # Short position
                    logger.info(
                        f"Take profit triggered for {ticker}: {current_price} <= {take_profit}")
                    return True

            # Stop loss
            stop_loss = position.get('stop_loss')
            if stop_loss:
                if quantity > 0 and current_price <= stop_loss:  # Long position
                    logger.info(
                        f"Stop loss triggered for {ticker}: {current_price} <= {stop_loss}")
                    return True
                elif quantity < 0 and current_price >= stop_loss:  # Short position
                    logger.info(
                        f"Stop loss triggered for {ticker}: {current_price} >= {stop_loss}")
                    return True

            # Trailing stop
            trailing_stop = position.get('trailing_stop_price')
            if trailing_stop:
                if quantity > 0 and current_price <= trailing_stop:  # Long position
                    logger.info(
                        f"Trailing stop triggered for {ticker}: {current_price} <= {trailing_stop}")
                    return True
                elif quantity < 0 and current_price >= trailing_stop:  # Short position
                    logger.info(
                        f"Trailing stop triggered for {ticker}: {current_price} >= {trailing_stop}")
                    return True

            # Position timeout
            if time.time() - entry_time > self.config['execution']['position_timeout']:
                logger.info(f"Position timeout for {ticker}")
                return True

            return False

        except Exception as e:
            logger.error(
                f"Error checking exit conditions for {ticker}: {str(e)}")
            return False

    def _exit_position(self, ticker, reason, price=None):
        """Exit a position"""
        try:
            if ticker not in self.active_positions:
                return

            position = self.active_positions[ticker]
            quantity = position['quantity']
            entry_price = position['entry_price']

            # Cancel any pending orders for this ticker
            self._cancel_position_orders(ticker)

            # Submit exit order
            side = 'sell' if quantity > 0 else 'buy'
            qty = abs(quantity)

            order = self.alpaca_client.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            logger.info(f"Exit order submitted for {ticker}: {order.id}")

            # Update position status
            position['exit_pending'] = True
            position['exit_order_id'] = order.id
            position['exit_reason'] = reason

            # Store in Redis
            self.redis.hset("positions:active", ticker,
                            json.dumps(position))

            # Wait for order to fill
            filled = False
            start_time = time.time()
            while not filled and time.time() - start_time < 60:  # Wait up to 60 seconds
                try:
                    # Check order status
                    order_status = self.alpaca_client.get_order(order.id)
                    if order_status.status == 'filled':
                        filled = True
                        break
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error checking exit order status: {str(e)}")
                    break

            # Get fill price
            exit_price = None
            try:
                if filled:
                    fill_info = self.alpaca_client.get_order(order.id)
                    exit_price = float(fill_info.filled_avg_price)
                else:
                    # Use current price if order didn't fill
                    exit_price = price or self._get_current_price(ticker)
            except Exception as e:
                logger.error(f"Error getting exit fill price: {str(e)}")
                exit_price = price or position['current_price']

            # Calculate P&L
            if quantity > 0:  # Long position
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = (exit_price / entry_price - 1) * 100
            else:  # Short position
                pnl = (entry_price - exit_price) * abs(quantity)
                pnl_pct = (entry_price / exit_price - 1) * 100

            # Update daily stats
            self.daily_stats['current_exposure'] -= position['current_value']
            self.daily_stats['total_pnl'] += pnl
            if pnl > 0:
                self.daily_stats['profitable_trades'] += 1
            self.redis.hmset("execution:daily_stats", self.daily_stats)

            # Remove from active positions
            if ticker in self.active_positions:
                del self.active_positions[ticker]
                self.redis.hdel("positions:active", ticker)

            # Add to position history
            position_history = {
                'ticker': ticker,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': position['entry_time'],
                'exit_time': time.time(),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            }
            self.redis.lpush("positions:history", json.dumps(position_history))
            # Keep last 100 positions
            self.redis.ltrim("positions:history", 0, 99)

            # Publish position update
            self.redis.publish("position_updates", json.dumps({
                "type": "position_closed",
                "ticker": ticker,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "timestamp": time.time()
            }))

            logger.info(
                f"Position closed for {ticker}: PnL ${pnl:.2f} ({pnl_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error exiting position for {ticker}: {str(e)}")

    def _exit_all_positions(self, reason):
        """Exit all positions"""
        try:
            # Get active positions
            positions = self.active_positions.copy()

            # Exit each position
            for ticker in positions:
                # Get current price
                current_price = self._get_current_price(ticker)
                if current_price:
                    # Exit position
                    self._exit_position(ticker, reason, current_price)

        except Exception as e:
            logger.error(f"Error exiting all positions: {str(e)}")

    def _submit_trailing_stop(self, ticker, position):
        """Submit a trailing stop order"""
        try:
            if 'trailing_stop' not in position or not position.get('quantity'):
                return

            # Calculate trailing stop percentage
            trailing_percent = position.get(
                'trailing_stop', self.config['execution']['trailing_stop_percent'])

            # Calculate initial stop price
            entry_price = position['entry_price']
            quantity = position['quantity']

            if quantity > 0:  # Long position
                # Initial stop price is entry price minus trailing percentage
                stop_price = entry_price * (1 - trailing_percent)
            else:  # Short position
                # Initial stop price is entry price plus trailing percentage
                stop_price = entry_price * (1 + trailing_percent)

            # Store trailing stop information
            position['trailing_stop_percent'] = trailing_percent
            position['trailing_stop_price'] = stop_price
            position['trailing_stop_high_price'] = entry_price if quantity > 0 else 0
            position['trailing_stop_low_price'] = 0 if quantity > 0 else entry_price

            # Store in Redis
            self.redis.hset("positions:active", ticker, json.dumps(position))

            logger.info(
                f"Set trailing stop for {ticker} at ${stop_price:.2f} ({trailing_percent*100:.1f}%)")

        except Exception as e:
            logger.error(f"Error setting trailing stop for {ticker}: {str(e)}")

    def _update_trailing_stop(self, ticker, position):
        """Update trailing stop price based on current price"""
        try:
            if 'trailing_stop_price' not in position:
                return

            current_price = position['current_price']
            quantity = position['quantity']
            trailing_percent = position['trailing_stop_percent']

            if quantity > 0:  # Long position
                # Track new high price
                high_price = position['trailing_stop_high_price']

                if current_price > high_price:
                    # Update high price
                    position['trailing_stop_high_price'] = current_price

                    # Calculate new stop price
                    new_stop = current_price * (1 - trailing_percent)

                    # Only raise stop price, never lower it
                    if new_stop > position['trailing_stop_price']:
                        position['trailing_stop_price'] = new_stop
                        logger.info(
                            f"Updated trailing stop for {ticker} to ${new_stop:.2f}")

            else:  # Short position
                # Track new low price
                low_price = position['trailing_stop_low_price']

                if current_price < low_price or low_price == 0:
                    # Update low price
                    position['trailing_stop_low_price'] = current_price

                    # Calculate new stop price
                    new_stop = current_price * (1 + trailing_percent)

                    # Only lower stop price, never raise it
                    if new_stop < position['trailing_stop_price'] or position['trailing_stop_price'] == 0:
                        position['trailing_stop_price'] = new_stop
                        logger.info(
                            f"Updated trailing stop for {ticker} to ${new_stop:.2f}")

        except Exception as e:
            logger.error(
                f"Error updating trailing stop for {ticker}: {str(e)}")

    def _cancel_position_orders(self, ticker):
        """Cancel all orders related to a position"""
        try:
            # Get open orders for the ticker
            orders = self.alpaca_client.list_orders(
                status='open',
                symbols=[ticker]
            )

            # Cancel each order
            for order in orders:
                try:
                    self.alpaca_client.cancel_order(order.id)
                    logger.info(f"Canceled order {order.id} for {ticker}")

                    # Remove from pending orders if present
                    if order.id in self.pending_orders:
                        del self.pending_orders[order.id]
                        self.redis.hdel("orders:pending", order.id)

                except Exception as e:
                    logger.error(f"Error canceling order {order.id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error canceling orders for {ticker}: {str(e)}")

    def _update_daily_stats(self):
        """Update daily statistics"""
        try:
            # Get current exposure
            current_exposure = sum(pos['current_value']
                                   for pos in self.active_positions.values())
            self.daily_stats['current_exposure'] = current_exposure

            # Calculate drawdown
            account = self._get_account_info()
            if account:
                equity = float(account['equity'])
                previous_equity = float(account.get('last_equity', equity))

                # Calculate drawdown as percentage
                drawdown = (previous_equity - equity) / \
                    previous_equity * 100 if previous_equity > 0 else 0

                # Update max drawdown
                if drawdown > self.daily_stats['max_drawdown']:
                    self.daily_stats['max_drawdown'] = drawdown

            # Update last update time
            self.daily_stats['last_update'] = time.time()

            # Store in Redis if available
            if self.redis:
                try:
                    self.redis.hmset("execution:daily_stats", self.daily_stats)
                except redis.RedisError as e:
                    logger.warning(
                        f"Error storing daily stats in Redis: {str(e)}")

        except Exception as e:
            logger.error(f"Error updating daily stats: {str(e)}")

    def _get_account_info(self):
        """Get account information"""
        try:
            account = self.alpaca_client.get_account()
            return {
                'id': account.id,
                'equity': account.equity,
                'cash': account.cash,
                'buying_power': account.buying_power,
                'last_equity': account.last_equity
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def _get_account_equity(self):
        """Get account equity"""
        account = self._get_account_info()
        return float(account['equity']) if account else 0

    #
    # Day Trading Methods from day_trading_system.py
    #

    async def calculate_intraday_profit_potential(self, ticker):
        """Calculate potential intraday profit for a ticker based on historical patterns"""
        try:
            # Get historical intraday data for the past 10 trading days
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            ten_days_ago = (datetime.datetime.now() -
                            datetime.timedelta(days=14)).strftime("%Y-%m-%d")

            # Check if data_pipeline is available
            if not self.data_pipeline:
                logger.warning(f"Data pipeline not available for {ticker}")
                return {
                    'average_range_percent': 0,
                    'profit_probability': 0,
                    'dollar_profit_potential': 0
                }

            # Get minute aggregates
            aggs = await self.data_pipeline.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_date=ten_days_ago,
                to_date=today
            )

            if not isinstance(aggs, pd.DataFrame) or aggs.empty:
                return {
                    'average_range_percent': 0,
                    'profit_probability': 0,
                    'dollar_profit_potential': 0
                }

            # Group by day
            aggs['date'] = pd.to_datetime(aggs['timestamp']).dt.date

            # Calculate daily stats
            daily_stats = []
            for date, group in aggs.groupby('date'):
                if len(group) < 30:  # Skip days with little data
                    continue

                day_open = group['open'].iloc[0]
                day_high = group['high'].max()
                day_low = group['low'].min()
                day_close = group['close'].iloc[-1]

                # Calculate range and movement
                day_range_percent = (day_high - day_low) / \
                    day_open * 100 if day_open > 0 else 0
                day_move_percent = (day_close - day_open) / \
                    day_open * 100 if day_open > 0 else 0

                daily_stats.append({
                    'date': date,
                    'range_percent': day_range_percent,
                    'move_percent': day_move_percent,
                    'positive_close': day_close > day_open
                })

            if not daily_stats:
                return {
                    'average_range_percent': 0,
                    'profit_probability': 0,
                    'dollar_profit_potential': 0
                }

            # Calculate average range and probability of positive close
            df_stats = pd.DataFrame(daily_stats)
            average_range = df_stats['range_percent'].mean()
            profit_probability = df_stats['positive_close'].mean() * 100

            # Get current price
            current_price = self._get_current_price(ticker)
            if not current_price:
                return {
                    'average_range_percent': average_range,
                    'profit_probability': profit_probability,
                    'dollar_profit_potential': 0
                }

            # Calculate dollar profit potential based on target percentage
            target_profit_percent = self.config['day_trading']['target_profit_percent']
            dollar_profit_potential = current_price * \
                (target_profit_percent / 100)

            # Adjust by probability
            adjusted_dollar_potential = dollar_profit_potential * \
                (profit_probability / 100)

            return {
                'average_range_percent': average_range,
                'profit_probability': profit_probability,
                'dollar_profit_potential': adjusted_dollar_potential,
                'target_profit_percent': target_profit_percent
            }

        except Exception as e:
            logger.error(
                f"Error calculating profit potential for {ticker}: {str(e)}")
            return {
                'average_range_percent': 0,
                'profit_probability': 0,
                'dollar_profit_potential': 0
            }

    async def calculate_dollar_profit_potential(self, ticker, price, budget=5000):
        """Calculate potential dollar profit for a ticker based on historical patterns"""
        try:
            # Check if data_pipeline is available
            if not self.data_pipeline:
                logger.warning(f"Data pipeline not available for {ticker}")
                return 0

            # Get average daily range (ATR)
            atr = await self.data_pipeline.get_atr(ticker)
            if not atr:
                return 0

            # Calculate optimal position size
            # Max 25% of budget per position, cap at 1000 shares
            max_shares = min(int(budget * 0.25 / price), 1000)

            # Calculate expected dollar profit based on average range
            # Expected dollar move based on ATR
            expected_move = price * (atr / price)
            dollar_profit = max_shares * expected_move * \
                0.5  # Assume we capture 50% of the move

            logger.info(f"Dollar profit potential for {ticker}: ${dollar_profit:.2f} " +
                        f"(Price: ${price:.2f}, ATR: ${atr:.2f}, Shares: {max_shares})")

            return dollar_profit
        except Exception as e:
            logger.error(
                f"Error calculating dollar profit potential for {ticker}: {str(e)}")
            return 0

    async def _update_day_trading_candidates(self):
        """Update the list of day trading candidates"""
        logger.info("Updating day trading candidates with $5000 position limit")

        candidates = []
        try:
            # Get current focused watchlist
            watchlist_data = self.redis.zrevrange(
                "watchlist:focused", 0, -1, withscores=True)

            if not watchlist_data:
                # Fall back to active watchlist if focused is empty
                watchlist_data = self.redis.zrevrange(
                    "watchlist:active", 0, -1, withscores=True)

            if not watchlist_data:
                logger.warning(
                    "No stocks in watchlist for day trading selection")
                return []

            # Convert to list of tuples
            watchlist = [(item[0].decode('utf-8') if isinstance(item[0], bytes) else item[0], item[1])
                         for item in watchlist_data]

            # Calculate day trading metrics for each stock
            for ticker, base_score in watchlist:
                # Get current price
                current_price = self._get_current_price(ticker)

                if not current_price or current_price <= 0:
                    continue

                # Skip if price is outside our range
                if current_price < self.config['stock_selection']['min_price'] or current_price > self.config['stock_selection']['max_price']:
                    continue

                # Calculate profit potential
                profit_potential = await self.calculate_intraday_profit_potential(ticker)

                # Calculate dollar profit potential
                dollar_profit = await self.calculate_dollar_profit_potential(ticker, current_price, self.config['day_trading']['max_total_position'])

                # Calculate optimal position size
                max_position = min(
                    # Max 25% in one stock
                    self.config['day_trading']['max_total_position'] *
                    self.config['day_trading']['max_position_percent'],
                    5000  # Hard limit of $5000 per position
                )

                # Calculate optimal shares
                optimal_shares = int(max_position / current_price)

                # Skip if we can't buy at least 10 shares
                if optimal_shares < 10:
                    continue

                # Calculate actual position value
                position_value = optimal_shares * current_price

                # Calculate stop loss and target prices
                stop_loss_percent = self.config['day_trading']['stop_loss_percent']
                target_profit_percent = self.config['day_trading']['target_profit_percent']

                stop_price = current_price * (1 - stop_loss_percent / 100)
                target_price = current_price * \
                    (1 + target_profit_percent / 100)

                # Calculate risk/reward ratio
                risk = current_price - stop_price
                reward = target_price - current_price
                risk_reward = reward / risk if risk > 0 else 0

                # Skip if risk/reward is too low
                # Lowered threshold to 1.5 to find more opportunities
                if risk_reward < 1.5:
                    continue

                # Calculate day trading score
                day_trading_score = self._calculate_day_trading_score(
                    base_score,
                    profit_potential,
                    risk_reward,
                    optimal_shares,
                    current_price
                )

                # Add to candidates
                candidates.append({
                    'ticker': ticker,
                    'price': current_price,
                    'optimal_shares': optimal_shares,
                    'max_position': position_value,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'risk_reward': risk_reward,
                    'score': day_trading_score,
                    'dollar_profit_potential': dollar_profit
                })

            # Sort by score (highest first)
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # Take top candidates up to max positions
            max_positions = self.config['day_trading']['max_positions']
            candidates = candidates[:min(max_positions, len(candidates))]

            if not candidates:
                logger.warning("No suitable day trading candidates found")
                return []

            # Store candidates in Redis
            pipeline = self.redis.pipeline()
            pipeline.delete("day_trading:active")

            for candidate in candidates:
                pipeline.zadd("day_trading:active", {
                              candidate['ticker']: candidate['score']})

                # Store detailed data
                position_data = {
                    'price': str(candidate['price']),
                    'shares': str(candidate['optimal_shares']),
                    'position_value': str(candidate['max_position']),
                    'stop_price': str(candidate['stop_price']),
                    'target_price': str(candidate['target_price']),
                    'risk_reward': str(candidate['risk_reward']),
                    'score': str(candidate['score']),
                    'timestamp': datetime.datetime.now().isoformat()
                }

                self.redis.hset(
                    f"day_trading:position:{candidate['ticker']}", mapping=position_data)

            # Store last update time
            now = datetime.datetime.now().isoformat()
            pipeline.set("day_trading:active:last_update", now)

            pipeline.execute()

            # Update local state
            self.day_trading_candidates = set(
                [c['ticker'] for c in candidates])

            logger.info(
                f"Day trading candidates updated with {len(candidates)} stocks")

            # Calculate total position
            total_position = sum(c['max_position'] for c in candidates)
            logger.info(
                f"Total day trading position: ${total_position:.2f} (Limit: $5,000)")

            # Log the candidates
            for candidate in candidates:
                logger.info(f"Day Trading Candidate: {candidate['ticker']}, "
                            f"Price: ${candidate['price']:.2f}, "
                            f"Shares: {candidate['optimal_shares']}, "
                            f"Position: ${candidate['max_position']:.2f}, "
                            f"Target: ${candidate['target_price']:.2f}, "
                            f"Stop: ${candidate['stop_price']:.2f}, "
                            f"Score: {candidate['score']:.2f}")

            return candidates

        except Exception as e:
            logger.error(
                f"Error updating day trading candidates: {str(e)}", exc_info=True)
            return []

    def _calculate_day_trading_score(self, base_score, profit_potential, risk_reward, shares, price):
        """Calculate day trading score with emphasis on dollar profit potential"""
        try:
            # Start with base score
            score = base_score

            # Calculate position sizing
            budget = self.config['day_trading']['max_total_position']
            max_shares = min(int(budget * 0.25 / price), 1000)

            # Get dollar profit potential from the profit_potential parameter
            dollar_profit_potential = 0

            # Use the profit_potential parameter which is already calculated
            if profit_potential:
                dollar_profit_potential = profit_potential.get(
                    'dollar_profit_potential', 0)

            # 1. Dollar profit potential (highest weight)
            if dollar_profit_potential > 200:
                score *= 1.8
            elif dollar_profit_potential > 100:
                score *= 1.5
            elif dollar_profit_potential > 50:
                score *= 1.2

            # 2. Traditional profit potential factors
            if profit_potential:
                avg_range = profit_potential.get('average_range_percent', 0)
                probability = profit_potential.get('profit_probability', 0)

                # Boost score for stocks with higher profit probability (reduced weight)
                if probability >= 70:
                    score *= 1.2
                elif probability >= 60:
                    score *= 1.15
                elif probability >= 50:
                    score *= 1.05

            # 3. Risk/reward ratio (second highest weight)
            if risk_reward >= 5.0:
                score *= 1.5
            elif risk_reward >= 4.0:
                score *= 1.4
            elif risk_reward >= 3.0:
                score *= 1.3
            elif risk_reward >= 2.0:
                score *= 1.2
            elif risk_reward >= 1.5:
                score *= 1.1

            # 4. Share count - prefer positions between 100-500 shares
            if 100 <= max_shares <= 500:
                score *= 1.2
            elif max_shares > 500:
                score *= 1.1
            elif 50 <= max_shares < 100:
                score *= 1.05

            # 5. Price range - slightly favor lower priced stocks
            if 5 <= price <= 20:  # Sweet spot for day trading
                score *= 1.2
            elif 20 < price <= 50:
                score *= 1.1
            elif 3 <= price < 5:
                score *= 1.05

            logger.info(f"Day trading score for ticker with price ${price:.2f}: {score:.2f} " +
                        f"(Risk/Reward: {risk_reward:.2f}, Shares: {max_shares})")

            return score

        except Exception as e:
            logger.error(f"Error calculating day trading score: {str(e)}")
            return base_score

    def _send_entry_signal(self, ticker, shares, entry_price, stop_price, target_price):
        """Send entry signal to execution system via Redis"""
        try:
            # Create signal
            signal = {
                'ticker': ticker,
                'direction': 'long',  # We only do long positions for day trading
                'signal_score': 80,   # High confidence for day trading signals
                'confidence': 0.8,
                'position_size': int(shares),
                'stop_loss': float(stop_price),
                'price_target': float(target_price),
                'signal_source': 'day_trading_system',
                'timestamp': datetime.datetime.now().timestamp()
            }

            # Publish to execution system
            self.redis.publish("execution:new_signal", json.dumps(signal))

            logger.info(
                f"Sent entry signal for {ticker}: {shares} shares at ~${entry_price}")
            return True
        except Exception as e:
            logger.error(f"Error sending entry signal for {ticker}: {str(e)}")
            return False

    def _send_exit_signal(self, ticker, shares, entry_price):
        """Send exit signal to execution system via Redis"""
        try:
            # Create signal
            signal = {
                'ticker': ticker,
                'direction': 'close',  # Special direction to close position
                'signal_score': 90,    # High priority for closing positions
                'confidence': 0.9,
                'position_size': int(shares),
                'signal_source': 'day_trading_system_close',
                'timestamp': datetime.datetime.now().timestamp()
            }

            # Publish to execution system
            self.redis.publish("execution:new_signal", json.dumps(signal))

            logger.info(
                f"Sent exit signal for {ticker}: {shares} shares, entry: ${entry_price}")
            return True
        except Exception as e:
            logger.error(f"Error sending exit signal for {ticker}: {str(e)}")
            return False

    async def close_all_day_trading_positions(self):
        """Close all day trading positions"""
        logger.info("Closing all day trading positions")
        try:
            # Get active day trading positions
            positions_data = self.redis.zrange(
                "day_trading:active", 0, -1, withscores=False)

            if not positions_data:
                logger.info("No day trading positions to close")
                return

            # Convert to list of tickers
            tickers = [pos.decode('utf-8') if isinstance(pos, bytes) else pos
                       for pos in positions_data]

            logger.info(f"Closing {len(tickers)} day trading positions")

            # Close each position
            for ticker in tickers:
                position_data = self.redis.hgetall(
                    f"day_trading:position:{ticker}")
                if not position_data:
                    continue

                # Convert bytes to strings
                position = {k.decode('utf-8') if isinstance(k, bytes) else k:
                            v.decode('utf-8') if isinstance(v, bytes) else v
                            for k, v in position_data.items()}

                # Get position details
                shares = int(float(position.get('shares', 0)))
                entry_price = float(position.get('price', 0))

                if shares > 0 and entry_price > 0:
                    # Send exit signal
                    self._send_exit_signal(ticker, shares, entry_price)

                    # Update position status
                    self.redis.hset(f"day_trading:position:{ticker}", mapping={
                        "status": "closed",
                        "exit_reason": "market_close",
                        "exit_time": datetime.datetime.now().isoformat()
                    })

                    # Remove from active positions
                    self.redis.zrem("day_trading:active", ticker)

                    # Add to closed positions
                    self.redis.zadd("day_trading:closed",
                                    {ticker: time.time()})

                    logger.info(f"Closed day trading position for {ticker}")

            logger.info("All day trading positions closed")

        except Exception as e:
            logger.error(f"Error closing day trading positions: {str(e)}")

    async def _check_real_time_opportunities(self, ranked_tickers):
        """Check for new day trading opportunities based on real-time data"""
        try:
            # Get current day trading positions
            active_positions = self.redis.zrange(
                "day_trading:active", 0, -1, withscores=True)
            active_tickers = {pos[0].decode('utf-8') if isinstance(pos[0], bytes) else pos[0]
                              for pos in active_positions}

            # Get current time (Eastern)
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))

            # Only look for new opportunities during certain times
            if not (9 <= now.hour < 15) or now.weekday() >= 5:  # 9 AM to 3 PM ET, weekdays only
                return

            # Check if we have capacity for new positions
            max_positions = self.config['day_trading']['max_positions']
            if len(active_tickers) >= max_positions:
                return

            # Get available capital
            max_total_position = self.config['day_trading']['max_total_position']
            used_capital = 0

            for ticker in active_tickers:
                position_data = self.redis.hgetall(
                    f"day_trading:position:{ticker}")
                if position_data:
                    position_value = float(position_data.get(
                        b'position_value', b'0').decode('utf-8'))
                    used_capital += position_value

            available_capital = max_total_position - used_capital

            # If we have less than $500 available, don't open new positions
            if available_capital < 500:
                return

            # Look for new opportunities
            for ticker, score in ranked_tickers:
                # Skip if already in active positions
                if ticker in active_tickers:
                    continue

                # Check if ticker has real-time data
                if (ticker not in self.real_time_data['trades'] or
                    ticker not in self.real_time_data['quotes'] or
                        ticker not in self.real_time_data['minute_aggs']):
                    continue

                # Check if ticker has real-time alerts
                has_volume_spike = ticker in self.real_time_metrics['volume_spikes']
                has_price_jump = ticker in self.real_time_metrics['price_jumps']
                has_momentum_shift = ticker in self.real_time_metrics['momentum_shifts']

                # Only consider tickers with at least one alert
                if not (has_volume_spike or has_price_jump or has_momentum_shift):
                    continue

                # Get current price
                latest_trades = self.real_time_data['trades'][ticker]
                if not latest_trades:
                    continue

                current_price = latest_trades[-1]['price']

                # Skip if price is outside our range
                if current_price < self.config['stock_selection']['min_price'] or current_price > self.config['stock_selection']['max_price']:
                    continue

                # Calculate position size
                max_position_size = min(
                    available_capital,
                    self.config['day_trading']['max_total_position'] *
                    (self.config['day_trading']['max_position_percent'])
                )

                # Calculate shares
                shares = int(max_position_size / current_price)

                # Skip if not enough shares
                if shares < 10:
                    continue

                # Calculate position value
                position_value = shares * current_price

                # Calculate stop loss and target prices
                stop_loss_percent = self.config['day_trading']['stop_loss_percent']
                target_profit_percent = self.config['day_trading']['target_profit_percent']

                stop_price = current_price * (1 - stop_loss_percent / 100)
                target_price = current_price * \
                    (1 + target_profit_percent / 100)

                # Calculate risk/reward ratio
                risk = current_price - stop_price
                reward = target_price - current_price
                risk_reward = reward / risk if risk > 0 else 0

                # Skip if risk/reward is too low
                if risk_reward < 2.0:
                    continue

                # We have a valid opportunity - add to day trading candidates
                logger.info(
                    f"Real-time opportunity detected for {ticker} at ${current_price:.2f}")

                # Add to day trading active list
                self.redis.zadd("day_trading:active", {ticker: score})

                # Send entry signal to execution system
                self._send_entry_signal(
                    ticker, shares, current_price, stop_price, target_price)

                # Store position details
                position_data = {
                    'price': str(current_price),
                    'shares': str(shares),
                    'position_value': str(position_value),
                    'stop_price': str(stop_price),
                    'target_price': str(target_price),
                    'risk_reward': str(risk_reward),
                    'score': str(score),
                    'entry_time': datetime.datetime.now().isoformat(),
                    'status': 'open',
                    'entry_reason': 'real_time_alert'
                }

                self.redis.hset(
                    f"day_trading:position:{ticker}", mapping=position_data)

                # Update local state
                self.day_trading_candidates.add(ticker)

                # Publish new position alert
                self.redis.publish("position_updates", json.dumps({
                    "type": "new_position",
                    "ticker": ticker,
                    "price": current_price,
                    "shares": shares,
                    "position_value": position_value,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "risk_reward": risk_reward,
                    "entry_reason": "real_time_alert"
                }))

                # Update available capital
                available_capital -= position_value

                # Stop if we've reached max positions or used all available capital
                if len(self.day_trading_candidates) >= max_positions or available_capital < 500:
                    break

        except Exception as e:
            logger.error(f"Error checking real-time opportunities: {str(e)}")

    #
    # Peak Detection Methods from peak_detection_monitor.py
    #

    async def _get_monitored_tickers(self) -> List[str]:
        """Get list of tickers to monitor for peak detection"""
        try:
            # Try to get tickers from Redis
            if not self.redis:
                logger.warning(
                    "Redis not available for getting monitored tickers")
                return ["SPY", "QQQ", "IWM", "DIA", "XLK"]

            tickers_json = self.redis.get("monitored_tickers")
            if tickers_json:
                # Handle bytes response from Redis
                if isinstance(tickers_json, bytes):
                    tickers_json = tickers_json.decode('utf-8')
                return json.loads(tickers_json)

            # Return default list if no Redis entry
            default_symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK"]
            return default_symbols
        except Exception as e:
            logger.error(f"Error getting monitored tickers: {e}")
            # Return default list on error
            return ["SPY", "QQQ", "IWM", "DIA", "XLK"]

    async def _monitor_ticker_peaks(self, ticker: str):
        """
        Monitor a specific ticker for peak patterns

        Args:
            ticker: Stock ticker symbol
        """
        logger.info(f"Starting to monitor peaks for {ticker}")

        while self.running:
            try:
                # Get price data
                price_data = await self._get_ticker_price_data(ticker)
                if not price_data or len(price_data) < 20:
                    await asyncio.sleep(15)
                    continue

                # Detect peaks and troughs
                peaks, troughs = self._detect_peaks_and_troughs(price_data)

                # Analyze patterns
                patterns = self._analyze_patterns(
                    ticker, price_data, peaks, troughs)

                # Store results
                if patterns:
                    self._store_patterns(ticker, patterns)

            except Exception as e:
                logger.error(f"Error monitoring peaks for {ticker}: {e}")

            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

    async def _get_ticker_price_data(self, ticker: str) -> List[float]:
        """
        Get historical price data for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of closing prices
        """
        try:
            # Try to get from Redis
            key = f"stock:{ticker}:candles:minute"
            candles = self.redis.hgetall(key)

            if not candles:
                logger.warning(f"No price data found for {ticker}")
                return []

            # Convert to list of prices
            prices = []
            for timestamp, candle_json in sorted(candles.items()):
                candle = json.loads(candle_json)
                prices.append(candle['close'])

            return prices
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return []

    def _detect_peaks_and_troughs(self, price_data: List[float]) -> Tuple[List[int], List[int]]:
        """
        Detect peaks and troughs in price data using GPU acceleration if available

        Args:
            price_data: List of prices

        Returns:
            Tuple of (peak indices, trough indices)
        """
        try:
            # Use GPU acceleration if available
            if self.gpu_initialized and CUPY_AVAILABLE and cp is not None:
                try:
                    # Transfer data to GPU
                    prices_gpu = cp.array(price_data)
                    price_mean = cp.mean(prices_gpu)
                    price_std = cp.std(prices_gpu)

                    # Normalize peaks for detection
                    normalized_heights = cp.abs(
                        prices_gpu - price_mean) / price_std if price_std > 0 else cp.abs(prices_gpu - price_mean)

                    # Find peaks using GPU
                    peak_indices = []
                    trough_indices = []

                    # Create arrays for comparison
                    prices_shifted_left = cp.roll(prices_gpu, 1)
                    prices_shifted_right = cp.roll(prices_gpu, -1)

                    # Set boundary values to avoid false positives
                    prices_shifted_left[0] = prices_gpu[0]
                    prices_shifted_right[-1] = prices_gpu[-1]

                    # Find potential peaks and troughs
                    potential_peaks = cp.where((prices_gpu > prices_shifted_left) &
                                               (prices_gpu > prices_shifted_right))[0]
                    potential_troughs = cp.where((prices_gpu < prices_shifted_left) &
                                                 (prices_gpu < prices_shifted_right))[0]

                    # Transfer results back to CPU for further processing
                    potential_peaks = cp.asnumpy(potential_peaks)
                    potential_troughs = cp.asnumpy(potential_troughs)
                    prices = cp.asnumpy(prices_gpu)

                    # Filter peaks by prominence
                    for i in potential_peaks:
                        if i > 0 and i < len(prices) - 1:  # Skip boundary points
                            prominence = min(
                                prices[i] - prices[i-1], prices[i] - prices[i+1])
                            if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                                # Check if far enough from previous peak
                                if not peak_indices or i - peak_indices[-1] >= MIN_PEAK_DISTANCE:
                                    peak_indices.append(i)

                    # Filter troughs by prominence
                    for i in potential_troughs:
                        if i > 0 and i < len(prices) - 1:  # Skip boundary points
                            prominence = min(
                                prices[i-1] - prices[i], prices[i+1] - prices[i])
                            if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                                # Check if far enough from previous trough
                                if not trough_indices or i - trough_indices[-1] >= MIN_PEAK_DISTANCE:
                                    trough_indices.append(i)

                    # Clear GPU memory
                    if self.gpu_memory_manager:
                        self.gpu_memory_manager.request_cleanup()

                    return peak_indices, trough_indices

                except Exception as e:
                    logger.warning(
                        f"GPU peak detection failed, falling back to CPU: {e}")
                    # Fall back to CPU implementation

            # CPU implementation
            prices = np.array(price_data)
            price_mean = np.mean(prices)
            price_std = np.std(prices)

            # Normalize peaks for detection
            normalized_heights = np.abs(
                prices - price_mean) / price_std if price_std > 0 else np.abs(prices - price_mean)

            # Find peaks
            peak_indices = []
            trough_indices = []

            # Simple peak detection with prominence
            for i in range(1, len(prices) - 1):
                # Peak detection
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    prominence = min(
                        prices[i] - prices[i-1], prices[i] - prices[i+1])
                    if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                        # Check if far enough from previous peak
                        if not peak_indices or i - peak_indices[-1] >= MIN_PEAK_DISTANCE:
                            peak_indices.append(i)

                # Trough detection
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    prominence = min(
                        prices[i-1] - prices[i], prices[i+1] - prices[i])
                    if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                        # Check if far enough from previous trough
                        if not trough_indices or i - trough_indices[-1] >= MIN_PEAK_DISTANCE:
                            trough_indices.append(i)

            return peak_indices, trough_indices
        except Exception as e:
            logger.error(f"Error detecting peaks: {e}")
            return [], []

    def _analyze_patterns(self, ticker: str, price_data: List[float],
                          peaks: List[int], troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Analyze patterns from peaks and troughs

        Args:
            ticker: Stock ticker symbol
            price_data: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of detected patterns
        """
        patterns = []

        if not peaks or not troughs or not price_data:
            return patterns

        try:
            # Find double top pattern
            double_tops = self._find_double_tops(price_data, peaks, troughs)
            patterns.extend(double_tops)

            # Find double bottom pattern
            double_bottoms = self._find_double_bottoms(
                price_data, peaks, troughs)
            patterns.extend(double_bottoms)

            # Find head and shoulders pattern
            head_shoulders = self._find_head_and_shoulders(
                price_data, peaks, troughs)
            patterns.extend(head_shoulders)

            # Add ticker to patterns
            for pattern in patterns:
                pattern['ticker'] = ticker
                pattern['timestamp'] = int(time.time())

            return patterns
        except Exception as e:
            logger.error(f"Error analyzing patterns for {ticker}: {e}")
            return []

    def _find_double_tops(self, prices: List[float], peaks: List[int],
                          troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find double top patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of double top patterns
        """
        patterns = []

        if len(peaks) < 2:
            return patterns

        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak1_price = prices[peak1_idx]

            for j in range(i + 1, len(peaks)):
                peak2_idx = peaks[j]
                peak2_price = prices[peak2_idx]

                # Check if peaks are at similar price levels (within 2%)
                price_diff_pct = abs(peak1_price - peak2_price) / peak1_price
                if price_diff_pct > 0.02:
                    continue

                # Check for trough in between
                trough_between = False
                trough_idx = None
                min_price = float('inf')

                for t in troughs:
                    if peak1_idx < t < peak2_idx:
                        trough_between = True
                        if prices[t] < min_price:
                            min_price = prices[t]
                            trough_idx = t

                if not trough_between:
                    continue

                # Calculate confirmation level (neckline)
                neckline = prices[trough_idx]

                # Check if the pattern has enough height (at least 3%)
                height_pct = (peak1_price - neckline) / neckline
                if height_pct < 0.03:
                    continue

                patterns.append({
                    'pattern': 'double_top',
                    'peak1_idx': peak1_idx,
                    'peak1_price': peak1_price,
                    'peak2_idx': peak2_idx,
                    'peak2_price': peak2_price,
                    'trough_idx': trough_idx,
                    'trough_price': prices[trough_idx],
                    'neckline': neckline,
                    'confidence': 0.7 * (1 - price_diff_pct * 10) + 0.3 * min(1, height_pct * 10)
                })

        return patterns

    def _find_double_bottoms(self, prices: List[float], peaks: List[int],
                             troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find double bottom patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of double bottom patterns
        """
        patterns = []

        if len(troughs) < 2:
            return patterns

        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough1_price = prices[trough1_idx]

            for j in range(i + 1, len(troughs)):
                trough2_idx = troughs[j]
                trough2_price = prices[trough2_idx]

                # Check if troughs are at similar price levels (within 2%)
                price_diff_pct = abs(
                    trough1_price - trough2_price) / trough1_price
                if price_diff_pct > 0.02:
                    continue

                # Check for peak in between
                peak_between = False
                peak_idx = None
                max_price = float('-inf')

                for p in peaks:
                    if trough1_idx < p < trough2_idx:
                        peak_between = True
                        if prices[p] > max_price:
                            max_price = prices[p]
                            peak_idx = p

                if not peak_between:
                    continue

                # Calculate confirmation level (resistance line)
                resistance = prices[peak_idx]

                # Check if the pattern has enough height (at least 3%)
                height_pct = (resistance - trough1_price) / trough1_price
                if height_pct < 0.03:
                    continue

                patterns.append({
                    'pattern': 'double_bottom',
                    'trough1_idx': trough1_idx,
                    'trough1_price': trough1_price,
                    'trough2_idx': trough2_idx,
                    'trough2_price': trough2_price,
                    'peak_idx': peak_idx,
                    'peak_price': prices[peak_idx],
                    'resistance': resistance,
                    'confidence': 0.7 * (1 - price_diff_pct * 10) + 0.3 * min(1, height_pct * 10)
                })

        return patterns

    def _find_head_and_shoulders(self, prices: List[float], peaks: List[int],
                                 troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find head and shoulders patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of head and shoulders patterns
        """
        patterns = []

        if len(peaks) < 3 or len(troughs) < 2:
            return patterns

        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            left_shoulder_price = prices[left_shoulder_idx]

            for j in range(i + 1, len(peaks) - 1):
                head_idx = peaks[j]
                head_price = prices[head_idx]

                # Head should be higher than left shoulder
                if head_price <= left_shoulder_price:
                    continue

                for k in range(j + 1, len(peaks)):
                    right_shoulder_idx = peaks[k]
                    right_shoulder_price = prices[right_shoulder_idx]

                    # Right shoulder should be similar to left shoulder
                    shoulder_diff_pct = abs(
                        left_shoulder_price - right_shoulder_price) / left_shoulder_price
                    if shoulder_diff_pct > 0.05:
                        continue

                    # Check for troughs between shoulders and head
                    left_trough_idx = None
                    right_trough_idx = None
                    left_trough_price = float('inf')
                    right_trough_price = float('inf')

                    for t in troughs:
                        if left_shoulder_idx < t < head_idx and prices[t] < left_trough_price:
                            left_trough_idx = t
                            left_trough_price = prices[t]
                        elif head_idx < t < right_shoulder_idx and prices[t] < right_trough_price:
                            right_trough_idx = t
                            right_trough_price = prices[t]

                    if left_trough_idx is None or right_trough_idx is None:
                        continue

                    # Calculate neckline as approximate line between troughs
                    neckline = (prices[left_trough_idx] +
                                prices[right_trough_idx]) / 2

                    # Pattern height
                    height_pct = (head_price - neckline) / neckline
                    if height_pct < 0.03:
                        continue

                    patterns.append({
                        'pattern': 'head_and_shoulders',
                        'left_shoulder_idx': left_shoulder_idx,
                        'left_shoulder_price': left_shoulder_price,
                        'head_idx': head_idx,
                        'head_price': head_price,
                        'right_shoulder_idx': right_shoulder_idx,
                        'right_shoulder_price': right_shoulder_price,
                        'left_trough_idx': left_trough_idx,
                        'left_trough_price': prices[left_trough_idx],
                        'right_trough_idx': right_trough_idx,
                        'right_trough_price': prices[right_trough_idx],
                        'neckline': neckline,
                        'confidence': 0.4 * (1 - shoulder_diff_pct * 5) + 0.6 * min(1, height_pct * 10)
                    })

        return patterns

    def _store_patterns(self, ticker: str, patterns: List[Dict[str, Any]]):
        """
        Store detected patterns in Redis

        Args:
            ticker: Stock ticker symbol
            patterns: List of detected patterns
        """
        if not patterns:
            return

        try:
            # Store in Redis
            now = int(time.time())
            key = f"patterns:{ticker}"

            pipeline = self.redis.pipeline()
            for pattern in patterns:
                pattern_id = f"{pattern['pattern']}:{now}:{pattern['confidence']}"
                pipeline.hset(key, pattern_id, json.dumps(pattern))

            # Publish pattern alert
            for pattern in patterns:
                self.redis.publish("pattern_alerts", json.dumps({
                    "ticker": ticker,
                    "pattern": pattern['pattern'],
                    "confidence": pattern['confidence'],
                    "timestamp": now
                }))

                # Generate trading signal if confidence is high enough
                if pattern['confidence'] >= 0.7:
                    self._generate_pattern_signal(ticker, pattern)

            # Expire after 24 hours
            pipeline.expire(key, 86400)
            pipeline.execute()

        except Exception as e:
            logger.error(f"Error storing patterns for {ticker}: {e}")

    def _generate_pattern_signal(self, ticker: str, pattern: Dict[str, Any]):
        """
        Generate trading signal from pattern

        Args:
            ticker: Stock ticker symbol
            pattern: Pattern data
        """
        try:
            # Get current price
            current_price = self._get_current_price(ticker)
            if not current_price:
                return

            # Determine signal direction
            direction = "long"
            if pattern['pattern'] == 'double_top' or pattern['pattern'] == 'head_and_shoulders':
                direction = "short"  # Bearish patterns
            elif pattern['pattern'] == 'double_bottom':
                direction = "long"   # Bullish pattern

            # Calculate stop loss and target
            stop_loss_pct = 0.02  # 2% default
            target_pct = 0.04     # 4% default

            # Customize based on pattern
            if pattern['pattern'] == 'double_top':
                # Stop above the second peak, target at neckline level
                stop_loss = pattern['peak2_price'] * 1.01
                target = pattern['neckline'] * 0.95
            elif pattern['pattern'] == 'double_bottom':
                # Stop below the second trough, target at resistance level
                stop_loss = pattern['trough2_price'] * 0.99
                target = pattern['resistance'] * 1.05
            elif pattern['pattern'] == 'head_and_shoulders':
                # Stop above the head, target below neckline
                stop_loss = pattern['head_price'] * 1.01
                target = pattern['neckline'] * 0.90
            else:
                # Default calculation
                if direction == "long":
                    stop_loss = current_price * (1 - stop_loss_pct)
                    target = current_price * (1 + target_pct)
                else:
                    stop_loss = current_price * (1 + stop_loss_pct)
                    target = current_price * (1 - target_pct)

            # Create signal
            signal = {
                'ticker': ticker,
                'direction': direction,
                'signal_score': int(pattern['confidence'] * 100),
                'confidence': pattern['confidence'],
                'stop_loss': float(stop_loss),
                'price_target': float(target),
                'signal_source': f"pattern_{pattern['pattern']}",
                'timestamp': time.time()
            }

            # Publish to execution system
            self.redis.publish("execution:new_signal", json.dumps(signal))
            logger.info(
                f"Generated {direction} signal for {ticker} based on {pattern['pattern']} pattern")

        except Exception as e:
            logger.error(f"Error generating pattern signal for {ticker}: {e}")


def main():
    """Main entry point for the Trading Engine"""
    parser = argparse.ArgumentParser(description='Trading Engine')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--action', type=str, default='start',
                        choices=['start', 'stop', 'status'], help='Action to perform')
    parser.add_argument('--enable-continual-learning',
                        action='store_true', help='Enable continual learning system')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize Redis
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6380)),
        db=int(os.environ.get('REDIS_DB', 0)),
        username=os.environ.get('REDIS_USERNAME', 'default'),
        password=os.environ.get('REDIS_PASSWORD', 'trading_system_2025')
    )

    # Create trading engine
    trading_engine = TradingEngine(
        redis_client=redis_client,
        config_path=args.config
    )

    # Override config with command line arguments
    if args.enable_continual_learning:
        trading_engine.config['system']['continual_learning']['enabled'] = True

    if args.no_gpu:
        trading_engine.config['system']['use_gpu'] = False

    # Perform requested action
    if args.action == 'start':
        trading_engine.start()

        # Set up asyncio event loop for async tasks
        loop = asyncio.get_event_loop()

        # Keep running until interrupted
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            # Stop the trading engine
            trading_engine.stop()

            # Close the event loop
            loop.close()

    elif args.action == 'stop':
        trading_engine.stop()

    elif args.action == 'status':
        status = trading_engine.status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
