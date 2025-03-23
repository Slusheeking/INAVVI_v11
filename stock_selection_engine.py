#!/usr/bin/env python3
"""
Unified Stock Selection Engine
==============================

This module consolidates all components of the GPU-accelerated stock selection system:

1. Core Selection System: Stock universe building and filtering
2. GPU Acceleration: Optimized data processing with NVIDIA GH200
3. WebSocket Integration: Real-time market data processing
4. Day Trading: Intraday trading strategies and execution
5. Data Helpers: Market data acquisition and preprocessing
6. Enhanced Selection: Multi-factor scoring and filtering

Components are optimized for the NVIDIA GH200 Grace Hopper Superchip architecture.
"""

import asyncio
import concurrent.futures
import contextlib
import datetime
import json
import logging
import os
import pickle
import time
from asyncio import Lock

import numpy as np
import pytz
from dotenv import load_dotenv

import redis
from config import config

# Load environment variables from .env file
load_dotenv()

# Import configuration

# Try to import GPU acceleration libraries with fallbacks
try:
    import cupy as cp
    import tensorflow as tf

    HAS_GPU_SUPPORT = True
except ImportError:
    cp = None
    tf = None
    HAS_GPU_SUPPORT = False
    logging.warning("GPU libraries not available. Using CPU fallbacks.")

# Import Prometheus client for metrics
try:
    import prometheus_client as prom

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning(
        "Prometheus client not available. Metrics will not be exposed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.environ.get(
                    "LOGS_DIR", "./logs"), "stock_selection_engine.log",
            ),
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("stock_selection_engine")

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # GPU metrics
    GPU_MEMORY_USAGE = prom.Gauge(
        "stock_selection_gpu_memory_usage_bytes",
        "GPU memory usage in bytes",
        ["device"],
    )

    GPU_UTILIZATION = prom.Gauge(
        "stock_selection_gpu_utilization_percent",
        "GPU utilization percentage",
        ["device"],
    )

    # Stock selection metrics
    UNIVERSE_SIZE = prom.Gauge(
        "stock_selection_universe_size", "Number of stocks in the universe",
    )

    WATCHLIST_SIZE = prom.Gauge(
        "stock_selection_watchlist_size", "Number of stocks in the watchlist",
    )

    FOCUSED_LIST_SIZE = prom.Gauge(
        "stock_selection_focused_list_size", "Number of stocks in the focused list",
    )

    # Performance metrics
    BATCH_PROCESSING_TIME = prom.Histogram(
        "stock_selection_batch_processing_seconds",
        "Time spent processing batches of stocks",
        ["operation"],
    )

    STOCK_SCORES = prom.Gauge(
        "stock_selection_stock_score", "Stock selection score", [
            "ticker", "score_type"],
    )

    # WebSocket metrics
    WEBSOCKET_MESSAGES = prom.Counter(
        "stock_selection_websocket_messages_total",
        "Number of WebSocket messages received",
        ["message_type"],
    )

    WEBSOCKET_ERRORS = prom.Counter(
        "stock_selection_websocket_errors_total",
        "Number of WebSocket errors",
        ["error_type"],
    )

    # Day trading metrics
    TRADING_OPPORTUNITIES = prom.Counter(
        "stock_selection_trading_opportunities_total",
        "Number of trading opportunities detected",
        ["signal_type"],
    )

    ACTIVE_POSITIONS = prom.Gauge(
        "stock_selection_active_positions", "Number of active trading positions",
    )

    POSITION_PNL = prom.Gauge(
        "stock_selection_position_pnl",
        "Profit and loss for trading positions",
        ["ticker"],
    )

    # Additional market quality metrics
    STOCK_SPREAD = prom.Gauge(
        "stock_selection_spread_percent",
        "Bid-ask spread as percentage of mid price",
        ["ticker"],
    )

    STOCK_VOLATILITY = prom.Gauge(
        "stock_selection_volatility_percent",
        "Stock price volatility (high-low range as percentage of open)",
        ["ticker"],
    )

    logger.info("Prometheus metrics initialized for Stock Selection Engine")

# Environment variables and constants
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
UNUSUAL_WHALES_API_KEY = os.environ.get("UNUSUAL_WHALES_API_KEY", "")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6380))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

# Technical analysis constants
SMA_PERIODS = [5, 10, 20, 50, 200]
RSI_PERIOD = 14
VOLUME_AVG_PERIODS = [5, 20]
ATR_PERIOD = 14

# =============================================================================
# SECTION 1: GPU Utilities
# =============================================================================


class GH200Accelerator:
    """
    NVIDIA GH200 Grace Hopper Superchip accelerator manager.
    Provides GPU memory management and optimization for the GH200 architecture.
    """

    def __init__(self) -> None:
        """Initialize the GH200 accelerator"""
        self.has_tensorflow_gpu = False
        self.has_cupy_gpu = False
        self.has_tensorrt = False
        self.device_name = "CPU (No GPU Available)"
        self.compute_capability = None
        self.total_memory_gb = 0
        self.tensorrt_config = None

        # Check for TensorFlow GPU
        if tf is not None:
            self._check_tensorflow_gpu()
            # Check for TensorRT
            self._check_tensorrt()

        # Check for CuPy GPU
        if cp is not None:
            self._check_cupy_gpu()

        # Configure optimal settings based on detected hardware
        self._configure_optimal_settings()

        logger.info(
            f"GPU Acceleration: TensorFlow={self.has_tensorflow_gpu}, CuPy={self.has_cupy_gpu}, TensorRT={self.has_tensorrt}",
        )
        logger.info(f"Using device: {self.device_name}")
        if self.compute_capability:
            logger.info(f"Compute Capability: {self.compute_capability}")
        if self.total_memory_gb > 0:
            logger.info(f"Total GPU Memory: {self.total_memory_gb:.2f} GB")

    def _check_tensorflow_gpu(self) -> None:
        """Check if TensorFlow can access GPUs"""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                self.has_tensorflow_gpu = True

                # Configure memory growth to prevent OOM errors
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu.name}")
                    except Exception as e:
                        logger.warning(f"Error setting memory growth: {e}")

                # Look for GH200 specifically
                from tensorflow.python.client import device_lib

                devices = device_lib.list_local_devices()
                for device in devices:
                    if device.device_type == "GPU":
                        self.device_name = f"TensorFlow: {device.physical_device_desc}"
                        # Extract compute capability if available
                        if "compute capability" in device.physical_device_desc:
                            import re

                            match = re.search(
                                r"compute capability: (\d+\.\d+)",
                                device.physical_device_desc,
                            )
                            if match:
                                self.compute_capability = match.group(1)
                        # Check specifically for GH200
                        if "GH200" in device.physical_device_desc:
                            logger.info(
                                "NVIDIA GH200 Grace Hopper Superchip detected")
                            return

                # If no GH200 found, use the first GPU
                self.device_name = f"TensorFlow: {gpus[0].name}"
            else:
                self.has_tensorflow_gpu = False
        except Exception as e:
            logger.exception(f"Error checking TensorFlow GPU: {e}")
            self.has_tensorflow_gpu = False

    def _check_tensorrt(self) -> bool | None:
        """Check if TensorRT is available"""
        try:
            if not self.has_tensorflow_gpu:
                return False

            # Check if TensorRT module is available
            if hasattr(tf, "experimental") and hasattr(tf.experimental, "tensorrt"):
                try:
                    from tensorflow.python.compiler.tensorrt import trt_convert as trt

                    # Create a simple TensorRT converter to verify functionality
                    trt.DEFAULT_TRT_CONVERSION_PARAMS
                    self.has_tensorrt = True

                    # Store TensorRT configuration
                    self.tensorrt_config = {
                        "precision_mode": os.environ.get(
                            "TENSORRT_PRECISION_MODE", "FP16",
                        ),
                        "max_workspace_size_bytes": 8000000000,  # 8GB
                        "maximum_cached_engines": 100,
                    }

                    logger.info("TensorRT is available and configured")
                    return True
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"TensorRT module found but not functional: {e}")
                    return False
            return False
        except Exception as e:
            logger.exception(f"Error checking TensorRT: {e}")
            return False

    def _check_cupy_gpu(self) -> None:
        """Check if CuPy can access GPUs"""
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                self.has_cupy_gpu = True

                # Look for GH200 specifically
                for i in range(num_gpus):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    name = props["name"].decode()

                    # Store compute capability
                    if not self.compute_capability:
                        major = props["major"]
                        minor = props["minor"]
                        self.compute_capability = f"{major}.{minor}"

                    # Store total memory
                    total_memory = props["totalGlobalMem"]
                    self.total_memory_gb = total_memory / (1024**3)

                    if "GH200" in name:
                        cp.cuda.Device(i).use()
                        self.device_name = f"CuPy: {name}"
                        logger.info(
                            f"Using NVIDIA GH200 with CuPy (Device {i})")
                        return

                # If no GH200 found, use the first GPU
                cp.cuda.Device(0).use()
                props = cp.cuda.runtime.getDeviceProperties(0)
                self.device_name = f"CuPy: {props['name'].decode()}"
                logger.info(
                    f"Using {props['name'].decode()} with CuPy (Device 0)")
            else:
                self.has_cupy_gpu = False
        except Exception as e:
            logger.exception(f"Error checking CuPy GPU: {e}")
            self.has_cupy_gpu = False

    def _configure_optimal_settings(self) -> None:
        """Configure optimal settings based on detected hardware"""
        if not (self.has_tensorflow_gpu or self.has_cupy_gpu):
            return

        try:
            # Configure CuPy for optimal performance
            if self.has_cupy_gpu:
                # Set pinned memory for faster host-device transfers
                cp.cuda.set_pinned_memory_allocator()

                # Configure CuPy to use the fastest algorithms
                if hasattr(cp.cuda, "set_allocator"):
                    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                    logger.info(
                        "Configured CuPy memory pool for better performance")

            # Configure TensorFlow for optimal performance
            if self.has_tensorflow_gpu:
                # Use mixed precision for better performance on GH200
                if hasattr(tf, "keras") and hasattr(tf.keras, "mixed_precision"):
                    try:
                        policy = tf.keras.mixed_precision.Policy(
                            "mixed_float16")
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info(
                            "Enabled mixed precision (float16) for TensorFlow")
                    except Exception as e:
                        logger.warning(
                            f"Could not set mixed precision policy: {e}")

                # Configure TensorFlow for optimal performance
                if self.compute_capability and float(self.compute_capability) >= 8.0:
                    # For Ampere (8.x) and Hopper (9.x) architectures
                    logger.info(
                        f"Optimizing for compute capability {self.compute_capability}",
                    )

                    # Enable TF32 for better performance
                    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

                    # Enable tensor cores
                    os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"
                    os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"

        except Exception as e:
            logger.exception(f"Error configuring optimal settings: {e}")

    def get_memory_info(self):
        """Get current GPU memory usage information"""
        if not self.has_cupy_gpu:
            return {"error": "No GPU available"}

        try:
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free

            # Record metrics in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    # Convert name to string if it's bytes
                    device_name = self.device_name
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode()

                    # Record memory usage
                    GPU_MEMORY_USAGE.labels(device=device_name).set(used)

                    # Get GPU utilization if possible
                    try:
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(
                            0)  # Use first GPU
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(
                            handle)
                        GPU_UTILIZATION.labels(
                            device=device_name).set(utilization.gpu)
                    except Exception as pynvml_error:
                        logger.debug(
                            f"Could not get GPU utilization: {pynvml_error}")
                except Exception as prom_e:
                    logger.warning(
                        f"Error recording GPU metrics in Prometheus: {prom_e}",
                    )

            return {
                "total_mb": total / (1024 * 1024),
                "total_gb": total / (1024 * 1024 * 1024),
                "used_mb": used / (1024 * 1024),
                "used_gb": used / (1024 * 1024 * 1024),
                "free_mb": free / (1024 * 1024),
                "free_gb": free / (1024 * 1024 * 1024),
                "utilization_pct": (used / total) * 100,
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_memory(self):
        """Clear GPU memory to prevent fragmentation"""
        success = False

        if self.has_cupy_gpu:
            try:
                # Clear CuPy memory pool
                cp.get_default_memory_pool().free_all_blocks()
                # Clear pinned memory pool if available
                if hasattr(cp.cuda, "pinned_memory_pool"):
                    cp.cuda.pinned_memory_pool.free_all_blocks()
                logger.info("CuPy memory pools cleared")
                success = True
            except Exception as e:
                logger.exception(f"Error clearing CuPy memory: {e}")

        if self.has_tensorflow_gpu:
            try:
                # Clear TensorFlow session
                tf.keras.backend.clear_session()

                # Force garbage collection
                import gc

                gc.collect()

                logger.info("TensorFlow session cleared and garbage collected")
                success = True
            except Exception as e:
                logger.exception(f"Error clearing TensorFlow memory: {e}")

        # Log memory info after clearing
        if success:
            try:
                mem_info = self.get_memory_info()
                if "error" not in mem_info:
                    logger.info(
                        f"GPU memory after clearing: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free ({mem_info['utilization_pct']:.1f}%)",
                    )
            except Exception:
                pass

        return success


def optimize_for_gh200() -> None:
    """Apply GH200-specific optimizations via environment variables"""
    # TensorFlow optimizations
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "16"  # Optimize for GH200
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"  # Use unified memory
    os.environ["TF_ENABLE_NUMA_AWARE_ALLOCATORS"] = "1"  # For multi-GPU

    # Enable XLA JIT compilation for better performance
    os.environ["TF_XLA_FLAGS"] = (
        "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
    )

    # Memory limits and thread configuration
    os.environ["TF_CUDA_HOST_MEM_LIMIT_IN_MB"] = "16000"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "4"  # Optimal for GH200

    # Prevent graph errors but allow some JIT compilation
    os.environ["TF_FUNCTION_JIT_COMPILE_DEFAULT"] = "0"

    # Enable deterministic operations for better stability
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Enable tensor op math for better performance
    os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"
    os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"
    os.environ["TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32"] = "1"

    # Enable persistent space for batchnorm operations
    os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"

    # Enable client streaming for better performance
    os.environ["TF_KERAS_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "True"

    # For CPU side of GH200
    os.environ["GOMP_CPU_AFFINITY"] = "0-15"  # Adjust based on Neoverse cores

    # GPU-direct optimizations
    os.environ["CUDA_AUTO_BOOST"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"

    # Set TensorRT precision mode
    os.environ["TENSORRT_PRECISION_MODE"] = "FP16"

    logger.info(
        "Applied GH200-specific optimizations for TensorFlow, TensorRT, and CuPy",
    )


# =============================================================================
# SECTION 2: Market Data Helpers
# =============================================================================


async def get_price_data(redis_client, polygon_client, ticker, days=5):
    """
    Get price data for a ticker from cache or API

    Args:
        redis_client: Redis client for caching
        polygon_client: Polygon API client
        ticker: Ticker symbol
        days: Number of days of data to retrieve

    Returns:
        List of price data dictionaries
    """
    cache_key = f"price_data:{ticker}:{days}"

    # Try to get from cache first
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return pickle.loads(cached_data)
            except Exception:
                pass  # Fall through to API call on error

    # Calculate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    # Format dates for API call
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")

    try:
        # Get data from Polygon API
        response = await polygon_client.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=from_date,
            to_date=to_date,
        )

        if response and "results" in response:
            # Cache the results
            if redis_client:
                redis_client.setex(
                    cache_key,
                    3600,  # Cache for 1 hour
                    pickle.dumps(response["results"]),
                )
            return response["results"]
    except Exception as e:
        logger.exception(f"Error getting price data for {ticker}: {e}")

    return []


async def get_volume_data(redis_client, polygon_client, ticker, days=20):
    """
    Get volume data for a ticker from cache or API

    Args:
        redis_client: Redis client for caching
        polygon_client: Polygon API client
        ticker: Ticker symbol
        days: Number of days of data to retrieve

    Returns:
        List of volume data values
    """
    # Use price data function but extract only volume
    price_data = await get_price_data(redis_client, polygon_client, ticker, days)

    if price_data:
        return [item.get("v", 0) for item in price_data]

    return []


async def get_ticker_details(redis_client, polygon_client, ticker):
    """
    Get ticker details from cache or API

    Args:
        redis_client: Redis client for caching
        polygon_client: Polygon API client
        ticker: Ticker symbol

    Returns:
        Dictionary of ticker details
    """
    cache_key = f"ticker_details:{ticker}"

    # Try to get from cache first
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return pickle.loads(cached_data)
            except Exception:
                pass  # Fall through to API call on error

    try:
        # Get data from Polygon API
        response = await polygon_client.get_ticker_details(ticker=ticker)

        if response and "results" in response:
            # Cache the results
            if redis_client:
                redis_client.setex(
                    cache_key,
                    86400,  # Cache for 1 day
                    pickle.dumps(response["results"]),
                )
            return response["results"]
    except Exception as e:
        logger.exception(f"Error getting ticker details for {ticker}: {e}")

    return {}


async def get_options_data(redis_client, unusual_whales_client, ticker):
    """
    Get options flow data for a ticker from cache or API

    Args:
        redis_client: Redis client for caching
        unusual_whales_client: Unusual Whales API client
        ticker: Ticker symbol

    Returns:
        Dictionary of options flow data
    """
    if not unusual_whales_client:
        return {}

    cache_key = f"options_flow:{ticker}"

    # Try to get from cache first
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return pickle.loads(cached_data)
            except Exception:
                pass  # Fall through to API call on error

    try:
        # Get data from Unusual Whales API
        response = await unusual_whales_client.get_flow_alerts(ticker=ticker)

        if response and "data" in response:
            # Cache the results
            if redis_client:
                redis_client.setex(
                    cache_key,
                    300,  # Cache for 5 minutes
                    pickle.dumps(response["data"]),
                )
            return response["data"]
    except Exception as e:
        logger.exception(f"Error getting options data for {ticker}: {e}")

    return {}


# =============================================================================
# SECTION 3: Stock Selection Core
# =============================================================================


class StockSelectionCore:
    """
    Core stock selection system for identifying and filtering tradable securities.
    Provides basic functionality that GPU-accelerated version builds upon.
    """

    def __init__(self) -> None:
        """Initialize the stock selection core"""
        self.full_universe = set()  # Full universe of tradable stocks
        self.active_watchlist = set()  # Active watchlist for monitoring
        self.focused_list = set()  # Focused list for trading

        self.config = {
            "universe_size": 2000,
            "watchlist_size": 100,
            "focused_list_size": 20,
            "min_price": 5.0,
            "max_price": 500.0,
            "min_volume": 500000,
            "min_relative_volume": 1.5,
            "min_atr_percent": 1.0,
            "refresh_interval": 900,  # 15 minutes
            "cache_expiry": 300,  # 5 minutes
            "weights": {
                "volume": 0.30,
                "volatility": 0.25,
                "momentum": 0.25,
                "options": 0.20,
            },
        }

        self.logger = logging.getLogger("stock_selection_core")

    async def build_initial_universe(self) -> None:
        """Build initial universe of tradable stocks"""
        self.logger.info("Building initial universe of tradable stocks")

        try:
            # Use default tickers from configuration
            default_tickers = self.config["stock_selection"]["universe"][
                "default_tickers"
            ]
            self.logger.info(
                f"Using default tickers from configuration: {default_tickers}",
            )

            self.full_universe = set(default_tickers)

            self.logger.info(
                f"Initial universe built with {len(self.full_universe)} stocks",
            )

        except Exception as e:
            self.logger.exception(f"Error building initial universe: {e!s}")
            raise

    async def refresh_watchlist(self) -> None:
        """Refresh the active watchlist based on various metrics"""
        self.logger.info("Refreshing active watchlist")

        try:
            # In the basic version, we just take the top N stocks from the universe
            # based on a simple alphabetical sort (for demonstration)
            sorted_universe = sorted(self.full_universe)
            watchlist_size = min(
                self.config["watchlist_size"], len(sorted_universe))

            self.active_watchlist = set(sorted_universe[:watchlist_size])

            self.logger.info(
                f"Watchlist refreshed with {len(self.active_watchlist)} stocks",
            )

        except Exception as e:
            self.logger.exception(f"Error refreshing watchlist: {e!s}")

    async def update_focused_list(self) -> None:
        """Update the focused list for trading"""
        self.logger.info("Updating focused list")

        try:
            # In the basic version, we just take the top N stocks from the watchlist
            # based on a simple alphabetical sort (for demonstration)
            sorted_watchlist = sorted(self.active_watchlist)
            focused_size = min(
                self.config["focused_list_size"], len(sorted_watchlist))

            self.focused_list = set(sorted_watchlist[:focused_size])

            self.logger.info(
                f"Focused list updated with {len(self.focused_list)} stocks",
            )

        except Exception as e:
            self.logger.exception(f"Error updating focused list: {e!s}")

    def calculate_score(self, ticker_data):
        """
        Calculate score for a ticker based on various metrics

        Args:
            ticker_data: Dictionary of ticker metrics

        Returns:
            Float score between 0 and 1
        """
        try:
            # Basic metrics to consider
            price = ticker_data.get("price", 0)
            volume = ticker_data.get("volume", 0)
            avg_volume = ticker_data.get("avg_volume", 1)
            volatility = ticker_data.get("volatility", 0)
            momentum = ticker_data.get("momentum", 0)
            options_flow = ticker_data.get("options_flow", 0)

            # Skip if key metrics are missing or invalid
            if price <= 0 or volume <= 0 or avg_volume <= 0:
                return 0.0

            # Calculate component scores (each 0-1)
            volume_score = min(1.0, volume / (3 * avg_volume))

            if "volatility" in ticker_data:
                # 5% daily volatility = 1.0
                volatility_score = min(1.0, volatility / 0.05)
            else:
                volatility_score = 0.5  # Default if not available

            if "momentum" in ticker_data:
                momentum_score = (momentum + 0.05) / \
                    0.1  # Scale -5% to +5% to 0-1
                momentum_score = max(0, min(1, momentum_score))
            else:
                momentum_score = 0.5  # Default if not available

            if "options_flow" in ticker_data:
                # 10+ unusual options = 1.0
                options_score = min(1.0, options_flow / 10.0)
            else:
                options_score = 0.0  # Default if not available

            # Combine scores using configured weights
            weights = self.config["weights"]
            return (
                weights["volume"] * volume_score
                + weights["volatility"] * volatility_score
                + weights["momentum"] * momentum_score
                + weights["options"] * options_score
            )

        except Exception as e:
            self.logger.exception(f"Error calculating score: {e!s}")
            return 0.0

    async def start(self) -> None:
        """Start the stock selection system"""
        self.logger.info("Starting stock selection system")

        # Build initial universe
        await self.build_initial_universe()

        # Refresh watchlist
        await self.refresh_watchlist()

        # Update focused list
        await self.update_focused_list()

        self.logger.info("Stock selection system started")

    async def stop(self) -> None:
        """Stop the stock selection system"""
        self.logger.info("Stopping stock selection system")
        # No special cleanup needed in the basic version
        self.logger.info("Stock selection system stopped")


# =============================================================================
# SECTION 4: GPU-Optimized Stock Selection
# =============================================================================


class GPUStockSelectionSystem(StockSelectionCore):
    """
    GPU-Optimized Stock Selection System for NVIDIA GH200

    Extends the base stock selection system with GPU acceleration
    for faster processing of large datasets and real-time filtering.
    """

    def __init__(
        self,
        redis_client=None,
        polygon_api_client=None,
        polygon_websocket_client=None,
        unusual_whales_client=None,
    ) -> None:
        """
        Initialize the GPU-optimized stock selection system

        Args:
            redis_client: Redis client for caching
            polygon_api_client: Polygon API client
            polygon_websocket_client: Polygon WebSocket client
            unusual_whales_client: Unusual Whales API client
        """
        # Initialize parent class
        super().__init__()

        # Store clients
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

        # Check if GPU is available
        self.gpu_available = (
            self.gh200_accelerator.has_tensorflow_gpu
            or self.gh200_accelerator.has_cupy_gpu
        )

        if not self.gpu_available and USE_GPU:
            logger.warning(
                "GPU acceleration requested but not available. Using CPU fallback.",
            )

        # Performance optimization with thread pools
        logger.info(
            f"Initializing thread pool with {min(os.cpu_count(), 20)} workers")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

        # Market data cache
        self.cache = {
            "market_data": {},
            "options_data": {},
            "technical_data": {},
            "last_refresh": {},
        }

        # Update configuration with more detailed settings
        self.config.update(
            {
                "default_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                "universe_size": 2000,
                "watchlist_size": 100,
                "focused_list_size": 30,
                "min_price": 3.0,
                "max_price": 100.0,
                "min_volume": 500000,
                "min_relative_volume": 1.5,
                "min_atr_percent": 1.0,
                "refresh_interval": 900,
                "cache_expiry": 300,
                "weights": {
                    "volume": 0.30,
                    "volatility": 0.25,
                    "momentum": 0.25,
                    "options": 0.20,
                },
                "batch_size": 1024,
                "max_workers": min(os.cpu_count(), 8),
                "day_trading": {
                    "enabled": True,
                    "max_total_position": 5000,
                    "max_positions": 5,
                    "target_profit_percent": 1.0,
                    "stop_loss_percent": 0.5,
                    "no_overnight_positions": True,
                    "min_liquidity_score": 70,
                },
            },
        )

        # Internal state
        self.running = False
        self.tasks = {}
        self.day_trading_candidates = set()

        # Additional locks for thread safety
        self._universe_lock = Lock()
        self._watchlist_lock = Lock()
        self._focused_lock = Lock()

        # Shared memory for inter-process communication on GPU
        self.shared_data = {}

        # Log GPU device information
        logger.info(f"Using GPU device: {self.gh200_accelerator.device_name}")

        logger.info("GPU-Optimized Stock Selection System initialized")

    async def _is_market_open(self) -> bool | None:
        """Check if market is currently open"""
        try:
            # Get current time (Eastern)
            now = datetime.datetime.now(pytz.timezone("US/Eastern"))

            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            # Check market hours (9:30 AM - 4:00 PM ET)
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)

            # If market is closed, check WebSocket connection status
            if not (market_open <= now <= market_close):
                if (
                    hasattr(self, "polygon_ws")
                    and self.polygon_ws
                    and hasattr(self.polygon_ws, "is_connected")
                    and self.polygon_ws.is_connected()
                ):
                    logger.info(
                        "Market closed but WebSocket connected - using extended hours data",
                    )
                    return True
                return False

            return True
        except Exception as e:
            logger.exception(f"Error checking market hours: {e}")
            return False

    async def build_initial_universe(self) -> None:
        """Build initial universe of tradable stocks with GPU acceleration"""
        async with self._universe_lock:  # Ensure thread safety
            logger.info("Building initial universe with GPU acceleration")

            try:
                # Get all active tickers from Polygon
                if self.polygon_api:
                    # Using the Polygon client to get ticker data
                    response = await self.polygon_api._make_request(
                        "v3/reference/tickers",
                        {"market": "stocks", "active": "true", "limit": 1000},
                    )

                    if response and "results" in response:
                        tickers = [
                            item["ticker"]
                            for item in response["results"]
                            if item.get("ticker")
                        ]
                        self.full_universe = set(tickers)
                    else:
                        # Fallback to basic initialization
                        await super().build_initial_universe()
                else:
                    # Fallback to basic initialization
                    await super().build_initial_universe()

                if self.gpu_available and self.full_universe:
                    # Apply GPU-accelerated filtering
                    logger.info("Applying GPU-accelerated filtering")

                    # Process tickers in batches to avoid memory issues
                    tickers = list(self.full_universe)
                    batch_size = self.config["batch_size"]
                    filtered_universe = set()

                    for i in range(0, len(tickers), batch_size):
                        batch = tickers[i: i + batch_size]
                        filtered_batch = await self._apply_gpu_filters(batch)
                        filtered_universe.update(filtered_batch)

                    self.full_universe = filtered_universe
                    logger.info(
                        f"GPU-accelerated universe built with {len(self.full_universe)} stocks",
                    )
                else:
                    logger.warning(
                        "GPU not available or empty universe, using basic filtering",
                    )

            except Exception as e:
                logger.exception(
                    f"Error building GPU-accelerated universe: {e!s}")
                raise

    async def _apply_gpu_filters(self, tickers):
        """
        Apply GPU-accelerated filters to a batch of tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Set of filtered ticker symbols
        """
        try:
            # Get market data for the batch
            data = []
            for ticker in tickers:
                price_data = await get_price_data(self.redis, self.polygon_api, ticker)
                volume_data = await get_volume_data(
                    self.redis, self.polygon_api, ticker,
                )

                if price_data and volume_data:
                    # Use the most recent data point
                    data.append(
                        {
                            "ticker": ticker,
                            "price": price_data[0].get("c", 0) if price_data else 0,
                            "volume": volume_data[0] if volume_data else 0,
                        },
                    )

            if not data:
                return set()

            if self.gpu_available and cp is not None:
                # Log GPU memory before processing
                if hasattr(self.gh200_accelerator, "get_memory_info"):
                    mem_info = self.gh200_accelerator.get_memory_info()
                    if "error" not in mem_info:
                        logger.debug(
                            f"GPU memory before filtering: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free",
                        )

                # Prepare arrays for GPU processing with proper data types
                prices = np.array([d["price"] for d in data], dtype=np.float32)
                volumes = np.array([d["volume"]
                                   for d in data], dtype=np.float32)

                # Move data to GPU with pinned memory for faster transfer
                with cp.cuda.Stream() as stream:
                    # Use pinned memory for faster host-to-device transfer
                    cp_prices = cp.asarray(prices, stream=stream)
                    cp_volumes = cp.asarray(volumes, stream=stream)

                    # Apply filters on GPU with optimized operations
                    # Use element-wise operations which are highly optimized on GPU
                    price_mask = cp.logical_and(
                        cp_prices >= self.config["min_price"],
                        cp_prices <= self.config["max_price"],
                    )
                    volume_mask = cp_volumes >= self.config["min_volume"]

                    # Combined mask using logical AND
                    combined_mask = cp.logical_and(price_mask, volume_mask)

                    # Get indices of eligible tickers
                    # Use CUB library for efficient parallel reduction
                    eligible_indices = cp.where(combined_mask)[0]

                    # Transfer results back to CPU
                    eligible_indices_cpu = cp.asnumpy(eligible_indices)

                # Get eligible tickers
                eligible_tickers = {data[idx]["ticker"]
                                    for idx in eligible_indices_cpu}

                # Log filtering results
                logger.info(
                    f"GPU filtering: {len(eligible_tickers)} of {len(data)} tickers passed filters",
                )

                # Clean up GPU memory explicitly
                del (
                    cp_prices,
                    cp_volumes,
                    price_mask,
                    volume_mask,
                    combined_mask,
                    eligible_indices,
                )

                # Force memory cleanup
                if hasattr(self.gh200_accelerator, "clear_memory"):
                    self.gh200_accelerator.clear_memory()

                # Log GPU memory after processing
                if hasattr(self.gh200_accelerator, "get_memory_info"):
                    mem_info = self.gh200_accelerator.get_memory_info()
                    if "error" not in mem_info:
                        logger.debug(
                            f"GPU memory after filtering: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free",
                        )

                return eligible_tickers
            # Fallback to CPU filtering with vectorized NumPy operations
            prices = np.array([d["price"] for d in data], dtype=np.float32)
            volumes = np.array([d["volume"] for d in data], dtype=np.float32)

            # Use NumPy's vectorized operations for better CPU performance
            price_mask = np.logical_and(
                prices >= self.config["min_price"],
                prices <= self.config["max_price"],
            )
            volume_mask = volumes >= self.config["min_volume"]
            combined_mask = np.logical_and(price_mask, volume_mask)
            eligible_indices = np.where(combined_mask)[0]

            # Get eligible tickers
            eligible_tickers = {data[idx]["ticker"]
                                for idx in eligible_indices}

            logger.info(
                f"CPU filtering: {len(eligible_tickers)} of {len(data)} tickers passed filters",
            )
            return eligible_tickers

        except Exception as e:
            logger.exception(f"Error in GPU filtering: {e!s}")
            # Fallback to returning all tickers
            return set(tickers)

    async def refresh_watchlist(self) -> None:
        """Refresh watchlist with GPU-accelerated scoring"""
        async with self._watchlist_lock:
            logger.info("Refreshing watchlist with GPU acceleration")

            try:
                if not self.full_universe:
                    logger.warning("Empty universe, cannot refresh watchlist")
                    return

                # Get all tickers
                tickers = list(self.full_universe)

                # Start timing for Prometheus metrics
                start_time = time.time()

                # Process in batches
                batch_size = self.config["batch_size"]
                all_scores = []

                for i in range(0, len(tickers), batch_size):
                    batch = tickers[i: i + batch_size]
                    batch_scores = await self._calculate_batch_scores(batch)
                    all_scores.extend(batch_scores)

                # Sort by score (descending)
                all_scores.sort(key=lambda x: x[1], reverse=True)

                # Take top N for watchlist
                watchlist_size = min(
                    self.config["watchlist_size"], len(all_scores))
                self.active_watchlist = {
                    ticker for ticker, _ in all_scores[:watchlist_size]
                }

                # Record metrics in Prometheus if available
                if PROMETHEUS_AVAILABLE:
                    try:
                        # Record processing time
                        processing_time = time.time() - start_time
                        BATCH_PROCESSING_TIME.labels(
                            operation="refresh_watchlist",
                        ).observe(processing_time)

                        # Record universe and watchlist sizes
                        UNIVERSE_SIZE.set(len(self.full_universe))
                        WATCHLIST_SIZE.set(len(self.active_watchlist))

                        # Record top stock scores
                        # Record top 10 scores
                        for ticker, score in all_scores[:10]:
                            STOCK_SCORES.labels(
                                ticker=ticker, score_type="watchlist",
                            ).set(score)
                    except Exception as prom_e:
                        logger.warning(
                            f"Error recording metrics in Prometheus: {prom_e}",
                        )

                logger.info(
                    f"Watchlist refreshed with {len(self.active_watchlist)} stocks",
                )

            except Exception as e:
                logger.exception(f"Error refreshing watchlist: {e!s}")

    async def _calculate_batch_scores(self, tickers):
        """
        Calculate scores for a batch of tickers with GPU acceleration

        Args:
            tickers: List of ticker symbols

        Returns:
            List of (ticker, score) tuples
        """
        try:
            # Gather data for all tickers
            ticker_data = {}
            for ticker in tickers:
                # Get price data
                price_data = await get_price_data(self.redis, self.polygon_api, ticker)

                if not price_data:
                    continue

                # Calculate basic metrics
                current_price = price_data[0].get("c", 0) if price_data else 0
                volume = price_data[0].get("v", 0) if price_data else 0

                # Calculate average volume (5-day)
                volumes = [d.get("v", 0) for d in price_data[:5]]
                avg_volume = sum(volumes) / len(volumes) if volumes else 0

                # Calculate volatility (ATR)
                if len(price_data) >= 14:
                    highs = [d.get("h", 0) for d in price_data[:14]]
                    lows = [d.get("l", 0) for d in price_data[:14]]
                    closes = [d.get("c", 0) for d in price_data[:14]]

                    # True Range calculations
                    tr_values = []
                    for i in range(1, 14):
                        hl = highs[i] - lows[i]
                        hpc = abs(highs[i] - closes[i - 1])
                        lpc = abs(lows[i] - closes[i - 1])
                        tr = max(hl, hpc, lpc)
                        tr_values.append(tr)

                    # Average True Range
                    atr = sum(tr_values) / len(tr_values) if tr_values else 0
                    atr_percent = (
                        (atr / current_price) * 100 if current_price > 0 else 0
                    )
                else:
                    atr_percent = 0

                # Calculate momentum (5-day return)
                if len(price_data) >= 5:
                    current = price_data[0].get("c", 0)
                    previous = price_data[4].get("c", 0)
                    momentum = (current - previous) / \
                        previous if previous > 0 else 0
                else:
                    momentum = 0

                # Get options data if available
                options_flow = 0
                if self.unusual_whales:
                    options_data = await get_options_data(
                        self.redis, self.unusual_whales, ticker,
                    )
                    options_flow = len(options_data) if options_data else 0

                # Store data
                ticker_data[ticker] = {
                    "price": current_price,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "volatility": atr_percent,
                    "momentum": momentum,
                    "options_flow": options_flow,
                }

            # Calculate scores
            if self.gpu_available and len(ticker_data) > 0:
                # Log GPU memory before processing
                if hasattr(self.gh200_accelerator, "get_memory_info"):
                    mem_info = self.gh200_accelerator.get_memory_info()
                    if "error" not in mem_info:
                        logger.debug(
                            f"GPU memory before scoring: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free",
                        )

                # Determine which GPU acceleration to use
                if (
                    self.gh200_accelerator.has_tensorflow_gpu
                    and tf is not None
                    and len(ticker_data) >= 100
                ):
                    # For larger batches, use TensorFlow with TensorRT optimization
                    logger.info(
                        f"Using TensorFlow for batch scoring of {len(ticker_data)} tickers",
                    )

                    # Prepare data
                    tickers_list = list(ticker_data.keys())

                    # Create input tensors
                    input_data = {
                        "price": np.array(
                            [ticker_data[t]["price"] for t in tickers_list],
                            dtype=np.float32,
                        ),
                        "volume": np.array(
                            [ticker_data[t]["volume"] for t in tickers_list],
                            dtype=np.float32,
                        ),
                        "avg_volume": np.array(
                            [ticker_data[t]["avg_volume"]
                                for t in tickers_list],
                            dtype=np.float32,
                        ),
                        "volatility": np.array(
                            [ticker_data[t]["volatility"]
                                for t in tickers_list],
                            dtype=np.float32,
                        ),
                        "momentum": np.array(
                            [ticker_data[t]["momentum"] for t in tickers_list],
                            dtype=np.float32,
                        ),
                        "options_flow": np.array(
                            [ticker_data[t]["options_flow"]
                                for t in tickers_list],
                            dtype=np.float32,
                        ),
                    }

                    # Get weights from config
                    weights = self.config["weights"]

                    # Create a simple TensorFlow model for scoring
                    # Use XLA compilation for better performance
                    @tf.function(jit_compile=True)
                    def calculate_scores(inputs, weights):
                        # Calculate component scores
                        volume_scores = tf.minimum(
                            1.0, inputs["volume"] / (3 * inputs["avg_volume"]),
                        )
                        volatility_scores = tf.minimum(
                            1.0, inputs["volatility"] / 5.0,
                        )  # 5% volatility = 1.0

                        # Momentum from -5% to +5% scaled to 0-1
                        momentum_scores = (inputs["momentum"] + 0.05) / 0.1
                        momentum_scores = tf.maximum(
                            0.0, tf.minimum(1.0, momentum_scores),
                        )

                        # Options flow score (10+ unusual options = 1.0)
                        options_scores = tf.minimum(
                            1.0, inputs["options_flow"] / 10.0)

                        # Final scores with weights
                        return (
                            weights["volume"] * volume_scores
                            + weights["volatility"] * volatility_scores
                            + weights["momentum"] * momentum_scores
                            + weights["options"] * options_scores
                        )

                    try:
                        # Run the calculation with TensorFlow
                        with tf.device("/GPU:0"):
                            final_scores = calculate_scores(
                                input_data, weights).numpy()

                        # Create result tuples
                        result = [
                            (tickers_list[i], float(final_scores[i]))
                            for i in range(len(tickers_list))
                        ]

                        # Clean up TensorFlow memory
                        tf.keras.backend.clear_session()

                        logger.info(
                            f"TensorFlow scoring completed for {len(result)} tickers",
                        )
                        return result

                    except Exception as tf_error:
                        logger.warning(
                            f"TensorFlow scoring failed, falling back to CuPy: {tf_error}",
                        )
                        # Fall through to CuPy implementation

                # Use CuPy for GPU acceleration (either as primary or fallback)
                if self.gh200_accelerator.has_cupy_gpu and cp is not None:
                    logger.info(
                        f"Using CuPy for batch scoring of {len(ticker_data)} tickers",
                    )

                    # Prepare arrays for GPU processing
                    tickers_list = list(ticker_data.keys())

                    # Use streams for asynchronous operations
                    with cp.cuda.Stream() as stream:
                        # Move data to GPU with pinned memory for faster transfer
                        cp_prices = cp.asarray(
                            np.array(
                                [ticker_data[t]["price"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )
                        cp_volumes = cp.asarray(
                            np.array(
                                [ticker_data[t]["volume"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )
                        cp_avg_volumes = cp.asarray(
                            np.array(
                                [ticker_data[t]["avg_volume"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )
                        cp_volatilities = cp.asarray(
                            np.array(
                                [ticker_data[t]["volatility"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )
                        cp_momentums = cp.asarray(
                            np.array(
                                [ticker_data[t]["momentum"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )
                        cp_options_flows = cp.asarray(
                            np.array(
                                [ticker_data[t]["options_flow"]
                                    for t in tickers_list],
                                dtype=np.float32,
                            ),
                            stream=stream,
                        )

                        # Get weights from config
                        weights = self.config["weights"]
                        volume_weight = weights["volume"]
                        volatility_weight = weights["volatility"]
                        momentum_weight = weights["momentum"]
                        options_weight = weights["options"]

                        # Calculate component scores with optimized operations
                        # Use fused operations where possible for better performance
                        cp_volume_scores = cp.minimum(
                            1.0, cp_volumes / (3 * cp_avg_volumes),
                        )
                        cp_volatility_scores = cp.minimum(
                            1.0, cp_volatilities / 5.0)

                        # Momentum from -5% to +5% scaled to 0-1
                        cp_momentum_scores = (cp_momentums + 0.05) / 0.1
                        cp_momentum_scores = cp.maximum(
                            0, cp.minimum(1, cp_momentum_scores),
                        )

                        # Options flow score (10+ unusual options = 1.0)
                        cp_options_scores = cp.minimum(
                            1.0, cp_options_flows / 10.0)

                        # Final scores with weights - use fused multiply-add for better
                        # performance
                        cp_final_scores = cp.zeros_like(cp_volume_scores)
                        cp_final_scores = cp.add(
                            cp_final_scores,
                            volume_weight * cp_volume_scores,
                            cp_final_scores,
                        )
                        cp_final_scores = cp.add(
                            cp_final_scores,
                            volatility_weight * cp_volatility_scores,
                            cp_final_scores,
                        )
                        cp_final_scores = cp.add(
                            cp_final_scores,
                            momentum_weight * cp_momentum_scores,
                            cp_final_scores,
                        )
                        cp_final_scores = cp.add(
                            cp_final_scores,
                            options_weight * cp_options_scores,
                            cp_final_scores,
                        )

                        # Move back to CPU
                        final_scores = cp.asnumpy(cp_final_scores)

                    # Clean up GPU memory explicitly
                    del (
                        cp_prices,
                        cp_volumes,
                        cp_avg_volumes,
                        cp_volatilities,
                        cp_momentums,
                        cp_options_flows,
                    )
                    del (
                        cp_volume_scores,
                        cp_volatility_scores,
                        cp_momentum_scores,
                        cp_options_scores,
                        cp_final_scores,
                    )

                    # Force memory cleanup
                    if hasattr(self.gh200_accelerator, "clear_memory"):
                        self.gh200_accelerator.clear_memory()

                    # Log GPU memory after processing
                    if hasattr(self.gh200_accelerator, "get_memory_info"):
                        mem_info = self.gh200_accelerator.get_memory_info()
                        if "error" not in mem_info:
                            logger.debug(
                                f"GPU memory after scoring: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free",
                            )

                    # Create result tuples
                    result = [
                        (tickers_list[i], final_scores[i])
                        for i in range(len(tickers_list))
                    ]

                    logger.info(
                        f"CuPy scoring completed for {len(result)} tickers")
                    return result
            else:
                # Fallback to CPU scoring
                return [
                    (ticker, self.calculate_score(data))
                    for ticker, data in ticker_data.items()
                ]

        except Exception as e:
            logger.exception(f"Error calculating batch scores: {e!s}")
            return [(ticker, 0.0) for ticker in tickers]

    async def update_focused_list(self) -> None:
        """Update focused list with additional real-time data"""
        async with self._focused_lock:
            logger.info("Updating focused list with real-time data")

            try:
                if not self.active_watchlist:
                    logger.warning(
                        "Empty watchlist, cannot update focused list")
                    return

                # Get all tickers from watchlist
                tickers = list(self.active_watchlist)

                # Get real-time data for each ticker
                realtime_data = {}
                for ticker in tickers:
                    # Get current price
                    current_price = await self._get_current_price(ticker)

                    # Skip if price not available
                    if current_price <= 0:
                        continue

                    # Get options data if available
                    options_interest = 0
                    if self.unusual_whales:
                        options_data = await get_options_data(
                            self.redis, self.unusual_whales, ticker,
                        )
                        options_interest = len(
                            options_data) if options_data else 0

                    # Get WebSocket data if available
                    trades_per_minute = 0
                    price_volatility = 0

                    if self.polygon_ws and hasattr(self.polygon_ws, "get_ticker_stats"):
                        ws_stats = await self.polygon_ws.get_ticker_stats(ticker)
                        trades_per_minute = ws_stats.get(
                            "trades_per_minute", 0)
                        price_volatility = ws_stats.get("price_volatility", 0)

                    # Store real-time data
                    realtime_data[ticker] = {
                        "current_price": current_price,
                        "options_interest": options_interest,
                        "trades_per_minute": trades_per_minute,
                        "price_volatility": price_volatility,
                    }

                # Calculate real-time scores
                realtime_scores = []
                for ticker, data in realtime_data.items():
                    # Score based on trading activity and options interest
                    # 100+ trades/min = 1.0
                    activity_score = min(
                        1.0, data["trades_per_minute"] / 100.0)
                    # 1% volatility = 1.0
                    volatility_score = min(
                        1.0, data["price_volatility"] / 0.01)
                    # 5+ options alerts = 1.0
                    options_score = min(1.0, data["options_interest"] / 5.0)

                    # Combined score
                    rt_score = (
                        0.4 * activity_score
                        + 0.4 * volatility_score
                        + 0.2 * options_score
                    )
                    realtime_scores.append((ticker, rt_score))

                # Sort by score (descending)
                realtime_scores.sort(key=lambda x: x[1], reverse=True)

                # Take top N for focused list
                focused_size = min(
                    self.config["focused_list_size"], len(realtime_scores),
                )
                self.focused_list = {
                    ticker for ticker, _ in realtime_scores[:focused_size]
                }

                logger.info(
                    f"Focused list updated with {len(self.focused_list)} stocks",
                )

            except Exception as e:
                logger.exception(f"Error updating focused list: {e!s}")

    async def _get_current_price(self, ticker):
        """
        Get current price for a ticker

        Args:
            ticker: Ticker symbol

        Returns:
            Current price or 0 if not available
        """
        try:
            # Try to get from WebSocket first
            if self.polygon_ws and hasattr(self.polygon_ws, "get_last_price"):
                price = await self.polygon_ws.get_last_price(ticker)
                if price > 0:
                    return price

            # Fall back to API
            if self.polygon_api:
                response = await self.polygon_api._make_request(
                    f"v2/last/trade/{ticker}", {},
                )
                if response and "last" in response and "price" in response["last"]:
                    return float(response["last"]["price"])

            return 0
        except Exception as e:
            logger.exception(f"Error getting current price for {ticker}: {e}")
            return 0

    async def _create_simulation_watchlist(self, tickers) -> None:
        """Create simulation watchlist for testing"""
        self.active_watchlist = set(tickers)
        self.focused_list = set(tickers[: min(5, len(tickers))])
        logger.info(
            f"Created simulation watchlist with {len(self.active_watchlist)} stocks",
        )

    async def start(self) -> None:
        """Start the GPU-optimized stock selection system"""
        if self.running:
            logger.warning("Stock selection system already running")
            return

        self.running = True
        logger.info("Starting GPU-optimized stock selection system")

        # Check if market is open
        market_open = await self._is_market_open()

        if market_open:
            # Normal market operation
            # Build initial universe
            await self.build_initial_universe()

            # Refresh watchlist
            await self.refresh_watchlist()

            # Update focused list
            await self.update_focused_list()

            # Start background tasks
            self.tasks["refresh_watchlist"] = asyncio.create_task(
                self._periodic_refresh_watchlist(),
            )
            self.tasks["update_focused"] = asyncio.create_task(
                self._periodic_update_focused(),
            )
        else:
            # Market closed - set up simulation mode
            await self._handle_market_closed()

        logger.info("GPU-optimized stock selection system started")

    async def _periodic_refresh_watchlist(self) -> None:
        """Periodically refresh the watchlist"""
        try:
            while self.running:
                # Wait for refresh interval
                await asyncio.sleep(self.config["refresh_interval"])

                # Check if market is still open
                if not await self._is_market_open():
                    logger.info("Market closed, stopping watchlist refresh")
                    break

                # Refresh watchlist
                await self.refresh_watchlist()
        except asyncio.CancelledError:
            logger.info("Watchlist refresh task cancelled")
        except Exception as e:
            logger.exception(f"Error in watchlist refresh task: {e}")

    async def _periodic_update_focused(self) -> None:
        """Periodically update the focused list"""
        try:
            while self.running:
                # Wait for update interval (1/3 of watchlist refresh)
                await asyncio.sleep(self.config["refresh_interval"] / 3)

                # Check if market is still open
                if not await self._is_market_open():
                    logger.info("Market closed, stopping focused list updates")
                    break

                # Update focused list
                await self.update_focused_list()
        except asyncio.CancelledError:
            logger.info("Focused list update task cancelled")
        except Exception as e:
            logger.exception(f"Error in focused list update task: {e}")

    async def stop(self) -> None:
        """Stop the GPU-optimized stock selection system"""
        if not self.running:
            logger.warning("Stock selection system not running")
            return

        self.running = False
        logger.info("Stopping GPU-optimized stock selection system")

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling {name} task")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Clean up GPU resources
        if self.gpu_available:
            logger.info("Cleaning up GPU resources")

            # Log GPU memory before cleanup
            if hasattr(self.gh200_accelerator, "get_memory_info"):
                mem_info = self.gh200_accelerator.get_memory_info()
                if "error" not in mem_info:
                    logger.info(
                        f"GPU memory before cleanup: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free ({mem_info['utilization_pct']:.1f}%)",
                    )

            # Clear CuPy memory
            if (
                hasattr(self.gh200_accelerator, "has_cupy_gpu")
                and self.gh200_accelerator.has_cupy_gpu
            ):
                try:
                    if cp is not None:
                        # Clear memory pool
                        cp.get_default_memory_pool().free_all_blocks()
                        # Clear pinned memory pool if available
                        if hasattr(cp.cuda, "pinned_memory_pool"):
                            cp.cuda.pinned_memory_pool.free_all_blocks()
                        logger.info("CuPy memory pools cleared")
                except Exception as e:
                    logger.exception(f"Error clearing CuPy memory: {e}")

            # Clear TensorFlow memory
            if (
                hasattr(self.gh200_accelerator, "has_tensorflow_gpu")
                and self.gh200_accelerator.has_tensorflow_gpu
            ):
                try:
                    if tf is not None:
                        # Clear TensorFlow session
                        tf.keras.backend.clear_session()
                        logger.info("TensorFlow session cleared")
                except Exception as e:
                    logger.exception(f"Error clearing TensorFlow memory: {e}")

            # Use accelerator's clear_memory method
            try:
                self.gh200_accelerator.clear_memory()
                logger.info("GPU accelerator memory cleared")
            except Exception as e:
                logger.exception(f"Error clearing GPU accelerator memory: {e}")

            # Force garbage collection
            try:
                import gc

                gc.collect()
                logger.info("Garbage collection completed")
            except Exception as e:
                logger.exception(f"Error during garbage collection: {e}")

            # Log GPU memory after cleanup
            if hasattr(self.gh200_accelerator, "get_memory_info"):
                mem_info = self.gh200_accelerator.get_memory_info()
                if "error" not in mem_info:
                    logger.info(
                        f"GPU memory after cleanup: {mem_info['used_gb']:.2f}GB used, {mem_info['free_gb']:.2f}GB free ({mem_info['utilization_pct']:.1f}%)",
                    )

        # Shutdown thread pool
        logger.info("Shutting down thread pool")
        self.executor.shutdown(wait=False)

        logger.info("GPU-optimized stock selection system stopped successfully")

    async def _handle_market_closed(self) -> None:
        """Handle market closed scenario by entering simulation mode"""
        logger.info("Market is closed. Entering simulation mode.")

        # Send notification to frontend with additional details
        self._send_frontend_notification(
            message="Market is closed. System entering simulation mode.",
            level="warning",
            category="market_status",
            details={
                "simulation_mode": True,
                "timestamp": time.time(),
                "default_tickers": self.config.get("default_tickers", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
            }
        )

        # Update system status in Redis for frontend
        if self.redis:
            system_status = json.loads(self.redis.get(
                "frontend:system:status") or "{}")
            system_status["market_status"] = "closed"
            system_status["simulation_mode"] = True
            system_status["timestamp"] = time.time()
            self.redis.set("frontend:system:status", json.dumps(system_status))

        # For simulation, create a simulation watchlist with default tickers from config
        default_tickers = self.config.get(
            "default_tickers", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        )
        await self._create_simulation_watchlist(default_tickers)


# =============================================================================
# SECTION 5: WebSocket Core
# =============================================================================


class WebSocketCore:
    """
    Core WebSocket functionality for real-time market data.
    Provides base functionality for the enhanced WebSocket-based stock selection.
    """

    def __init__(self, api_key, redis_client=None) -> None:
        """
        Initialize WebSocket core

        Args:
            api_key: API key for authentication
            redis_client: Redis client for caching and PubSub
        """
        self.api_key = api_key
        self.redis = redis_client
        self.connection = None
        self.is_connected = False
        self.subscriptions = set()
        self.message_handlers = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2.0  # seconds
        self.heartbeat_interval = 30
        self.last_message_time = 0
        self.running = False
        self.tasks = {}
        self.logger = logging.getLogger("websocket_core")

    async def connect(self) -> None:
        """Connect to WebSocket API"""
        self.logger.info("Connecting to WebSocket API")
        # To be implemented by child classes

    async def disconnect(self) -> None:
        """Disconnect from WebSocket API"""
        self.logger.info("Disconnecting from WebSocket API")
        # To be implemented by child classes

    async def subscribe(self, channels) -> None:
        """
        Subscribe to WebSocket channels

        Args:
            channels: List of channels to subscribe to
        """
        self.logger.info(f"Subscribing to channels: {channels}")
        # To be implemented by child classes

    async def unsubscribe(self, channels) -> None:
        """
        Unsubscribe from WebSocket channels

        Args:
            channels: List of channels to unsubscribe from
        """
        self.logger.info(f"Unsubscribing from channels: {channels}")
        # To be implemented by child classes

    async def _process_message(self, message) -> None:
        """
        Process incoming WebSocket message

        Args:
            message: WebSocket message
        """
        # To be implemented by child classes

    async def _heartbeat(self) -> None:
        """Send heartbeat to keep connection alive"""
        # To be implemented by child classes

    async def _reconnect(self) -> bool | None:
        """Reconnect to WebSocket API"""
        self.logger.info("Attempting to reconnect")
        self.reconnect_attempts += 1

        if self.reconnect_attempts > self.max_reconnect_attempts:
            self.logger.error("Max reconnect attempts reached")
            self.running = False
            return False

        try:
            # Disconnect first
            await self.disconnect()

            # Exponential backoff
            delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            self.logger.info(f"Waiting {delay:.2f}s before reconnecting")
            await asyncio.sleep(delay)

            # Reconnect
            await self.connect()

            # Resubscribe to channels
            if self.subscriptions:
                await self.subscribe(list(self.subscriptions))

            self.reconnect_attempts = 0
            return True
        except Exception as e:
            self.logger.exception(f"Reconnection failed: {e}")
            return False

    def add_message_handler(self, event_type, handler) -> None:
        """
        Add message handler for specific event type

        Args:
            event_type: Event type to handle
            handler: Callback function to handle event
        """
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)
        self.logger.info(f"Added message handler for {event_type}")

    def remove_message_handler(self, event_type, handler) -> None:
        """
        Remove message handler for specific event type

        Args:
            event_type: Event type to handle
            handler: Callback function to remove
        """
        if event_type in self.message_handlers:
            if handler in self.message_handlers[event_type]:
                self.message_handlers[event_type].remove(handler)
                self.logger.info(f"Removed message handler for {event_type}")


# =============================================================================
# SECTION 6: Day Trading System
# =============================================================================


class DayTradingSystem:
    """
    Day Trading System for high-frequency intraday trading.

    Provides functionality for:
    - Real-time opportunity detection
    - Entry and exit signal generation
    - Risk management and position sizing
    - Pattern recognition and momentum analysis
    """

    def __init__(
        self,
        redis_client=None,
        polygon_client=None,
        websocket_client=None,
        execution_system=None,
    ) -> None:
        """
        Initialize the day trading system

        Args:
            redis_client: Redis client for caching and messaging
            polygon_client: Polygon API client
            websocket_client: WebSocket client for real-time data
            execution_system: Execution system for order placement
        """
        self.redis = redis_client
        self.polygon = polygon_client
        self.websocket = websocket_client
        self.execution = execution_system

        # Configuration
        self.config = {
            "max_positions": 5,
            "max_position_size": 5000,  # $ per position
            "position_timeout": 3600,  # 1 hour
            "profit_target_pct": 1.0,  # 1%
            "stop_loss_pct": 0.5,  # 0.5%
            "entry_signals": ["volume_spike", "momentum_shift", "breakout", "bounce"],
            "min_volume": 500000,
            "min_dollar_volume": 5000000,
            "min_volatility": 1.0,  # Minimum ATR %
            "batch_size": 100,
        }

        # Internal state
        self.active_positions = {}
        self.candidates = set()
        self.monitoring = set()
        self.pattern_cache = {}
        self.stats = {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

        # Runtime state
        self.running = False
        self.tasks = {}

        self.logger = logging.getLogger("day_trading_system")
        self.logger.info("Day Trading System initialized")

    async def start(self) -> None:
        """Start the day trading system"""
        if self.running:
            self.logger.warning("Day trading system already running")
            return

        self.running = True
        self.logger.info("Starting day trading system")

        # Initialize monitoring set
        await self._initialize_candidates()

        # Start background tasks
        self.tasks["opportunity_detection"] = asyncio.create_task(
            self._opportunity_detection_loop(),
        )
        self.tasks["position_monitoring"] = asyncio.create_task(
            self._position_monitoring_loop(),
        )
        self.tasks["candidate_refresh"] = asyncio.create_task(
            self._candidate_refresh_loop(),
        )

        # Set up WebSocket handlers
        if self.websocket:
            self.websocket.add_message_handler("trade", self._handle_trade)
            self.websocket.add_message_handler("quote", self._handle_quote)
            self.websocket.add_message_handler("agg", self._handle_agg)

        self.logger.info("Day trading system started")

    async def stop(self) -> None:
        """Stop the day trading system"""
        if not self.running:
            self.logger.warning("Day trading system not running")
            return

        self.running = False
        self.logger.info("Stopping day trading system")

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                self.logger.info(f"Cancelling {name} task")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Remove WebSocket handlers
        if self.websocket:
            self.websocket.remove_message_handler("trade", self._handle_trade)
            self.websocket.remove_message_handler("quote", self._handle_quote)
            self.websocket.remove_message_handler("agg", self._handle_agg)

        # Close any open positions
        await self._close_all_positions("System shutdown")

        self.logger.info("Day trading system stopped")

    async def _initialize_candidates(self) -> None:
        """Initialize candidate stocks for day trading"""
        self.logger.info("Initializing day trading candidates")

        try:
            # Get active stocks with high volume
            if self.polygon:
                response = await self.polygon._make_request(
                    "v2/snapshot/locale/us/markets/stocks/tickers",
                    {"tickers": "AAPL,MSFT,TSLA,AMZN,NVDA,AMD,META,GOOGL,QQQ,SPY"},
                )

                if response and "tickers" in response:
                    for ticker_data in response["tickers"]:
                        ticker = ticker_data.get("ticker", "")
                        if ticker:
                            self.candidates.add(ticker)

                if not self.candidates:
                    # Fallback to default tickers from configuration
                    self.candidates = set(
                        self.config["stock_selection"]["universe"]["default_tickers"],
                    )
                    self.logger.info(
                        f"Using default tickers from configuration as fallback: {self.candidates}",
                    )
            else:
                # Fallback to default tickers from configuration
                self.candidates = set(
                    self.config["stock_selection"]["universe"]["default_tickers"],
                )
                self.logger.info(
                    f"Using default tickers from configuration as fallback: {self.candidates}",
                )

            # Subscribe to WebSocket channels
            if self.websocket:
                for ticker in self.candidates:
                    trade_channel = f"T.{ticker}"
                    quote_channel = f"Q.{ticker}"
                    agg_channel = f"AM.{ticker}"

                    await self.websocket.subscribe(
                        [trade_channel, quote_channel, agg_channel],
                    )

            self.logger.info(
                f"Initialized with {len(self.candidates)} candidates")

        except Exception as e:
            self.logger.exception(f"Error initializing candidates: {e}")
            # Fallback to default tickers from configuration
            self.candidates = set(
                self.config["stock_selection"]["universe"]["default_tickers"],
            )
            self.logger.info(
                f"Using default tickers from configuration as fallback: {self.candidates}",
            )

    async def _opportunity_detection_loop(self) -> None:
        """Main loop for opportunity detection"""
        try:
            while self.running:
                # Check for new opportunities
                opportunities = await self._scan_for_opportunities()

                # Process opportunities
                for ticker, signal in opportunities:
                    await self._process_opportunity(ticker, signal)

                # Sleep to avoid high CPU usage
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Opportunity detection loop cancelled")
        except Exception as e:
            self.logger.exception(f"Error in opportunity detection loop: {e}")

    async def _scan_for_opportunities(self):
        """
        Scan for trading opportunities

        Returns:
            List of (ticker, signal) tuples
        """
        opportunities = []

        try:
            # Get candidates to scan
            candidates = list(self.candidates)

            # Process in batches
            batch_size = self.config["batch_size"]
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i: i + batch_size]

                # Scan batch for opportunities
                batch_opportunities = await self._scan_batch(batch)
                opportunities.extend(batch_opportunities)

            return opportunities
        except Exception as e:
            self.logger.exception(f"Error scanning for opportunities: {e}")
            return []

    async def _scan_batch(self, tickers):
        """
        Scan a batch of tickers for opportunities

        Args:
            tickers: List of ticker symbols

        Returns:
            List of (ticker, signal) tuples
        """
        batch_opportunities = []

        try:
            for ticker in tickers:
                # Skip if already in a position
                if ticker in self.active_positions:
                    continue

                # Check each signal type
                for signal_type in self.config["entry_signals"]:
                    if await getattr(self, f"_check_{signal_type}")(ticker):
                        batch_opportunities.append((ticker, signal_type))
                        break  # Only one signal per ticker

            return batch_opportunities
        except Exception as e:
            self.logger.exception(f"Error scanning batch: {e}")
            return []

    async def _check_volume_spike(self, ticker) -> bool | None:
        """
        Check for volume spike signal

        Args:
            ticker: Ticker symbol

        Returns:
            True if signal detected, False otherwise
        """
        try:
            # Get recent volume data
            volume_data = await get_volume_data(self.redis, self.polygon, ticker)

            if not volume_data or len(volume_data) < 10:
                return False

            # Current volume
            current_volume = volume_data[0]

            # Calculate average volume (5-day)
            avg_volume = sum(volume_data[1:6]) / 5

            # Check for spike (3x average)
            if (
                current_volume > avg_volume * 3
                and current_volume > self.config["min_volume"]
            ):
                self.logger.info(f"Volume spike detected for {ticker}")
                return True

            return False
        except Exception as e:
            self.logger.exception(
                f"Error checking volume spike for {ticker}: {e}")
            return False

    async def _check_momentum_shift(self, ticker) -> bool | None:
        """
        Check for momentum shift signal

        Args:
            ticker: Ticker symbol

        Returns:
            True if signal detected, False otherwise
        """
        try:
            # Get recent price data
            price_data = await get_price_data(self.redis, self.polygon, ticker)

            if not price_data or len(price_data) < 10:
                return False

            # Get recent closing prices
            closes = [item.get("c", 0) for item in price_data[:10]]

            # Calculate short-term momentum (1-day)
            short_term = closes[0] / closes[1] - 1 if closes[1] > 0 else 0

            # Calculate medium-term momentum (5-day)
            medium_term = closes[0] / closes[5] - 1 if closes[5] > 0 else 0

            # Check for momentum shift (short-term momentum > 1% and opposite to
            # medium-term)
            if abs(short_term) > 0.01 and short_term * medium_term < 0:
                self.logger.info(f"Momentum shift detected for {ticker}")
                return True

            return False
        except Exception as e:
            self.logger.exception(
                f"Error checking momentum shift for {ticker}: {e}")
            return False

    async def _check_breakout(self, ticker) -> bool | None:
        """
        Check for breakout signal

        Args:
            ticker: Ticker symbol

        Returns:
            True if signal detected, False otherwise
        """
        try:
            # Get recent price data
            price_data = await get_price_data(self.redis, self.polygon, ticker)

            if not price_data or len(price_data) < 20:
                return False

            # Get highs and lows
            highs = [item.get("h", 0) for item in price_data[:20]]
            [item.get("l", 0) for item in price_data[:20]]
            closes = [item.get("c", 0) for item in price_data[:20]]

            # Current price
            current_price = closes[0]

            # Calculate resistance (highest high in last 20 days excluding today)
            resistance = max(highs[1:])

            # Check for breakout (price > resistance)
            if current_price > resistance * 1.02:  # 2% above resistance
                self.logger.info(f"Breakout detected for {ticker}")
                return True

            return False
        except Exception as e:
            self.logger.exception(f"Error checking breakout for {ticker}: {e}")
            return False

    async def _check_bounce(self, ticker) -> bool | None:
        """
        Check for bounce signal

        Args:
            ticker: Ticker symbol

        Returns:
            True if signal detected, False otherwise
        """
        try:
            # Get recent price data
            price_data = await get_price_data(self.redis, self.polygon, ticker)

            if not price_data or len(price_data) < 20:
                return False

            # Get highs and lows
            lows = [item.get("l", 0) for item in price_data[:20]]
            closes = [item.get("c", 0) for item in price_data[:20]]

            # Current price
            current_price = closes[0]

            # Calculate support (lowest low in last 20 days excluding today)
            support = min(lows[1:])

            # Check for bounce (price bounced off support)
            if lows[0] <= support * 1.02 and current_price > lows[0] * 1.02:
                self.logger.info(f"Bounce detected for {ticker}")
                return True

            return False
        except Exception as e:
            self.logger.exception(f"Error checking bounce for {ticker}: {e}")
            return False

    async def _process_opportunity(self, ticker, signal) -> None:
        """
        Process a trading opportunity

        Args:
            ticker: Ticker symbol
            signal: Signal type
        """
        try:
            self.logger.info(f"Processing {signal} opportunity for {ticker}")

            # Check if we can take another position
            if len(self.active_positions) >= self.config["max_positions"]:
                self.logger.info(f"Max positions reached, skipping {ticker}")
                return

            # Get current price
            current_price = await self._get_current_price(ticker)

            if current_price <= 0:
                self.logger.warning(f"Invalid price for {ticker}, skipping")
                return

            # Calculate position size
            position_size = await self._calculate_position_size(ticker, current_price)

            if position_size <= 0:
                self.logger.warning(
                    f"Invalid position size for {ticker}, skipping")
                return

            # Calculate entry/exit prices
            stop_price = current_price * \
                (1 - self.config["stop_loss_pct"] / 100)
            target_price = current_price * \
                (1 + self.config["profit_target_pct"] / 100)

            # Send notification to frontend about the opportunity with enhanced details
            details = {
                "ticker": ticker,
                "signal_type": signal,
                "current_price": current_price,
                "position_size": position_size,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_reward_ratio": (target_price - current_price) / (current_price - stop_price) if (current_price - stop_price) > 0 else 0,
                "potential_profit": (target_price - current_price) * position_size,
                "max_loss": (current_price - stop_price) * position_size,
                "detection_time": time.time()
            }

            self._send_frontend_notification(
                message=f"Trading opportunity detected: {signal} signal for {ticker} at ${current_price:.2f}",
                level="info",
                category="opportunity",
                details=details
            )

            # Record in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    TRADING_OPPORTUNITIES.labels(signal_type=signal).inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording opportunity in Prometheus: {prom_e}")

            # Generate entry signal
            await self._generate_entry_signal(
                ticker, position_size, current_price, stop_price, target_price, signal,
            )

        except Exception as e:
            self.logger.exception(
                f"Error processing opportunity for {ticker}: {e}")

    async def _generate_entry_signal(
        self, ticker, size, entry_price, stop_price, target_price, signal_type,
    ) -> None:
        """
        Generate entry signal

        Args:
            ticker: Ticker symbol
            size: Position size in shares
            entry_price: Entry price
            stop_price: Stop-loss price
            target_price: Profit target price
            signal_type: Signal type
        """
        try:
            # Create signal
            signal = {
                "ticker": ticker,
                "direction": "buy",  # day trading is long-only for now
                "size": size,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "signal_type": signal_type,
                "timestamp": time.time(),
                "expiry": time.time() + self.config["position_timeout"],
            }

            # Send signal to execution system
            if self.execution:
                await self.execution.execute_signal(signal)
                self.logger.info(f"Entry signal sent for {ticker}")

                # Add to active positions
                self.active_positions[ticker] = signal
            elif self.redis:
                # Publish to Redis if no execution system
                self.redis.publish("trading:entry_signal", json.dumps(signal))
                self.logger.info(f"Entry signal published for {ticker}")

                # Send notification to frontend
                self._send_frontend_notification(
                    message=f"Entry signal for {ticker}: {size} shares at ${entry_price:.2f}",
                    level="info",
                    category="trade_entry"
                )

                # Add to active positions
                self.active_positions[ticker] = signal
            else:
                self.logger.warning(
                    f"No execution system or Redis, cannot send entry signal for {ticker}",
                )

        except Exception as e:
            self.logger.exception(
                f"Error generating entry signal for {ticker}: {e}")

    async def _position_monitoring_loop(self) -> None:
        """Monitor active positions for exit conditions"""
        try:
            while self.running:
                # Check all active positions
                positions_to_exit = []

                for ticker, position in self.active_positions.items():
                    if await self._check_exit_conditions(ticker, position):
                        positions_to_exit.append(ticker)

                # Exit positions
                for ticker in positions_to_exit:
                    await self._exit_position(ticker)

                # Sleep to avoid high CPU usage
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Position monitoring loop cancelled")
        except Exception as e:
            self.logger.exception(f"Error in position monitoring loop: {e}")

    async def _check_exit_conditions(self, ticker, position) -> bool | None:
        """
        Check exit conditions for a position

        Args:
            ticker: Ticker symbol
            position: Position data

        Returns:
            True if exit conditions met, False otherwise
        """
        try:
            # Get current price
            current_price = await self._get_current_price(ticker)

            if current_price <= 0:
                self.logger.warning(
                    f"Invalid price for {ticker}, skipping exit check")
                return False

            # Check stop loss
            if current_price <= position["stop_price"]:
                self.logger.info(f"Stop loss triggered for {ticker}")
                return True

            # Check profit target
            if current_price >= position["target_price"]:
                self.logger.info(f"Profit target reached for {ticker}")
                return True

            # Check timeout
            if time.time() >= position["expiry"]:
                self.logger.info(f"Position timeout for {ticker}")
                return True

            return False
        except Exception as e:
            self.logger.exception(
                f"Error checking exit conditions for {ticker}: {e}")
            return False

    async def _exit_position(self, ticker) -> None:
        """
        Exit a position

        Args:
            ticker: Ticker symbol
        """
        try:
            # Get position data
            position = self.active_positions.get(ticker)

            if not position:
                self.logger.warning(f"No position found for {ticker}")
                return

            # Get current price
            current_price = await self._get_current_price(ticker)

            if current_price <= 0:
                self.logger.warning(
                    f"Invalid price for {ticker}, using entry price")
                current_price = position["entry_price"]

            # Generate exit signal
            await self._generate_exit_signal(ticker, position["size"], current_price)

            # Update stats
            self.stats["trades"] += 1

            pnl = (current_price - position["entry_price"]) * position["size"]
            self.stats["total_pnl"] += pnl

            if current_price > position["entry_price"]:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1

            # Remove from active positions
            del self.active_positions[ticker]

            self.logger.info(f"Position exited for {ticker}, PnL: ${pnl:.2f}")

        except Exception as e:
            self.logger.exception(f"Error exiting position for {ticker}: {e}")

    async def _generate_exit_signal(self, ticker, size, exit_price) -> None:
        """
        Generate exit signal

        Args:
            ticker: Ticker symbol
            size: Position size in shares
            exit_price: Exit price
        """
        try:
            # Create signal
            signal = {
                "ticker": ticker,
                "direction": "sell",
                "size": size,
                "exit_price": exit_price,
                "timestamp": time.time(),
            }

            # Send signal to execution system
            if self.execution:
                await self.execution.execute_signal(signal)
                self.logger.info(f"Exit signal sent for {ticker}")
            elif self.redis:
                # Publish to Redis if no execution system
                self.redis.publish("trading:exit_signal", json.dumps(signal))
                self.logger.info(f"Exit signal published for {ticker}")

                # Send notification to frontend with enhanced details
                pnl = (exit_price -
                       self.active_positions[ticker]["entry_price"]) * size
                pnl_percent = (
                    (exit_price / self.active_positions[ticker]["entry_price"]) - 1) * 100

                self._send_frontend_notification(
                    message=f"Exit signal for {ticker}: {size} shares at ${exit_price:.2f}",
                    level="info" if pnl >= 0 else "warning",
                    category="trade_exit",
                    details={
                        "ticker": ticker,
                        "size": size,
                        "entry_price": self.active_positions[ticker]["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "trade_duration": time.time() - self.active_positions[ticker]["timestamp"],
                        "signal_type": self.active_positions[ticker]["signal_type"],
                        "exit_time": time.time()
                    }
                )
            else:
                self.logger.warning(
                    f"No execution system or Redis, cannot send exit signal for {ticker}",
                )

        except Exception as e:
            self.logger.exception(
                f"Error generating exit signal for {ticker}: {e}")

    async def _calculate_position_size(self, ticker, price):
        """
        Calculate position size based on risk parameters

        Args:
            ticker: Ticker symbol
            price: Current price

        Returns:
            Position size in shares
        """
        try:
            # Maximum dollar amount per position
            max_position = self.config["max_position_size"]

            # Calculate shares based on price
            shares = int(max_position / price)

            # Ensure minimum share count
            if shares < 1:
                return 0

            return shares
        except Exception as e:
            self.logger.exception(
                f"Error calculating position size for {ticker}: {e}")
            return 0

    async def _get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            # Try WebSocket first
            if self.websocket:
                return self.websocket.get_latest_price(ticker)

            # Fall back to Redis
            if self.redis:
                price_key = f"price:{ticker}:latest"
                price_data = self.redis.get(price_key)
                if price_data:
                    return float(price_data)

            # Fall back to API
            if self.polygon:
                return await self._get_price_from_api(ticker)

            return 0
        except Exception as e:
            self.logger.exception(
                f"Error getting current price for {ticker}: {e}")
            return 0

    async def _get_price_from_api(self, ticker):
        """Get price from API"""
        try:
            # Use the Polygon client to get the latest price
            response = await self.polygon._make_request(f"v2/last/trade/{ticker}", {})

            if response and "results" in response:
                return float(response["results"]["p"])

            return 0
        except Exception as e:
            self.logger.exception(
                f"Error getting price from API for {ticker}: {e}")
            return 0

    async def _candidate_refresh_loop(self) -> None:
        """Periodically refresh the candidate list"""
        try:
            while self.running:
                # Update candidate list
                await self._update_candidate_list()

                # Sleep for 15 minutes
                await asyncio.sleep(900)
        except asyncio.CancelledError:
            self.logger.info("Candidate refresh loop cancelled")
        except Exception as e:
            self.logger.exception(f"Error in candidate refresh loop: {e}")

    async def _update_candidate_list(self) -> None:
        """Update the list of day trading candidates"""
        try:
            # Get universe of stocks from API
            universe = await self._get_candidate_universe()

            # Filter for liquidity
            liquid_stocks = await self._filter_for_liquidity(universe)

            # Update candidate set
            self.candidates = set(liquid_stocks)

            # Update WebSocket subscriptions
            await self._update_subscriptions()

            self.logger.info(
                f"Updated candidate list with {len(self.candidates)} stocks",
            )
        except Exception as e:
            self.logger.exception(f"Error updating candidate list: {e}")

    async def _get_candidate_universe(self):
        """Get universe of candidate stocks"""
        try:
            # Use the Polygon client to get active stocks
            response = await self.polygon._make_request(
                "v3/reference/tickers",
                {
                    "market": "stocks",
                    "active": "true",
                    "sort": "volume",
                    "order": "desc",
                    "limit": 200,
                },
            )

            if response and "results" in response:
                return [
                    item["ticker"] for item in response["results"] if item.get("ticker")
                ]

            # Fallback to default tickers from configuration
            default_tickers = self.config["stock_selection"]["universe"][
                "default_tickers"
            ]
            self.logger.info(
                f"Using default tickers from configuration as fallback: {default_tickers}",
            )
            return default_tickers
        except Exception as e:
            self.logger.exception(f"Error getting candidate universe: {e}")
            # Fallback to default tickers from configuration
            default_tickers = self.config["stock_selection"]["universe"][
                "default_tickers"
            ]
            self.logger.info(
                f"Using default tickers from configuration as fallback: {default_tickers}",
            )
            return default_tickers

    async def _filter_for_liquidity(self, tickers):
        """Filter tickers for liquidity"""
        try:
            liquid_tickers = []

            for ticker in tickers:
                # Get volume data
                volume_data = await get_volume_data(self.redis, self.polygon, ticker)

                if not volume_data:
                    continue

                # Calculate average volume
                avg_volume = (
                    sum(volume_data[:5]) / 5
                    if len(volume_data) >= 5
                    else volume_data[0]
                )

                # Check liquidity criteria
                if avg_volume >= self.config["min_volume"]:
                    liquid_tickers.append(ticker)

            return liquid_tickers
        except Exception as e:
            self.logger.exception(f"Error filtering for liquidity: {e}")
            return tickers

    async def _update_subscriptions(self) -> None:
        """Update WebSocket subscriptions"""
        try:
            if not self.websocket:
                return

            # Unsubscribe from old channels
            old_subscriptions = self.subscriptions.copy()

            # Create new subscriptions
            new_subscriptions = set()
            for ticker in self.candidates:
                trade_channel = f"T.{ticker}"
                quote_channel = f"Q.{ticker}"
                agg_channel = f"AM.{ticker}"

                new_subscriptions.update(
                    [trade_channel, quote_channel, agg_channel])

            # Find channels to unsubscribe from
            to_unsubscribe = old_subscriptions - new_subscriptions

            # Find channels to subscribe to
            to_subscribe = new_subscriptions - old_subscriptions

            # Update subscriptions
            if to_unsubscribe:
                await self.websocket.unsubscribe(list(to_unsubscribe))

            if to_subscribe:
                await self.websocket.subscribe(list(to_subscribe))

            # Update subscription set
            self.subscriptions = new_subscriptions

            self.logger.info(
                f"Updated WebSocket subscriptions: {len(to_unsubscribe)} removed, {len(to_subscribe)} added",
            )
        except Exception as e:
            self.logger.exception(f"Error updating subscriptions: {e}")

    async def _close_all_positions(self, reason) -> None:
        """Close all open positions"""
        try:
            self.logger.info(f"Closing all positions due to: {reason}")

            # Exit each position
            positions_to_exit = list(self.active_positions.keys())

            for ticker in positions_to_exit:
                await self._exit_position(ticker)

            self.logger.info(f"Closed {len(positions_to_exit)} positions")
        except Exception as e:
            self.logger.exception(f"Error closing all positions: {e}")

    def _handle_trade(self, message) -> None:
        """Handle trade message from WebSocket"""
        try:
            # Record WebSocket message in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_MESSAGES.labels(message_type="trade").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket message in Prometheus: {prom_e}",
                    )

            # Extract ticker and trade data
            ticker = message.get("sym", "")
            price = message.get("p", 0)
            size = message.get("s", 0)
            timestamp = message.get("t", 0)

            if not ticker or price <= 0:
                return

            # Store in real-time data cache
            if ticker not in self.real_time_data:
                self.real_time_data[ticker] = {
                    "trades": [],
                    "quotes": [],
                    "price": price,
                    "volume": 0,
                    "last_update": timestamp,
                }

            # Update price and volume
            self.real_time_data[ticker]["price"] = price
            self.real_time_data[ticker]["volume"] += size
            self.real_time_data[ticker]["last_update"] = timestamp

            # Add to trades list (keep only the last 100)
            self.real_time_data[ticker]["trades"].append(
                {"price": price, "size": size, "timestamp": timestamp},
            )

            if len(self.real_time_data[ticker]["trades"]) > 100:
                self.real_time_data[ticker]["trades"].pop(0)

            # Check for volume spike
            if self._check_volume_spike_rt(ticker, size):
                self.real_time_metrics["volume_spikes"].add(ticker)

                # Record volume spike in Prometheus if available
                if PROMETHEUS_AVAILABLE:
                    try:
                        TRADING_OPPORTUNITIES.labels(
                            signal_type="volume_spike").inc()
                    except Exception as prom_e:
                        self.logger.debug(
                            f"Error recording volume spike in Prometheus: {prom_e}",
                        )

            # Update active positions count in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    ACTIVE_POSITIONS.set(len(self.active_positions))
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording active positions in Prometheus: {prom_e}",
                    )

        except Exception as e:
            self.logger.exception(f"Error handling trade: {e}")

            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_ERRORS.labels(
                        error_type="trade_processing").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket error in Prometheus: {prom_e}",
                    )

    def _handle_quote(self, message) -> None:
        """Handle quote message from WebSocket"""
        try:
            # Record WebSocket message in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_MESSAGES.labels(message_type="quote").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket message in Prometheus: {prom_e}",
                    )

            # Extract ticker and quote data
            ticker = message.get("sym", "")
            bid_price = message.get("bp", 0)
            ask_price = message.get("ap", 0)
            bid_size = message.get("bs", 0)
            ask_size = message.get("as", 0)
            timestamp = message.get("t", 0)

            if not ticker or bid_price <= 0 or ask_price <= 0:
                return

            # Store in real-time data cache
            if ticker not in self.real_time_data:
                self.real_time_data[ticker] = {
                    "trades": [],
                    "quotes": [],
                    "price": (bid_price + ask_price) / 2,
                    "volume": 0,
                    "last_update": timestamp,
                }

            # Update mid price
            self.real_time_data[ticker]["price"] = (bid_price + ask_price) / 2
            self.real_time_data[ticker]["last_update"] = timestamp

            # Add to quotes list (keep only the last 100)
            self.real_time_data[ticker]["quotes"].append(
                {
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "timestamp": timestamp,
                },
            )

            if len(self.real_time_data[ticker]["quotes"]) > 100:
                self.real_time_data[ticker]["quotes"].pop(0)

            # Calculate spread and record in Prometheus if available
            if PROMETHEUS_AVAILABLE and ticker in self.active_positions:
                try:
                    spread = ask_price - bid_price
                    spread_pct = (spread / ((bid_price + ask_price) / 2)) * 100

                    # Record spread as a custom metric
                    # This is useful for liquidity monitoring
                    STOCK_SPREAD.labels(ticker=ticker).set(spread_pct)
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording spread metrics in Prometheus: {prom_e}",
                    )

        except Exception as e:
            self.logger.exception(f"Error handling quote: {e}")

            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_ERRORS.labels(
                        error_type="quote_processing").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket error in Prometheus: {prom_e}",
                    )

    def _handle_agg(self, message) -> None:
        """Handle aggregate (minute bar) message from WebSocket"""
        try:
            # Record WebSocket message in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_MESSAGES.labels(message_type="agg").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket message in Prometheus: {prom_e}",
                    )

            # Extract ticker and aggregate data
            ticker = message.get("sym", "")
            open_price = message.get("o", 0)
            high_price = message.get("h", 0)
            low_price = message.get("l", 0)
            close_price = message.get("c", 0)
            volume = message.get("v", 0)
            timestamp = message.get("t", 0)

            if not ticker or close_price <= 0:
                return

            # Store in real-time data cache
            if ticker not in self.real_time_data:
                self.real_time_data[ticker] = {
                    "trades": [],
                    "quotes": [],
                    "minute_bars": [],
                    "price": close_price,
                    "volume": 0,
                    "last_update": timestamp,
                }

            # Update price
            self.real_time_data[ticker]["price"] = close_price
            self.real_time_data[ticker]["volume"] += volume
            self.real_time_data[ticker]["last_update"] = timestamp

            # Add to minute bars list (keep only the last 100)
            self.real_time_data[ticker]["minute_bars"].append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "timestamp": timestamp,
                },
            )

            if len(self.real_time_data[ticker]["minute_bars"]) > 100:
                self.real_time_data[ticker]["minute_bars"].pop(0)

            # Check for price jump
            if self._check_price_jump_rt(ticker, close_price):
                self.real_time_metrics["price_jumps"].add(ticker)

                # Record price jump in Prometheus if available
                if PROMETHEUS_AVAILABLE:
                    try:
                        TRADING_OPPORTUNITIES.labels(
                            signal_type="price_jump").inc()
                    except Exception as prom_e:
                        self.logger.debug(
                            f"Error recording price jump in Prometheus: {prom_e}",
                        )

            # Record volatility metrics in Prometheus if available
            if PROMETHEUS_AVAILABLE and ticker in self.active_positions:
                try:
                    # Calculate volatility (high-low range as percentage of open)
                    if open_price > 0:
                        volatility = (
                            (high_price - low_price) / open_price) * 100

                        # Record volatility as a custom metric
                        STOCK_VOLATILITY.labels(ticker=ticker).set(volatility)
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording volatility metrics in Prometheus: {prom_e}",
                    )

        except Exception as e:
            self.logger.exception(f"Error handling aggregate: {e}")

            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    WEBSOCKET_ERRORS.labels(error_type="agg_processing").inc()
                except Exception as prom_e:
                    self.logger.debug(
                        f"Error recording WebSocket error in Prometheus: {prom_e}",
                    )

    def _check_volume_spike_rt(self, ticker, size):
        """Check for volume spike in real-time data"""
        try:
            # Need at least 10 trades to establish baseline
            if (
                ticker not in self.real_time_data
                or len(self.real_time_data[ticker]["trades"]) < 10
            ):
                return False

            # Calculate average trade size
            trades = self.real_time_data[ticker]["trades"]
            avg_size = sum(t["size"] for t in trades) / len(trades)

            # Check for spike (5x average)
            return size > avg_size * 5 and size > 1000
        except Exception as e:
            self.logger.exception(f"Error checking volume spike: {e}")
            return False

    def _send_frontend_notification(self, message, level="info", category="system", details=None):
        """
        Send notification to frontend via Redis

        Args:
            message: Notification message
            level: Notification level (info, warning, error, success)
            category: Notification category (system, trade, pattern, etc.)
            details: Optional additional details for the notification
        """
        if not self.redis:
            return

        try:
            notification = {
                "type": category,
                "message": message,
                "level": level,
                "timestamp": time.time()
            }

            # Add details if provided
            if details:
                notification["details"] = details

            # Push to notifications list
            self.redis.lpush("frontend:notifications",
                             json.dumps(notification))
            # Keep last 100 notifications
            self.redis.ltrim("frontend:notifications", 0, 99)

            # Also store in category-specific list if appropriate
            if category in ["trade", "trade_entry", "trade_exit", "system_startup", "system_shutdown", "opportunity", "market_status"]:
                self.redis.lpush(
                    f"frontend:{category}", json.dumps(notification))
                # Keep last 50 category-specific items
                self.redis.ltrim(f"frontend:{category}", 0, 49)

            # Log the notification
            if level == "error":
                self.logger.error(f"Frontend notification: {message}")
            elif level == "warning":
                self.logger.warning(f"Frontend notification: {message}")
            else:
                self.logger.info(f"Frontend notification: {message}")

        except Exception as e:
            self.logger.error(f"Error sending frontend notification: {e}")

    def _check_price_jump_rt(self, ticker, price):
        """Check for price jump in real-time data"""
        try:
            # Need at least 5 minute bars to establish baseline
            if (
                ticker not in self.real_time_data
                or "minute_bars" not in self.real_time_data[ticker]
                or len(self.real_time_data[ticker]["minute_bars"]) < 5
            ):
                return False

            # Get previous minute bar
            bars = self.real_time_data[ticker]["minute_bars"]
            prev_close = bars[-2]["close"] if len(bars) >= 2 else None

            if not prev_close:
                return False

            # Calculate percent change
            pct_change = (price - prev_close) / \
                prev_close if prev_close > 0 else 0

            # Check for jump (1% in a minute)
            return abs(pct_change) > 0.01
        except Exception as e:
            self.logger.exception(f"Error checking price jump: {e}")
            return False


# =============================================================================
# SECTION 7: Enhanced WebSocket Integration
# =============================================================================


class WebSocketEnhancedStockSelection:
    """
    Enhanced WebSocket-based stock selection system.

    Integrates WebSocket real-time data with the stock selection system
    for dynamic adjustment of watchlists based on real-time market conditions.
    """

    def __init__(
        self, redis_client=None, polygon_websocket=None, stock_selection_system=None,
    ) -> None:
        """
        Initialize WebSocket-enhanced stock selection

        Args:
            redis_client: Redis client for caching and messaging
            polygon_websocket: Polygon WebSocket client
            stock_selection_system: Stock selection system instance
        """
        self.redis = redis_client
        self.websocket = polygon_websocket
        self.selection_system = stock_selection_system

        # Internal state
        self.tickers_last_update = {}
        self.real_time_scores = {}
        self.real_time_metrics = {
            "volume_spikes": set(),
            "momentum_shifts": set(),
            "price_gaps": set(),
            "volatility_surges": set(),
        }

        # Configuration
        self.config = {
            "metrics_update_interval": 60,  # 1 minute
            "max_alerts": 20,
            "websocket_channels": ["T.*", "Q.*", "AM.*"],
            "min_price": 5.0,
        }

        # Runtime state
        self.running = False
        self.tasks = {}

        self.logger = logging.getLogger("websocket_enhanced_selection")
        self.logger.info("WebSocket-Enhanced Stock Selection initialized")

    async def start(self) -> None:
        """Start WebSocket-enhanced stock selection"""
        if self.running:
            self.logger.warning("WebSocket-enhanced selection already running")
            return

        self.running = True
        self.logger.info("Starting WebSocket-enhanced stock selection")

        # Start WebSocket if not already running
        if (
            self.websocket
            and hasattr(self.websocket, "connect")
            and not self.websocket.is_connected
        ):
            await self.websocket.connect()

        # Subscribe to WebSocket channels
        if self.websocket and hasattr(self.websocket, "subscribe"):
            try:
                if (
                    self.config.get("websocket_channels")
                ):
                    await self.websocket.subscribe(self.config["websocket_channels"])
                else:
                    self.logger.warning(
                        "No WebSocket channels configured for subscription",
                    )
            except Exception as e:
                self.logger.exception(
                    f"Error subscribing to WebSocket channels: {e}")
        # Start metrics update task
        self.tasks["metrics_update"] = asyncio.create_task(
            self._metrics_update_loop())

        # Add message handlers
        if self.websocket:
            self.websocket.add_message_handler("T", self._handle_trade)
            self.websocket.add_message_handler("Q", self._handle_quote)
            self.websocket.add_message_handler("AM", self._handle_agg)

        self.logger.info("WebSocket-enhanced stock selection started")

    async def stop(self) -> None:
        """Stop WebSocket-enhanced stock selection"""
        if not self.running:
            self.logger.warning("WebSocket-enhanced selection not running")
            return

        self.running = False
        self.logger.info("Stopping WebSocket-enhanced stock selection")

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                self.logger.info(f"Cancelling {name} task")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Remove WebSocket handlers
        if self.websocket:
            self.websocket.remove_message_handler("T", self._handle_trade)
            self.websocket.remove_message_handler("Q", self._handle_quote)
            self.websocket.remove_message_handler("AM", self._handle_agg)

        self.logger.info("WebSocket-enhanced stock selection stopped")

    async def _metrics_update_loop(self) -> None:
        """Update metrics and enhance stock selection"""
        try:
            while self.running:
                # Calculate real-time scores
                await self._calculate_real_time_scores()

                # Update stock selection system
                await self._update_selection_system()

                # Sleep for the update interval
                await asyncio.sleep(self.config["metrics_update_interval"])
        except asyncio.CancelledError:
            self.logger.info("Metrics update loop cancelled")
        except Exception as e:
            self.logger.exception(f"Error in metrics update loop: {e}")

    async def _calculate_real_time_scores(self) -> None:
        """Calculate real-time scores for tickers"""
        try:
            # Combine all real-time metrics
            active_tickers = set()
            active_tickers.update(self.real_time_metrics["volume_spikes"])
            active_tickers.update(self.real_time_metrics["momentum_shifts"])
            active_tickers.update(self.real_time_metrics["price_gaps"])
            active_tickers.update(self.real_time_metrics["volatility_surges"])

            # Calculate scores
            for ticker in active_tickers:
                score = 0.0

                # Add score components
                if ticker in self.real_time_metrics["volume_spikes"]:
                    score += 0.25

                if ticker in self.real_time_metrics["momentum_shifts"]:
                    score += 0.25

                if ticker in self.real_time_metrics["price_gaps"]:
                    score += 0.25

                if ticker in self.real_time_metrics["volatility_surges"]:
                    score += 0.25

                # Store score
                self.real_time_scores[ticker] = score

            # Log top scores
            top_tickers = sorted(
                self.real_time_scores.items(), key=lambda x: x[1], reverse=True,
            )[:5]
            if top_tickers:
                self.logger.info(f"Top real-time scores: {top_tickers}")
        except Exception as e:
            self.logger.exception(f"Error calculating real-time scores: {e}")

    async def _update_selection_system(self) -> None:
        """Update stock selection system with real-time metrics"""
        try:
            if not self.selection_system:
                return

            # Get top tickers by real-time score
            top_tickers = sorted(
                self.real_time_scores.items(), key=lambda x: x[1], reverse=True,
            )

            # Take top N tickers (max alerts)
            top_tickers = top_tickers[: self.config["max_alerts"]]

            # Add to focused list
            top_ticker_set = {ticker for ticker, _ in top_tickers if _ > 0.5}

            if not top_ticker_set:
                return

            # Update selection system's focused list
            if hasattr(self.selection_system, "focused_list") and isinstance(
                self.selection_system.focused_list, set,
            ):
                # Create a new focused list with both existing focused stocks and
                # real-time top tickers
                new_focused = self.selection_system.focused_list.copy()
                new_focused.update(top_ticker_set)

                # Limit size to focused_list_size
                if (
                    hasattr(self.selection_system, "config")
                    and "focused_list_size" in self.selection_system.config
                ):
                    max_size = self.selection_system.config["focused_list_size"]

                    if len(new_focused) > max_size:
                        # Prioritize real-time top tickers
                        focused_from_existing = (
                            self.selection_system.focused_list - top_ticker_set
                        )
                        # Sort by score if possible
                        if hasattr(
                            self.selection_system, "ticker_scores",
                        ) and isinstance(self.selection_system.ticker_scores, dict):
                            focused_from_existing = sorted(
                                focused_from_existing,
                                key=lambda t: self.selection_system.ticker_scores.get(
                                    t, 0,
                                ),
                                reverse=True,
                            )

                        # Take top N-len(top_ticker_set) from existing focused
                        top_existing = list(focused_from_existing)[
                            : max_size - len(top_ticker_set)
                        ]

                        # Combine
                        new_focused = set(top_existing) | top_ticker_set

                # Update focused list
                self.selection_system.focused_list = new_focused

                self.logger.info(
                    f"Updated focused list with {len(top_ticker_set)} real-time top tickers",
                )
        except Exception as e:
            self.logger.exception(f"Error updating selection system: {e}")

    def _handle_trade(self, message) -> None:
        """Handle trade message from WebSocket"""
        try:
            # Extract ticker and trade data
            ticker = message.get("sym", "")
            price = message.get("p", 0)
            size = message.get("s", 0)

            if not ticker or price <= 0 or price < self.config["min_price"]:
                return

            # Check for volume spike
            if size > 10000:  # Large trade
                self.real_time_metrics["volume_spikes"].add(ticker)
        except Exception as e:
            self.logger.exception(f"Error handling trade: {e}")

    def _handle_quote(self, message) -> None:
        """Handle quote message from WebSocket"""
        try:
            # Extract ticker and quote data
            ticker = message.get("sym", "")
            bid_price = message.get("bp", 0)
            ask_price = message.get("ap", 0)

            if (
                not ticker
                or bid_price <= 0
                or ask_price <= 0
                or (bid_price + ask_price) / 2 < self.config["min_price"]
            ):
                return

            # Track last update time
            self.tickers_last_update[ticker] = time.time()
        except Exception as e:
            self.logger.exception(f"Error handling quote: {e}")

    def _handle_agg(self, message) -> None:
        """Handle aggregate (minute bar) message from WebSocket"""
        try:
            # Extract ticker and aggregate data
            ticker = message.get("sym", "")
            open_price = message.get("o", 0)
            close_price = message.get("c", 0)
            message.get("v", 0)

            if not ticker or close_price <= 0 or close_price < self.config["min_price"]:
                return

            # Check for price movement
            if open_price > 0:
                pct_change = (close_price - open_price) / open_price

                # Check for momentum shift
                if abs(pct_change) > 0.01:  # 1% move in a minute
                    self.real_time_metrics["momentum_shifts"].add(ticker)

                # Check for volatility surge
                if abs(pct_change) > 0.02:  # 2% move in a minute
                    self.real_time_metrics["volatility_surges"].add(ticker)

            # Track last update time
            self.tickers_last_update[ticker] = time.time()
        except Exception as e:
            self.logger.exception(f"Error handling aggregate: {e}")


# =============================================================================
# SECTION 8: Main Entry Point
# =============================================================================


async def main() -> int:
    """Main entry point for the stock selection engine"""
    logger.info("Starting Unified Stock Selection Engine")

    try:
        # Initialize Redis connection
        try:
            # Get Redis configuration from config.py
            redis_config = config["redis"]

            # Check if password is provided
            if not redis_config["password"]:
                logger.warning("Redis password not provided in configuration")

            logger.info(
                f"Connecting to Redis at {redis_config['host']}:{redis_config['port']}",
            )

            redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                # Pass password even if empty
                password=redis_config["password"],
                username=redis_config.get("username", "default"),
                ssl=redis_config.get("ssl", False),
                decode_responses=False,
                socket_connect_timeout=redis_config.get("timeout", 5.0),
                socket_timeout=redis_config.get("timeout", 5.0),
            )

            # Test Redis connection
            redis_client.ping()
            logger.info("Connected to Redis")
        except redis.exceptions.AuthenticationError as auth_err:
            logger.exception(f"Redis authentication failed: {auth_err}")
            logger.exception(
                "Please check REDIS_PASSWORD environment variable")
            # Create a mock Redis client for testing/development
            redis_client = None
            raise
        except redis.exceptions.ConnectionError as conn_err:
            logger.exception(f"Redis connection error: {conn_err}")
            logger.exception(
                f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
            # Create a mock Redis client for testing/development
            redis_client = None
            raise
        except Exception as e:
            logger.exception(f"Redis initialization error: {e}")
            # Create a mock Redis client for testing/development
            redis_client = None
            raise

        # Initialize Polygon API client
        from api_clients import (
            PolygonRESTClient,
            PolygonWebSocketClient,
            UnusualWhalesClient,
        )

        # If Redis connection failed, we can still proceed with API clients
        # but without caching capabilities
        if redis_client is None:
            logger.warning("Proceeding without Redis caching")

        polygon_client = PolygonRESTClient(
            api_key=POLYGON_API_KEY, redis_client=redis_client, use_gpu=USE_GPU,
        )

        polygon_ws_client = PolygonWebSocketClient(
            api_key=POLYGON_API_KEY, redis_client=redis_client, use_gpu=USE_GPU,
        )

        unusual_whales_client = (
            UnusualWhalesClient(
                api_key=UNUSUAL_WHALES_API_KEY,
                redis_client=redis_client,
                use_gpu=USE_GPU,
            )
            if UNUSUAL_WHALES_API_KEY
            else None
        )

        # Initialize stock selection system
        stock_selection = GPUStockSelectionSystem(
            redis_client=redis_client,
            polygon_api_client=polygon_client,
            polygon_websocket_client=polygon_ws_client,
            unusual_whales_client=unusual_whales_client,
        )

        # Initialize WebSocket-enhanced selection
        ws_enhanced = WebSocketEnhancedStockSelection(
            redis_client=redis_client,
            polygon_websocket=polygon_ws_client,
            stock_selection_system=stock_selection,
        )

        # Initialize day trading system
        day_trading = DayTradingSystem(
            redis_client=redis_client,
            polygon_client=polygon_client,
            websocket_client=polygon_ws_client,
        )

        # Start systems
        await stock_selection.start()
        await ws_enhanced.start()
        await day_trading.start()

        logger.info("All systems started successfully")

        # Send notification to frontend about successful startup
        if redis_client:
            try:
                # Create a notification for the frontend
                notification = {
                    "type": "system_startup",
                    "message": "Stock Selection Engine started successfully",
                    "level": "success",
                    "timestamp": time.time(),
                    "details": {
                        "gpu_available": stock_selection.gpu_available,
                        "device_name": stock_selection.gh200_accelerator.device_name if hasattr(stock_selection, "gh200_accelerator") else "CPU",
                        "components": ["StockSelection", "WebSocketEnhanced", "DayTrading"],
                        "startup_time": time.time()
                    }
                }

                # Push to notifications list
                redis_client.lpush("frontend:notifications",
                                   json.dumps(notification))
                redis_client.ltrim("frontend:notifications", 0, 99)

                # Also store in system_startup category
                redis_client.lpush("frontend:system_startup",
                                   json.dumps(notification))
                redis_client.ltrim("frontend:system_startup", 0, 49)

                logger.info("Startup notification sent to frontend")
            except Exception as e:
                logger.error(f"Error sending startup notification: {e}")

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            # Stop all systems
            await day_trading.stop()
            await ws_enhanced.stop()
            await stock_selection.stop()

            logger.info("All systems stopped")

            # Send notification to frontend about shutdown
            if redis_client:
                try:
                    # Create a notification for the frontend
                    notification = {
                        "type": "system_shutdown",
                        "message": "Stock Selection Engine stopped gracefully",
                        "level": "info",
                        "timestamp": time.time(),
                        "details": {
                            "components": ["StockSelection", "WebSocketEnhanced", "DayTrading"],
                            "shutdown_time": time.time(),
                            "shutdown_reason": "User initiated" if 'KeyboardInterrupt' in locals() else "System initiated"
                        }
                    }

                    # Push to notifications list
                    redis_client.lpush(
                        "frontend:notifications", json.dumps(notification))
                    redis_client.ltrim("frontend:notifications", 0, 99)

                    # Also store in system_shutdown category
                    redis_client.lpush(
                        "frontend:system_shutdown", json.dumps(notification))
                    redis_client.ltrim("frontend:system_shutdown", 0, 49)

                    # Update system status
                    system_status = json.loads(redis_client.get(
                        "frontend:system:status") or "{}")
                    system_status["running"] = False
                    system_status["timestamp"] = time.time()
                    system_status["shutdown_time"] = time.time()
                    redis_client.set("frontend:system:status",
                                     json.dumps(system_status))

                    logger.info("Shutdown notification sent to frontend")
                except Exception as e:
                    logger.error(f"Error sending shutdown notification: {e}")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
