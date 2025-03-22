#!/usr/bin/env python3
"""
Unified Trading System

This is the main entry point for the integrated GPU-accelerated trading system that brings
together all components:
1. Data Pipeline (market data, WebSockets, order book)
2. ML Engine (GPU-accelerated models and inference)
3. Execution System (order management and routing)
4. Monitoring & Reporting
5. Continual Learning

The system is designed for the NVIDIA GH200 Grace Hopper Superchip but works with
any NVIDIA GPU with suitable optimizations applied automatically.

Usage:
    python unified_system.py --config config.json
    python unified_system.py --debug --no-gpu
"""

from monitoring_system.monitoring_system import MonitoringSystem
from learning_engine import ContinualLearningSystem
from stock_selection_engine import GPUStockSelectionSystem
from trading_engine import TradingEngine
from ml_engine import MLModelTrainer
from data_pipeline import DataPipeline
from gpu_utils import get_gpu_memory_usage
import tensorflow as tf
import redis
import os
import sys
import time
import json
import logging
import argparse
import threading
import signal
import asyncio
import redis
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/unified_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("unified_system")

# Import custom modules


class UnifiedTradingSystem:
    """
    Unified Trading System that integrates all components of the platform.
    """

    def __init__(self, config_path=None, debug=False, use_gpu=True):
        """
        Initialize the unified trading system.

        Args:
            config_path: Path to configuration file (JSON)
            debug: Enable debug logging
            use_gpu: Whether to use GPU acceleration
        """
        # Setup logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize GPU environment if enabled
        self.use_gpu = use_gpu
        if self.use_gpu:
            logger.info("Initializing GPU environment")
            self.gpu_accelerator = self._initialize_gpu()
        else:
            logger.info("GPU acceleration disabled")
            self.gpu_accelerator = None

        # Initialize Redis connection
        self.redis = self._initialize_redis()

        # Initialize components
        self.components = {}
        self.running = False
        self.event_loop = None
        self.thread_executor = ThreadPoolExecutor(max_workers=10)

        logger.info("Unified Trading System initialized")

    def _load_config(self, config_path):
        """Load configuration from file with defaults"""
        default_config = {
            'redis': {
                'host': os.environ.get('REDIS_HOST', 'localhost'),
                'port': int(os.environ.get('REDIS_PORT', 6380)),
                'db': int(os.environ.get('REDIS_DB', 0)),
                'username': os.environ.get('REDIS_USERNAME', 'default'),
                'password': os.environ.get('REDIS_PASSWORD', 'trading_system_2025')
            },
            'api_keys': {
                'polygon': os.environ.get('POLYGON_API_KEY', ''),
                'unusual_whales': os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
                'alpaca': {
                    'api_key': os.environ.get('ALPACA_API_KEY', ''),
                    'api_secret': os.environ.get('ALPACA_API_SECRET', ''),
                    'base_url': os.environ.get('ALPACA_API_URL', 'https://paper-api.alpaca.markets')
                }
            },
            'system': {
                'max_exposure': float(os.environ.get('MAX_EXPOSURE', '5000.0')),
                'market_hours_only': True,
                'data_dir': os.environ.get('DATA_DIR', './data'),
                'models_dir': os.environ.get('MODELS_DIR', './models'),
                'logs_dir': os.environ.get('LOGS_DIR', './logs'),
                'use_gpu': self.use_gpu,
                'continual_learning': {
                    'enabled': True,
                    'daily_update_time': '23:30',
                    'full_retrain_time': '00:30'
                }
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)

                # Recursive merge configs
                self._merge_configs(default_config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(
                    f"Error loading config from {config_path}: {str(e)}")

        # Ensure directories exist
        for dir_path in [
            default_config['system']['data_dir'],
            default_config['system']['models_dir'],
            default_config['system']['logs_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)

        return default_config

    def _merge_configs(self, base, override):
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _initialize_gpu(self):
        """Initialize GPU environment with optimizations for GH200"""
        try:
            # Import GPU utilities
            from gpu_utils import (
                optimize_for_gh200,
                initialize_gpu,
                get_gpu_memory_usage,
                optimize_tensorflow_model,
                clear_gpu_memory,
                gpu_array,
                to_numpy
            )
            from ml_engine import GH200Accelerator, configure_tensorflow_for_gh200

            # Apply global optimizations for GH200
            logger.info("Applying GH200-specific optimizations")
            optimize_for_gh200()

            # Initialize GPU with base utilities
            gpu_initialized = initialize_gpu()

            if not gpu_initialized:
                logger.warning(
                    "No compatible GPU detected, falling back to CPU")
                return False

            logger.info("GPU hardware initialized successfully")

            # Check for TensorFlow, TensorRT, and CuPy availability
            try:
                # Check TensorFlow
                import tensorflow as tf
                has_tensorflow = True
                tf_version = tf.__version__
                logger.info(f"TensorFlow {tf_version} detected")

                # Check for TensorRT support in TensorFlow
                try:
                    from tensorflow.python.compiler.tensorrt import trt_convert as trt
                    has_tensorrt = True
                    tensorrt_version = trt.__version__ if hasattr(
                        trt, '__version__') else "Unknown"
                    logger.info(
                        f"TensorRT integration for TensorFlow detected (version: {tensorrt_version})")
                except ImportError:
                    has_tensorrt = False
                    logger.warning("TensorRT not available for TensorFlow")

                # Check for GPU devices in TensorFlow
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    logger.info(
                        f"TensorFlow detected {len(physical_devices)} GPU devices")
                    # Log device details
                    for i, device in enumerate(physical_devices):
                        logger.info(f"GPU {i}: {device.name}")

                    # Configure memory growth to prevent TensorFlow from allocating all GPU memory
                    for device in physical_devices:
                        try:
                            tf.config.experimental.set_memory_growth(
                                device, True)
                            logger.info(
                                f"Enabled memory growth for {device.name}")
                        except Exception as e:
                            logger.warning(
                                f"Error setting memory growth for {device.name}: {e}")
                else:
                    logger.warning("No GPU devices detected by TensorFlow")
            except ImportError:
                has_tensorflow = False
                has_tensorrt = False
                logger.warning("TensorFlow not available")

            # Check for CuPy
            try:
                import cupy as cp
                has_cupy = True
                cupy_version = cp.__version__
                logger.info(f"CuPy {cupy_version} detected")

                # Test CuPy with a simple operation
                try:
                    x = cp.array([1, 2, 3])
                    y = cp.array([4, 5, 6])
                    z = cp.add(x, y)
                    logger.info("CuPy GPU operations test successful")

                    # Configure CuPy memory pool
                    mempool = cp.get_default_memory_pool()
                    # Use up to 80% of GPU memory
                    mempool.set_limit(fraction=0.8)
                    logger.info("CuPy memory pool configured")
                except Exception as e:
                    logger.warning(f"CuPy GPU operations test failed: {e}")
            except ImportError:
                has_cupy = False
                logger.warning("CuPy not available")

            # Initialize TensorFlow with GH200-specific configurations
            logger.info("Configuring TensorFlow for GH200")
            tf_configured = configure_tensorflow_for_gh200()
            if tf_configured:
                logger.info("TensorFlow configured successfully for GH200")
            else:
                logger.warning(
                    "TensorFlow configuration for GH200 was not fully successful")

            # Create GH200Accelerator for advanced GPU capabilities
            logger.info("Initializing GH200Accelerator")
            accelerator = GH200Accelerator()

            # Set additional properties based on our checks
            if not hasattr(accelerator, 'has_tensorflow_gpu'):
                accelerator.has_tensorflow_gpu = has_tensorflow and bool(
                    physical_devices) if 'physical_devices' in locals() else False

            if not hasattr(accelerator, 'has_tensorrt'):
                accelerator.has_tensorrt = has_tensorrt

            if not hasattr(accelerator, 'has_cupy_gpu'):
                accelerator.has_cupy_gpu = has_cupy

            # Add TensorRT configuration if available
            if has_tensorrt:
                accelerator.tensorrt_config = {
                    'version': tensorrt_version if 'tensorrt_version' in locals() else "Unknown",
                    'precision_modes': ['FP32', 'FP16', 'INT8'],
                    'max_workspace_size_bytes': 8 * (1024**3),  # 8GB
                    'max_batch_size': 64
                }

            # Add TensorFlow version info
            if has_tensorflow:
                accelerator.tensorflow_version = tf_version if 'tf_version' in locals() else "Unknown"

            # Add CuPy version info
            if has_cupy:
                accelerator.cupy_version = cupy_version if 'cupy_version' in locals() else "Unknown"

            # Log GPU capabilities
            capabilities = []
            if accelerator.has_tensorflow_gpu:
                capabilities.append("TensorFlow GPU")
            if accelerator.has_cupy_gpu:
                capabilities.append("CuPy")
            if accelerator.has_tensorrt:
                capabilities.append("TensorRT")

            logger.info(
                f"GPU acceleration enabled with: {', '.join(capabilities)}")

            # Report GPU memory information
            mem_info = get_gpu_memory_usage()
            if mem_info:
                used_mb, total_mb = mem_info
                free_mb = total_mb - used_mb
                logger.info(
                    f"GPU memory: {free_mb/(1024**2):.2f}GB free / {total_mb/(1024**2):.2f}GB total")

                # Store in Redis for monitoring if available
                if self.redis:
                    self.redis.hset("system:gpu", "total_memory_mb", total_mb)
                    self.redis.hset("system:gpu", "free_memory_mb", free_mb)
                    self.redis.hset("system:gpu", "used_memory_mb", used_mb)
                    self.redis.hset("system:gpu", "tensorflow_gpu", str(
                        accelerator.has_tensorflow_gpu))
                    self.redis.hset("system:gpu", "cupy_gpu",
                                    str(accelerator.has_cupy_gpu))
                    self.redis.hset("system:gpu", "tensorrt",
                                    str(accelerator.has_tensorrt))
                    self.redis.hset("system:gpu", "device_name",
                                    accelerator.device_name or "unknown")

                    # Store additional information
                    if has_tensorflow:
                        self.redis.hset("system:gpu", "tensorflow_version",
                                        accelerator.tensorflow_version if hasattr(accelerator, 'tensorflow_version') else "Unknown")
                    if has_tensorrt:
                        self.redis.hset("system:gpu", "tensorrt_version",
                                        accelerator.tensorrt_config['version'] if hasattr(accelerator, 'tensorrt_config') else "Unknown")
                    if has_cupy:
                        self.redis.hset("system:gpu", "cupy_version",
                                        accelerator.cupy_version if hasattr(accelerator, 'cupy_version') else "Unknown")

            return accelerator
        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}")
            return None

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            # Ensure password is properly set
            password = self.config['redis']['password']
            if not password and 'REDIS_PASSWORD' in os.environ:
                password = os.environ['REDIS_PASSWORD']

            redis_client = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                username=self.config['redis']['username'],
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            redis_client.ping()
            logger.info(
                f"Connected to Redis at {self.config['redis']['host']}:{self.config['redis']['port']}")
            return redis_client
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            logger.error(
                f"Redis config: host={self.config['redis']['host']}, port={self.config['redis']['port']}, password={'*****' if self.config['redis']['password'] else 'None'}")
            # Use a mock Redis client for testing/development
            return None

    def _handle_signal(self, signum, frame):
        """Handle termination signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    async def _initialize_async_components(self):
        """Initialize async components"""
        try:
            # Data pipeline - handles all market data streams
            logger.info("Initializing data pipeline...")
            data_pipeline = DataPipeline(
                redis_client=self.redis,
                use_gpu=self.use_gpu
            )
            self.components['data_pipeline'] = data_pipeline

            # Peak detection monitor - identifies trading patterns
            logger.info("Initializing trading engine...")
            trading_engine = TradingEngine(
                redis_client=self.redis,
                data_pipeline=self.components.get('data_pipeline')
            )
            self.components['trading_engine'] = trading_engine

            # Stock selection system - identifies trading candidates
            logger.info("Initializing stock selection system...")
            stock_selection = GPUStockSelectionSystem(
                redis_client=self.redis,
                use_gpu=self.use_gpu
            )
            self.components['stock_selection'] = stock_selection

            logger.info("Async components initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing async components: {str(e)}")
            return False

    def _initialize_components(self):
        """Initialize system components"""
        try:
            # ML model trainer - manages all ML models
            logger.info("Initializing ML model trainer...")
            model_trainer = MLModelTrainer(
                redis_client=self.redis,
                data_loader=None  # Will be set after data pipeline is initialized
            )
            self.components['model_trainer'] = model_trainer

            # Note: Trading system and execution system functionality is now integrated in TradingEngine
            # The trading_engine component initialized earlier handles all trading and execution logic

            # Monitoring system - metrics and alerting
            logger.info("Initializing monitoring system...")
            monitoring_system = MonitoringSystem(
                redis_client=self.redis,
                config=self.config
            )
            self.components['monitoring_system'] = monitoring_system

            # Continual learning system - automated model updates
            if self.config['system']['continual_learning']['enabled']:
                logger.info("Initializing continual learning system...")
                continual_learning = ContinualLearningSystem(
                    redis_client=self.redis,
                    data_loader=None,  # Will be set after data pipeline is initialized
                    model_trainer=model_trainer
                )
                self.components['continual_learning'] = continual_learning

            logger.info("All components initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False

    async def _start_async_components(self):
        """Start asynchronous components"""
        try:
            for name, component in self.components.items():
                if hasattr(component, 'start') and asyncio.iscoroutinefunction(component.start):
                    logger.info(f"Starting async component: {name}")
                    await component.start()

            logger.info("All async components started")
            return True
        except Exception as e:
            logger.error(f"Error starting async components: {str(e)}")
            return False

    def _start_components(self):
        """Start synchronous components"""
        try:
            for name, component in self.components.items():
                if hasattr(component, 'start') and not asyncio.iscoroutinefunction(component.start):
                    logger.info(f"Starting component: {name}")
                    component.start()

            logger.info("All components started")
            return True
        except Exception as e:
            logger.error(f"Error starting components: {str(e)}")
            return False

    async def _stop_async_components(self):
        """Stop asynchronous components"""
        for name, component in reversed(list(self.components.items())):
            if hasattr(component, 'stop') and asyncio.iscoroutinefunction(component.stop):
                logger.info(f"Stopping async component: {name}")
                try:
                    await component.stop()
                except Exception as e:
                    logger.error(f"Error stopping {name}: {str(e)}")

    def _stop_components(self):
        """Stop synchronous components"""
        for name, component in reversed(list(self.components.items())):
            if hasattr(component, 'stop') and not asyncio.iscoroutinefunction(component.stop):
                logger.info(f"Stopping component: {name}")
                try:
                    component.stop()
                except Exception as e:
                    logger.error(f"Error stopping {name}: {str(e)}")

    async def _main_loop(self):
        """Main async event loop"""
        logger.info("Starting main event loop")

        # Check system health periodically
        while self.running:
            try:
                # Monitor GPU memory if available
                if self.use_gpu and self.gpu_accelerator:
                    try:
                        from gpu_utils import get_gpu_memory_usage, clear_gpu_memory

                        # Get memory usage
                        mem_info = get_gpu_memory_usage()
                        if mem_info:
                            used_mb, total_mb = mem_info
                            usage_pct = (used_mb / total_mb) * \
                                100 if total_mb > 0 else 0

                            # Log memory usage at different levels based on usage
                            if usage_pct > 80.0:
                                logger.warning(
                                    f"GPU Memory: {used_mb/(1024**2):.2f}GB/{total_mb/(1024**2):.2f}GB ({usage_pct:.1f}%)")
                            else:
                                logger.debug(
                                    f"GPU Memory: {used_mb/(1024**2):.2f}GB/{total_mb/(1024**2):.2f}GB ({usage_pct:.1f}%)")

                            # Store in Redis for monitoring with timestamp
                            if self.redis:
                                timestamp = int(time.time())
                                memory_data = {
                                    "timestamp": timestamp,
                                    "used_mb": int(used_mb),
                                    "total_mb": int(total_mb),
                                    "usage_pct": float(usage_pct),
                                    "free_mb": int(total_mb - used_mb)
                                }

                                # Store current metrics
                                self.redis.hset(
                                    "system:metrics", "gpu_memory_used_mb", used_mb)
                                self.redis.hset(
                                    "system:metrics", "gpu_memory_total_mb", total_mb)
                                self.redis.hset(
                                    "system:metrics", "gpu_memory_pct", usage_pct)

                                # Store historical data (keep last 100 points)
                                self.redis.lpush(
                                    "system:gpu_memory_history", json.dumps(memory_data))
                                self.redis.ltrim(
                                    "system:gpu_memory_history", 0, 99)

                                # Store framework-specific memory usage if available
                                if hasattr(self.gpu_accelerator, 'has_tensorflow_gpu') and self.gpu_accelerator.has_tensorflow_gpu:
                                    try:
                                        import tensorflow as tf
                                        # Get TensorFlow memory info
                                        tf_mem_info = {}
                                        for device in tf.config.list_physical_devices('GPU'):
                                            tf_mem_info[device.name] = tf.config.experimental.get_memory_info(
                                                device)

                                        if tf_mem_info:
                                            self.redis.hset(
                                                "system:metrics", "tensorflow_memory_info", json.dumps(tf_mem_info))
                                    except Exception as e:
                                        logger.debug(
                                            f"Error getting TensorFlow memory info: {e}")

                                if hasattr(self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu:
                                    try:
                                        import cupy as cp
                                        # Get CuPy memory pool info
                                        mempool = cp.get_default_memory_pool()
                                        pinned_mempool = cp.get_default_pinned_memory_pool()

                                        cupy_mem_info = {
                                            "used_bytes": mempool.used_bytes(),
                                            "total_bytes": mempool.total_bytes(),
                                            "pinned_used_bytes": pinned_mempool.used_bytes(),
                                            "pinned_total_bytes": pinned_mempool.total_bytes()
                                        }

                                        self.redis.hset(
                                            "system:metrics", "cupy_memory_info", json.dumps(cupy_mem_info))
                                    except Exception as e:
                                        logger.debug(
                                            f"Error getting CuPy memory info: {e}")

                            # Progressive GPU memory management based on usage
                            if usage_pct > 95.0:
                                logger.warning(
                                    f"Critical GPU memory usage: {usage_pct:.1f}%, performing emergency cleanup")
                                # Force aggressive cleanup
                                clear_gpu_memory(force_full_cleanup=True)

                                # Framework-specific cleanup
                                if hasattr(self.gpu_accelerator, 'has_tensorflow_gpu') and self.gpu_accelerator.has_tensorflow_gpu:
                                    try:
                                        import tensorflow as tf
                                        # Clear TensorFlow memory
                                        tf.keras.backend.clear_session()

                                        # Reset TensorFlow GPU memory stats for each device
                                        for device in tf.config.list_physical_devices('GPU'):
                                            try:
                                                tf.config.experimental.reset_memory_stats(
                                                    device)
                                            except:
                                                pass

                                        logger.info(
                                            "TensorFlow memory cleared")
                                    except Exception as e:
                                        logger.warning(
                                            f"Error clearing TensorFlow memory: {e}")

                                if hasattr(self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu:
                                    try:
                                        import cupy as cp
                                        # Clear CuPy memory pools
                                        cp.get_default_memory_pool().free_all_blocks()
                                        cp.get_default_pinned_memory_pool().free_all_blocks()
                                        logger.info(
                                            "CuPy memory pools cleared")
                                    except Exception as e:
                                        logger.warning(
                                            f"Error clearing CuPy memory: {e}")

                                # If we have an accelerator with TensorFlow, clear its session too
                                if hasattr(self.gpu_accelerator, 'clear_gpu_memory'):
                                    self.gpu_accelerator.clear_gpu_memory()

                                # Force garbage collection
                                import gc
                                for i in range(3):  # Run multiple cycles for better cleanup
                                    gc.collect(i)
                                logger.info("Garbage collection completed")

                            elif usage_pct > 85.0:
                                logger.warning(
                                    f"High GPU memory usage: {usage_pct:.1f}%, clearing cache")
                                # Standard cleanup
                                clear_gpu_memory()

                                # Framework-specific gentle cleanup
                                if hasattr(self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu:
                                    try:
                                        import cupy as cp
                                        # Clear unused memory blocks
                                        cp.get_default_memory_pool().free_all_blocks()
                                        logger.info(
                                            "CuPy memory blocks cleared")
                                    except Exception as e:
                                        logger.warning(
                                            f"Error clearing CuPy memory blocks: {e}")

                            elif usage_pct > 75.0:
                                # Gentle cleanup during scheduled maintenance
                                scheduled_cleanup_interval = int(
                                    os.environ.get('GPU_MEMORY_CLEANUP_INTERVAL', 300))
                                current_time = int(time.time())
                                last_cleanup_time = int(self.redis.get(
                                    "system:last_gpu_cleanup") or 0) if self.redis else 0

                                if current_time - last_cleanup_time > scheduled_cleanup_interval:
                                    logger.info(
                                        f"Performing scheduled GPU memory cleanup ({usage_pct:.1f}%)")
                                    clear_gpu_memory()
                                    if self.redis:
                                        self.redis.set(
                                            "system:last_gpu_cleanup", current_time)
                    except Exception as e:
                        logger.error(f"Error monitoring GPU memory: {str(e)}")

                # Check health of components
                for name, component in self.components.items():
                    if hasattr(component, 'running'):
                        if not component.running:
                            logger.warning(
                                f"Component {name} is not running, attempting to restart")
                            if hasattr(component, 'start'):
                                if asyncio.iscoroutinefunction(component.start):
                                    await component.start()
                                else:
                                    component.start()

                # Wait before next check
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)

        logger.info("Main event loop ended")

    async def _run_async(self):
        """Run the async parts of the system"""
        try:
            # Initialize and start async components
            if await self._initialize_async_components():
                if await self._start_async_components():
                    # Start the main loop
                    await self._main_loop()

            # Stop async components
            await self._stop_async_components()
        except Exception as e:
            logger.error(f"Error in async operation: {str(e)}")
        finally:
            self.running = False

    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(self._run_async())
        except Exception as e:
            logger.error(f"Error in event loop thread: {str(e)}")
        finally:
            self.event_loop.close()

    def start(self):
        """Start the unified trading system"""
        if self.running:
            logger.warning("System is already running")
            return False

        logger.info("Starting unified trading system")
        self.running = True

        # Initialize components
        if not self._initialize_components():
            logger.error("Failed to initialize components")
            self.running = False
            return False

        # Start components
        if not self._start_components():
            logger.error("Failed to start components")
            self.running = False
            return False

        # Start the async event loop in a separate thread
        self.event_loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True)
        self.event_loop_thread.start()

        logger.info("Unified trading system started successfully")
        return True

    def stop(self):
        """Stop the unified trading system"""
        if not self.running:
            logger.warning("System is not running")
            return False

        logger.info("Stopping unified trading system")
        self.running = False

        # Stop the event loop
        if self.event_loop:
            for task in asyncio.all_tasks(self.event_loop):
                task.cancel()

            # Wait for the event loop thread to finish
            if self.event_loop_thread and self.event_loop_thread.is_alive():
                self.event_loop_thread.join(timeout=10.0)

        # Stop synchronous components
        self._stop_components()

        # Clean up GPU resources if using GPU
        if self.use_gpu and self.gpu_accelerator:
            logger.info("Cleaning up GPU resources before shutdown")

            try:
                # Import necessary utilities
                from gpu_utils import clear_gpu_memory

                # Framework-specific cleanup first
                # TensorFlow cleanup
                if hasattr(self.gpu_accelerator, 'has_tensorflow_gpu') and self.gpu_accelerator.has_tensorflow_gpu:
                    try:
                        import tensorflow as tf

                        # Clear TensorFlow session
                        tf.keras.backend.clear_session()
                        logger.info("TensorFlow session cleared")

                        # Reset TensorFlow GPU memory stats for each device
                        for device in tf.config.list_physical_devices('GPU'):
                            try:
                                tf.config.experimental.reset_memory_stats(
                                    device)
                                logger.info(
                                    f"Reset memory stats for {device.name}")
                            except:
                                pass

                        # For a more thorough cleanup, reset the entire graph
                        try:
                            import tensorflow.compat.v1 as tf1
                            tf1.reset_default_graph()
                            if hasattr(tf1, 'Session'):
                                session = tf1.Session()
                                session.close()
                            logger.info("TensorFlow graph reset")
                        except Exception as e:
                            logger.debug(
                                f"TensorFlow v1 graph reset not available: {e}")
                    except Exception as e:
                        logger.warning(
                            f"Error cleaning up TensorFlow resources: {e}")

                # TensorRT cleanup
                if hasattr(self.gpu_accelerator, 'has_tensorrt') and self.gpu_accelerator.has_tensorrt:
                    try:
                        # TensorRT cleanup is mostly handled through TensorFlow
                        # But we can explicitly clear any cached engines
                        if hasattr(self.gpu_accelerator, 'tensorrt_config'):
                            logger.info("TensorRT resources released")
                    except Exception as e:
                        logger.warning(
                            f"Error cleaning up TensorRT resources: {e}")

                # CuPy cleanup
                if hasattr(self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu:
                    try:
                        import cupy as cp

                        # Clear memory pools
                        before_free = 0
                        try:
                            # Get memory usage before clearing
                            before_free, before_total = cp.cuda.runtime.memGetInfo()
                        except:
                            pass

                        # Clear all memory pools
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()

                        # Get memory usage after clearing
                        try:
                            after_free, after_total = cp.cuda.runtime.memGetInfo()
                            freed_mb = (after_free - before_free) / \
                                (1024 * 1024)
                            logger.info(
                                f"CuPy memory pools cleared: {freed_mb:.2f}MB freed")
                        except:
                            logger.info("CuPy memory pools cleared")
                    except Exception as e:
                        logger.warning(
                            f"Error cleaning up CuPy resources: {e}")

                # General GPU memory cleanup
                clear_gpu_memory(force_full_cleanup=True)
                logger.info("GPU memory cleared")

                # If accelerator has its own cleanup method, use it
                if hasattr(self.gpu_accelerator, 'clear_gpu_memory'):
                    self.gpu_accelerator.clear_gpu_memory()
                    logger.info("Accelerator GPU memory cleared")

                # Force garbage collection - multiple passes for thorough cleanup
                import gc
                for i in range(3):
                    collected = gc.collect(i)
                logger.info(
                    f"Garbage collection completed: {collected} objects collected")

                # Record final GPU memory state
                try:
                    mem_info = get_gpu_memory_usage()
                    if mem_info and self.redis:
                        used_mb, total_mb = mem_info
                        free_mb = total_mb - used_mb
                        usage_pct = (used_mb / total_mb) * \
                            100 if total_mb > 0 else 0

                        self.redis.hset("system:shutdown",
                                        "gpu_memory_used_mb", int(used_mb))
                        self.redis.hset("system:shutdown",
                                        "gpu_memory_free_mb", int(free_mb))
                        self.redis.hset("system:shutdown",
                                        "gpu_memory_total_mb", int(total_mb))
                        self.redis.hset("system:shutdown",
                                        "gpu_memory_pct", float(usage_pct))
                        self.redis.hset("system:shutdown",
                                        "timestamp", int(time.time()))

                        # Store framework-specific cleanup status
                        cleanup_status = {
                            "tensorflow": hasattr(self.gpu_accelerator, 'has_tensorflow_gpu') and self.gpu_accelerator.has_tensorflow_gpu,
                            "tensorrt": hasattr(self.gpu_accelerator, 'has_tensorrt') and self.gpu_accelerator.has_tensorrt,
                            "cupy": hasattr(self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu,
                            "timestamp": int(time.time())
                        }
                        self.redis.hset(
                            "system:shutdown", "cleanup_status", json.dumps(cleanup_status))

                        logger.info(
                            f"Final GPU memory state: {used_mb/(1024**2):.2f}GB used, {free_mb/(1024**2):.2f}GB free")
                except Exception as e:
                    logger.warning(
                        f"Error recording final GPU memory state: {e}")
            except Exception as e:
                logger.error(f"Error cleaning up GPU resources: {e}")

        # Shutdown thread executor
        self.thread_executor.shutdown(wait=False)

        logger.info("Unified trading system stopped")
        return True

    def status(self):
        """Get the status of the unified trading system"""
        status = {
            "system": "running" if self.running else "stopped",
            "components": {},
            "gpu": {}
        }

        # Component status
        for name, component in self.components.items():
            if hasattr(component, 'running'):
                status["components"][name] = "running" if component.running else "stopped"
            else:
                status["components"][name] = "unknown"

        # GPU status if enabled
        if self.use_gpu and self.gpu_accelerator:
            status["gpu"]["acceleration"] = "enabled"

            # Add detailed GPU information
            if hasattr(self.gpu_accelerator, 'strategy'):
                status["gpu"]["strategy"] = self.gpu_accelerator.strategy

            # Framework support
            status["gpu"]["tensorflow"] = "enabled" if hasattr(
                self.gpu_accelerator, 'has_tensorflow_gpu') and self.gpu_accelerator.has_tensorflow_gpu else "disabled"
            status["gpu"]["cupy"] = "enabled" if hasattr(
                self.gpu_accelerator, 'has_cupy_gpu') and self.gpu_accelerator.has_cupy_gpu else "disabled"
            status["gpu"]["tensorrt"] = "enabled" if hasattr(
                self.gpu_accelerator, 'has_tensorrt') and self.gpu_accelerator.has_tensorrt else "disabled"

            # Device information
            if hasattr(self.gpu_accelerator, 'device_name') and self.gpu_accelerator.device_name:
                status["gpu"]["device_name"] = self.gpu_accelerator.device_name

            # TensorRT configuration if available
            if hasattr(self.gpu_accelerator, 'tensorrt_config'):
                status["gpu"]["tensorrt_config"] = self.gpu_accelerator.tensorrt_config

            # Memory usage
            mem_info = get_gpu_memory_usage()
            if mem_info:
                used_mb, total_mb = mem_info
                free_mb = total_mb - used_mb
                usage_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0

                status["gpu"]["memory"] = {
                    "used_mb": int(used_mb),
                    "total_mb": int(total_mb),
                    "free_mb": int(free_mb),
                    "used_gb": round(used_mb / 1024**2, 2),
                    "total_gb": round(total_mb / 1024**2, 2),
                    "free_gb": round(free_mb / 1024**2, 2),
                    "usage_pct": round(usage_pct, 2)
                }

                # Memory health status
                if usage_pct > 90:
                    status["gpu"]["memory"]["health"] = "critical"
                elif usage_pct > 75:
                    status["gpu"]["memory"]["health"] = "warning"
                else:
                    status["gpu"]["memory"]["health"] = "good"

            # Get historical memory usage from Redis if available
            if self.redis and self.redis.exists("system:gpu_memory_history"):
                try:
                    # Get last 10 memory usage points
                    history_data = self.redis.lrange(
                        "system:gpu_memory_history", 0, 9)
                    if history_data:
                        memory_history = [json.loads(item)
                                          for item in history_data]
                        status["gpu"]["memory_history"] = memory_history
                except Exception as e:
                    logger.error(f"Error getting GPU memory history: {e}")

            # Get last cleanup time
            if self.redis and self.redis.exists("system:last_gpu_cleanup"):
                try:
                    last_cleanup = int(self.redis.get(
                        "system:last_gpu_cleanup") or 0)
                    status["gpu"]["last_cleanup"] = {
                        "timestamp": last_cleanup,
                        "time_ago_seconds": int(time.time()) - last_cleanup
                    }
                except Exception as e:
                    logger.error(f"Error getting last GPU cleanup time: {e}")
        else:
            status["gpu"]["acceleration"] = "disabled"
            status["gpu"]["reason"] = "GPU acceleration was disabled by configuration" if not self.use_gpu else "No compatible GPU accelerator was found"

        return status


def main():
    """Main entry point for the unified trading system"""
    parser = argparse.ArgumentParser(description="Unified Trading System")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    args = parser.parse_args()

    # Create and start the unified trading system
    system = UnifiedTradingSystem(
        config_path=args.config,
        debug=args.debug,
        use_gpu=not args.no_gpu
    )

    if system.start():
        # Keep running until interrupted
        try:
            while system.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            system.stop()
    else:
        logger.error("Failed to start the unified trading system")
        sys.exit(1)


if __name__ == "__main__":
    main()
