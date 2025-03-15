"""
System Components for Autonomous Trading System

This module provides the core system components for the autonomous trading system,
including the SystemController, StateManager, HealthMonitor, RecoveryManager,
TimescaleDBManager, and GPU configuration functions.
"""

import os
import json
import time
import threading
import copy
import numpy as np
from sqlalchemy import create_engine, text

from src.utils.logging.logger import get_logger

# Set up logger for this module
logger = get_logger("system_components")

# System Controller Components
class StateManager:
    """Manages system state persistence."""
    
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = self.load_state() or {}
        self.lock = threading.Lock()
        self.logger = get_logger("system_components.state_manager")
        
    def get_state(self):
        """Get the current state."""
        with self.lock:
            return copy.deepcopy(self.state)
        
    def update_state(self, new_state):
        """Update the state."""
        with self.lock:
            self.state.update(new_state)
            self.persist_state()
            
    def persist_state(self):
        """Persist the state to disk."""
        with self.lock:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f)
            except Exception as e:
                self.logger.error(f"Error persisting state: {e}")
                
    def load_state(self):
        """Load the state from disk."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        return {}
        
    def reset_state(self):
        """Reset the state."""
        with self.lock:
            self.state = {}
            self.persist_state()

class HealthMonitor:
    """Monitors the health of all subsystems."""
    
    def __init__(self, check_interval=30):
        self.subsystems = {}
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.logger = get_logger("system_components.health_monitor")
        
    def register_subsystem(self, name, subsystem):
        """Register a subsystem for health monitoring."""
        self.subsystems[name] = subsystem
        
    def start(self):
        """Start health monitoring."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Health monitoring started")
        
    def stop(self):
        """Stop health monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
        
    def get_health_status(self):
        """Get the health status of all subsystems."""
        return {name: self._check_subsystem_health(subsystem) 
                for name, subsystem in self.subsystems.items()}
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                for name, subsystem in self.subsystems.items():
                    health = self._check_subsystem_health(subsystem)
                    if not health["healthy"]:
                        self.logger.warning(f"Subsystem {name} is unhealthy: {health['reason']}")
                        # Notify system controller
                        # This would be an event or callback in a real implementation
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                
            time.sleep(self.check_interval)
            
    def _check_subsystem_health(self, subsystem):
        """Check the health of a subsystem."""
        try:
            status = subsystem.get_health()
            return status
        except Exception as e:
            return {"healthy": False, "reason": str(e)}

class RecoveryManager:
    """Manages recovery from subsystem failures."""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger("system_components.recovery_manager")
        
    def handle_failure(self, subsystem_name, error):
        """Handle a subsystem failure."""
        self.logger.error(f"Handling failure in subsystem {subsystem_name}: {error}")
        
        # Determine recovery strategy based on error type
        strategy = self._get_recovery_strategy(subsystem_name, error)
        
        if strategy == "restart":
            self.logger.info(f"Attempting to restart subsystem {subsystem_name}")
            # In a real implementation, this would call back to the system controller
            return True
        elif strategy == "reconfigure":
            self.logger.info(f"Attempting to reconfigure subsystem {subsystem_name}")
            # Reconfiguration logic would go here
            return True
        else:
            self.logger.warning(f"No recovery strategy for subsystem {subsystem_name}")
            return False
            
    def _get_recovery_strategy(self, subsystem_name, error):
        """Determine the recovery strategy based on the error."""
        # This would be more sophisticated in a real implementation
        if "connection" in str(error).lower():
            return "restart"
        elif "configuration" in str(error).lower():
            return "reconfigure"
        else:
            return "restart"  # Default to restart

class SystemController:
    """Central orchestration component for the Autonomous Trading System."""
    
    def __init__(self, config):
        self.config = config
        self.subsystems = {}
        self.state_manager = StateManager(state_file="system_state.json")
        self.health_monitor = HealthMonitor(check_interval=30)  # 30 second checks
        self.recovery_manager = RecoveryManager(config=config)
        self.logger = get_logger("system_components.system_controller")
        
    def register_subsystem(self, name, subsystem):
        """Register a subsystem with the controller."""
        self.subsystems[name] = subsystem
        self.health_monitor.register_subsystem(name, subsystem)
        
    def start(self):
        """Start all subsystems in the correct order."""
        self.logger.info("Starting system controller")
        self._initialize_subsystems()
        self._start_health_monitoring()
        
    def _initialize_subsystems(self):
        """Initialize all subsystems in the correct order."""
        # Order matters: data acquisition → feature engineering → model training → trading strategy
        subsystem_order = [
            "data_acquisition", "feature_engineering", 
            "model_training", "trading_strategy"
        ]
        
        for name in subsystem_order:
            if name in self.subsystems:
                try:
                    self.logger.info(f"Initializing subsystem: {name}")
                    self.subsystems[name].initialize()
                except Exception as e:
                    self.logger.error(f"Error initializing subsystem {name}: {e}")
                    self.recovery_manager.handle_failure(name, e)
    
    def _start_health_monitoring(self):
        """Start health monitoring for all subsystems."""
        self.health_monitor.start()
        
    def get_system_status(self):
        """Get the current status of all subsystems."""
        return {
            "system_controller": "active",
            "subsystems": {name: subsystem.get_status() for name, subsystem in self.subsystems.items()},
            "health": self.health_monitor.get_health_status(),
            "state": self.state_manager.get_state()
        }
        
    def handle_subsystem_failure(self, subsystem_name, error):
        """Handle a subsystem failure."""
        self.logger.error(f"Subsystem failure detected: {subsystem_name}, Error: {error}")
        self.recovery_manager.handle_failure(subsystem_name, error)
        
    def restart_subsystem(self, subsystem_name):
        """Restart a specific subsystem."""
        if subsystem_name in self.subsystems:
            try:
                self.logger.info(f"Restarting subsystem: {subsystem_name}")
                self.subsystems[subsystem_name].shutdown()
                self.subsystems[subsystem_name].initialize()
                return True
            except Exception as e:
                self.logger.error(f"Error restarting subsystem {subsystem_name}: {e}")
                return False
        return False
        
    def shutdown(self):
        """Shutdown all subsystems in the reverse order."""
        self.logger.info("Shutting down system controller")
        self.health_monitor.stop()
        
        # Reverse order for shutdown
        subsystem_order = [
            "trading_strategy", "model_training", 
            "feature_engineering", "data_acquisition"
        ]
        
        for name in subsystem_order:
            if name in self.subsystems:
                try:
                    self.logger.info(f"Shutting down subsystem: {name}")
                    self.subsystems[name].shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down subsystem {name}: {e}")

# Enhanced TimescaleDB Connection
class TimescaleDBManager:
    """Manages connections to TimescaleDB."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.engine = None
        self.connection_pool = None
        self.logger = get_logger("system_components.timescaledb_manager")
        self.performance_metrics = None
        
    def set_performance_metrics(self, metrics):
        """Set the performance metrics collector."""
        self.performance_metrics = metrics
        
    def initialize(self):
        """Initialize the database connection."""
        try:
            start_time = time.time()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                self.connection_string,
                pool_size=20,
                max_overflow=10,
                pool_recycle=3600,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                self.logger.info(f"Connected to TimescaleDB: {version}")
                
            # Create connection pool
            self.connection_pool = self.engine.pool
            
            # Check if TimescaleDB extension is installed
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
                ))
                if result.rowcount == 0:
                    self.logger.warning("TimescaleDB extension not installed")
                else:
                    self.logger.info("TimescaleDB extension installed")
            
            # Record metrics
            if self.performance_metrics:
                self.performance_metrics.record_system_latency(
                    "database_connection", time.time() - start_time
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Error initializing TimescaleDB connection: {e}")
            return False
            
    def get_connection(self):
        """Get a connection from the pool."""
        if not self.engine:
            self.initialize()
        return self.engine.connect()
        
    def execute_query(self, query, params=None):
        """Execute a query and return the results."""
        try:
            start_time = time.time()
            
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                
            # Record metrics
            if self.performance_metrics:
                self.performance_metrics.record_data_processing_time(
                    "database_query", time.time() - start_time
                )
                
            return result
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
            
    def get_table_info(self):
        """Get information about all tables in the database."""
        query = """
        SELECT 
            table_name, 
            (SELECT count(*) FROM information_schema.columns 
             WHERE table_name=t.table_name) as column_count,
            pg_total_relation_size(table_name) as size_bytes
        FROM information_schema.tables t
        WHERE table_schema = 'public'
        ORDER BY size_bytes DESC
        """
        try:
            result = self.execute_query(query)
            tables = [dict(row) for row in result]
            return tables
        except Exception as e:
            self.logger.error(f"Error getting table info: {e}")
            return []
            
    def get_hypertable_info(self):
        """Get information about all hypertables in the database."""
        query = """
        SELECT h.table_name, 
               h.schema_name,
               h.created,
               h.chunk_time_interval,
               pg_total_relation_size(format('%I.%I', h.schema_name, h.table_name)) as size_bytes
        FROM _timescaledb_catalog.hypertable h
        ORDER BY size_bytes DESC
        """
        try:
            result = self.execute_query(query)
            hypertables = [dict(row) for row in result]
            return hypertables
        except Exception as e:
            self.logger.error(f"Error getting hypertable info: {e}")
            return []
            
    def close(self):
        """Close all connections."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("TimescaleDB connections closed")

# GPU Acceleration Configuration
def configure_tensorflow_gpu():
    """Configure TensorFlow for GPU acceleration using NVIDIA TensorFlow containers."""
    try:
        import tensorflow as tf
        gpu_logger = get_logger("system_components.gpu_config")
        
        # Check if running in NVIDIA container
        is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
        if is_nvidia_container:
            gpu_logger.info("Running in NVIDIA TensorFlow container")
        
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                gpu_logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
                
                # Set visible devices if specific GPUs should be used
                # This is useful in multi-GPU environments
                gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
                if gpu_ids:
                    gpu_logger.info(f"Using GPUs: {gpu_ids}")
            except RuntimeError as e:
                gpu_logger.error(f"Error configuring GPU memory growth: {e}")
                
        # Enable mixed precision training (optimized for NVIDIA Tensor Cores)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        gpu_logger.info("Mixed precision training enabled (optimized for Tensor Cores)")
        
        # Enable XLA compilation for better performance
        tf.config.optimizer.set_jit(True)
        gpu_logger.info("XLA compilation enabled")
        
        # Apply optimizations for NVIDIA GPUs
        os.environ["TF_USE_CUDNN"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "0"  # Disable for better performance
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"   # Enable autotuning
        os.environ["TF_CUDNN_RESET_RNN_DESCRIPTOR"] = "1"  # Better memory usage for RNNs
        
        # Additional optimizations for NVIDIA containers
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Dedicate GPU threads
        os.environ["TF_GPU_THREAD_COUNT"] = "2"  # Number of GPU threads
        os.environ["TF_USE_CUDA_MALLOC_ASYNC"] = "1"  # Async memory allocation
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable oneDNN optimizations
        
        gpu_logger.info("NVIDIA GPU optimizations applied")
        
        # Log TensorFlow configuration
        gpu_logger.info(f"TensorFlow version: {tf.__version__}")
        gpu_logger.info(f"CUDA available: {tf.test.is_built_with_cuda()}")
        gpu_logger.info(f"GPU available: {tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else tf.config.list_physical_devices('GPU')}")
        
        # Log GPU information
        if gpus:
            try:
                # Get GPU device details
                for i, gpu in enumerate(gpus):
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_logger.info(f"GPU {i}: {gpu_details}")
            except Exception as e:
                gpu_logger.debug(f"Could not get detailed GPU information: {e}")
        
        return gpus
    except ImportError:
        logger.warning("TensorFlow not installed, GPU acceleration not available")
        return []
    except Exception as e:
        logger.error(f"Error configuring TensorFlow GPU: {e}")
        return []

def configure_xgboost_gpu():
    """Configure XGBoost for GPU acceleration with NVIDIA containers."""
    try:
        import xgboost as xgb
        gpu_logger = get_logger("system_components.gpu_config")
        
        # Check if running in NVIDIA container
        is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
        if is_nvidia_container:
            gpu_logger.info("Running in NVIDIA container with XGBoost support")
        
        # Check CUDA availability
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            gpu_logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        
        # Check if GPU is available for XGBoost
        try:
            # Create test parameters with optimal settings for NVIDIA GPUs
            params = {
                'tree_method': 'gpu_hist',           # Use GPU histogram for tree construction
                'predictor': 'gpu_predictor',        # Use GPU for prediction
                'gpu_id': 0,                         # Use first GPU
                'max_bin': 256,                      # Optimal for GPU memory usage
                'sampling_method': 'gradient_based', # Better performance on GPU
                'n_jobs': -1                         # Use all available cores
            }
            
            # Create small test dataset
            test_data = xgb.DMatrix(np.random.rand(10, 10), label=np.random.rand(10))
            
            # Try training a small model
            gpu_logger.info("Testing XGBoost GPU acceleration...")
            # Train model and verify it works
            xgb.train(params, test_data, num_boost_round=2)
            
            # Get GPU memory usage if possible
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_logger.info(f"GPU {i} Memory: {info.used/1024/1024:.2f}MB used / {info.total/1024/1024:.2f}MB total")
            except Exception as e:
                gpu_logger.debug(f"Could not get GPU memory information: {e}")
                
            gpu_logger.info("XGBoost GPU acceleration successfully configured")
            return True
            
        except Exception as e:
            gpu_logger.warning(f"XGBoost GPU acceleration not available: {e}")
            gpu_logger.info("Falling back to CPU for XGBoost with optimized settings")
            
            # Configure optimized CPU settings
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
            os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
            os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
            
            return False
            
    except ImportError:
        logger.warning("XGBoost not installed, GPU acceleration not available")
        return False
    except Exception as e:
        logger.error(f"Error configuring XGBoost GPU: {e}")
        return False