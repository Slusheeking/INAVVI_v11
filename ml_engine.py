#!/usr/bin/env python3
"""
ML Engine Module

This module provides a unified machine learning engine for the trading system:
1. GPU acceleration and optimization specifically for NVIDIA GH200 
2. Technical indicator calculation functions
3. Model training for various prediction tasks
4. Utility functions for feature selection, time series CV, and diagnostics
5. Reporting and monitoring functionality

The ML Engine is optimized for high-performance on NVIDIA GH200 Grace Hopper Superchips.
"""

import os
import re
import json
import time
import logging
import datetime
import numpy as np
import pandas as pd
import subprocess
import pickle
import threading

# Import joblib for saving scalers
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import ks_2samp

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, LSTM
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.getLogger('ml_engine').warning(
        "TensorFlow not available. Some functionality will be limited.")

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.getLogger('ml_engine').warning(
        "XGBoost not available. Some functionality will be limited.")

# Import CuPy with error handling
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.getLogger('ml_engine').warning(
        "CuPy not available. Some functionality will be limited.")

# Import Optuna for hyperparameter optimization if available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.getLogger('ml_engine').warning(
        "Optuna not available. Hyperparameter optimization will be disabled.")

# Import Prometheus client for metrics
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.getLogger('ml_engine').warning(
        "Prometheus client not available. Metrics will not be exposed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.environ.get(
            'LOGS_DIR', './logs'), 'ml_engine.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ml_engine')

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # Model training metrics
    MODEL_TRAINING_TIME = prom.Histogram(
        'ml_engine_model_training_time_seconds',
        'Time spent training models',
        ['model_name', 'model_type']
    )

    MODEL_EVALUATION_METRICS = prom.Gauge(
        'ml_engine_model_metrics',
        'Model evaluation metrics',
        ['model_name', 'metric']
    )

    GPU_MEMORY_USAGE = prom.Gauge(
        'ml_engine_gpu_memory_usage_bytes',
        'GPU memory usage in bytes',
        ['device']
    )

    GPU_UTILIZATION = prom.Gauge(
        'ml_engine_gpu_utilization_percent',
        'GPU utilization percentage',
        ['device']
    )

    FEATURE_IMPORTANCE = prom.Gauge(
        'ml_engine_feature_importance',
        'Feature importance values',
        ['model_name', 'feature']
    )

    PREDICTION_LATENCY = prom.Histogram(
        'ml_engine_prediction_latency_seconds',
        'Time taken to make predictions',
        ['model_name']
    )

    DRIFT_DETECTION = prom.Counter(
        'ml_engine_drift_detection_total',
        'Number of drift detections',
        ['model_name', 'result']
    )

    logger.info("Prometheus metrics initialized for ML Engine")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


#########################################
# SECTION 1: GPU ACCELERATION FUNCTIONS #
#########################################

def optimize_for_gh200():
    """Apply GH200-specific optimizations"""
    # Environment variables for GH200
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation

    # Disable aggressive XLA optimization
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=0"

    # For ARM CPU side of GH200
    os.environ["GOMP_CPU_AFFINITY"] = "0-15"  # Adjust based on Neoverse cores

    # Optimize memory transfer
    os.environ["CUDA_AUTO_BOOST"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"  # Reduced for stability

    # NVLink optimizations
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"

    # Prevent aggressive graph optimizations
    os.environ["TF_FUNCTION_JIT_COMPILE_DEFAULT"] = "0"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    logger.info("Applied GH200-specific optimizations")
    return True


def configure_tensorflow_for_gh200():
    """Configure TensorFlow specifically for GH200 architecture"""
    try:
        # Check if we're running in the TensorFlow container
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
            logger.info(
                "Running in TensorFlow container, using container's configuration")

            # Even in container, apply some critical optimizations
            # Enable TF32 computation for better performance
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

            # Enable tensor op math for better performance
            os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = "1"
            os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = "1"
            os.environ["TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32"] = "1"

            # Set TensorRT precision mode
            os.environ["TENSORRT_PRECISION_MODE"] = "FP16"

            return True

        # Set basic environment variables for GH200
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        # Configure XLA for better performance
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

        # Disable eager optimization that can cause graph update errors
        os.environ["TF_FUNCTION_JIT_COMPILE_DEFAULT"] = "0"

        # Enable eager execution for stability
        tf.config.run_functions_eagerly(True)

        # For Grace Hopper specifically
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation

        # Prevent OOM errors
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        # Increased for GH200
        os.environ["TF_GPU_HOST_MEM_LIMIT_IN_MB"] = "16000"

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

        # GPU-direct optimizations
        os.environ["CUDA_AUTO_BOOST"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_P2P_LEVEL"] = "NVL"

        # Configure memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    # More reliable than just set_memory_growth
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            # Limit to 90GB (slightly less than total for stability)
                            memory_limit=int(1024 * 1024 * 90)
                        )]  # Use 90GB as the memory limit
                    )
                    logger.info(f"Set virtual device configuration for {gpu}")
                except Exception as e:
                    logger.warning(
                        f"Error setting virtual device configuration: {e}")
                    # Fallback to memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Enable mixed precision which works well on Hopper
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                # Use mixed_float16 for better performance on GH200
                set_global_policy('mixed_float16')
                logger.info(
                    "TensorFlow configured for GH200 with mixed_float16 precision")
            except ImportError:
                logger.warning("Could not set mixed precision policy")
            return True
        else:
            logger.warning(
                "No GPUs detected by TensorFlow, using TensorFlow container")
            return False
    except Exception as e:
        logger.warning(f"Error configuring TensorFlow for GH200: {str(e)}")
        return False


class GH200Accelerator:
    """Unified class to handle GPU acceleration on GH200"""

    def __init__(self):
        self.has_tensorflow_gpu = False
        self.has_cupy_gpu = False
        self.has_tensorrt = False
        self.device_name = None
        self.device_memory = None
        self.trt_converter = None

        # Configure all frameworks
        self._configure_tensorflow()
        self._configure_cupy()
        self._configure_tensorrt()

        # Set optimal execution strategy
        self._set_execution_strategy()

    def _configure_tensorflow(self):
        """Configure TensorFlow for GH200"""
        self.has_tensorflow_gpu = configure_tensorflow_for_gh200()
        if self.has_tensorflow_gpu:
            self.device_name = tf.test.gpu_device_name()
            logger.info(f"TensorFlow using GPU device: {self.device_name}")

    def _configure_cupy(self):
        """Configure CuPy for GH200"""
        try:
            if not CUPY_AVAILABLE:
                logger.warning("CuPy not available")
                return

            # Check if CuPy can see any GPU
            if cp.cuda.runtime.getDeviceCount() > 0:
                self.has_cupy_gpu = True

                # Find and use GH200 if available
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    device_name = props["name"].decode()
                    logger.info(f"CuPy found GPU device {i}: {device_name}")
                    if "GH200" in device_name:
                        cp.cuda.Device(i).use()
                        self.device_name = device_name

                        # Configure for unified memory
                        cp.cuda.set_allocator(cp.cuda.MemoryPool(
                            cp.cuda.malloc_managed).malloc)

                        # Get memory info
                        free, total = cp.cuda.runtime.memGetInfo()
                        self.device_memory = (free, total)
                        logger.info(
                            f"Using GH200 device with {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total memory")
                        break
        except Exception as e:
            logger.warning(f"Error configuring CuPy: {e}")

    def _configure_tensorrt(self):
        """Configure TensorRT for optimized model inference"""
        try:
            if not TF_AVAILABLE:
                logger.warning(
                    "TensorFlow not available, skipping TensorRT configuration")
                return

            # Get TensorRT precision mode from environment or use FP16 as default
            precision_mode = os.environ.get("TENSORRT_PRECISION_MODE", "FP16")
            logger.info(f"Using TensorRT precision mode: {precision_mode}")

            # Initialize TensorRT converter with optimized settings for GH200
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                max_workspace_size_bytes=12000000000,  # 12GB workspace for GH200
                precision_mode=precision_mode,  # Use environment variable or default to FP16
                maximum_cached_engines=200,  # Increased for GH200's larger memory
                use_calibration=precision_mode == "INT8",  # Only needed for INT8
                # Optimize for Hopper architecture
                minimum_segment_size=3,  # Smaller segments for more conversion opportunities
                allow_build_at_runtime=True  # Allow building engines at runtime
            )

            self.trt_converter = trt.TrtGraphConverterV2(
                conversion_params=conversion_params,
                use_dynamic_shape=True
            )

            # Store additional TensorRT configuration
            self.tensorrt_config = {
                "precision_mode": precision_mode,
                "workspace_size": conversion_params.max_workspace_size_bytes,
                "max_cached_engines": conversion_params.maximum_cached_engines,
                "use_dynamic_shape": True,
                "allow_build_at_runtime": True
            }

            self.has_tensorrt = True
            logger.info(
                f"TensorRT configured successfully with {precision_mode} precision")
        except Exception as e:
            logger.warning(f"Error configuring TensorRT: {e}")
            self.has_tensorrt = False

    def _set_execution_strategy(self):
        """Set the optimal execution strategy based on available hardware"""
        if self.has_tensorflow_gpu and self.has_tensorrt:
            self.strategy = "tensorrt_optimized"
        elif self.has_tensorflow_gpu:
            self.strategy = "tensorflow_gpu_eager"
        elif self.has_cupy_gpu:
            self.strategy = "cupy_gpu_tensorflow_cpu"
        else:
            self.strategy = "cpu_only"

        logger.info(f"Using execution strategy: {self.strategy}")

    def optimize_model(self, model):
        """Optimize a TensorFlow model using TensorRT"""
        if not self.has_tensorrt:
            logger.warning("TensorRT not available, returning original model")
            return model

        try:
            # Create a unique model directory with timestamp to avoid conflicts
            import time
            model_id = int(time.time())
            temp_saved_model = f"/tmp/model_{model_id}"

            logger.info(
                f"Saving model to {temp_saved_model} for TensorRT optimization")

            # Save model with input and output signatures for better TensorRT conversion
            if hasattr(model, 'input') and hasattr(model, 'output'):
                # For Keras models, use signatures
                input_signature = []
                for input_tensor in model.inputs:
                    input_signature.append(tf.TensorSpec(
                        shape=input_tensor.shape,
                        dtype=input_tensor.dtype,
                        name=input_tensor.name.split(':')[0]
                    ))

                @tf.function(input_signature=input_signature)
                def serving_fn(*input_tensors):
                    return model(input_tensors[0] if len(input_tensors) == 1 else input_tensors)

                # Save with signatures
                tf.saved_model.save(
                    model,
                    temp_saved_model,
                    signatures={
                        'serving_default': serving_fn
                    }
                )
            else:
                # For non-Keras models or SavedModels
                tf.saved_model.save(model, temp_saved_model)

            logger.info(
                "Model saved successfully, starting TensorRT conversion")

            # Configure TensorRT conversion
            precision_mode = self.tensorrt_config["precision_mode"]

            # Create a new converter with the saved model
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=temp_saved_model,
                conversion_params=self.trt_converter.conversion_params
            )

            # Convert the model - this creates TensorRT engines
            logger.info(
                f"Converting model to TensorRT with {precision_mode} precision")
            converter.convert()

            # Build engines for common input shapes to avoid runtime compilation
            if hasattr(model, 'input_shape'):
                try:
                    # Get typical input shape from model
                    input_shape = model.input_shape
                    batch_sizes = [1, 8, 32]  # Common batch sizes

                    logger.info(
                        f"Building TensorRT engines for common batch sizes: {batch_sizes}")
                    for batch_size in batch_sizes:
                        # Create a sample input with the right shape
                        if len(input_shape) > 1:
                            # Replace batch dimension with current batch size
                            concrete_shape = list(input_shape)
                            concrete_shape[0] = batch_size
                            concrete_input = tf.random.normal(concrete_shape)

                            # Build engine for this shape
                            converter.build(input_fn=lambda: [concrete_input])
                            logger.info(
                                f"Built TensorRT engine for batch size {batch_size}")
                except Exception as shape_error:
                    logger.warning(
                        f"Error building engines for specific shapes: {shape_error}")

            # Save the optimized model
            trt_saved_model_path = f"{temp_saved_model}_trt"
            logger.info(
                f"Saving TensorRT optimized model to {trt_saved_model_path}")
            converter.save(trt_saved_model_path)

            # Load the optimized model
            logger.info("Loading TensorRT optimized model")
            optimized_model = tf.saved_model.load(trt_saved_model_path)

            # Clean up temporary directories to avoid disk space issues
            try:
                import shutil
                shutil.rmtree(temp_saved_model, ignore_errors=True)
                logger.info(
                    f"Cleaned up temporary directory {temp_saved_model}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Error cleaning up temporary directory: {cleanup_error}")

            logger.info(
                f"Model successfully optimized with TensorRT using {precision_mode} precision")
            return optimized_model
        except Exception as e:
            logger.error(f"Error optimizing model with TensorRT: {e}")
            logger.info("Returning original model as fallback")
            return model

    def get_optimal_batch_size(self):
        """Calculate optimal batch size based on GPU memory"""
        if not self.device_memory:
            return 128  # Conservative default

        free_memory = self.device_memory[0]
        # Even more conservative: use only 5% of free memory to avoid OOM
        memory_per_sample = 8000000  # Increase bytes per sample estimate
        return min(1024, max(64, int(free_memory * 0.1 / memory_per_sample)))

    def clear_gpu_memory(self):
        """Clear GPU memory to prevent fragmentation"""
        try:
            if self.has_tensorflow_gpu and TF_AVAILABLE:
                tf.keras.backend.clear_session()

            if self.has_cupy_gpu and CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

            # Force garbage collection
            import gc
            gc.collect()

            logger.info("Cleared GPU memory")
            return True
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {str(e)}")
            return False

    def create_safe_model_config(self):
        """Create a safer model configuration less prone to GPU graph errors"""
        return {
            'batch_size': 32,  # Small batch size for stability
            'learning_rate': 0.0001,  # Very low learning rate
            'optimizer': 'adam',
            'use_early_stopping': True,
            'patience': 3,  # Stop quickly if not improving
            'use_reduce_lr': True,
            'use_model_checkpoint': False,  # Disable for stability
            'max_epochs': 5  # Limit training time
        }


class GPUStatsTracker:
    """Track GPU statistics during model training"""

    def __init__(self, polling_interval=10.0):
        self.polling_interval = polling_interval
        self.stats = []
        self.running = False
        self.thread = None

    def start(self):
        """Start tracking GPU statistics"""
        if self.running:
            return

        self.running = True
        self.stats = []

        # Start tracking in a separate thread
        self.thread = threading.Thread(target=self._track_stats)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop tracking GPU statistics and return results"""
        if not self.running:
            return self.stats

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        return self.stats

    def _track_stats(self):
        """Track GPU statistics in a loop"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            while self.running:
                timestamp = time.time()
                devices_stats = []

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    device_stats = {
                        "device_id": i,
                        "name": name,
                        "memory_used": memory.used,
                        "memory_total": memory.total,
                        "gpu_utilization": utilization.gpu,
                        "memory_utilization": utilization.memory
                    }

                    devices_stats.append(device_stats)

                    # Record metrics in Prometheus if available
                    if PROMETHEUS_AVAILABLE:
                        try:
                            # Convert name to string if it's bytes
                            device_name = name.decode() if isinstance(name, bytes) else name

                            # Record memory usage
                            GPU_MEMORY_USAGE.labels(
                                device=device_name).set(memory.used)

                            # Record GPU utilization
                            GPU_UTILIZATION.labels(
                                device=device_name).set(utilization.gpu)
                        except Exception as prom_e:
                            logger.warning(
                                f"Error recording GPU metrics in Prometheus: {prom_e}")

                self.stats.append({
                    "timestamp": timestamp,
                    "devices": devices_stats
                })

                time.sleep(self.polling_interval)

        except ImportError:
            logger.warning("pynvml not available, cannot track GPU statistics")
        except Exception as e:
            logger.error(f"Error tracking GPU statistics: {str(e)}")
        finally:
            self.running = False


class SlackReporter:
    """Report model training results to Slack"""

    def __init__(self, webhook_url=None, bot_token=None, channel='#system-notifications'):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel = channel

    def report_error(self, message, phase=None):
        """Report an error to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = f"❌ *Error*: {message}"
            if phase:
                text += f"\n*Phase*: {phase}"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")

    def report_model_metrics(self, model_name, metrics, training_time):
        """Report model metrics to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = f"✅ *{model_name.replace('_', ' ').title()} Model Trained*"
            text += f"\n*Training Time*: {training_time:.2f} seconds"

            # Add metrics
            text += "\n*Metrics*:("
            for key, value in metrics.items():
                if key != 'feature_importance' and key != 'training_history':
                    if isinstance(value, float):
                        text += f"\n- {key}: {value:.4f}"
                    else:
                        text += f"\n- {key}: {value}"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")

    def report_training_complete(self, total_time, models_status, gpu_stats=None):
        """Report training completion to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = "🎉 *Model Training Complete*"
            text += f"\n*Total Training Time*: {total_time:.2f} seconds"

            # Add model status
            text += "\n*Models Status*:("
            for model_name, status in models_status.items():
                if status.get('success', False):
                    text += f"\n- {model_name}: ✅"
                else:
                    text += f"\n- {model_name}: ❌ ({status.get('error', 'Unknown error')})"

            # Add GPU stats if available
            if gpu_stats:
                text += "\n*GPU Statistics*:("

                # Calculate average utilization
                avg_gpu_util = np.mean(
                    [stat['devices'][0]['gpu_utilization'] for stat in gpu_stats if stat['devices']])
                max_gpu_util = np.max(
                    [stat['devices'][0]['gpu_utilization'] for stat in gpu_stats if stat['devices']])
                avg_mem_util = np.mean(
                    [stat['devices'][0]['memory_utilization'] for stat in gpu_stats if stat['devices']])
                max_mem_used = np.max(
                    [stat['devices'][0]['memory_used'] for stat in gpu_stats if stat['devices']])

                text += f"\n- Average GPU Utilization: {avg_gpu_util:.1f}%"
                text += f"\n- Peak GPU Utilization: {max_gpu_util:.1f}%"
                text += f"\n- Average Memory Utilization: {avg_mem_util:.1f}%"
                text += f"\n- Peak Memory Used: {max_mem_used / (1024**2):.1f} MB"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")


def run_diagnostics():
    """Run diagnostics to identify GPU configuration issues"""
    logger.info("=== Running GPU Diagnostics ===")

    results = {
        "nvidia_smi": None,
        "tensorflow_gpu": None,
        "cuda_version": None,
        "cudnn_version": None,
        "cupy_version": None,
        "tensorflow_build_info": None,
        "nvcc_version": None,
        "gh200_specific": None,
        "system_libraries": None,
        "latency_benchmark": None
    }

    # Check NVIDIA driver
    try:
        results["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi"]).decode()
        logger.info("NVIDIA driver detected")
    except Exception as e:
        results["nvidia_smi"] = f"Failed to run nvidia-smi: {str(e)}"
        logger.warning(f"NVIDIA driver issue: {str(e)}")

    # Check TensorFlow GPU
    try:
        if TF_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            results["tensorflow_gpu"] = [gpu.name for gpu in gpus]
            if gpus:
                logger.info(
                    f"TensorFlow detected {len(gpus)} GPU(s): {results['tensorflow_gpu']}")
            else:
                logger.warning("TensorFlow did not detect any GPUs")
    except Exception as e:
        results["tensorflow_gpu"] = f"Error: {str(e)}"
        logger.error(f"TensorFlow GPU detection error: {str(e)}")

    # Check CUDA version
    try:
        if CUPY_AVAILABLE:
            results["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
            results["cupy_version"] = cp.__version__
            logger.info(
                f"CUDA version: {results['cuda_version']}, CuPy version: {results['cupy_version']}")

            # Check for GH200 specifically
            gh200_info = []
            for i in range(cp.cuda.runtime.getDeviceCount()):
                props = cp.cuda.runtime.getDeviceProperties(i)
                device_name = props["name"].decode()
                if "GH200" in device_name:
                    gh200_info.append({
                        "device_id": i,
                        "name": device_name,
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "total_memory": props["totalGlobalMem"]
                    })

            if gh200_info:
                results["gh200_specific"] = gh200_info
                logger.info(f"GH200 GPU detected: {gh200_info}")

                # Run basic performance benchmark if GH200 is detected
                try:
                    # Create a dummy matrix for benchmark
                    size = 10000
                    a_gpu = cp.random.rand(size, size, dtype=cp.float32)
                    b_gpu = cp.random.rand(size, size, dtype=cp.float32)

                    # Warm up
                    _ = cp.dot(a_gpu, b_gpu)
                    cp.cuda.Stream.null.synchronize()

                    # Benchmark
                    start = time.time()
                    _ = cp.dot(a_gpu, b_gpu)
                    cp.cuda.Stream.null.synchronize()
                    end = time.time()

                    latency = (end - start) * 1000  # ms
                    results["latency_benchmark"] = {
                        "operation": f"Matrix multiplication ({size}x{size})",
                        "time_ms": latency,
                        "throughput_gflops": 2 * size**3 / (end - start) / 1e9
                    }

                    logger.info(
                        f"GH200 Benchmark: Matrix multiplication {size}x{size} took {latency:.2f} ms")
                except Exception as bench_error:
                    logger.warning(f"Benchmark error: {str(bench_error)}")
            else:
                logger.warning("No GH200 GPU detected")
    except ImportError:
        results["cuda_version"] = "CuPy not available"
        logger.warning("CuPy not available, cannot check CUDA version")
    except Exception as e:
        results["cuda_version"] = f"Error: {str(e)}"
        logger.error(f"Error checking CUDA version: {str(e)}")

    # Check system libraries relevant to GPU operation
    try:
        # Get library versions using ldconfig
        lib_output = subprocess.check_output(["ldconfig", "-p"]).decode()
        libraries = {
            "libcuda": None,
            "libcudart": None,
            "libcudnn": None,
            "libnccl": None,
            "libtensorflow": None
        }

        for lib in libraries.keys():
            match = re.search(f"{lib}[^ ]* => ([^ ]+)", lib_output)
            if match:
                lib_path = match.group(1)
                libraries[lib] = lib_path

        results["system_libraries"] = libraries
    except Exception as e:
        logger.warning(f"Error checking system libraries: {str(e)}")

    return results


###########################################
# SECTION 2: TECHNICAL INDICATOR FUNCTIONS #
###########################################

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return np.array([np.nan] * len(data))

    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(data)
    ema[period-1] = np.mean(data[:period])

    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(data) < slow_period + signal_period:
        return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram (MACD line - signal line)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(data) < period:
        return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

    # Calculate rolling mean (middle band)
    rolling_mean = np.array([np.nan] * len(data))
    for i in range(period-1, len(data)):
        rolling_mean[i] = np.mean(data[i-period+1:i+1])

    # Calculate rolling standard deviation
    rolling_std = np.array([np.nan] * len(data))
    for i in range(period-1, len(data)):
        rolling_std[i] = np.std(data[i-period+1:i+1])

    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, rolling_mean, lower_band


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(close) < period + 1:
        return np.array([np.nan] * len(close))

    # Calculate True Range (TR)
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate +DM and -DM
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))

    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = np.zeros(len(close))
    smoothed_plus_dm = np.zeros(len(close))
    smoothed_minus_dm = np.zeros(len(close))

    # Initialize with simple averages
    smoothed_tr[period] = np.sum(tr[1:period+1])
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

    # Calculate smoothed values
    for i in range(period+1, len(close)):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - \
            (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - \
            (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    # Calculate +DI and -DI
    # Add small epsilon (1e-8) to prevent division by zero
    plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-8))
    minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-8))

    # Calculate DX and ADX
    # Add small epsilon (1e-8) to prevent division by zero
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

    # Calculate ADX (smoothed DX)
    adx = np.zeros(len(close))
    adx[2*period-1] = np.mean(dx[period:2*period])

    for i in range(2*period, len(close)):
        adx[i] = ((adx[i-1] * (period-1)) + dx[i]) / period

    return adx


def calculate_obv(close, volume):
    """Calculate On-Balance Volume (OBV)"""
    obv = np.zeros(len(close))

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    return obv


############################################
# SECTION 3: ML UTILITY FUNCTIONS          #
############################################

def select_features(X, y, problem_type='regression', method='importance', threshold=0.01, n_features=20):
    """Select most important features using specified method"""
    logger.info(f"Performing feature selection using {method} method")

    try:
        if method == 'importance':
            # Use a simple model to get feature importances
            if problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42)
            else:
                model = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=42)

            model.fit(X, y)
            importances = model.feature_importances_

            # Select features with importance above threshold
            selected_features = X.columns[importances > threshold]

            # Ensure we have at least 5 features
            if len(selected_features) < 5:
                # Take top 5 features by importance
                selected_features = X.columns[np.argsort(importances)[-5:]]

            logger.info(
                f"Selected {len(selected_features)} features using importance threshold")

            return X[selected_features]

        elif method == 'rfe':
            # Use Recursive Feature Elimination
            if problem_type == 'classification':
                estimator = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42)
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=42)

            selector = RFE(estimator, n_features_to_select=min(
                n_features, X.shape[1]), step=1)
            selector = selector.fit(X, y)

            selected_features = X.columns[selector.support_]
            logger.info(
                f"Selected {len(selected_features)} features using RFE")

            return X[selected_features]

        elif method == 'mutual_info':
            # Use mutual information
            if problem_type == 'classification':
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            # Select features with MI score above threshold
            selected_features = X.columns[mi_scores > threshold]

            # Ensure we have at least 5 features
            if len(selected_features) < 5:
                # Take top 5 features by MI score
                selected_features = X.columns[np.argsort(mi_scores)[-5:]]

            logger.info(
                f"Selected {len(selected_features)} features using mutual information")

            return X[selected_features]

        else:
            logger.warning(
                f"Unknown feature selection method: {method}. Using all features.")
            return X

    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return X


def create_time_series_splits(X, y, n_splits=5, embargo_size=10):
    """Create time series cross-validation splits with embargo period"""
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Apply embargo period to prevent data leakage
    purged_splits = []
    for train_idx, test_idx in tscv.split(X):
        # Apply embargo: remove samples at the end of train that are too close to test
        if embargo_size > 0:
            min_test_idx = min(test_idx)
            embargo_idx = range(
                max(min_test_idx - embargo_size, 0), min_test_idx)
            train_idx = np.setdiff1d(train_idx, embargo_idx)

        purged_splits.append((train_idx, test_idx))

    return purged_splits


def detect_feature_drift(current_data, reference_data, threshold=0.05):
    """Detect drift in feature distributions using KS test"""
    try:
        # Select numeric features only
        numeric_features = reference_data.select_dtypes(
            include=[np.number]).columns

        drift_detected = False
        drift_features = {}

        for feature in numeric_features:
            if feature in current_data.columns:
                # Get clean samples from both datasets
                ref_values = reference_data[feature].dropna().values
                cur_values = current_data[feature].dropna().values

                if len(ref_values) > 10 and len(cur_values) > 10:
                    # Perform KS test
                    ks_statistic, p_value = ks_2samp(ref_values, cur_values)

                    if p_value < threshold:
                        drift_detected = True
                        drift_features[feature] = {'ks_statistic': float(
                            ks_statistic), 'p_value': float(p_value)}

        return drift_detected, drift_features

    except Exception as e:
        logger.error(f"Error detecting feature drift: {str(e)}")
        return False, {}


def optimize_hyperparameters(data, model_type, config, data_processor=None):
    """Run hyperparameter optimization for a specific model type"""
    if not OPTUNA_AVAILABLE:
        logger.warning(
            "Optuna not available. Skipping hyperparameter optimization.")
        return None

    try:
        logger.info(f"Optimizing hyperparameters for {model_type} model")

        if model_type == 'signal_detection':
            return optimize_signal_detection_hyperparams(data, config, data_processor)
        elif model_type == 'price_prediction':
            return optimize_price_prediction_hyperparams(data, config, data_processor)
        elif model_type == 'exit_strategy':
            return optimize_exit_strategy_hyperparams(data, config, data_processor)
        else:
            logger.warning(
                f"Hyperparameter optimization not implemented for {model_type}")
            return None

    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        return None


def optimize_signal_detection_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for signal detection model"""
    try:
        # Prepare data
        features, target = data_processor.prepare_signal_detection_data(data)

        if len(features) == 0 or len(target) == 0:
            logger.error(
                "No valid data for signal detection hyperparameter optimization")
            return None

        # Apply feature selection if enabled
        if config['feature_selection']['enabled']:
            features = select_features(
                features, target, 'classification',
                config['feature_selection']['method'],
                config['feature_selection']['threshold'],
                config['feature_selection']['n_features']
            )

        # Create time series splits for cross-validation
        splits = create_time_series_splits(
            features, target,
            config['time_series_cv']['n_splits'],
            config['time_series_cv']['embargo_size']
        )

        # Define the objective function for optimization
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'booster': trial.suggest_categorical('booster', ['gbtree']),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1)
            }

            # Cross-validation scores
            cv_scores = []

            # Import XGBoost here to avoid issues if not available
            if not XGB_AVAILABLE:
                logger.error(
                    "XGBoost not available for hyperparameter optimization")
                return 0.0

            for train_idx, test_idx in splits:
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                dtest = xgb.DMatrix(X_test_scaled, label=y_test)

                # Train model
                model = xgb.train(
                    {k: v for k, v in params.items() if k != 'n_estimators'},
                    dtrain=dtrain,
                    num_boost_round=params['n_estimators'],
                    early_stopping_rounds=20,
                    evals=[(dtest, 'test')],
                    verbose_eval=False
                )

                # Evaluate
                y_pred = model.predict(dtest)
                auc = roc_auc_score(y_test, y_pred)
                cv_scores.append(auc)

            return np.mean(cv_scores)

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        # Get best parameters
        best_params = study.best_params
        logger.info(
            f"Best hyperparameters for signal detection: {best_params}")

        # Save best parameters
        params_path = os.path.join(
            config['models_dir'], 'signal_detection_optimized_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f)

        return best_params

    except Exception as e:
        logger.error(
            f"Error optimizing signal detection hyperparameters: {str(e)}")
        return None


def optimize_price_prediction_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for price prediction model"""
    # Implementation would be similar to signal_detection but for LSTM model
    logger.warning(
        "Price prediction hyperparameter optimization not implemented")
    return None


def optimize_exit_strategy_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for exit strategy model"""
    # Implementation would be similar to signal_detection but for regression
    logger.warning("Exit strategy hyperparameter optimization not implemented")
    return None


############################################
# SECTION 4: MODEL TRAINERS                #
############################################


class SignalDetectionTrainer:
    """Trainer for signal detection model using XGBoost"""

    def __init__(self, config, redis_client=None, slack_reporter=None):
        self.config = config
        self.redis = redis_client
        self.slack_reporter = slack_reporter
        self.model_type = 'signal_detection'

        # Check if XGBoost is available
        if not XGB_AVAILABLE:
            logger.error(
                "XGBoost is not available. Cannot train signal detection model.")
            raise ImportError("XGBoost is required for signal detection model")

    def train(self, features, target, data_processor=None):
        """Train signal detection model"""
        logger.info("Training signal detection model")

        try:
            if len(features) == 0 or len(target) == 0:
                logger.error("No valid data for signal detection model")
                return False

            # Apply feature selection if enabled
            if self.config['feature_selection']['enabled'] and data_processor:
                features = data_processor.select_features(
                    features, target, 'classification')

            # Use time series cross-validation if enabled
            if self.config['time_series_cv']['enabled'] and data_processor:
                # Create time series split
                splits = data_processor.create_time_series_splits(
                    features, target)

                # Use the last split for final evaluation
                train_idx, test_idx = splits[-1]
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            else:
                # Use traditional train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target,
                    test_size=self.config['test_size'],
                    random_state=self.config['random_state']
                )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save scaler
            scaler_path = os.path.join(
                self.config['models_dir'], 'signal_detection_scaler.pkl')
            joblib.dump(scaler, scaler_path)

            # Get model config
            model_config = self.config['model_configs']['signal_detection']

            # Check for optimized parameters
            optimized_params_path = os.path.join(
                self.config['models_dir'], 'signal_detection_optimized_params.json')
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path, 'r') as f:
                    model_config['params'].update(json.load(f))

            # Train XGBoost model
            logger.info("Training XGBoost signal detection model")
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)

            eval_list = [(dtrain, 'train'), (dtest, 'test')]

            # Create a copy of params without n_estimators to avoid warning
            model = xgb.train(
                params={
                    k: v for k, v in model_config['params'].items() if k != 'n_estimators'},
                dtrain=dtrain,
                evals=eval_list,
                num_boost_round=model_config['params'].get(
                    'n_estimators', 200),
                early_stopping_rounds=20,
                verbose_eval=False
            )

            # Evaluate model
            y_pred = model.predict(dtest)
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Check if we have multiple classes in the test set
            unique_classes = np.unique(y_test)
            if len(unique_classes) < 2:
                logger.warning(
                    f"Only one class present in test set: {unique_classes}. Using simplified metrics.")
                accuracy = accuracy_score(y_test, y_pred_binary)
                metrics = {
                    'accuracy': float(accuracy)
                }
                logger.info(
                    f"Signal detection model metrics - Accuracy: {accuracy:.4f}")
            else:
                accuracy = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                f1 = f1_score(y_test, y_pred_binary)
                auc = roc_auc_score(y_test, y_pred)
                metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'auc': float(auc),
                    'feature_importance': {str(k): float(v) for k, v in model.get_score(importance_type='gain').items()}
                }
                logger.info(
                    f"Signal detection model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

            # Save model
            model_path = os.path.join(
                self.config['models_dir'], 'signal_detection_model.xgb')
            model.save_model(model_path)

            # Save metrics
            metrics_path = os.path.join(
                self.config['models_dir'], 'signal_detection_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

            # Update Redis
            if self.redis:
                self.redis.hset(
                    "models:metrics",
                    "signal_detection",
                    json.dumps(metrics)
                )

            # Report metrics to Slack if available
            if self.slack_reporter:
                self.slack_reporter.report_model_metrics(
                    "signal_detection",
                    metrics,
                    0  # Training time not available here
                )

            logger.info("Signal detection model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training signal detection model: {str(e)}", exc_info=True)
            return False


class PricePredictionTrainer:
    """Trainer for price prediction model using TensorFlow"""

    def __init__(self, config, redis_client=None, slack_reporter=None, accelerator=None):
        self.config = config
        self.redis = redis_client
        self.slack_reporter = slack_reporter
        self.model_type = 'price_prediction'
        self.accelerator = accelerator
        self.use_gpu = accelerator is not None

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            logger.error(
                "TensorFlow is not available. Cannot train price prediction model.")
            raise ImportError(
                "TensorFlow is required for price prediction model")

    def train(self, sequences, targets):
        """Train price prediction model"""
        logger.info("Training price prediction model")

        try:
            if len(sequences) == 0 or len(targets) == 0:
                logger.error("No valid data for price prediction model")
                return False

            # Clear GPU memory if available to reduce fragmentation
            if self.use_gpu and self.accelerator:
                self.accelerator.clear_gpu_memory()
                logger.info("Cleared GPU memory before training")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )

            # Get model config
            model_config = self.config['model_configs']['price_prediction']

            # Determine optimal batch size based on GPU memory
            batch_size = 32  # Default
            if self.use_gpu and self.accelerator:
                # Use more conservative batch size
                batch_size = min(
                    128, self.accelerator.get_optimal_batch_size())
                logger.info(
                    f"Using optimal batch size for GH200: {batch_size}")

            # Set TensorFlow GPU memory limitations to avoid graph errors
            if self.use_gpu and self.accelerator:
                if self.accelerator.has_tensorflow_gpu:
                    # Configure TensorFlow for optimal performance
                    logger.info(
                        "Configuring TensorFlow for optimal GPU performance")

                    # Use mixed precision for better performance on GH200
                    try:
                        from tensorflow.keras.mixed_precision import set_global_policy
                        set_global_policy('mixed_float16')
                        logger.info(
                            "Enabled mixed_float16 precision for training")
                    except Exception as e:
                        logger.warning(f"Could not set mixed precision: {e}")

                    # Disable JIT compilation for more stable training
                    tf.config.optimizer.set_jit(False)

                    # Set memory growth to prevent OOM errors
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            try:
                                tf.config.experimental.set_memory_growth(
                                    gpu, True)
                            except Exception as e:
                                logger.warning(
                                    f"Could not set memory growth: {e}")
                else:
                    logger.warning(
                        "TensorFlow GPU not available on GH200, but GPU acceleration is enabled")
                    logger.info("Using TensorFlow CPU for model training")

            # Create a model architecture optimized for GH200
            logger.info("Creating model architecture optimized for GH200")
            model = Sequential()

            # Use LSTM layers for better sequence modeling
            # LSTM is well-optimized on GH200 with TensorFlow
            model.add(LSTM(
                64,  # Moderate size for balance of performance and accuracy
                return_sequences=True,
                input_shape=(sequences.shape[1], sequences.shape[2]),
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))

            model.add(BatchNormalization())

            model.add(LSTM(
                32,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))

            model.add(BatchNormalization())
            model.add(Dropout(0.3))

            # Dense layers for final prediction
            model.add(Dense(
                16,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))

            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            # Output layer
            model.add(Dense(
                targets.shape[1],
                activation='linear',
                kernel_initializer='glorot_uniform'
            ))

            # Use a stable optimizer configuration with mixed precision
            if self.use_gpu and self.accelerator and self.accelerator.has_tensorflow_gpu:
                # Use mixed precision optimizer for better performance
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                    Adam(
                        learning_rate=0.001,
                        clipnorm=1.0  # Gradient clipping for stability
                    )
                )
            else:
                # Standard optimizer for CPU or when mixed precision not available
                optimizer = Adam(
                    learning_rate=0.001,
                    clipnorm=1.0
                )

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']  # Add mean absolute error as additional metric
            )

            # Print model summary
            model.summary()

            # Custom callback to stop training if NaN loss is encountered
            class TerminateOnNaN(tf.keras.callbacks.Callback):
                def on_batch_end(self, batch, logs=None):
                    logs = logs or {}
                    loss = logs.get('loss')
                    if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                        logger.warning(
                            f'Batch {batch}: Invalid loss, terminating training')
                        self.model.stop_training = True

            # Define checkpoint path
            checkpoint_path = os.path.join(
                self.config['models_dir'], 'price_prediction_best.keras')

            # Create callbacks
            callbacks = [
                TerminateOnNaN(),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    min_delta=0.0001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]

            # Add checkpoint callback if path exists
            if os.path.exists(os.path.dirname(checkpoint_path)):
                callbacks.append(
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=False
                    )
                )

            # Train model with robust error handling
            try:
                # Clear GPU memory before starting
                if self.use_gpu and self.accelerator:
                    self.accelerator.clear_gpu_memory()
                    logger.info("Cleared GPU memory before training start")

                # Start with full dataset training
                logger.info("Starting model training with full dataset")
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                logger.info("Model training completed successfully")

            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.info("Attempting training with reduced dataset size")

                # Try training with progressively smaller subsets if needed
                try_sizes = [0.5, 0.25, 0.1]
                history = None

                for size_factor in try_sizes:
                    try:
                        # Clear memory before trying with smaller size
                        if self.use_gpu and self.accelerator:
                            self.accelerator.clear_gpu_memory()

                        logger.info(
                            f"Trying with {size_factor*100}% of training data")
                        subset_size = int(len(X_train) * size_factor)
                        # Use sequential indices for better memory locality
                        indices = np.arange(subset_size)
                        X_train_subset = X_train[indices]
                        y_train_subset = y_train[indices]

                        # Reduce batch size for smaller dataset
                        reduced_batch_size = min(batch_size, max(
                            16, int(batch_size * size_factor)))

                        # Train with reduced dataset
                        history = model.fit(
                            X_train_subset, y_train_subset,
                            epochs=5,
                            batch_size=reduced_batch_size,
                            validation_split=0.2,
                            callbacks=callbacks,
                            verbose=1
                        )
                        logger.info(
                            f"Training succeeded with {size_factor*100}% of data")
                        break
                    except Exception as subset_error:
                        logger.warning(
                            f"Error training with {size_factor*100}% of data: {str(subset_error)}")

                        if size_factor == try_sizes[-1]:
                            # Create emergency model as fallback
                            logger.warning(
                                "All training attempts failed, creating emergency fallback model")
                            model = Sequential([
                                Flatten(input_shape=(
                                    sequences.shape[1], sequences.shape[2])),
                                Dense(8, activation='relu'),
                                Dense(targets.shape[1], activation='linear')
                            ])
                            model.compile(optimizer='adam', loss='mse')
                            history = model.fit(
                                X_train[:100], y_train[:100],
                                epochs=1, batch_size=16, verbose=1
                            )
                            break

            # Evaluate model
            logger.info("Evaluating model performance")

            # Use batched prediction to avoid memory issues
            def predict_in_batches(model, data, batch_size=32):
                predictions = []
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    batch_pred = model.predict(batch, verbose=0)
                    predictions.append(batch_pred)
                return np.vstack(predictions)

            # Evaluate on test set
            test_loss = model.evaluate(
                X_test, y_test, batch_size=32, verbose=0)

            # Get predictions
            y_pred = predict_in_batches(model, X_test, batch_size=32)

            # Calculate metrics
            direction_accuracy = np.mean(
                (y_pred[:, 0] > 0) == (y_test[:, 0] > 0))
            mse = np.mean((y_pred - y_test) ** 2)
            mae = np.mean(np.abs(y_pred - y_test))

            logger.info(
                f"Price prediction model metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, Direction Accuracy: {direction_accuracy:.4f}")

            # Save and optimize model with TensorRT if GPU is available
            model_path = os.path.join(
                self.config['models_dir'], 'price_prediction_model.keras')

            if self.use_gpu and self.accelerator and self.accelerator.has_tensorrt:
                logger.info("Optimizing model with TensorRT...")
                try:
                    # Save original model first as backup
                    model.save(model_path + ".backup")
                    logger.info(
                        f"Saved original model backup to {model_path}.backup")

                    # Optimize with TensorRT
                    optimized_model = self.accelerator.optimize_model(model)
                    logger.info("Model successfully optimized with TensorRT")

                    # Save optimized model
                    tf.saved_model.save(optimized_model, model_path)
                    logger.info(
                        f"Saved TensorRT optimized model to {model_path}")
                except Exception as e:
                    logger.error(f"Error optimizing model with TensorRT: {e}")
                    # Fallback to saving original model
                    model.save(model_path)
                    logger.info(
                        f"Saved original model as fallback to {model_path}")
            else:
                # Save original model if TensorRT not available
                model.save(model_path)
                logger.info(
                    f"Saved original model to {model_path} (TensorRT not available)")

            # Clear GPU memory after training
            if self.use_gpu and self.accelerator:
                self.accelerator.clear_gpu_memory()
                logger.info("Cleared GPU memory after training")

            # Save metrics
            metrics = {
                'test_loss': float(test_loss) if isinstance(test_loss, (int, float)) else float(test_loss[0]),
                'direction_accuracy': float(direction_accuracy),
                'mse': float(mse),
                'mae': float(mae),
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])] if 'val_loss' in history.history else [],
                    'mae': [float(x) for x in history.history.get('mae', [])] if 'mae' in history.history else [],
                    'val_mae': [float(x) for x in history.history.get('val_mae', [])] if 'val_mae' in history.history else []
                },
                'gpu_used': self.use_gpu,
                'tensorrt_optimized': self.use_gpu and self.accelerator and self.accelerator.has_tensorrt
            }

            metrics_path = os.path.join(
                self.config['models_dir'], 'price_prediction_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            logger.info(f"Saved model metrics to {metrics_path}")

            # Update Redis
            if self.redis:
                self.redis.hset("models:metrics",
                                "price_prediction", json.dumps(metrics))
                logger.info("Updated Redis with model metrics")

            logger.info("Price prediction model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training price prediction model: {str(e)}", exc_info=True)
            return False


class RiskAssessmentTrainer:
    """Trainer for risk assessment model using Random Forest"""

    def __init__(self, config, redis_client=None, slack_reporter=None):
        self.config = config
        self.redis = redis_client
        self.slack_reporter = slack_reporter
        self.model_type = 'risk_assessment'

    def train(self, features, targets, data_processor=None):
        """Train risk assessment model"""
        logger.info("Training risk assessment model")

        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for risk assessment model")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save scaler
            scaler_path = os.path.join(
                self.config['models_dir'], 'risk_assessment_scaler.pkl')
            joblib.dump(scaler, scaler_path)

            # Get model config
            model_config = self.config['model_configs']['risk_assessment']

            # Create model
            model = RandomForestRegressor(
                n_estimators=model_config['params']['n_estimators'],
                max_depth=min(model_config['params']['max_depth'], 4),
                min_samples_leaf=max(
                    model_config['params']['min_samples_leaf'], 50),
                max_features='sqrt',
                random_state=self.config['random_state']
            )

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            r2 = model.score(X_test_scaled, y_test)

            logger.info(
                f"Risk assessment model metrics - MSE: {mse:.6f}, R²: {r2:.4f}")

            # Save model
            model_path = os.path.join(
                self.config['models_dir'], 'risk_assessment_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save metrics
            metrics = {
                'mse': float(mse),
                'r2': float(r2),
                'feature_importance': {str(i): float(v) for i, v in enumerate(model.feature_importances_)}
            }

            metrics_path = os.path.join(
                self.config['models_dir'], 'risk_assessment_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

            # Update Redis
            if self.redis:
                self.redis.hset("models:metrics",
                                "risk_assessment", json.dumps(metrics))

            logger.info("Risk assessment model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training risk assessment model: {str(e)}", exc_info=True)
            return False


class ExitStrategyTrainer:
    """Trainer for exit strategy model using XGBoost"""

    def __init__(self, config, redis_client=None, slack_reporter=None):
        self.config = config
        self.redis = redis_client
        self.slack_reporter = slack_reporter
        self.model_type = 'exit_strategy'

        # Check if XGBoost is available
        if not XGB_AVAILABLE:
            logger.error(
                "XGBoost is not available. Cannot train exit strategy model.")
            raise ImportError("XGBoost is required for exit strategy model")

    def train(self, features, targets, data_processor=None):
        """Train exit strategy model"""
        logger.info("Training exit strategy model")

        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for exit strategy model")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save scaler
            scaler_path = os.path.join(
                self.config['models_dir'], 'exit_strategy_scaler.pkl')
            joblib.dump(scaler, scaler_path)

            # Get model config
            model_config = self.config['model_configs']['exit_strategy']

            # Train XGBoost model
            logger.info("Training XGBoost exit strategy model")
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)

            eval_list = [(dtrain, 'train'), (dtest, 'test')]

            # Create a copy of params without n_estimators to avoid warning
            xgb_params = {
                k: v for k, v in model_config['params'].items() if k != 'n_estimators'}

            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=model_config['params'].get(
                    'n_estimators', 200),
                evals=eval_list,
                early_stopping_rounds=20,
                verbose_eval=False
            )

            # Evaluate model
            y_pred = model.predict(dtest)

            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)

            logger.info(
                f"Exit strategy model metrics - RMSE: {rmse:.6f}, Mean Target: {mean_actual:.6f}")

            # Save model
            model_path = os.path.join(
                self.config['models_dir'], 'exit_strategy_model.xgb')
            model.save_model(model_path)

            # Save metrics
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'feature_importance': {str(k): float(v) for k, v in model.get_score(importance_type='gain').items()}
            }

            metrics_path = os.path.join(
                self.config['models_dir'], 'exit_strategy_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

            # Update Redis
            if self.redis:
                self.redis.hset("models:metrics",
                                "exit_strategy", json.dumps(metrics))

            logger.info("Exit strategy model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training exit strategy model: {str(e)}", exc_info=True)
            return False


class MarketRegimeTrainer:
    """Trainer for market regime model using KMeans clustering"""

    def __init__(self, config, redis_client=None, slack_reporter=None):
        self.config = config
        self.redis = redis_client
        self.slack_reporter = slack_reporter
        self.model_type = 'market_regime'

    def train(self, features, data_processor=None):
        """Train market regime classifier model"""
        logger.info("Training market regime model")

        try:
            if len(features) == 0:
                logger.error("No valid data for market regime model")
                return False

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Save scaler
            scaler_path = os.path.join(
                self.config['models_dir'], 'market_regime_scaler.pkl')
            joblib.dump(scaler, scaler_path)

            # Get model config
            model_config = self.config['model_configs']['market_regime']

            # Create model
            model = KMeans(
                n_clusters=model_config['params']['n_clusters'],
                random_state=model_config['params']['random_state']
            )

            # Train model
            model.fit(features_scaled)

            # Calculate metrics
            inertia = model.inertia_
            cluster_counts = np.bincount(model.labels_)

            logger.info(
                f"Market regime model metrics - Inertia: {inertia:.2f}, Cluster counts: {cluster_counts}")

            # Save model
            model_path = os.path.join(
                self.config['models_dir'], 'market_regime_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save metrics
            metrics = {
                'inertia': float(inertia),
                'cluster_counts': [int(count) for count in cluster_counts],
                'cluster_centers': [[float(value) for value in center] for center in model.cluster_centers_]
            }

            metrics_path = os.path.join(
                self.config['models_dir'], 'market_regime_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

            # Update Redis
            if self.redis:
                self.redis.hset("models:metrics",
                                "market_regime", json.dumps(metrics))

            logger.info("Market regime model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training market regime model: {str(e)}", exc_info=True)

            # If training failed, create a simple fallback model
            try:
                logger.info(
                    "Creating fallback market regime model with default parameters")

                # Create a simple KMeans model with default parameters
                model = KMeans(n_clusters=4, random_state=42)

                # Fit on a small dummy dataset to initialize the model
                dummy_data = np.random.rand(10, 5)  # 10 samples, 5 features
                model.fit(dummy_data)

                # Save the model
                model_path = os.path.join(
                    self.config['models_dir'], 'market_regime_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(
                    f"Created and saved fallback market regime model to {model_path}")

                # Update Redis with minimal model info
                if self.redis:
                    self.redis.hset("models:metrics", "market_regime",
                                    json.dumps({"fallback": True}))
            except Exception as fallback_error:
                logger.error(
                    f"Error creating fallback market regime model: {str(fallback_error)}")

            return False


class MLDataProcessor:
    """Data processor for ML model training"""

    def __init__(self, data_loader, redis_client=None, config=None):
        self.data_loader = data_loader
        self.redis = redis_client
        self.config = config
        self.reference_data = None

    def load_historical_data(self):
        """Load historical data for model training"""
        try:
            # Get lookback days from config
            lookback_days = self.config.get('lookback_days', 30)

            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)

            # Use data loader to get price data
            price_data = self.data_loader.load_price_data(
                tickers=self.data_loader.get_watchlist_tickers(),
                start_date=start_date,
                end_date=end_date,
                timeframe='1m'
            )

            # Get options data if available
            options_data = self.data_loader.load_options_data(
                tickers=self.data_loader.get_watchlist_tickers(),
                start_date=start_date,
                end_date=end_date
            )

            # Get market data
            market_data = self.data_loader.load_market_data(
                start_date=start_date,
                end_date=end_date
            )

            # Prepare combined dataset
            combined_data = self.data_loader.prepare_training_data(
                price_data=price_data,
                options_data=options_data,
                market_data=market_data
            )

            logger.info(
                f"Loaded historical data: {len(combined_data)} samples")
            return combined_data

        except Exception as e:
            logger.error(
                f"Error loading historical data: {str(e)}", exc_info=True)
            return None

    def store_reference_data(self, data):
        """Store reference data for drift detection"""
        try:
            # Create a copy to avoid modifying the original
            self.reference_data = data.copy()

            # Save to disk for persistence
            if self.config and 'monitoring_dir' in self.config:
                ref_path = os.path.join(
                    self.config['monitoring_dir'], 'reference_data.pkl')
                with open(ref_path, 'wb') as f:
                    pickle.dump(self.reference_data, f)

            logger.info(
                f"Stored reference data: {len(self.reference_data)} samples")
            return True
        except Exception as e:
            logger.error(
                f"Error storing reference data: {str(e)}", exc_info=True)
            return False

    def prepare_signal_detection_data(self, data):
        """Prepare data for signal detection model"""
        try:
            # Select features
            feature_columns = [
                # Price-based features
                'close', 'open', 'high', 'low', 'volume',

                # Technical indicators
                'sma5', 'sma10', 'sma20',
                'ema5', 'ema10', 'ema20',
                'macd', 'macd_signal', 'macd_hist',
                'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20',
                'mom1', 'mom5', 'mom10',
                'volatility', 'volume_ratio', 'rsi',
                'bb_width',

                # Market features (if available)
                'spy_close', 'vix_close', 'spy_change', 'vix_change',

                # Options features (if available)
                'put_call_ratio', 'implied_volatility', 'option_volume'
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 5:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()

            # Select data
            X = data[available_columns].copy()
            y = data['signal_target'].copy()

            # Drop rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            # Handle any remaining infinity values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.mean(), inplace=True)

            logger.info(
                f"Prepared signal detection data with {len(X)} samples and {len(available_columns)} features")

            return X, y

        except Exception as e:
            logger.error(
                f"Error preparing signal detection data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def prepare_price_prediction_data(self, data):
        """Prepare data for price prediction model (LSTM)"""
        try:
            # Select features
            feature_columns = [
                # Price-based features
                'close', 'high', 'low', 'volume',

                # Technical indicators
                'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20',
                'macd', 'rsi', 'volatility',

                # Market features (if available)
                'spy_close', 'vix_close'
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 4:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return np.array([]), np.array([])

            # Target columns
            target_columns = ['future_return_5min',
                              'future_return_10min', 'future_return_30min']
            available_targets = [
                col for col in target_columns if col in data.columns]

            if len(available_targets) == 0:
                logger.warning("No target variables available")
                return np.array([]), np.array([])

            # Group by ticker to create sequences
            sequences = []
            targets = []

            for ticker, group in data.groupby('ticker'):
                # Sort by timestamp
                group = group.sort_index()

                # Select features and targets
                X = group[available_columns].values
                y = group[available_targets].values

                # Create sequences (lookback of 20 intervals)
                for i in range(20, len(X)):
                    sequences.append(X[i-20:i])
                    targets.append(y[i])

            # Convert to numpy arrays
            X_array = np.array(sequences)
            y_array = np.array(targets)

            # Handle NaN or infinite values
            if np.isnan(X_array).any() or np.isinf(X_array).any() or np.isnan(y_array).any() or np.isinf(y_array).any():
                logger.warning(
                    "NaN or infinite values detected. Performing robust cleaning...")

                # Identify rows with NaN or inf in either X or y
                X_has_invalid = np.any(
                    np.isnan(X_array) | np.isinf(X_array), axis=(1, 2))
                y_has_invalid = np.any(
                    np.isnan(y_array) | np.isinf(y_array), axis=1)
                valid_indices = ~(X_has_invalid | y_has_invalid)

                # Filter out invalid rows if enough valid data
                if np.sum(valid_indices) > 100:
                    X_array = X_array[valid_indices]
                    y_array = y_array[valid_indices]
                else:
                    # Replace NaN and inf with zeros/means if not enough valid data
                    X_array = np.nan_to_num(
                        X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    y_array = np.nan_to_num(
                        y_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_array.shape
            X_reshaped = X_array.reshape(n_samples * n_timesteps, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Store scaler in memory
            self.price_prediction_scaler = scaler

            logger.info(
                f"Prepared price prediction data with {len(sequences)} sequences")

            return X_array, y_array

        except Exception as e:
            logger.error(
                f"Error preparing price prediction data: {str(e)}", exc_info=True)
            return np.array([]), np.array([])

    def create_time_series_splits(self, X, y):
        """Create time series cross-validation splits"""
        return create_time_series_splits(
            X, y,
            self.config['time_series_cv']['n_splits'],
            self.config['time_series_cv']['embargo_size']
        )

    def select_features(self, X, y, problem_type='classification'):
        """Select important features based on feature selection method"""
        return select_features(
            X, y, problem_type,
            self.config['feature_selection']['method'],
            self.config['feature_selection']['threshold'],
            self.config['feature_selection']['n_features']
        )

    def detect_drift(self, current_data):
        """Detect drift in feature distributions"""
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return False, {}

        # Record drift detection attempt in Prometheus if available
        if PROMETHEUS_AVAILABLE:
            try:
                # Start timing the drift detection
                start_time = time.time()

                # Perform drift detection
                drift_detected, drift_features = detect_feature_drift(
                    current_data,
                    self.reference_data,
                    self.config['monitoring']['drift_threshold']
                )

                # Record the result in Prometheus
                DRIFT_DETECTION.labels(
                    model_name='data_features',
                    result='detected' if drift_detected else 'not_detected'
                ).inc()

                # Record detection time
                detection_time = time.time() - start_time
                logger.info(
                    f"Drift detection completed in {detection_time:.4f} seconds")

                return drift_detected, drift_features
            except Exception as e:
                logger.error(f"Error in drift detection with Prometheus: {e}")
                # Record error in Prometheus
                DRIFT_DETECTION.labels(
                    model_name='data_features',
                    result='error'
                ).inc()

                # Fall back to regular detection
                return detect_feature_drift(
                    current_data,
                    self.reference_data,
                    self.config['monitoring']['drift_threshold']
                )
        else:
            # Regular drift detection without Prometheus
            return detect_feature_drift(
                current_data,
                self.reference_data,
                self.config['monitoring']['drift_threshold']
            )


class MLModelTrainer:
    """
    ML Model Trainer for trading system
    Builds and trains models using live market data
    """

    def __init__(self, redis_client, data_loader):
        self.redis = redis_client
        self.data_loader = data_loader

        # Apply GH200 optimizations
        optimize_for_gh200()

        # Initialize GPU acceleration
        self.use_gpu = os.environ.get('USE_GPU', 'true').lower() == 'true'
        if self.use_gpu:
            # Initialize GH200 accelerator
            self.accelerator = GH200Accelerator()
            self.cupy_gpu_available = self.accelerator.has_cupy_gpu
            self.tf_gpu_available = self.accelerator.has_tensorflow_gpu
            logger.info(
                f"GH200 acceleration enabled: {self.use_gpu}, TensorFlow GPU available: {self.tf_gpu_available}, CuPy GPU available: {self.accelerator.has_cupy_gpu}")

        # Configuration
        self.config = {
            'models_dir': os.environ.get('MODELS_DIR', './models'),
            'monitoring_dir': os.environ.get('MONITORING_DIR', './monitoring'),
            'data_dir': os.environ.get('DATA_DIR', './data'),
            'min_samples': 1000,
            'lookback_days': 30,
            'feature_selection': {
                'enabled': True,
                'method': 'importance',  # 'importance', 'rfe', 'mutual_info'
                'threshold': 0.01,  # For importance-based selection
                'n_features': 20    # For RFE
            },
            'time_series_cv': {
                'enabled': True,
                'n_splits': 5,
                'embargo_size': 10  # Number of samples to exclude between train and test
            },
            'monitoring': {'enabled': True, 'drift_threshold': 0.05},
            'test_size': 0.2,
            'random_state': 42,
            'model_configs': {
                'signal_detection': {
                    'type': 'xgboost',
                    'params': {
                        'max_depth': 6,
                        'learning_rate': 0.03,
                        'subsample': 0.8,
                        'n_estimators': 200,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc'
                    }
                },
                'price_prediction': {
                    'type': 'lstm',
                    'params': {
                        'units': [64, 32],
                        'dropout': 0.3,
                        'epochs': 50,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    }
                },
                'risk_assessment': {
                    'type': 'random_forest',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'max_features': 'sqrt',
                        'min_samples_leaf': 30
                    }
                },
                'exit_strategy': {
                    'type': 'xgboost',
                    'params': {
                        'max_depth': 5,
                        'learning_rate': 0.02,
                        'subsample': 0.8,
                        'n_estimators': 150,
                        'objective': 'reg:squarederror'
                    }
                },
                'market_regime': {
                    'type': 'kmeans',
                    'params': {
                        'n_clusters': 4,
                        'random_state': 42
                    }
                }
            },
        }

        # Initialize data processor
        self.data_processor = MLDataProcessor(
            data_loader=self.data_loader,
            redis_client=self.redis,
            config=self.config
        )

        # Initialize Slack reporting
        self.slack_reporter = None
        self.gpu_tracker = None
        self.model_training_times = {}
        self.training_start_time = None

        # Initialize Slack reporter using environment variables or defaults
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL', '')
        bot_token = os.environ.get('SLACK_BOT_TOKEN', '')
        channel = os.environ.get('SLACK_CHANNEL', '#system-notifications')

        if webhook_url or bot_token:
            self.slack_reporter = SlackReporter(
                webhook_url=webhook_url, bot_token=bot_token, channel=channel)
            self.gpu_tracker = GPUStatsTracker(
                polling_interval=10.0)  # Poll every 10 seconds

        # Initialize model trainers
        self._init_model_trainers()

        logger.info("ML Model Trainer initialized")

    def _init_model_trainers(self):
        """Initialize model trainers"""
        try:
            self.trainers = {
                'signal_detection': SignalDetectionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'price_prediction': PricePredictionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter,
                    accelerator=self.accelerator if hasattr(
                        self, 'accelerator') else None
                ),
                'risk_assessment': RiskAssessmentTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'exit_strategy': ExitStrategyTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'market_regime': MarketRegimeTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                )
            }
        except ImportError as e:
            logger.warning(
                f"Could not initialize all model trainers: {str(e)}")
            # Continue with available trainers

    def train_all_models(self):
        """Train all trading models"""
        logger.info("Starting training for all models")

        # Start tracking total training time
        self.training_start_time = time.time()

        # Start GPU tracking if available
        if self.gpu_tracker:
            self.gpu_tracker.start()

        # Run hyperparameter optimization if enabled
        if os.environ.get('OPTIMIZE_HYPERPARAMS', 'false').lower() == 'true':
            logger.info("Running hyperparameter optimization")
            if OPTUNA_AVAILABLE:
                # Load historical data
                historical_data = self.data_processor.load_historical_data()

                if historical_data is not None and not historical_data.empty:
                    # Optimize signal detection model
                    optimize_hyperparameters(
                        historical_data,
                        'signal_detection',
                        self.config,
                        self.data_processor
                    )
            else:
                logger.warning(
                    "Optuna not available. Skipping hyperparameter optimization.")

        # Continue with regular training
        self._train_all_models()

    def _train_all_models(self):
        """Internal method to train all models with current hyperparameters"""
        try:
            # Load historical data
            logger.info("Loading historical data")
            historical_data = self.data_processor.load_historical_data()

            if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty):
                logger.error("Failed to load sufficient historical data")

                # Report error to Slack if available
                if self.slack_reporter:
                    self.slack_reporter.report_error(
                        "Failed to load sufficient historical data",
                        phase="data loading"
                    )

                return False

            # Store reference data for drift detection
            self.data_processor.store_reference_data(historical_data)

            # Train each model
            model_results = {}

            # Signal detection model
            if 'signal_detection' in self.trainers:
                start_time = time.time()
                features, target = self.data_processor.prepare_signal_detection_data(
                    historical_data)
                success = self.trainers['signal_detection'].train(
                    features, target, self.data_processor)
                training_time = time.time() - start_time
                self.model_training_times['signal_detection'] = training_time
                model_results['signal_detection'] = {
                    'success': success, 'time': training_time}

            # Price prediction model
            if 'price_prediction' in self.trainers:
                start_time = time.time()
                sequences, targets = self.data_processor.prepare_price_prediction_data(
                    historical_data)
                success = self.trainers['price_prediction'].train(
                    sequences, targets)
                training_time = time.time() - start_time
                self.model_training_times['price_prediction'] = training_time
                model_results['price_prediction'] = {
                    'success': success, 'time': training_time}

            # Risk assessment model
            if 'risk_assessment' in self.trainers:
                start_time = time.time()
                features, targets = self.data_processor.prepare_signal_detection_data(
                    historical_data)  # Use same features but different target
                if 'atr_pct' in historical_data.columns:
                    targets = historical_data['atr_pct']
                    success = self.trainers['risk_assessment'].train(
                        features, targets, self.data_processor)
                    training_time = time.time() - start_time
                    self.model_training_times['risk_assessment'] = training_time
                    model_results['risk_assessment'] = {
                        'success': success, 'time': training_time}
                else:
                    logger.warning(
                        "No risk target variable (atr_pct) available")
                    model_results['risk_assessment'] = {
                        'success': False, 'error': 'No target variable'}

            # Exit strategy model
            if 'exit_strategy' in self.trainers:
                start_time = time.time()
                features, _ = self.data_processor.prepare_signal_detection_data(
                    historical_data)  # Use same features but different target
                if 'optimal_exit' in historical_data.columns:
                    targets = historical_data['optimal_exit']
                    success = self.trainers['exit_strategy'].train(
                        features, targets, self.data_processor)
                    training_time = time.time() - start_time
                    self.model_training_times['exit_strategy'] = training_time
                    model_results['exit_strategy'] = {
                        'success': success, 'time': training_time}
                else:
                    logger.warning(
                        "No exit strategy target variable (optimal_exit) available")
                    model_results['exit_strategy'] = {
                        'success': False, 'error': 'No target variable'}

            # Market regime model
            if 'market_regime' in self.trainers:
                start_time = time.time()
                # Extract market features
                market_features = [
                    col for col in historical_data.columns if 'spy_' in col or 'vix_' in col]
                if market_features:
                    market_data = historical_data[market_features].dropna()
                    success = self.trainers['market_regime'].train(market_data)
                    training_time = time.time() - start_time
                    self.model_training_times['market_regime'] = training_time
                    model_results['market_regime'] = {
                        'success': success, 'time': training_time}
                else:
                    logger.warning(
                        "No market features available for regime detection")
                    model_results['market_regime'] = {
                        'success': False, 'error': 'No market features'}

            # Update Redis with model info
            self.update_model_info()

            # Calculate total training time
            total_training_time = time.time() - self.training_start_time

            # Stop GPU tracking if available
            gpu_stats = None
            if self.gpu_tracker:
                gpu_stats = self.gpu_tracker.stop()

            # Report training completion to Slack if available
            if self.slack_reporter:
                self.slack_reporter.report_training_complete(
                    total_training_time,
                    model_results,
                    gpu_stats
                )

            logger.info(
                f"All models trained successfully in {total_training_time:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}", exc_info=True)
            return False

    def update_model_info(self):
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}

            for model_name, config in self.config['model_configs'].items():
                model_path = os.path.join(
                    self.config['models_dir'], f"{model_name}_model.{ 'xgb' if config['type'] == 'xgboost' else 'pkl' if config['type'] in ['random_forest', 'kmeans'] else 'keras' }"
                )

                if os.path.exists(model_path):
                    file_stats = os.stat(model_path)

                    models_info[model_name] = {
                        'type': config['type'],
                        'path': model_path,
                        'size_bytes': file_stats.st_size,
                        'last_modified': int(file_stats.st_mtime),
                        'last_modified_str': datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }

            # Update Redis
            if self.redis:
                self.redis.set("models:info", json.dumps(models_info))

            logger.info(f"Updated model info for {len(models_info)} models")

        except Exception as e:
            logger.error(f"Error updating model info: {str(e)}", exc_info=True)

    def predict_signals(self, market_data):
        """
        Predict trading signals using trained models

        Args:
            market_data: DataFrame with latest market data

        Returns:
            Dictionary of ticker -> prediction results
        """
        try:
            start_time = time.time()
            logger.info(f"Making predictions with {len(market_data)} samples")

            # Prepare data
            features, _ = self.data_processor.prepare_signal_detection_data(
                market_data)

            if features.empty:
                logger.warning("No valid features for prediction")
                return {}

            # Check for model files
            signal_model_path = os.path.join(
                self.config['models_dir'], 'signal_detection_model.xgb')
            signal_scaler_path = os.path.join(
                self.config['models_dir'], 'signal_detection_scaler.pkl')

            if not os.path.exists(signal_model_path) or not os.path.exists(signal_scaler_path):
                logger.error("Signal detection model or scaler not found")
                return {}

            # Load model and scaler
            signal_model = xgb.Booster()
            signal_model.load_model(signal_model_path)
            signal_scaler = joblib.load(signal_scaler_path)

            # Scale features
            features_scaled = signal_scaler.transform(features)

            # Make predictions
            dmatrix = xgb.DMatrix(features_scaled)
            signal_scores = signal_model.predict(dmatrix)

            # Organize predictions by ticker
            predictions = {}

            # Add ticker info
            if 'ticker' in market_data.columns:
                for i, ticker in enumerate(market_data['ticker']):
                    if i < len(signal_scores):
                        if ticker not in predictions:
                            predictions[ticker] = {
                                'signal_score': float(signal_scores[i]),
                                'signal': 1 if signal_scores[i] > 0.5 else 0,
                                'timestamp': datetime.datetime.now().isoformat()
                            }

            # Record prediction latency in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                prediction_time = time.time() - start_time
                PREDICTION_LATENCY.labels(
                    model_name='signal_detection').observe(prediction_time)
                logger.info(
                    f"Prediction latency: {prediction_time:.4f} seconds")

            logger.info(
                f"Generated predictions for {len(predictions)} tickers")

            # Update predictions in Redis if available
            if self.redis:
                for ticker, pred in predictions.items():
                    self.redis.hset(
                        f"predictions:{ticker}", "signal", json.dumps(pred))

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                DRIFT_DETECTION.labels(
                    model_name='signal_detection',
                    result='error'
                ).inc()
            return {}


# Main execution
if __name__ == "__main__":
    import redis
    from data_pipeline import DataPipeline

    try:
        # Run GPU diagnostics
        diagnostics_results = run_diagnostics()

        # Create Redis client
        redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6380)),
            db=int(os.environ.get('REDIS_DB', 0)),
            username=os.environ.get('REDIS_USERNAME', 'default'),
            password=os.environ.get('REDIS_PASSWORD', 'trading_system_2025')
        )

        # Create data pipeline
        data_loader = DataPipeline(
            redis_client=redis_client,
            polygon_client=None,  # Import and initialize here
            polygon_ws=None,      # Import and initialize here
            unusual_whales_client=None,  # Import and initialize here
            use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
        )

        # Create model trainer
        model_trainer = MLModelTrainer(redis_client, data_loader)

        # Train all models
        model_trainer.train_all_models()

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
