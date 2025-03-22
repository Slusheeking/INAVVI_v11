#!/usr/bin/env python3
"""
Unified GPU Utilities for NVIDIA GH200

This module provides comprehensive GPU acceleration utilities for the trading system:
1. GPU initialization and configuration for TensorFlow, CuPy, and TensorRT
2. Memory management and optimization for GH200 Grace Hopper Superchips
3. Model optimization and performance monitoring
4. Batch size optimization and throughput maximization
5. GPU-accelerated data processing functions

The utilities are designed specifically for the NVIDIA GH200 Grace Hopper Superchip architecture
but include fallbacks for other GPU types and CPU-only operation.
"""

import os
import time
import logging
import gc
import threading
import numpy as np
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_utils')

# Import GPU libraries with fallbacks
try:
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    TF_AVAILABLE = True
    logger.info("TensorFlow is available")

    # Suppress TensorFlow info and warning messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow is not available")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy is available")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    logger.warning("CuPy is not available, GPU acceleration will be limited")

# Import optional TensorRT if available
try:
    import tensorrt as trt_standalone
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT standalone is available")
except ImportError:
    TENSORRT_AVAILABLE = False
    trt_standalone = None
    logger.warning("TensorRT standalone is not available")

# Import pynvml for GPU memory monitoring if available
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning(
        "pynvml not available. GPU memory monitoring will be limited")

# Environment variables for GPU configuration
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
GPU_MEMORY_LIMIT = int(os.environ.get('TF_CUDA_HOST_MEM_LIMIT_IN_MB', 16000))
TF_XLA_FLAGS = os.environ.get(
    'TF_XLA_FLAGS', '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit')


#########################
# INITIALIZATION & SETUP #
#########################

def detect_gpus() -> List[Dict[str, Any]]:
    """
    Detect available GPUs and their properties

    Returns:
        List of dictionaries with GPU information
    """
    gpus = []

    # Try TensorFlow detection first
    if TF_AVAILABLE:
        try:
            tf_gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(tf_gpus):
                gpus.append({
                    'index': i,
                    'name': gpu.name,
                    'source': 'tensorflow',
                    'device_type': 'GPU',
                })
            logger.info(f"TensorFlow detected {len(tf_gpus)} GPUs")
        except Exception as e:
            logger.warning(f"Error detecting GPUs with TensorFlow: {e}")

    # Try CuPy detection if available
    if CUPY_AVAILABLE:
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            for i in range(num_gpus):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode()

                # Check if this GPU is already in the list (from TensorFlow)
                if not any(gpu['index'] == i and gpu['source'] == 'cupy' for gpu in gpus):
                    gpus.append({
                        'index': i,
                        'name': name,
                        'source': 'cupy',
                        'device_type': 'GPU',
                        'total_memory': props["totalGlobalMem"],
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'clock_rate': props["clockRate"],
                        'multi_processor_count': props["multiProcessorCount"],
                    })
            logger.info(f"CuPy detected {num_gpus} GPUs")
        except Exception as e:
            logger.warning(f"Error detecting GPUs with CuPy: {e}")

    # If no GPUs detected, add CPU info
    if not gpus:
        gpus.append({
            'index': 0,
            'name': 'CPU',
            'source': 'system',
            'device_type': 'CPU',
            'cpu_count': os.cpu_count(),
        })
        logger.warning("No GPUs detected, using CPU")

    return gpus


def select_best_gpu() -> Optional[int]:
    """
    Select the best GPU based on capability and memory availability

    Returns:
        GPU index or None if no suitable GPU found
    """
    if not USE_GPU:
        logger.info("GPU usage is disabled, not selecting GPU")
        return None

    # Detect available GPUs
    gpus = detect_gpus()

    # Filter to actual GPUs (not CPUs)
    gpus = [gpu for gpu in gpus if gpu['device_type'] == 'GPU']

    if not gpus:
        logger.warning("No GPUs available to select")
        return None

    # Check for GH200 GPUs first
    gh200_gpus = [gpu for gpu in gpus if 'GH200' in gpu.get('name', '')]
    if gh200_gpus:
        selected_gpu = gh200_gpus[0]
        logger.info(
            f"Selected GH200 GPU: {selected_gpu['name']} (index: {selected_gpu['index']})")
        return selected_gpu['index']

    # If no GH200, select based on memory if available
    if CUPY_AVAILABLE:
        try:
            # Get memory info for each GPU
            for i, gpu in enumerate(gpus):
                if gpu['source'] == 'cupy':
                    cp.cuda.Device(gpu['index']).use()
                    free, total = cp.cuda.runtime.memGetInfo()
                    gpu['free_memory'] = free
                    gpu['memory_pct_free'] = (free / total) * 100

            # Sort by free memory (descending)
            gpus_with_memory = [gpu for gpu in gpus if 'free_memory' in gpu]
            if gpus_with_memory:
                gpus_with_memory.sort(
                    key=lambda g: g['free_memory'], reverse=True)
                selected_gpu = gpus_with_memory[0]
                logger.info(
                    f"Selected GPU by memory: {selected_gpu['name']} (index: {selected_gpu['index']}, free: {selected_gpu['free_memory']/(1024**3):.2f} GB)")
                return selected_gpu['index']
        except Exception as e:
            logger.warning(f"Error selecting GPU by memory: {e}")

    # Default to first GPU
    selected_gpu = gpus[0]
    logger.info(
        f"Selected first available GPU: {selected_gpu['name']} (index: {selected_gpu['index']})")
    return selected_gpu['index']


def initialize_gpu() -> bool:
    """
    Initialize GPU for optimal performance with TensorFlow, CuPy, and TensorRT

    This function:
    1. Detects available GPUs
    2. Selects the best GPU (preferring GH200 if available)
    3. Configures TensorFlow for optimal performance
    4. Configures CuPy with unified memory
    5. Applies GH200-specific optimizations

    Returns:
        bool: True if GPU initialization succeeded, False otherwise
    """
    if not USE_GPU:
        logger.info("GPU usage is disabled by configuration")
        return False

    try:
        # Select best GPU
        gpu_index = select_best_gpu()
        if gpu_index is None:
            logger.warning("No suitable GPU found for initialization")
            return False

        # Apply GH200-specific optimizations first
        optimize_for_gh200()

        # Configure TensorFlow if available
        if TF_AVAILABLE:
            configure_tensorflow(gpu_index)

        # Configure CuPy if available
        if CUPY_AVAILABLE:
            configure_cupy(gpu_index)

        # Initial GPU memory cleanup to start fresh
        clear_gpu_memory()

        logger.info("GPU initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing GPU: {e}")
        return False


def optimize_for_gh200() -> None:
    """
    Apply GH200-specific optimizations via environment variables
    """
    # TensorFlow optimizations
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "16"  # Optimize for GH200
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"  # Use unified memory
    os.environ["TF_ENABLE_NUMA_AWARE_ALLOCATORS"] = "1"  # For multi-GPU

    # XLA optimization settings
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

    # Enable XLA JIT compilation
    if TF_AVAILABLE:
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("Enabled XLA JIT compilation")
        except Exception as e:
            logger.warning(f"Could not enable XLA JIT compilation: {e}")

    # Memory limits and thread configuration
    os.environ["TF_CUDA_HOST_MEM_LIMIT_IN_MB"] = str(GPU_MEMORY_LIMIT)
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "4"  # Optimal for GH200

    # Memory allocator settings
    # Use async allocator for better performance
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Prevent graph errors
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

    # Enable auto-tuning for cuDNN
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"

    # Disable layout optimizer which can cause issues with TensorRT
    os.environ["TF_LAYOUT_OPTIMIZER_DISABLE"] = "1"

    # Disable oneDNN optimizations which can conflict with TensorRT
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # For CPU side of GH200
    os.environ["GOMP_CPU_AFFINITY"] = "0-15"  # Adjust based on Neoverse cores

    # GPU-direct optimizations
    os.environ["CUDA_AUTO_BOOST"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"

    # Memory allocation optimizations
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

    # Set TensorRT precision mode
    os.environ["TENSORRT_PRECISION_MODE"] = "FP16"

    # Configure CuPy for GH200
    if CUPY_AVAILABLE:
        try:
            # Set CuPy to use unified memory
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            logger.info("Configured CuPy to use unified memory for GH200")
        except Exception as e:
            logger.warning(f"Could not configure CuPy unified memory: {e}")

    logger.info("Applied GH200-specific optimizations via environment variables")


def configure_tensorflow(gpu_index: int) -> bool:
    """
    Configure TensorFlow for optimal performance on selected GPU

    Args:
        gpu_index: GPU index to use

    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available, skipping configuration")
        return False

    try:
        # List physical GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs detected by TensorFlow")
            return False

        logger.info(f"TensorFlow detected {len(gpus)} GPUs")

        # Enable memory growth to prevent OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                logger.warning(f"Error setting memory growth: {e}")

        # Set visible GPU (if multiple GPUs available)
        if len(gpus) > 1:
            try:
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                logger.info(f"Set visible GPU to index {gpu_index}")
            except Exception as e:
                logger.warning(f"Error setting visible GPU: {e}")

        # Configure mixed precision
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("TensorFlow configured with mixed_float16 precision")
        except Exception as e:
            logger.warning(f"Error setting mixed precision: {e}")

        # Configure TensorRT if needed
        if TENSORRT_AVAILABLE:
            try:
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                    max_workspace_size_bytes=8000000000,
                    precision_mode="FP16",
                    maximum_cached_engines=100
                )
                logger.info("TensorRT configured for TensorFlow")
            except Exception as e:
                logger.warning(
                    f"Error configuring TensorRT for TensorFlow: {e}")

        logger.info("TensorFlow GPU configuration completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error configuring TensorFlow: {e}")
        return False


def configure_cupy(gpu_index: int) -> bool:
    """
    Configure CuPy for optimal performance on selected GPU

    Args:
        gpu_index: GPU index to use

    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    if not CUPY_AVAILABLE:
        logger.warning("CuPy not available, skipping configuration")
        return False

    try:
        # Set device
        cp.cuda.Device(gpu_index).use()

        # Use unified memory for better performance
        cp.cuda.set_allocator(cp.cuda.MemoryPool(
            cp.cuda.malloc_managed).malloc)

        # Get device properties
        props = cp.cuda.runtime.getDeviceProperties(gpu_index)
        device_name = props["name"].decode()
        logger.info(f"CuPy using GPU: {device_name} (index: {gpu_index})")

        # Get memory info
        free, total = cp.cuda.runtime.memGetInfo()
        logger.info(
            f"GPU Memory: {free/(1024**3):.2f} GB free, {total/(1024**3):.2f} GB total")

        # Set pinned memory limit (for async operations)
        try:
            cp.cuda.set_pinned_memory_allocator()
        except Exception as e:
            logger.warning(f"Error setting pinned memory allocator: {e}")

        logger.info("CuPy configured successfully")
        return True

    except Exception as e:
        logger.error(f"Error configuring CuPy: {e}")
        return False


#########################
# MEMORY MANAGEMENT    #
#########################

def log_memory_usage(location_tag: str) -> Dict[str, float]:
    """
    Log CPU and GPU memory usage

    Args:
        location_tag: Tag to identify the logging location

    Returns:
        Dictionary with memory usage statistics
    """
    memory_stats = {
        'cpu_memory_mb': 0,
        'gpu_memory_used_mb': 0,
        'gpu_memory_free_mb': 0,
        'gpu_memory_total_mb': 0,
        'gpu_memory_percent': 0,
    }

    try:
        # Log CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 * 1024)
        memory_stats['cpu_memory_mb'] = cpu_mem

        # Log GPU memory if available
        if CUPY_AVAILABLE:
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                free, total = mem_info[0], mem_info[1]
                used = total - free

                memory_stats['gpu_memory_used_mb'] = used / (1024 * 1024)
                memory_stats['gpu_memory_free_mb'] = free / (1024 * 1024)
                memory_stats['gpu_memory_total_mb'] = total / (1024 * 1024)
                memory_stats['gpu_memory_percent'] = (used / total) * 100

                logger.info(
                    f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, "
                    f"GPU: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, "
                    f"Total={total/(1024**2):.2f}MB, Used={used/total*100:.2f}%"
                )
            except Exception as e:
                logger.warning(f"Error getting GPU memory info: {e}")
        else:
            logger.info(
                f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Not available")

    except Exception as e:
        logger.error(f"Failed to log memory usage at {location_tag}: {e}")

    return memory_stats


def get_gpu_memory_usage() -> tuple:
    """
    Get GPU memory usage information

    Returns:
        tuple: (used_memory_bytes, total_memory_bytes) or None if not available
    """
    # Try using pynvml first (most accurate)
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used, info.total)
        except Exception as e:
            logger.warning(f"Error getting GPU memory with pynvml: {e}")
            # Continue to try other methods
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    # Try using CuPy if available
    if CUPY_AVAILABLE:
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free, total = mem_info
            used = total - free
            return (used, total)
        except Exception as e:
            logger.warning(f"Error getting GPU memory with CuPy: {e}")

    # Try using TensorFlow if available
    if TF_AVAILABLE:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # TensorFlow doesn't provide exact memory usage, so we estimate
                # This is a rough approximation
                gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
                if 'current' in gpu_mem and 'peak' in gpu_mem:
                    # Return current as used and peak as total (approximation)
                    return (gpu_mem['current'], gpu_mem['peak'])
        except Exception as e:
            logger.warning(f"Error getting GPU memory with TensorFlow: {e}")

    # If all methods fail, return None
    logger.warning("Could not get GPU memory usage information")
    return None


def clear_gpu_memory(force_full_cleanup: bool = False) -> bool:
    """
    Clear GPU memory to prevent fragmentation

    Args:
        force_full_cleanup: Whether to force a full cleanup including TensorFlow session reset

    Returns:
        bool: True if operation succeeded, False otherwise
    """
    success = True
    start_time = time.time()

    # Log initial memory state
    initial_memory = get_gpu_memory_usage()
    if initial_memory:
        used_mb = initial_memory[0] / (1024 * 1024)
        total_mb = initial_memory[1] / (1024 * 1024)
        logger.debug(
            f"Initial GPU memory: {used_mb:.2f}MB used / {total_mb:.2f}MB total ({used_mb/total_mb*100:.2f}%)")

    # Clear TensorFlow memory
    if TF_AVAILABLE:
        try:
            # Clear TensorFlow memory caches
            tf.keras.backend.clear_session()

            # Reset TensorFlow GPU memory stats
            for device in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass

            # For a full cleanup, recreate the TensorFlow session
            if force_full_cleanup:
                # This is more aggressive and will completely reset TensorFlow
                import tensorflow.compat.v1 as tf1
                tf1.reset_default_graph()
                if hasattr(tf1, 'Session'):
                    try:
                        session = tf1.Session()
                        session.close()
                    except:
                        pass

            logger.debug("TensorFlow memory cleared")
        except Exception as e:
            logger.warning(f"Error clearing TensorFlow memory: {e}")
            success = False

    # Clear CuPy memory
    if CUPY_AVAILABLE:
        try:
            # Get current memory usage before clearing
            try:
                before_free, before_total = cp.cuda.runtime.memGetInfo()
                before_used = before_total - before_free
                before_used_mb = before_used / (1024 * 1024)
            except:
                before_used_mb = 0

            # Clear memory pools
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            # Get memory usage after clearing
            try:
                after_free, after_total = cp.cuda.runtime.memGetInfo()
                after_used = after_total - after_free
                after_used_mb = after_used / (1024 * 1024)
                freed_mb = before_used_mb - after_used_mb

                if freed_mb > 0:
                    logger.debug(
                        f"CuPy memory cleared: {freed_mb:.2f}MB freed")
                else:
                    logger.debug("CuPy memory cleared (no change in usage)")
            except:
                logger.debug("CuPy memory cleared")

        except Exception as e:
            logger.warning(f"Error clearing CuPy memory: {e}")
            success = False

    # Force garbage collection
    try:
        # Run multiple cycles of garbage collection for better cleanup
        for i in range(3):
            collected = gc.collect(i)
        logger.debug(
            f"Garbage collection completed: {collected} objects collected")
    except Exception as e:
        logger.warning(f"Error during garbage collection: {e}")
        success = False

    # Log final memory state
    final_memory = get_gpu_memory_usage()
    if final_memory and initial_memory:
        initial_used = initial_memory[0]
        final_used = final_memory[0]
        freed = (initial_used - final_used) / (1024 * 1024)
        total_mb = final_memory[1] / (1024 * 1024)
        final_used_mb = final_used / (1024 * 1024)

        if freed > 0:
            logger.info(
                f"GPU memory cleanup: {freed:.2f}MB freed, now at {final_used_mb:.2f}MB / {total_mb:.2f}MB ({final_used_mb/total_mb*100:.2f}%)")
        else:
            logger.info(
                f"GPU memory cleanup completed, current usage: {final_used_mb:.2f}MB / {total_mb:.2f}MB ({final_used_mb/total_mb*100:.2f}%)")

    # Log time taken
    elapsed = time.time() - start_time
    logger.debug(f"Memory cleanup completed in {elapsed*1000:.2f}ms")

    return success


def get_optimal_batch_size(model_size_mb: float = 500,
                           input_element_size_bytes: float = 4,
                           input_shape: Tuple[int, ...] = None,
                           min_batch: int = 8,
                           max_batch: int = 1024) -> int:
    """
    Calculate optimal batch size based on GPU memory availability

    Args:
        model_size_mb: Estimated model size in MB
        input_element_size_bytes: Size of each input element in bytes (e.g., 4 for float32)
        input_shape: Shape of a single input example (excluding batch dimension)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    # Conservative default
    default_batch = 64

    if not USE_GPU or not CUPY_AVAILABLE:
        return default_batch

    try:
        # Get available memory
        free, total = cp.cuda.runtime.memGetInfo()
        free_mb = free / (1024 * 1024)

        # Leave some memory for overhead (use only 80% of free memory)
        available_mb = free_mb * 0.8

        # Calculate memory needed per batch item
        if input_shape:
            # Calculate from input shape
            input_size_bytes = input_element_size_bytes
            for dim in input_shape:
                input_size_bytes *= dim
            input_size_mb = input_size_bytes / (1024 * 1024)
        else:
            # Use a conservative estimate
            input_size_mb = 0.5  # 0.5 MB per example

        # We need memory for:
        # 1. Model (weights + activations)
        # 2. Input data
        # 3. Output data (typically smaller than input)
        # 4. Intermediate activations (roughly 2-3x model size)

        # Calculate optimal batch size
        batch_size = int((available_mb - model_size_mb) / (input_size_mb * 5))

        # Ensure batch size is within bounds
        batch_size = max(min_batch, min(batch_size, max_batch))

        logger.info(
            f"Calculated optimal batch size: {batch_size} (available memory: {available_mb:.2f} MB)")
        return batch_size

    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}")
        return default_batch


class GPUMemoryManager:
    """GPU memory manager for automatic cleanup and monitoring"""

    def __init__(self, cleanup_interval: int = 300,
                 memory_threshold: float = 0.9):
        """
        Initialize GPU memory manager

        Args:
            cleanup_interval: Memory cleanup interval in seconds
            memory_threshold: Memory usage threshold (0.0-1.0) to trigger cleanup
        """
        self.cleanup_interval = cleanup_interval
        self.memory_threshold = memory_threshold
        self.running = False
        self.thread = None
        self.stats_history = []
        self.max_history = 100  # Keep last 100 memory snapshots

    def start(self) -> None:
        """Start the memory manager"""
        if self.running:
            logger.warning("Memory manager is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(
            f"GPU memory manager started (cleanup interval: {self.cleanup_interval}s, threshold: {self.memory_threshold*100}%)")

    def stop(self) -> None:
        """Stop the memory manager"""
        if not self.running:
            return

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("GPU memory manager stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Check if CUPY is available and GPU is used
                if USE_GPU and CUPY_AVAILABLE:
                    # Check memory usage
                    free, total = cp.cuda.runtime.memGetInfo()
                    used = total - free
                    used_ratio = used / total

                    # Record stats
                    timestamp = datetime.now().isoformat()
                    self.stats_history.append({
                        'timestamp': timestamp,
                        'used_bytes': int(used),
                        'free_bytes': int(free),
                        'total_bytes': int(total),
                        'used_ratio': float(used_ratio)
                    })

                    # Limit history length
                    if len(self.stats_history) > self.max_history:
                        self.stats_history = self.stats_history[-self.max_history:]

                    # Check if cleanup is needed
                    if used_ratio > self.memory_threshold:
                        logger.info(
                            f"GPU memory usage ({used_ratio:.2%}) exceeded threshold ({self.memory_threshold:.2%}), performing cleanup")
                        clear_gpu_memory()
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")

            # Sleep until next check
            for _ in range(min(60, self.cleanup_interval)):
                if not self.running:
                    break
                time.sleep(1)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics

        Returns:
            Dictionary with memory statistics
        """
        if not self.stats_history:
            return {
                'current': None,
                'average': None,
                'peak': None,
                'history_length': 0
            }

        # Calculate statistics
        current = self.stats_history[-1] if self.stats_history else None

        # Calculate averages
        if self.stats_history:
            avg_used = sum(stat['used_bytes']
                           for stat in self.stats_history) / len(self.stats_history)
            avg_ratio = sum(stat['used_ratio']
                            for stat in self.stats_history) / len(self.stats_history)
        else:
            avg_used = 0
            avg_ratio = 0

        # Find peak usage
        if self.stats_history:
            peak = max(self.stats_history, key=lambda x: x['used_ratio'])
        else:
            peak = None

        return {
            'current': current,
            'average': {
                'used_bytes': avg_used,
                'used_ratio': avg_ratio
            },
            'peak': peak,
            'history_length': len(self.stats_history)
        }


#########################
# MODEL OPTIMIZATION   #
#########################

def optimize_tensorflow_model(model: 'tf.keras.Model',
                              precision: str = 'float16',
                              use_tensorrt: bool = True,
                              dynamic_shapes: bool = True,
                              input_shapes: dict = None,
                              optimization_level: int = 3) -> 'tf.keras.Model':
    """
    Optimize a TensorFlow model for GPU inference with TensorRT

    Args:
        model: TensorFlow model to optimize
        precision: Precision to use ('float32', 'float16', or 'int8')
        use_tensorrt: Whether to use TensorRT for optimization
        dynamic_shapes: Whether to use dynamic shapes for TensorRT conversion
        input_shapes: Dictionary of input shapes for optimization (optional)
        optimization_level: TensorRT optimization level (1-5, higher = more aggressive)

    Returns:
        Optimized model
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available, cannot optimize model")
        return model

    if not USE_GPU:
        logger.info("GPU acceleration disabled, skipping model optimization")
        return model

    try:
        # Log initial model information
        logger.info(
            f"Optimizing model: {model.name if hasattr(model, 'name') else 'unnamed'}")
        logger.info(
            f"Model input shape: {model.input_shape if hasattr(model, 'input_shape') else 'unknown'}")
        logger.info(
            f"Model output shape: {model.output_shape if hasattr(model, 'output_shape') else 'unknown'}")

        # Set precision
        if precision == 'float16':
            # Convert to mixed precision
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
                logger.info("Set mixed_float16 precision policy")
            except Exception as e:
                logger.warning(f"Error setting mixed precision: {e}")

        # Apply TensorRT optimization if available and requested
        if use_tensorrt and TENSORRT_AVAILABLE:
            try:
                # Create a unique temporary directory
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="trt_model_")
                temp_saved_model = os.path.join(temp_dir, "saved_model")

                # Convert to SavedModel for TensorRT conversion
                logger.info(
                    f"Saving model to {temp_saved_model} for TensorRT conversion")
                tf.saved_model.save(model, temp_saved_model)

                # Configure TensorRT conversion parameters
                if precision == 'float16':
                    precision_mode = "FP16"
                elif precision == 'int8':
                    precision_mode = "INT8"
                else:
                    precision_mode = "FP32"

                # Adjust workspace size based on model complexity and available memory
                gpu_memory = get_gpu_memory_usage()
                if gpu_memory:
                    total_memory = gpu_memory[1]
                    # Use up to 75% of total GPU memory for workspace
                    workspace_size = min(int(total_memory * 0.75), 8000000000)
                else:
                    workspace_size = 8000000000  # Default 8GB

                logger.info(
                    f"Using TensorRT workspace size: {workspace_size/(1024**3):.2f} GB")

                # Configure conversion parameters
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                    max_workspace_size_bytes=workspace_size,
                    precision_mode=precision_mode,
                    maximum_cached_engines=100,
                    use_calibration=precision == 'int8',
                    allow_build_at_runtime=True
                )

                # Set optimization level
                if optimization_level > 0:
                    conversion_params = conversion_params._replace(
                        optimization_level=min(optimization_level, 5)
                    )
                    logger.info(
                        f"Set TensorRT optimization level to {optimization_level}")

                # Create TensorRT converter
                logger.info("Creating TensorRT converter...")
                converter_kwargs = {
                    'input_saved_model_dir': temp_saved_model,
                    'conversion_params': conversion_params
                }

                # Add dynamic shapes if requested
                if dynamic_shapes:
                    converter_kwargs['use_dynamic_shape'] = True
                    logger.info(
                        "Enabled dynamic shapes for TensorRT conversion")

                # Add input shapes if provided
                if input_shapes:
                    converter_kwargs['input_shape_map'] = input_shapes
                    logger.info(f"Using custom input shapes: {input_shapes}")

                converter = trt.TrtGraphConverterV2(**converter_kwargs)

                # Convert the model
                logger.info("Converting model with TensorRT...")
                start_time = time.time()
                converter.convert()
                conversion_time = time.time() - start_time
                logger.info(
                    f"TensorRT conversion completed in {conversion_time:.2f} seconds")

                # Save and load the optimized model
                optimized_model_path = os.path.join(temp_dir, "trt_model")
                logger.info(
                    f"Saving TensorRT optimized model to {optimized_model_path}")
                converter.save(optimized_model_path)

                # Load the optimized model
                logger.info("Loading TensorRT optimized model...")
                optimized_model = tf.saved_model.load(optimized_model_path)
                logger.info(
                    f"Model successfully optimized with TensorRT (precision: {precision_mode})")

                # Test inference with the optimized model
                try:
                    # Create a sample input for testing
                    if hasattr(model, 'input_shape') and model.input_shape:
                        input_shape = list(model.input_shape)
                        if input_shape[0] is None:  # Batch dimension
                            input_shape[0] = 1
                        sample_input = tf.random.normal(input_shape)

                        # Run inference
                        start_time = time.time()
                        if hasattr(optimized_model, '__call__'):
                            optimized_model(sample_input)
                        inference_time = time.time() - start_time
                        logger.info(
                            f"TensorRT model inference test successful. Inference time: {inference_time*1000:.2f} ms")
                except Exception as e:
                    logger.warning(
                        f"Error testing TensorRT model inference: {e}")

                # Clean up temporary directories
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up temporary directory: {e}")

                return optimized_model

            except Exception as e:
                logger.warning(f"Error optimizing model with TensorRT: {e}")
                logger.info("Falling back to standard TensorFlow model")
                return model

        else:
            # Apply basic TensorFlow optimizations
            try:
                # Ensure the model uses mixed precision
                if precision == 'float16':
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')

                # Apply graph optimization
                try:
                    # Convert to a SavedModel
                    temp_dir = tempfile.mkdtemp(prefix="tf_opt_model_")
                    temp_saved_model = os.path.join(temp_dir, "saved_model")

                    # Save the model
                    tf.saved_model.save(model, temp_saved_model)

                    # Load with optimization options
                    optimized_model = tf.saved_model.load(
                        temp_saved_model,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )

                    # Clean up
                    shutil.rmtree(temp_dir)

                    logger.info(
                        f"Applied TensorFlow graph optimizations (precision: {precision})")
                    return optimized_model
                except Exception as e:
                    logger.warning(
                        f"Error applying TensorFlow graph optimizations: {e}")
                    # Fall back to original model with basic optimizations

                logger.info(
                    f"Applied basic TensorFlow optimizations (precision: {precision})")
                return model

            except Exception as e:
                logger.warning(f"Error applying TensorFlow optimizations: {e}")
                return model

    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        return model


#########################
# GPU-ACCELERATED MATH #
#########################

def gpu_array(data: Union[List, np.ndarray]) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Convert data to GPU array if CuPy is available, otherwise use NumPy

    Args:
        data: Data to convert (list or NumPy array)

    Returns:
        CuPy array or NumPy array
    """
    if not USE_GPU or not CUPY_AVAILABLE:
        return np.array(data)

    try:
        return cp.array(data)
    except Exception as e:
        logger.warning(f"Error creating GPU array: {e}")
        return np.array(data)


def to_numpy(array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """
    Convert CuPy array to NumPy array if needed

    Args:
        array: Array to convert

    Returns:
        NumPy array
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def gpu_apply(array: Union[np.ndarray, 'cp.ndarray'],
              func: Callable,
              to_numpy_result: bool = True) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Apply a function to an array using GPU acceleration if available

    Args:
        array: Input array
        func: Function to apply (should work with both NumPy and CuPy)
        to_numpy_result: Whether to convert result to NumPy array

    Returns:
        Result array
    """
    if not USE_GPU or not CUPY_AVAILABLE:
        return func(array)

    try:
        # Convert to CuPy array if not already
        if not isinstance(array, cp.ndarray):
            cp_array = cp.array(array)
        else:
            cp_array = array

        # Apply function
        result = func(cp_array)

        # Convert back to NumPy if requested
        if to_numpy_result:
            return cp.asnumpy(result)
        return result

    except Exception as e:
        logger.warning(f"Error in GPU function application: {e}")
        # Fall back to NumPy
        if isinstance(array, cp.ndarray):
            array = cp.asnumpy(array)
        return func(array)


def gpu_matrix_multiply(a: Union[np.ndarray, 'cp.ndarray'],
                        b: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """
    Perform matrix multiplication using GPU acceleration if available

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        Result matrix as NumPy array
    """
    return gpu_apply(a, lambda x: cp.matmul(x, b) if CUPY_AVAILABLE else np.matmul(x, b), True)


def gpu_rolling_mean(array: Union[np.ndarray, 'cp.ndarray'],
                     window: int,
                     axis: int = 0) -> np.ndarray:
    """
    Calculate rolling mean using GPU acceleration if available

    Args:
        array: Input array
        window: Window size
        axis: Axis along which to calculate

    Returns:
        Rolling mean as NumPy array
    """
    if not USE_GPU or not CUPY_AVAILABLE or window <= 1:
        return np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window)/window, mode='valid'),
            axis, array
        )

    try:
        # Convert to CuPy array
        if not isinstance(array, cp.ndarray):
            array = cp.array(array)

        # Use CuPy's convolve function for rolling mean
        def rolling_func(x):
            # Create a ones kernel for the moving average
            kernel = cp.ones(window, dtype=cp.float32) / window
            return cp.convolve(x, kernel, mode='valid')

        result = cp.apply_along_axis(rolling_func, axis, array)
        return cp.asnumpy(result)

    except Exception as e:
        logger.warning(f"Error calculating GPU rolling mean: {e}")

        # Fall back to NumPy
        if isinstance(array, cp.ndarray):
            array = cp.asnumpy(array)

        return np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window)/window, mode='valid'),
            axis, array
        )


def gpu_batch_process(func: Callable,
                      data: List[Any],
                      batch_size: int = None) -> List[Any]:
    """
    Process data in batches using GPU

    Args:
        func: Function to apply to each batch
        data: List of data items to process
        batch_size: Batch size (will calculate optimal if None)

    Returns:
        List of processed results
    """
    if not data:
        return []

    # Determine batch size if not provided
    if batch_size is None:
        batch_size = get_optimal_batch_size()

    # Process in batches
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_results = func(batch)
        results.extend(batch_results)

        # Periodically clear GPU memory
        if i > 0 and i % (batch_size * 10) == 0:
            clear_gpu_memory()

    return results


#########################
#  TESTING & UTILITIES  #
#########################

def test_gpu_performance() -> Dict[str, Any]:
    """
    Test GPU performance with various operations

    Returns:
        Dictionary with performance metrics
    """
    results = {
        'gpu_available': USE_GPU,
        'cupy_available': CUPY_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'tensorrt_available': TENSORRT_AVAILABLE,
        'tests': {}
    }

    # Basic matrix multiplication test
    try:
        # Create test matrices
        size = 2000
        a_np = np.random.rand(size, size).astype(np.float32)
        b_np = np.random.rand(size, size).astype(np.float32)

        # Test with NumPy (CPU)
        start_time = time.time()
        np.matmul(a_np, b_np)
        cpu_time = time.time() - start_time
        results['tests']['matmul_cpu_time'] = cpu_time

        # Test with CuPy (GPU)
        if CUPY_AVAILABLE:
            a_cp = cp.array(a_np)
            b_cp = cp.array(b_np)

            # Warm-up run
            cp.matmul(a_cp, b_cp)
            cp.cuda.Stream.null.synchronize()

            # Timed run
            start_time = time.time()
            cp.matmul(a_cp, b_cp)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time

            results['tests']['matmul_gpu_time'] = gpu_time
            results['tests']['matmul_speedup'] = cpu_time / \
                gpu_time if gpu_time > 0 else 0

            logger.info(
                f"Matrix multiplication test: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={cpu_time/gpu_time:.2f}x")
        else:
            logger.info(
                f"Matrix multiplication test (CPU only): {cpu_time:.4f}s")
    except Exception as e:
        logger.error(f"Error in matrix multiplication test: {e}")
        results['tests']['matmul_error'] = str(e)

    # TensorFlow test if available
    if TF_AVAILABLE:
        try:
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    128, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy')

            # Create test data
            x = np.random.rand(1000, 100).astype(np.float32)
            y = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)

            # Warm-up run
            model.predict(x[:10])

            # Test inference speed
            start_time = time.time()
            model.predict(x)
            inference_time = time.time() - start_time

            results['tests']['tf_inference_time'] = inference_time
            results['tests']['tf_inference_examples_per_second'] = len(
                x) / inference_time

            logger.info(
                f"TensorFlow inference test: {inference_time:.4f}s ({len(x)/inference_time:.2f} examples/sec)")
        except Exception as e:
            logger.error(f"Error in TensorFlow test: {e}")
            results['tests']['tf_error'] = str(e)

    # Memory usage
    results['memory'] = log_memory_usage("Performance Test")

    return results


if __name__ == "__main__":
    # Test GPU performance
    print("Testing GPU performance...")

    # Initialize GPU
    if initialize_gpu():
        print("GPU initialization successful")
    else:
        print("GPU initialization failed")

    # Run performance test
    perf_results = test_gpu_performance()

    if perf_results['cupy_available']:
        print(
            f"Matrix multiplication speedup: {perf_results['tests'].get('matmul_speedup', 0):.2f}x")

    if perf_results['tensorflow_available']:
        print(
            f"TensorFlow inference: {perf_results['tests'].get('tf_inference_examples_per_second', 0):.2f} examples/sec")

    # Print memory usage
    if 'memory' in perf_results:
        if 'gpu_memory_used_mb' in perf_results['memory']:
            print(
                f"GPU memory used: {perf_results['memory']['gpu_memory_used_mb']:.2f} MB")
            print(
                f"GPU memory free: {perf_results['memory']['gpu_memory_free_mb']:.2f} MB")

        print(
            f"CPU memory used: {perf_results['memory']['cpu_memory_mb']:.2f} MB")
