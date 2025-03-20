#!/usr/bin/env python3
"""
GPU Utilities for NVIDIA GH200
This module provides utility functions for GPU configuration and memory tracking.
"""

import logging
import psutil
import cupy as cp
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_utils')


def log_memory_usage(location_tag: str) -> None:
    """Log CPU and GPU memory usage"""
    try:
        # Log CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 * 1024)

        # Log GPU memory
        mem_info = cp.cuda.runtime.memGetInfo()
        free, total = mem_info[0], mem_info[1]
        used = total - free

        logger.info(
            f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
    except Exception as e:
        logger.error(f"Failed to log memory usage at {location_tag}: {e}")


def configure_gpu() -> bool:
    """Configure GPU for optimal performance with TensorFlow, CuPy, and TensorRT"""
    try:
        # Apply GH200-specific optimizations first
        gpu_found = False
        # Configure TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {len(gpus)} GPUs")

                # Set TensorFlow to use mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                # Configure TensorRT
                conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                    max_workspace_size_bytes=8000000000,
                    precision_mode="FP16",
                    maximum_cached_engines=100
                )
                logger.info("TensorRT configured with FP16 precision")

                logger.info(
                    "TensorFlow and TensorRT configured with mixed precision")
                gpu_found = True
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")

        # Configure CuPy
        try:
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            logger.info("CuPy configured with unified memory")

            # Get device count and properties
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    device_name = device_props["name"].decode()
                    gpu_found = True
                    logger.info(f"GPU {i}: {device_name}")

                    # If GH200 is available, use it
                    if "GH200" in device_name:
                        cp.cuda.Device(i).use()
                        logger.info(f"Using GH200 GPU at index {i}")
                        break

            if not gpu_found:
                raise RuntimeError(
                    "No GPU devices found. This system requires NVIDIA GH200 GPU acceleration.")
            return True
        except Exception as e:
            logger.error(f"CuPy configuration failed: {e}")
            return False

    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        return False
