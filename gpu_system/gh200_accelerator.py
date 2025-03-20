#!/usr/bin/env python3
"""
GH200 Accelerator Module

This module provides GPU acceleration and optimization specifically for the NVIDIA GH200 Grace Hopper Superchip.
It configures TensorFlow and CuPy for optimal performance on this architecture.
"""

import os
import logging
import ctypes
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gh200_accelerator')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


def configure_tensorflow_for_gh200():
    """Configure TensorFlow specifically for GH200 architecture"""
    try:
        # Check if we're running in the TensorFlow container
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
            logger.info(
                "Running in TensorFlow container, using container's configuration")
            return True

        # Set basic environment variables for GH200
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        # Disable XLA auto-clustering which can cause graph update errors
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=0"

        # Disable eager optimization that can cause graph update errors
        os.environ["TF_FUNCTION_JIT_COMPILE_DEFAULT"] = "0"

        # Enable eager execution for stability
        tf.config.run_functions_eagerly(True)

        # For Grace Hopper specifically
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation

        # Prevent OOM errors
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ["TF_GPU_HOST_MEM_LIMIT_IN_MB"] = "4096"

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
                            # Limit to 95GB (slightly less than total)
                            memory_limit=int(1024 * 1024 * 95)
                        )]  # Use 95GB as the memory limit
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
                # Use float32 instead of mixed_float16 for more stable training
                set_global_policy('float32')
                logger.info(
                    "TensorFlow configured for GH200 with float32 precision")
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


def register_gh200_device():
    """Register GH200 as a special device for TensorFlow"""
    try:
        # Load CUDA libraries explicitly
        try:
            ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
            logger.info("Successfully loaded CUDA libraries")
        except Exception as e:
            logger.warning(f"Could not load CUDA libraries: {str(e)}")

        # Force device discovery
        physical_devices = tf.config.list_physical_devices()
        if not any(device.device_type == 'GPU' for device in physical_devices):
            # If no GPU found, try manual registration
            logger.info(
                "No GPU found by TensorFlow, attempting manual device registration...")
            return False
        return True
    except Exception as e:
        logger.error(f"Error registering GH200 device: {str(e)}")
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
            import cupy as cp

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
        except ImportError:
            logger.warning("CuPy not available")

    def _configure_tensorrt(self):
        """Configure TensorRT for optimized model inference"""
        try:
            # Initialize TensorRT converter with optimized settings for GH200
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                max_workspace_size_bytes=8000000000,  # 8GB workspace
                precision_mode="FP16",  # Use FP16 for better performance
                maximum_cached_engines=100,
                use_calibration=True
            )

            self.trt_converter = trt.TrtGraphConverterV2(
                conversion_params=conversion_params,
                use_dynamic_shape=True
            )

            self.has_tensorrt = True
            logger.info("TensorRT configured successfully")
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
            # Convert model to SavedModel format
            temp_saved_model = "/tmp/model"
            tf.saved_model.save(model, temp_saved_model)

            # Convert to TensorRT
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=temp_saved_model,
                conversion_params=self.trt_converter.conversion_params
            )
            converter.convert()

            # Save and load the optimized model
            converter.save(temp_saved_model + "_trt")
            optimized_model = tf.saved_model.load(temp_saved_model + "_trt")

            logger.info("Model successfully optimized with TensorRT")
            return optimized_model
        except Exception as e:
            logger.error(f"Error optimizing model with TensorRT: {e}")
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
            if self.has_tensorflow_gpu:
                tf.keras.backend.clear_session()

                # More aggressive memory cleanup
                # Reset memory stats removed - was causing error:
                # tf.config.experimental.reset_memory_stats not compatible with device objects

            if self.has_cupy_gpu:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except ImportError:
                    pass

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


if __name__ == "__main__":
    # Test the GH200 accelerator
    print("Testing GH200 Accelerator...")
    accelerator = GH200Accelerator()

    if accelerator.has_tensorflow_gpu:
        print("TensorFlow GPU acceleration is available")

        # Test a simple matrix multiplication
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = tf.timestamp()
            c = tf.matmul(a, b)
            end_time = tf.timestamp()
            print(
                f"GPU Matrix multiplication time: {(end_time - start_time).numpy() * 1000:.2f} ms")
    else:
        print("TensorFlow GPU acceleration is NOT available")

    if accelerator.has_cupy_gpu:
        print("CuPy GPU acceleration is available")

        # Test CuPy if available
        try:
            import cupy as cp
            a_cp = cp.random.normal(0, 1, (1000, 1000))
            b_cp = cp.random.normal(0, 1, (1000, 1000))
            start_time = time.time()
            c_cp = cp.matmul(a_cp, b_cp)
            cp.cuda.Stream.null.synchronize()
            end_time = time.time()
            print(
                f"CuPy Matrix multiplication time: {(end_time - start_time) * 1000:.2f} ms")
        except:
            print("Error testing CuPy")
    else:
        print("CuPy GPU acceleration is NOT available")
