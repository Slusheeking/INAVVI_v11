#!/usr/bin/env python3
"""
GPU Setup Script for ML Trading System

This script configures the environment for optimal GPU usage with TensorFlow and CuPy,
similar to the settings in the Docker container. It sets environment variables,
verifies GPU availability, and provides detailed information about the GPU configuration.
"""

import os
import sys
import logging
import subprocess
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_gpu')

def set_environment_variables():
    """Set environment variables for optimal GPU performance"""
    # TensorFlow GPU optimization
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    os.environ['TF_CUDA_HOST_MEM_LIMIT_IN_MB'] = '80000'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '8'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_USE_CUDA_GRAPHS'] = '0'
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
    os.environ['TF_LAYOUT_OPTIMIZER_DISABLE'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # NVIDIA visibility
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
    
    # XLA flags for better performance
    if platform.system() == 'Linux':
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
    
    logger.info("Environment variables set for optimal GPU performance")

def check_nvidia_smi():
    """Check if nvidia-smi is available and get GPU information"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA System Management Interface (nvidia-smi) is available")
            logger.info("GPU Information:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            return True
        else:
            logger.warning("nvidia-smi command failed with error:")
            logger.warning(result.stderr)
            return False
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found. NVIDIA drivers may not be installed.")
        return False
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow can access the GPU"""
    try:
        # Set environment variables for TensorFlow
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '8'
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
        
        import tensorflow as tf
        
        # Try to force TensorFlow to see the GPU
        physical_devices = tf.config.list_physical_devices()
        logger.info(f"All physical devices: {physical_devices}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow can access {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu}")
            
            # Configure memory growth for all GPUs
            for gpu in gpus:
                try:
                    # First try to set memory growth
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"  Memory growth enabled for {gpu}")
                    except Exception as e:
                        logger.warning(f"  Unable to set memory growth for {gpu}: {e}")
                        
                    # Then try to set virtual device configuration
                    try:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                        )
                        logger.info(f"  Virtual device configuration set for {gpu}")
                    except Exception as e:
                        logger.warning(f"  Unable to set virtual device configuration for {gpu}: {e}")
                except RuntimeError as e:
                    logger.warning(f"  Unable to set memory growth for {gpu}: {e}")
            
            # Get GPU device details
            for i, gpu in enumerate(gpus):
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    logger.info(f"  GPU {i} details: {gpu_details}")
                except Exception as e:
                    logger.warning(f"  Unable to get details for GPU {i}: {e}")
            
            # Run a simple test operation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                    c = tf.matmul(a, b)
                    result = c.numpy()
                logger.info(f"  Test matrix multiplication on GPU successful: {result}")
            except Exception as e:
                logger.warning(f"  Test operation on GPU failed: {e}")
            
            return True
        else:
            logger.warning("No GPU detected by TensorFlow")
            logger.info("TensorFlow will use CPU for computations")
            return False
    except ImportError:
        logger.error("TensorFlow is not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking TensorFlow GPU support: {e}")
        return False

def check_cupy():
    """Check if CuPy is available and can access the GPU"""
    try:
        import cupy as cp  
        
        # Explicitly select the first GPU device
        cp.cuda.Device(0).use()
        logger.info("Explicitly selected GPU device 0")
        
        # Configure CuPy for optimal performance
        try:
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            logger.info("CuPy configured with unified memory for GH200")
        except Exception as e:
            logger.warning(f"Unable to configure CuPy memory pool: {e}")
        
        # Get GPU information
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            logger.info(f"CuPy can access {num_gpus} GPU(s)")
            
            for i in range(num_gpus):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                logger.info(f"  GPU {i}: {device_props['name'].decode('utf-8')}")
                logger.info(f"    Compute Capability: {device_props['major']}.{device_props['minor']}")
                logger.info(f"    Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
            
            # Get memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            free, total = mem_info[0], mem_info[1]
            used = total - free
            logger.info(f"  GPU Memory: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
            
            # Run a simple test operation
            try:
                a = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
                b = cp.array([[5, 6], [7, 8]], dtype=cp.float32)
                c = cp.matmul(a, b)
                result = c.get()
                logger.info(f"  Test matrix multiplication on GPU with CuPy successful: {result}")
            except Exception as e:
                logger.warning(f"  Test operation on GPU with CuPy failed: {e}")
            
            return True
        except Exception as e:
            logger.warning(f"Error getting CuPy GPU information: {e}")
            return False
    except ImportError:
        logger.warning("CuPy is not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking CuPy GPU support: {e}")
        return False

def main():
    """Main function"""
    logger.info("Setting up GPU environment for ML Trading System...")
    
    # Set environment variables
    set_environment_variables()
    
    # Check NVIDIA drivers
    nvidia_ok = check_nvidia_smi()
    
    # Check TensorFlow GPU support
    tf_gpu_ok = check_tensorflow_gpu()
    
    # Check CuPy GPU support
    cupy_ok = check_cupy()
    
    # Print summary
    logger.info("\nGPU Setup Summary:")
    logger.info(f"NVIDIA Drivers: {'✓' if nvidia_ok else '✗'}")
    logger.info(f"TensorFlow GPU: {'✓' if tf_gpu_ok else '✗'}")
    logger.info(f"CuPy GPU: {'✓' if cupy_ok else '✗'}")
    
    if nvidia_ok and tf_gpu_ok and cupy_ok:
        logger.info("\n✓ GPU setup is complete and ready for ML Trading System")
        return 0
    elif tf_gpu_ok or cupy_ok:
        logger.info("\n⚠ GPU setup is partially complete. Some GPU features may be limited.")
        return 0
    else:
        logger.error("\n✗ GPU setup failed. The system will use CPU for computations.")
        logger.info("To use GPU acceleration, please ensure that:")
        logger.info("  1. NVIDIA drivers are installed")
        logger.info("  2. CUDA toolkit is installed")
        logger.info("  3. TensorFlow and CuPy are installed with GPU support")
        logger.info("  4. The system has a compatible NVIDIA GPU")
        return 1

if __name__ == "__main__":
    sys.exit(main())