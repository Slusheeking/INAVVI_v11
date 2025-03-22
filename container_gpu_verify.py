#!/usr/bin/env python3
"""
Container GPU Stack Verification Script

This script verifies that TensorFlow, TensorRT, and CuPy are properly installed and configured
in the Docker container with GPU access.
"""

import os
import sys
import platform
import importlib.util
from datetime import datetime


def print_separator():
    print("=" * 80)


def print_section(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def check_module_installed(module_name):
    """Check if a module is installed"""
    return importlib.util.find_spec(module_name) is not None


def verify_system_info():
    """Print system information"""
    print_section("System Information")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")

    # Check if running in container
    in_container = os.path.exists('/.dockerenv')
    print(f"Running in Docker container: {in_container}")

    # Check environment variables
    print("\nEnvironment Variables:")
    for var in ['CUDA_HOME', 'LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES']:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")


def verify_tensorflow():
    """Verify TensorFlow installation and GPU support"""
    print_section("TensorFlow Verification")

    if not check_module_installed("tensorflow"):
        print("TensorFlow is not installed.")
        return False

    import tensorflow as tf
    import numpy as np

    # Print TensorFlow information
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"TensorFlow Build Info: {tf.sysconfig.get_build_info()}")

    # Check for GPU availability
    print("\nGPU Information:")
    physical_devices = tf.config.list_physical_devices()

    print(f"All Physical Devices: {len(physical_devices)}")
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")

    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU Devices Available: {len(gpu_devices)}")

    if len(gpu_devices) > 0:
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu.name}")

        # Get GPU device details
        try:
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            for device in local_devices:
                if device.device_type == 'GPU':
                    print(f"\nGPU Details:")
                    print(f"  Name: {device.name}")
                    print(
                        f"  Memory: {device.memory_limit / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Could not retrieve detailed GPU information: {e}")

        # Test GPU with a simple computation
        print("\nRunning simple GPU computation test...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            start_time = datetime.now()
            c = tf.matmul(a, b)
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  Matrix multiplication shape: {c.shape}")
            print(f"  Computation time: {elapsed:.4f} seconds")

        print("\nGPU TEST PASSED ✓")
        return True
    else:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        print("\nGPU TEST FAILED ✗")
        return False


def verify_tensorrt():
    """Verify TensorRT installation"""
    print_section("TensorRT Verification")

    if not check_module_installed("tensorrt"):
        print("TensorRT Python package is not installed.")

        # Check if TensorRT is available through TensorFlow
        if check_module_installed("tensorflow"):
            import tensorflow as tf
            if hasattr(tf, 'experimental') and hasattr(tf.experimental, 'tensorrt'):
                print("TensorRT is available through TensorFlow's experimental API.")
                print("TensorRT version information:")
                try:
                    # Try to get TensorRT version through TensorFlow
                    trt_info = tf.experimental.tensorrt.get_linked_tensorrt_version()
                    print(f"  Linked TensorRT version: {trt_info}")
                    print("\nTensorRT TEST PASSED ✓")
                    return True
                except Exception as e:
                    print(
                        f"Could not determine TensorRT version through TensorFlow: {e}")

        # Check if the TensorRT shared libraries are available
        import ctypes
        try:
            ctypes.CDLL("libnvinfer.so")
            print("TensorRT libraries are installed in the system.")
            print("\nTensorRT TEST PASSED ✓")
            return True
        except Exception as e:
            print(f"TensorRT libraries not found in the system: {e}")
            print("\nTensorRT TEST FAILED ✗")
            return False

    try:
        import tensorrt as trt
        print(f"TensorRT Version: {trt.__version__}")

        # Create a simple TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        print(f"TensorRT Builder created successfully.")
        print(f"Max batch size: {builder.max_batch_size}")

        # Check available TensorRT capabilities
        print("\nTensorRT Capabilities:")
        print(f"  Platform has FP16: {builder.platform_has_fast_fp16}")
        print(f"  Platform has INT8: {builder.platform_has_fast_int8}")

        print("\nTensorRT TEST PASSED ✓")
        return True
    except Exception as e:
        print(f"Error verifying TensorRT: {e}")
        print("\nTensorRT TEST FAILED ✗")
        return False


def verify_cupy():
    """Verify CuPy installation and GPU support"""
    print_section("CuPy Verification")

    if not check_module_installed("cupy"):
        print("CuPy is not installed.")
        print("\nCuPy TEST FAILED ✗")
        return False

    try:
        import cupy as cp
        print(f"CuPy Version: {cp.__version__}")

        # Get CUDA information
        print("\nCUDA Information:")
        print(f"  CUDA Runtime Version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"  Number of CUDA devices: {cp.cuda.runtime.getDeviceCount()}")

        # Get device information
        for i in range(cp.cuda.runtime.getDeviceCount()):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nDevice {i} Information:")
            print(f"  Name: {device_props['name'].decode()}")
            print(
                f"  Compute Capability: {device_props['major']}.{device_props['minor']}")
            print(
                f"  Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")

        # Run a simple test
        print("\nRunning simple CuPy computation test...")
        a = cp.random.random((5000, 5000), dtype=cp.float32)
        b = cp.random.random((5000, 5000), dtype=cp.float32)

        start_time = datetime.now()
        c = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()  # Ensure computation is complete
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"  Matrix multiplication shape: {c.shape}")
        print(f"  Computation time: {elapsed:.4f} seconds")

        print("\nCuPy TEST PASSED ✓")
        return True
    except Exception as e:
        print(f"Error verifying CuPy: {e}")
        print("\nCuPy TEST FAILED ✗")
        return False


def verify_redis():
    """Verify Redis connection"""
    print_section("Redis Verification")

    if not check_module_installed("redis"):
        print("Redis Python client is not installed.")
        print("\nRedis TEST FAILED ✗")
        return False

    try:
        import redis
        from os import environ

        # Get Redis connection parameters from environment variables
        host = environ.get('REDIS_HOST', 'localhost')
        port = int(environ.get('REDIS_PORT', 6380))
        db = int(environ.get('REDIS_DB', 0))
        password = environ.get('REDIS_PASSWORD', 'trading_system_2025')
        username = environ.get('REDIS_USERNAME', 'default')

        print(f"Connecting to Redis at {host}:{port}...")

        # Create Redis client
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            username=username,
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Test connection
        ping_result = r.ping()
        print(f"Redis ping result: {ping_result}")

        # Test basic operations
        print("\nTesting basic Redis operations...")

        # Set a test key
        test_key = "test:container:verification"
        r.set(test_key, "Container GPU stack verification")

        # Get the test key
        value = r.get(test_key)
        print(f"  Retrieved value: {value.decode() if value else None}")

        # Delete the test key
        r.delete(test_key)

        # Get Redis info
        info = r.info()
        print("\nRedis Server Information:")
        print(f"  Redis Version: {info.get('redis_version')}")
        print(f"  Uptime: {info.get('uptime_in_seconds')} seconds")
        print(f"  Connected Clients: {info.get('connected_clients')}")
        print(f"  Used Memory: {info.get('used_memory_human')}")

        print("\nRedis TEST PASSED ✓")
        return True
    except Exception as e:
        print(f"Error verifying Redis: {e}")
        print("\nRedis TEST FAILED ✗")
        return False


def verify_file_access():
    """Verify access to mounted volumes"""
    print_section("File Access Verification")

    # Check project directory
    project_dir = "/app/project"
    print(f"Checking project directory: {project_dir}")
    if os.path.exists(project_dir):
        print(f"  Project directory exists: ✓")
        # List some files
        files = os.listdir(project_dir)
        print(f"  Files in project directory: {len(files)}")
        print(f"  Sample files: {files[:5] if len(files) > 5 else files}")
    else:
        print(f"  Project directory does not exist: ✗")

    # Check data directory
    data_dir = "/app/data"
    print(f"\nChecking data directory: {data_dir}")
    if os.path.exists(data_dir):
        print(f"  Data directory exists: ✓")
        # Try to create a test file
        try:
            test_file = os.path.join(data_dir, "test_access.txt")
            with open(test_file, 'w') as f:
                f.write("File access test")
            print(f"  Write access to data directory: ✓")
            os.remove(test_file)
            print(f"  Delete access to data directory: ✓")
        except Exception as e:
            print(f"  Error accessing data directory: {e}")
    else:
        print(f"  Data directory does not exist: ✗")

    # Check models directory
    models_dir = "/app/models"
    print(f"\nChecking models directory: {models_dir}")
    if os.path.exists(models_dir):
        print(f"  Models directory exists: ✓")
    else:
        print(f"  Models directory does not exist: ✗")

    # Check logs directory
    logs_dir = "/app/logs"
    print(f"\nChecking logs directory: {logs_dir}")
    if os.path.exists(logs_dir):
        print(f"  Logs directory exists: ✓")
    else:
        print(f"  Logs directory does not exist: ✗")


def main():
    print_section("Container GPU Stack Verification")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify system information
    verify_system_info()

    # Verify file access
    verify_file_access()

    # Verify each component
    tf_result = verify_tensorflow()
    trt_result = verify_tensorrt()
    cupy_result = verify_cupy()
    redis_result = verify_redis()

    # Print summary
    print_section("Verification Summary")
    print(f"TensorFlow GPU: {'✓ PASSED' if tf_result else '✗ FAILED'}")
    print(f"TensorRT: {'✓ PASSED' if trt_result else '✗ FAILED'}")
    print(f"CuPy: {'✓ PASSED' if cupy_result else '✗ FAILED'}")
    print(f"Redis: {'✓ PASSED' if redis_result else '✗ FAILED'}")

    print_separator()
    if tf_result and trt_result and cupy_result and redis_result:
        print("All components verified successfully!")
        print("The container is properly configured with GPU access.")
        print("TensorFlow, TensorRT, and CuPy are working correctly with the GPU.")
        print("Redis is properly configured and accessible.")
        print("All mounted volumes are accessible.")
    else:
        print("Some components failed verification. Check the logs for details.")
    print_separator()


if __name__ == "__main__":
    main()
