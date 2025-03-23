#!/usr/bin/env python3
"""
TensorFlow GPU Verification Script

This script verifies that TensorFlow is properly installed and can access the GPU.
It prints information about the TensorFlow version, available devices, and GPU capabilities.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime


def verify_tensorflow():
    """Verify TensorFlow installation and GPU access."""
    print("=" * 80)
    print("TensorFlow GPU Verification")
    print("=" * 80)

    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Print available devices
    print("\nAvailable devices:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"  {device.device_type}: {device.name}")

    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        return False

    print(f"\nFound {len(gpus)} GPU device(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # Print GPU details
    print("\nGPU details:")
    try:
        for gpu_id, gpu in enumerate(gpus):
            # Get GPU memory info
            try:
                gpu_mem = tf.config.experimental.get_memory_info(
                    f'GPU:{gpu_id}')
                print(f"  GPU {gpu_id} memory:")
                print(f"    Current: {gpu_mem['current'] / (1024**3):.2f} GB")
                print(f"    Peak: {gpu_mem['peak'] / (1024**3):.2f} GB")
            except (ValueError, tf.errors.NotFoundError) as e:
                print(f"  Could not get memory info for GPU {gpu_id}: {e}")
    except Exception as e:
        print(f"  Error getting GPU details: {e}")

    # Run a simple test
    print("\nRunning a simple matrix multiplication test on GPU...")
    try:
        with tf.device('/GPU:0'):
            start_time = datetime.now()

            # Create large matrices
            matrix_size = 5000
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])

            # Perform matrix multiplication
            c = tf.matmul(a, b)

            # Force execution and synchronization
            result = c.numpy()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(
                f"  Matrix multiplication of {matrix_size}x{matrix_size} completed in {duration:.2f} seconds")
            print(f"  Result shape: {result.shape}")
            print(f"  Result mean: {np.mean(result):.6f}")
            print(f"  Result std: {np.std(result):.6f}")

            print("\nGPU test successful!")
            return True
    except Exception as e:
        print(f"  Error during GPU test: {e}")
        return False


if __name__ == "__main__":
    success = verify_tensorflow()
    sys.exit(0 if success else 1)
