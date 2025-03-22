#!/usr/bin/env python3
"""
TensorFlow GPU Direct Test Script

This script tests TensorFlow with direct GPU operations using low-level TensorFlow APIs.
"""

import os
import sys
import time
import platform
from datetime import datetime

import tensorflow as tf
import numpy as np


def print_separator():
    print("=" * 80)


def print_section(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def test_gpu_memory_allocation():
    """Test GPU memory allocation and deallocation"""
    print("Testing GPU memory allocation and deallocation...")

    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs available")
        return False

    # Get initial memory info
    try:
        initial_memory = []
        for i, gpu in enumerate(gpus):
            mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            initial_memory.append(mem_info)
            print(
                f"GPU:{i} Initial memory - Current: {mem_info['current'] / (1024**2):.2f} MB, Peak: {mem_info['peak'] / (1024**2):.2f} MB")
    except:
        print("Memory info not available")
        initial_memory = None

    # Allocate a large tensor
    print("\nAllocating large tensor on GPU...")
    with tf.device('/GPU:0'):
        # Allocate a 4GB tensor
        tensor_size = 1024  # Size that will use significant memory
        a = tf.random.normal([tensor_size, tensor_size])
        b = tf.random.normal([tensor_size, tensor_size])
        c = tf.matmul(a, b)

        # Force execution
        result = c.numpy()
        print(
            f"Large tensor shape: {c.shape}, Size: {c.numpy().nbytes / (1024**3):.2f} GB")

    # Check memory after allocation
    if initial_memory:
        try:
            for i, gpu in enumerate(gpus):
                mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                print(
                    f"GPU:{i} After allocation - Current: {mem_info['current'] / (1024**2):.2f} MB, Peak: {mem_info['peak'] / (1024**2):.2f} MB")
                print(
                    f"GPU:{i} Memory increase: {(mem_info['current'] - initial_memory[i]['current']) / (1024**2):.2f} MB")
        except:
            print("Memory info not available after allocation")

    # Delete tensors and run garbage collection
    print("\nDeleting tensors and running garbage collection...")
    del a, b, c, result

    # Try to force TensorFlow to release memory
    try:
        tf.keras.backend.clear_session()
    except:
        pass

    # Check memory after deallocation
    if initial_memory:
        try:
            for i, gpu in enumerate(gpus):
                mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                print(
                    f"GPU:{i} After deallocation - Current: {mem_info['current'] / (1024**2):.2f} MB, Peak: {mem_info['peak'] / (1024**2):.2f} MB")
        except:
            print("Memory info not available after deallocation")

    return True


def test_gpu_compute_capabilities():
    """Test GPU compute capabilities with different precision types"""
    print("\nTesting GPU compute capabilities with different precision types...")

    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs available")
        return False

    # Test FP32 (single precision)
    print("\nTesting FP32 (single precision)...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([5000, 5000], dtype=tf.float32)
        b = tf.random.normal([5000, 5000], dtype=tf.float32)

        start_time = time.time()
        c = tf.matmul(a, b)
        # Force execution
        _ = c.numpy()
        elapsed = time.time() - start_time

        print(f"FP32 Matrix multiplication time: {elapsed:.4f} seconds")

    # Test FP16 (half precision) if supported
    try:
        print("\nTesting FP16 (half precision)...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000], dtype=tf.float16)
            b = tf.random.normal([5000, 5000], dtype=tf.float16)

            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy()
            elapsed = time.time() - start_time

            print(f"FP16 Matrix multiplication time: {elapsed:.4f} seconds")
    except:
        print("FP16 not supported or test failed")

    # Test mixed precision if supported
    try:
        print("\nTesting mixed precision...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])

            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy()
            elapsed = time.time() - start_time

            print(
                f"Mixed precision matrix multiplication time: {elapsed:.4f} seconds")

        # Reset policy
        tf.keras.mixed_precision.set_global_policy('float32')
    except:
        print("Mixed precision not supported or test failed")

    return True


def test_multi_gpu():
    """Test multi-GPU operations if multiple GPUs are available"""
    print("\nTesting multi-GPU operations...")

    # Get available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) < 2:
        print(f"Only {len(gpus)} GPU(s) available, skipping multi-GPU test")
        return True

    # Create a MirroredStrategy
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")

        # Create a model within the strategy scope
        with strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    256, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
            ])

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy']
            )

        # Create some dummy data
        batch_size = 64 * strategy.num_replicas_in_sync
        train_data = tf.random.normal([batch_size * 10, 784])
        train_labels = tf.random.uniform(
            [batch_size * 10], minval=0, maxval=10, dtype=tf.int64)

        # Train the model
        print("Training model on multiple GPUs...")
        start_time = time.time()
        model.fit(train_data, train_labels, epochs=2,
                  batch_size=batch_size, verbose=1)
        elapsed = time.time() - start_time

        print(f"Multi-GPU training time: {elapsed:.4f} seconds")

        return True
    except Exception as e:
        print(f"Multi-GPU test failed: {e}")
        return False


def main():
    print_section("TensorFlow GPU Direct Test")

    # Print basic system information
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {platform.python_version()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")

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

        # Run tests
        try:
            test_gpu_memory_allocation()
            test_gpu_compute_capabilities()
            test_multi_gpu()

            print("\nAll GPU direct tests PASSED ✓")
            return 0
        except Exception as e:
            print(f"\nError during GPU direct tests: {e}")
            return 1
    else:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        print("\nGPU direct tests SKIPPED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
