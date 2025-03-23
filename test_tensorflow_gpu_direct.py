#!/usr/bin/env python3
"""
TensorFlow GPU Direct Test Script

This script tests TensorFlow's ability to directly access the GPU using low-level operations.
It focuses on testing CUDA operations, memory transfers, and kernel execution.
"""

import os
import sys
import time
import tensorflow as tf
import numpy as np
from datetime import datetime


def test_tensorflow_gpu_direct():
    """Test TensorFlow's direct GPU access capabilities."""
    print("=" * 80)
    print("TensorFlow GPU Direct Access Test")
    print("=" * 80)

    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        return False

    print(f"\nFound {len(gpus)} GPU device(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # Test direct GPU memory allocation and operations
    print("\nTesting direct GPU memory operations...")

    try:
        with tf.device('/GPU:0'):
            # Test 1: Large tensor creation and basic operations
            print("\nTest 1: Large tensor creation and basic operations")
            start_time = time.time()

            # Create large tensors directly on GPU
            size = 10000
            a = tf.random.normal([size, size], dtype=tf.float32)
            b = tf.random.normal([size, size], dtype=tf.float32)

            # Perform operations
            c = a + b
            d = tf.matmul(a, b)
            e = tf.nn.relu(d)

            # Force execution
            result = e.numpy()

            end_time = time.time()
            print(f"  Completed in {end_time - start_time:.2f} seconds")
            print(f"  Result shape: {result.shape}")
            print(f"  Result mean: {np.mean(result):.6f}")

            # Test 2: Memory transfer speed
            print("\nTest 2: CPU-GPU memory transfer speed")

            # Create large tensor on CPU
            # 1 billion elements (4GB for float32)
            cpu_tensor_size = 1000000000
            print(f"  Creating CPU tensor with {cpu_tensor_size} elements...")

            # Use smaller size if memory is limited
            try:
                cpu_tensor = np.random.random(
                    cpu_tensor_size).astype(np.float32)
            except MemoryError:
                print("  Memory error with large tensor, reducing size...")
                cpu_tensor_size = 100000000  # 100 million elements (400MB)
                cpu_tensor = np.random.random(
                    cpu_tensor_size).astype(np.float32)

            print(f"  CPU tensor created with shape: {cpu_tensor.shape}")

            # Measure transfer to GPU
            start_time = time.time()
            gpu_tensor = tf.constant(cpu_tensor)
            # Force execution
            _ = gpu_tensor.numpy()
            end_time = time.time()

            transfer_time = end_time - start_time
            transfer_size_gb = cpu_tensor.nbytes / (1024**3)
            transfer_speed = transfer_size_gb / transfer_time

            print(
                f"  CPU to GPU transfer: {transfer_size_gb:.2f} GB in {transfer_time:.2f} seconds")
            print(f"  Transfer speed: {transfer_speed:.2f} GB/s")

            # Test 3: Kernel execution performance
            print("\nTest 3: CUDA kernel execution performance")

            # Create data for convolution
            input_size = 1024
            batch_size = 16
            channels = 64

            inputs = tf.random.normal(
                [batch_size, input_size, input_size, channels])
            filters = tf.random.normal([3, 3, channels, channels*2])

            # Warm up
            _ = tf.nn.conv2d(inputs, filters, strides=[
                             1, 1, 1, 1], padding='SAME')

            # Measure convolution performance
            iterations = 10
            start_time = time.time()

            for i in range(iterations):
                output = tf.nn.conv2d(inputs, filters, strides=[
                                      1, 1, 1, 1], padding='SAME')
                # Force execution
                _ = output.numpy()

            end_time = time.time()
            avg_time = (end_time - start_time) / iterations

            print(f"  Convolution shape: {inputs.shape} * {filters.shape}")
            print(f"  Average time per convolution: {avg_time:.4f} seconds")
            print(f"  Output shape: {output.shape}")

            print("\nGPU direct access test successful!")
            return True

    except Exception as e:
        print(f"\nError during GPU direct access test: {e}")
        return False


if __name__ == "__main__":
    success = test_tensorflow_gpu_direct()
    sys.exit(0 if success else 1)
