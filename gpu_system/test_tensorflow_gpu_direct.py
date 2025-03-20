#!/usr/bin/env python3
"""
Direct TensorFlow GPU Test
Tests direct GPU operations using TensorFlow
"""

import tensorflow as tf
import time
import sys


def test_gpu_direct():
    print("Testing direct GPU operations with TensorFlow...")
    print(f"TensorFlow version: {tf.__version__}")

    # Get list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"Found GPU: {gpu}")

    if not gpus:
        print("No GPU found. Test failed.")
        return False

    try:
        # Create a large matrix operation
        print("\nPerforming matrix multiplication on GPU...")
        with tf.device('/GPU:0'):
            # Create two large matrices
            size = 5000
            start_time = time.time()

            # Matrix multiplication test
            matrix1 = tf.random.normal([size, size])
            matrix2 = tf.random.normal([size, size])
            result = tf.matmul(matrix1, matrix2)

            # Force execution and GPU sync
            _ = result.numpy()

            end_time = time.time()

        print(
            f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
        print("GPU test successful!")
        return True

    except Exception as e:
        print(f"Error during GPU test: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_gpu_direct()
    sys.exit(0 if success else 1)
