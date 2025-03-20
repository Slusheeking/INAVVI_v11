#!/usr/bin/env python3
"""
TensorFlow GPU Integration Test
Tests GPU operations in an integrated environment
"""

import tensorflow as tf
import numpy as np
import time
import sys


def test_tensorflow_gpu():
    print("Running TensorFlow GPU integration test...")
    print(f"TensorFlow version: {tf.__version__}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"Found GPU: {gpu}")

        # Get device details
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"Device details: {details}")
        except:
            print("Device details not available")

    if not gpus:
        print("No GPU found. Test failed.")
        return False

    try:
        # Test 1: Basic tensor operations
        print("\nTest 1: Basic tensor operations")
        with tf.device('/GPU:0'):
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])
            z = tf.matmul(x, y)
            print("Basic tensor operations successful")

        # Test 2: Memory management
        print("\nTest 2: Memory management")
        with tf.device('/GPU:0'):
            # Allocate and deallocate large tensors
            for _ in range(5):
                large_tensor = tf.random.normal([5000, 5000])
                del large_tensor
            print("Memory management test successful")

        # Test 3: Performance benchmark
        print("\nTest 3: Performance benchmark")
        with tf.device('/GPU:0'):
            start_time = time.time()

            # Create and train a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            x_train = tf.random.normal([10000, 1000])
            y_train = tf.random.uniform(
                [10000], minval=0, maxval=10, dtype=tf.int32)
            y_train = tf.one_hot(y_train, 10)

            model.compile(optimizer='adam', loss='categorical_crossentropy')
            model.fit(x_train, y_train, epochs=1, verbose=0)

            end_time = time.time()
            print(
                f"Performance benchmark completed in {end_time - start_time:.2f} seconds")

        print("\nAll GPU tests passed successfully!")
        return True

    except Exception as e:
        print(f"Error during GPU test: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_tensorflow_gpu()
    sys.exit(0 if success else 1)
