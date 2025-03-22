#!/usr/bin/env python3
"""
TensorFlow GPU Test Script

This script runs a more comprehensive test of TensorFlow with GPU support.
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


def test_basic_operations():
    """Test basic TensorFlow operations on GPU"""
    print("Testing basic TensorFlow operations on GPU...")

    # Create some tensors
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)

        # Print the result
        print(f"Matrix multiplication result:\n{c.numpy()}")

    return True


def test_neural_network():
    """Test a simple neural network on GPU"""
    print("\nTesting simple neural network on GPU...")

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Take a small subset for quick testing
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:100]
    y_test = y_test[:100]

    # Build a simple model
    with tf.device('/GPU:0'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        print("Training model...")
        start_time = time.time()
        model.fit(x_train, y_train, epochs=2, verbose=1)
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")

        # Evaluate the model
        print("\nEvaluating model...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test accuracy: {test_acc:.4f}")

    return True


def test_gpu_performance():
    """Test GPU performance with a large matrix multiplication"""
    print("\nTesting GPU performance with large matrix multiplication...")

    # Create large matrices
    with tf.device('/GPU:0'):
        # Warm up
        a_small = tf.random.normal([1000, 1000])
        b_small = tf.random.normal([1000, 1000])
        c_small = tf.matmul(a_small, b_small)

        # Large matrix multiplication
        matrix_size = 10000
        print(f"Matrix size: {matrix_size}x{matrix_size}")

        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])

        # Time the operation
        start_time = time.time()
        c = tf.matmul(a, b)
        # Force execution to complete
        _ = c.numpy()
        elapsed = time.time() - start_time

        print(f"Matrix multiplication time: {elapsed:.4f} seconds")
        print(
            f"Performance: {(2 * matrix_size**3) / (elapsed * 1e9):.2f} TFLOPS")

    return True


def main():
    print_section("TensorFlow GPU Test")

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

        # Run tests
        try:
            test_basic_operations()
            test_neural_network()
            test_gpu_performance()

            print("\nAll GPU tests PASSED ✓")
            return 0
        except Exception as e:
            print(f"\nError during GPU tests: {e}")
            return 1
    else:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        print("\nGPU tests SKIPPED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
