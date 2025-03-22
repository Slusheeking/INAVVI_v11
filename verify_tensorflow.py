#!/usr/bin/env python3
"""
TensorFlow GPU Verification Script

This script verifies that TensorFlow is properly installed and configured with GPU support.
"""

import os
import sys
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


def main():
    print_section("TensorFlow GPU Verification")

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
        except:
            print("Could not retrieve detailed GPU information")

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
        return 0
    else:
        print("\nNo GPU devices found. TensorFlow will run on CPU only.")
        print("\nRunning simple CPU computation test...")
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        start_time = datetime.now()
        c = tf.matmul(a, b)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  Matrix multiplication shape: {c.shape}")
        print(f"  Computation time: {elapsed:.4f} seconds")

        print("\nCPU TEST PASSED ✓")
        return 1


if __name__ == "__main__":
    sys.exit(main())
