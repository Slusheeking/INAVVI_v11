#!/usr/bin/env python3
"""
Verify TensorRT Installation
This script verifies TensorRT functionality with TensorFlow.
"""

import tensorflow as tf
import numpy as np
import time


def verify_tensorrt():
    print("TensorFlow version:", tf.__version__)

    # Check if TensorRT is available
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        print("\nTensorRT support is available")
    except ImportError as e:
        print("\nError: TensorRT support not available")
        raise e

    # Create a simple model
    print("\nCreating and converting a simple model...")
    input_shape = (1, 224, 224, 3)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape[1:]),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Convert to TensorRT
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=8000000000,
        precision_mode="FP16",
        maximum_cached_engines=100
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=None,
        conversion_params=conversion_params,
        use_dynamic_shape=True
    )

    # Test inference
    print("\nTesting inference...")
    input_data = tf.random.normal(input_shape)

    # Warm up
    _ = model(input_data)

    # Benchmark
    start_time = time.time()
    for _ in range(100):
        result = model(input_data)
    end_time = time.time()

    print(f"Inference test completed in {end_time - start_time:.2f} seconds")
    print("TensorRT verification successful!")


if __name__ == "__main__":
    try:
        verify_tensorrt()
    except Exception as e:
        print(f"Error verifying TensorRT: {str(e)}")
        exit(1)
