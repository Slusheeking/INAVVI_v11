#!/usr/bin/env python3
"""
Verify GPU Framework Integration
This script verifies that TensorFlow, CuPy, and TensorRT are properly configured and working together.
"""

import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def verify_frameworks():
    """Verify all GPU frameworks are working properly"""
    print("Starting comprehensive GPU framework verification...")

    # Test TensorFlow
    print("\n=== TensorFlow Verification ===")
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")

    if len(gpus) > 0:
        print("\nTesting TensorFlow GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start_time = time.time()
            c = tf.matmul(a, b)
            tf_time = time.time() - start_time
            print(
                f"TensorFlow matrix multiplication time: {tf_time*1000:.2f}ms")

    # Test CuPy
    print("\n=== CuPy Verification ===")
    try:
        import cupy as cp
        print("CuPy version:", cp.__version__)

        # Get device info
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"Using GPU: {props['name'].decode()}")

        # Test computation
        print("\nTesting CuPy GPU computation...")
        a_cp = cp.random.normal(0, 1, (1000, 1000))
        b_cp = cp.random.normal(0, 1, (1000, 1000))
        start_time = time.time()
        c_cp = cp.matmul(a_cp, b_cp)
        cp.cuda.Stream.null.synchronize()
        cupy_time = time.time() - start_time
        print(f"CuPy matrix multiplication time: {cupy_time*1000:.2f}ms")
    except ImportError:
        print("CuPy not available")

    # Test TensorRT
    print("\n=== TensorRT Verification ===")
    try:
        # Create a more complex model that better represents real usage
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Convert to TensorRT
        print("Testing TensorRT model optimization...")
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            max_workspace_size_bytes=8000000000,
            precision_mode="FP16",
            maximum_cached_engines=100
        )

        # Test inference with larger batch size
        batch_size = 32
        input_data = tf.random.normal([batch_size, 224, 224, 3])

        # Warmup
        _ = model(input_data)

        # Test original model with multiple batches
        start_time = time.time()
        num_batches = 50
        for _ in range(num_batches):
            _ = model(input_data)
        original_time = time.time() - start_time
        print(
            f"Original model inference time (total): {original_time*1000:.2f}ms")
        print(
            f"Original model inference time (per batch): {(original_time/num_batches)*1000:.2f}ms")

        # Convert model to TensorRT
        temp_saved_model = "/tmp/model"
        tf.saved_model.save(model, temp_saved_model)

        # Convert to TensorRT
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=temp_saved_model,
            conversion_params=conversion_params
        )
        converter.convert()

        # Save converted model
        converter.save(temp_saved_model + "_trt")

        # Load and test optimized model
        trt_model = tf.saved_model.load(temp_saved_model + "_trt")
        trt_infer = trt_model.signatures['serving_default']

        # Prepare input tensor
        input_tensor = tf.constant(input_data.numpy())

        # Warmup TensorRT engine
        _ = trt_infer(input_1=input_tensor)

        # Test optimized model with multiple batches
        start_time = time.time()
        for _ in range(num_batches):
            _ = trt_infer(input_1=input_tensor)
        optimized_time = time.time() - start_time
        print(
            f"TensorRT optimized model inference time (total): {optimized_time*1000:.2f}ms")
        print(
            f"TensorRT optimized model inference time (per batch): {(optimized_time/num_batches)*1000:.2f}ms")
        print(
            f"Speed improvement: {(original_time/optimized_time - 1)*100:.1f}%")

        print("TensorRT optimization successful!")
    except Exception as e:
        print(f"Error testing TensorRT: {str(e)}")


if __name__ == "__main__":
    try:
        verify_frameworks()
        sys.exit(0)
    except Exception as e:
        print(f"Error verifying TensorFlow: {str(e)}")
        sys.exit(1)
