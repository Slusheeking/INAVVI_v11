#!/usr/bin/env python3
"""
TensorFlow GPU Test Script

This script performs a more comprehensive test of TensorFlow GPU capabilities
by running a simple neural network training task on the GPU.
"""

import os
import sys
import time
import tensorflow as tf
import numpy as np
from datetime import datetime


def test_tensorflow_gpu():
    """Test TensorFlow GPU capabilities with a simple neural network."""
    print("=" * 80)
    print("TensorFlow GPU Comprehensive Test")
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

    # Enable memory growth to avoid allocating all GPU memory at once
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\nEnabled memory growth for all GPUs")
    except Exception as e:
        print(f"\nError setting memory growth: {e}")

    # Create a simple dataset
    print("\nCreating synthetic dataset...")
    num_samples = 100000
    num_features = 1000

    # Generate random data
    x_train = np.random.random((num_samples, num_features)).astype(np.float32)
    y_train = np.random.random((num_samples, 1)).astype(np.float32)

    # Create a simple model
    print("\nCreating neural network model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu',
                              input_shape=(num_features,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Print model summary
    model.summary()

    # Train the model
    print("\nTraining model on GPU...")
    start_time = datetime.now()

    try:
        with tf.device('/GPU:0'):
            history = model.fit(
                x_train, y_train,
                epochs=5,
                batch_size=1024,
                validation_split=0.2,
                verbose=1
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"\nTraining completed in {duration:.2f} seconds")
            print(f"Final loss: {history.history['loss'][-1]:.6f}")
            print(
                f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

            # Test inference speed
            print("\nTesting inference speed...")
            inference_start = time.time()
            predictions = model.predict(x_train[:1000], verbose=0)
            inference_time = time.time() - inference_start

            print(
                f"Inference time for 1000 samples: {inference_time:.4f} seconds")
            print(
                f"Average inference time per sample: {inference_time/1000*1000:.2f} ms")

            print("\nGPU test successful!")
            return True
    except Exception as e:
        print(f"\nError during GPU test: {e}")
        return False


if __name__ == "__main__":
    success = test_tensorflow_gpu()
    sys.exit(0 if success else 1)
