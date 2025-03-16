import tensorflow as tf
import os
import time
import numpy as np

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check for GPU availability
print(
    f"GPU is{' not' if not tf.config.list_physical_devices('GPU') else ''} available")

# Print GPU devices
# Enable memory growth to avoid allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(f"Found GPU: {gpu}")

# Print GPU details
if gpus:
    # Get GPU memory info
    try:
        for gpu in gpus:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"GPU Details: {gpu_details}")
    except:
        print("Could not get detailed GPU information")

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")
# Run a simple benchmark
if gpus:
    print("\nRunning simple matrix multiplication benchmark...")

    # Create two large matrices
    with tf.device('/GPU:0'):
        # Matrix multiplication benchmark
        start_time = time.time()

        # Create large matrices
        matrix_size = 10000
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])

        # Perform matrix multiplication
        # Use XLA compilation for better performance
        @tf.function(jit_compile=True)
        def matmul_fn(x, y):
            return tf.matmul(x, y)

        c = matmul_fn(a, b)

        # Force execution and measure time
        result = c.numpy()
        end_time = time.time()

        print(
            f"Matrix multiplication of {matrix_size}x{matrix_size} with XLA took {end_time - start_time:.2f} seconds")

        # Test with pinned memory for faster CPU-GPU transfers
        print("\nTesting pinned memory transfers...")
        start_time = time.time()

        # Create data on CPU with pinned memory
        with tf.device('/CPU:0'):
            cpu_data = tf.constant(np.random.random(
                (5000, 5000)).astype(np.float32))

        # Transfer to GPU
        with tf.device('/GPU:0'):
            gpu_data = tf.identity(cpu_data)
            result = tf.reduce_sum(gpu_data).numpy()  # Force execution

        print(
            f"Pinned memory transfer took {time.time() - start_time:.2f} seconds")
else:
    print("Skipping benchmark as no GPU is available")

print("\nTensorFlow GPU test completed")
