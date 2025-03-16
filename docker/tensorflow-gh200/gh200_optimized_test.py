import tensorflow as tf
import numpy as np
import time
import os
import gc
from tensorflow.keras import mixed_precision

print(f"TensorFlow version: {tf.__version__}")

# Disable CUDA graphs which are causing errors
os.environ["TF_USE_CUDA_GRAPHS"] = "0"

# Enable mixed precision for better performance on GH200
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")

# Enable XLA JIT compilation for specific operations, not globally
# tf.config.optimizer.set_jit(True)  # This can cause issues with CUDA graphs

# Enable memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(
            f"Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")

        # Get GPU details
        for i, device in enumerate(physical_devices):
            details = tf.config.experimental.get_device_details(device)
            print(f"GPU {i} details: {details}")
    except Exception as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPUs found")
    exit(1)

print("\n" + "="*50)
print("NVIDIA GH200 OPTIMIZED TEST")
print("="*50 + "\n")

# Function to get GPU memory usage


def get_gpu_memory():
    try:
        result = os.popen(
            'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').readlines()
        return int(result[0])
    except:
        return 0

# Function to clear memory between tests


def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()


# 1. Matrix Multiplication Test
print("\n1. MATRIX MULTIPLICATION TEST")
print("-"*40)

# Test with different matrix sizes
matrix_sizes = [5000, 10000, 15000]

for size in matrix_sizes:
    print(f"\nMatrix size: {size}x{size}")

    # CPU test
    with tf.device('/CPU:0'):
        start_time = time.time()

        # Create matrices on CPU
        a = tf.random.normal([size, size], dtype=tf.float32)
        b = tf.random.normal([size, size], dtype=tf.float32)

        # Perform matrix multiplication
        c = tf.matmul(a, b)

        # Force execution and measure time
        result = c.numpy()
        cpu_time = time.time() - start_time

        print(f"CPU time: {cpu_time:.2f} seconds")

    clear_memory()

    # GPU test with explicit XLA compilation for this operation
    with tf.device('/GPU:0'):
        # Use XLA compilation for better performance
        @tf.function(jit_compile=True)
        def matmul_fn(x, y):
            return tf.matmul(x, y)

        # Record initial GPU memory
        initial_gpu_memory = get_gpu_memory()

        # Warmup run to compile XLA
        a_small = tf.random.normal([1000, 1000], dtype=tf.float16)
        b_small = tf.random.normal([1000, 1000], dtype=tf.float16)
        _ = matmul_fn(a_small, b_small)

        start_time = time.time()

        # Create matrices on GPU with float16 for better performance
        a = tf.random.normal([size, size], dtype=tf.float16)
        b = tf.random.normal([size, size], dtype=tf.float16)

        # Perform matrix multiplication
        c = matmul_fn(a, b)

        # Force execution and measure time
        result = c.numpy()
        gpu_time = time.time() - start_time

        # Record peak GPU memory
        peak_gpu_memory = get_gpu_memory()

        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time/max(0.001, gpu_time):.2f}x")
        print(f"GPU memory used: {peak_gpu_memory - initial_gpu_memory} MB")

    clear_memory()

# 2. Simple Neural Network Test with smaller dataset
print("\n\n2. SIMPLE NEURAL NETWORK TEST")
print("-"*40)

# Generate a small synthetic dataset
num_samples = 5000
feature_dim = 20
num_classes = 10

print(f"Generating synthetic dataset with {num_samples} samples...")

# Generate synthetic data
X = np.random.random((num_samples, feature_dim)).astype(np.float32)
y = np.random.randint(0, num_classes, size=(num_samples,))
y = tf.keras.utils.to_categorical(y, num_classes)

# Split into train and test
X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]

# Create optimized datasets
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Building simple neural network model...")

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(feature_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Use a lower learning rate for stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Disable XLA for training to avoid CUDA graph errors
tf.config.optimizer.set_jit(False)

# Train the model
print("Training simple neural network model...")
start_time = time.time()

# Use a callback to stop if we encounter errors


class ErrorCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if np.isnan(logs.get('loss', 0)):
            print("NaN loss detected, stopping training")
            self.model.stop_training = True


try:
    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=test_dataset,
        verbose=1,
        callbacks=[ErrorCallback()]
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(
        f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
except Exception as e:
    print(f"Error during training: {e}")

print("\n" + "="*50)
print("OPTIMIZED TEST COMPLETE")
print("="*50)
