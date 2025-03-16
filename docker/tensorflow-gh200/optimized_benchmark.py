import tensorflow as tf
import numpy as np
import time
import os
import psutil
import gc
from tensorflow.keras import mixed_precision

# Enable mixed precision for better performance on GH200
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation globally

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
print("NVIDIA GH200 OPTIMIZED BENCHMARK")
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

# Function to create an optimized dataset


def create_optimized_dataset(data, labels, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# 1. LARGE MATRIX MULTIPLICATION TEST
print("\n1. LARGE MATRIX MULTIPLICATION TEST")
print("-"*40)

# Test with different matrix sizes
matrix_sizes = [15000, 20000, 25000]

for size in matrix_sizes:
    print(f"\nMatrix size: {size}x{size}")

    # CPU test
    with tf.device('/CPU:0'):
        start_time = time.time()

        # Create matrices on CPU
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])

        # Perform matrix multiplication
        c = tf.matmul(a, b)

        # Force execution and measure time
        result = c.numpy()
        cpu_time = time.time() - start_time

        print(f"CPU time: {cpu_time:.2f} seconds")

    clear_memory()

    # GPU test
    with tf.device('/GPU:0'):
        # Use XLA compilation for better performance
        @tf.function(jit_compile=True)
        def matmul_fn(x, y):
            return tf.matmul(x, y)

        # Record initial GPU memory
        initial_gpu_memory = get_gpu_memory()
        start_time = time.time()

        # Create matrices on GPU
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])

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

# 2. CONVOLUTIONAL NEURAL NETWORK TRAINING
print("\n\n2. CONVOLUTIONAL NEURAL NETWORK TRAINING")
print("-"*40)

# Generate synthetic image data (100,000 32x32 RGB images)
num_samples = 100000
img_size = 32
num_classes = 10

print(f"Generating synthetic dataset with {num_samples} images...")

# Generate synthetic data
X = np.random.random((num_samples, img_size, img_size, 3)).astype(np.float32)
y = np.random.randint(0, num_classes, size=(num_samples,))
y = tf.keras.utils.to_categorical(y, num_classes)

# Split into train and test
X_train, X_test = X[:90000], X[90000:]
y_train, y_test = y[:90000], y[90000:]

# Create optimized datasets
train_dataset = create_optimized_dataset(X_train, y_train)
test_dataset = create_optimized_dataset(X_test, y_test)

print("Building CNN model...")

# Build a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Record initial GPU memory
initial_gpu_memory = get_gpu_memory()

# Train the model
print("Training CNN model...")
start_time = time.time()

# Use the optimized dataset
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset,
    verbose=1
)

training_time = time.time() - start_time

# Record peak GPU memory
peak_gpu_memory = get_gpu_memory()

print(f"\nCNN Training completed in {training_time:.2f} seconds")
print(f"GPU memory used: {peak_gpu_memory - initial_gpu_memory} MB")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# 3. TRANSFORMER MODEL TEST
print("\n\n3. TRANSFORMER MODEL TEST")
print("-"*40)

# Generate synthetic sequence data
seq_length = 128
vocab_size = 10000
num_samples = 50000

print(f"Generating synthetic sequence dataset with {num_samples} samples...")

# Generate synthetic data
X = np.random.randint(1, vocab_size, size=(num_samples, seq_length))
y = np.random.randint(0, 2, size=(num_samples, 1))  # Binary classification

# Split into train and test
X_train, X_test = X[:45000], X[45000:]
y_train, y_test = y[:45000], y[45000:]

# Create optimized datasets for transformer
train_dataset = create_optimized_dataset(X_train, y_train, batch_size=128)
test_dataset = create_optimized_dataset(X_test, y_test, batch_size=128)

print("Building Transformer model...")

# Build a Transformer model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.Conv1D(
        filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


# Build the model
inputs = tf.keras.layers.Input(shape=(seq_length,))
embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=256)(inputs)
embedding_layer = tf.keras.layers.Dropout(0.1)(embedding_layer)
embedding_layer = tf.keras.layers.LayerNormalization(
    epsilon=1e-6)(embedding_layer)

x = embedding_layer
for _ in range(4):
    x = transformer_encoder(x, 64, 4, 512, 0.1)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

transformer_model = tf.keras.Model(inputs=inputs, outputs=outputs)

transformer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Record initial GPU memory
initial_gpu_memory = get_gpu_memory()

# Train the model
print("Training Transformer model...")
start_time = time.time()

# Use the optimized dataset
transformer_history = transformer_model.fit(
    train_dataset,
    epochs=3,
    validation_data=test_dataset,
    verbose=1
)

transformer_time = time.time() - start_time

# Record peak GPU memory
peak_gpu_memory = get_gpu_memory()

print(f"\nTransformer Training completed in {transformer_time:.2f} seconds")
print(f"GPU memory used: {peak_gpu_memory - initial_gpu_memory} MB")
print(
    f"Final validation accuracy: {transformer_history.history['val_accuracy'][-1]:.4f}")

# 4. MEMORY BANDWIDTH TEST
print("\n\n4. MEMORY BANDWIDTH TEST")
print("-"*40)

# Test memory bandwidth by copying large tensors
sizes = [1000, 5000, 10000, 20000]

for size in sizes:
    tensor_size_gb = (size * size * 4) / \
        (1024 * 1024 * 1024)  # Size in GB (float32)
    print(f"\nTensor size: {size}x{size} ({tensor_size_gb:.2f} GB)")
    tf.keras.backend.clear_session()  # Clear session to free memory

    # GPU test
    with tf.device('/GPU:0'):
        # Create a large tensor
        start_time = time.time()
        a = tf.random.normal([size, size])
        b = tf.identity(a)  # Copy the tensor
        b = tf.identity(b)  # Copy again to ensure operation completes
        tf.debugging.assert_equal(a, b)  # Force execution
        end_time = time.time()

        bandwidth = tensor_size_gb / (end_time - start_time)
        print(f"Memory bandwidth: {bandwidth:.2f} GB/s")

    clear_memory()

# 5. OPTIMIZED DATA PIPELINE TEST
print("\n\n5. OPTIMIZED DATA PIPELINE TEST")
print("-"*40)

# Test data pipeline performance
print("Testing optimized data pipeline performance...")

# Generate a large dataset
num_samples = 1000000
feature_dim = 50

print(f"Generating dataset with {num_samples} samples...")
data = np.random.random((num_samples, feature_dim)).astype(np.float32)
labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

# Create an optimized dataset
start_time = time.time()
optimized_dataset = create_optimized_dataset(data, labels, batch_size=1024)
print(
    f"Time to process {num_samples} samples: {time.time() - start_time:.2f} seconds")

print("\n" + "="*50)
print("OPTIMIZED BENCHMARK COMPLETE")
print("="*50)
