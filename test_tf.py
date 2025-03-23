import tensorflow as tf
import time

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

if len(tf.config.list_physical_devices('GPU')) > 0:
    # Get GPU device properties
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    print(f"GPU device: {gpu_device}")

    # Run a matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark...")
    size = 5000
    print(f"Matrix size: {size}x{size}")

    # Create random matrices
    a = tf.random.normal((size, size))
    b = tf.random.normal((size, size))

    # Warm-up run
    c = tf.matmul(a, b)

    # Benchmark
    start = time.time()
    c = tf.matmul(a, b)
    # Force execution to complete
    _ = c.numpy()
    elapsed = time.time() - start

    print(f"Matrix multiplication completed in {elapsed:.4f} seconds")

    # Calculate GFLOPS
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    gflops = flops / elapsed / 1e9
    print(f"Performance: {gflops:.2f} GFLOPS")

    # Test XLA compilation
    print("\nTesting XLA compilation...")

    @tf.function(jit_compile=True)
    def matmul_fn(x, y):
        return tf.matmul(x, y)

    # Warm-up
    c_xla = matmul_fn(a, b)

    # Benchmark with XLA
    start = time.time()
    c_xla = matmul_fn(a, b)
    # Force execution to complete
    _ = c_xla.numpy()
    elapsed_xla = time.time() - start

    print(f"XLA matrix multiplication completed in {elapsed_xla:.4f} seconds")
    gflops_xla = flops / elapsed_xla / 1e9
    print(f"XLA Performance: {gflops_xla:.2f} GFLOPS")
    print(f"Speedup from XLA: {elapsed/elapsed_xla:.2f}x")
else:
    print("No GPU available for TensorFlow")
