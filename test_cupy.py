import cupy as cp
import numpy as np
import time

print(f'CuPy version: {cp.__version__}')
print(f'CUDA available: {cp.cuda.is_available()}')
print(f'GPU count: {cp.cuda.runtime.getDeviceCount()}')

if cp.cuda.is_available():
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f'GPU Name: {props["name"]}')
    print(f'Compute Capability: {props["major"]}.{props["minor"]}')
    print(f'Total Memory: {props["totalGlobalMem"] / (1024**3):.1f} GB')
    print(f'CUDA Version: {cp.cuda.runtime.driverGetVersion()}')

    # Run a matrix multiplication benchmark
    print("\nRunning matrix multiplication benchmark...")
    size = 5000
    print(f"Matrix size: {size}x{size}")

    # Create random matrices on GPU
    a_gpu = cp.random.random((size, size), dtype=cp.float32)
    b_gpu = cp.random.random((size, size), dtype=cp.float32)

    # Warm-up
    c_gpu = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.time()
    c_gpu = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start

    print(f"GPU matrix multiplication completed in {gpu_time:.4f} seconds")

    # Calculate GFLOPS
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    gflops = flops / gpu_time / 1e9
    print(f"Performance: {gflops:.2f} GFLOPS")
else:
    print("CUDA is not available. Cannot run GPU tests.")
