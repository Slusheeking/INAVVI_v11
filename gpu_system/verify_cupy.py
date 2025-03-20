#!/usr/bin/env python3
"""
Verify CuPy Installation
This script verifies that CuPy can access the GPU.
"""

import cupy as cp
import numpy as np
import time


def verify_cupy():
    print("CuPy version:", cp.__version__)
    print("\nCUDA Device Info:")
    print(cp.cuda.runtime.getDeviceProperties(0))

    print("\nTesting CuPy computation...")
    # Create large arrays
    size = 5000
    start_time = time.time()

    # Create arrays and perform computation on GPU
    x_gpu = cp.random.random((size, size), dtype=cp.float32)
    y_gpu = cp.random.random((size, size), dtype=cp.float32)
    z_gpu = cp.matmul(x_gpu, y_gpu)

    # Sync to ensure computation is complete
    cp.cuda.stream.get_current_stream().synchronize()

    end_time = time.time()
    print(
        f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
    print("CuPy GPU computation test successful!")


if __name__ == "__main__":
    try:
        verify_cupy()
    except Exception as e:
        print(f"Error verifying CuPy: {str(e)}")
        exit(1)
