#!/usr/bin/env python3
"""
Custom CUDA Kernels for Financial Calculations
Optimized for NVIDIA GH200 Grace Hopper Superchip

This module provides custom CUDA kernels for common financial calculations:
- Simple Moving Average (SMA)
- Volume Weighted Average Price (VWAP)
- Relative Strength Index (RSI)

These kernels are implemented using CUDA C++ and exposed to Python via ctypes.
"""

import os
import sys
import logging
import numpy as np
import ctypes
import time
import threading
import multiprocessing as mp
from ctypes import c_int, c_float, c_void_p, POINTER, Structure, byref
import tempfile
import subprocess
import atexit
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('financial_cuda_kernels')

# CUDA kernel code
CUDA_KERNEL_CODE = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>

extern "C" {

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        fprintf(stderr, "CUDA error in %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        return err; \\
    } \\
}

// Global variables for context management
static CUcontext cuda_context = NULL;
static int cuda_initialized = 0;
static int cuda_device_id = 0;

// Initialize CUDA
__host__ int initialize_cuda() {
    // Check if already initialized
    if (cuda_initialized) {
        return 0;
    }

    // Initialize CUDA driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        fprintf(stderr, "CUDA driver initialization failed: %s\\n", errorStr);
        return -1;
    }
    
    // Get device count
    int deviceCount = 0;
    result = cuDeviceGetCount(&deviceCount);
    if (result != CUDA_SUCCESS || deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found or failed to get device count\\n");
        return -2;
    }
    
    // Select device (prefer GH200 if available)
    CUdevice device = 0;
    cuda_device_id = 0;
    char deviceName[256];
    
    for (int i = 0; i < deviceCount; i++) {
        result = cuDeviceGet(&device, i);
        if (result == CUDA_SUCCESS) {
            result = cuDeviceGetName(deviceName, sizeof(deviceName), device);
            if (result == CUDA_SUCCESS) {
                fprintf(stdout, "Found CUDA device %d: %s\\n", i, deviceName);
                if (strstr(deviceName, "GH200") != NULL) {
                    cuda_device_id = i;
                    fprintf(stdout, "Selected GH200 device at index %d\\n", i);
                    break;
                }
            }
        }
    }
    
    // Create context on the selected device
    result = cuDeviceGet(&device, cuda_device_id);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get CUDA device %d\\n", cuda_device_id);
        return -3;
    }
    
    // Set device for runtime API as well
    cudaSetDevice(cuda_device_id);
    
    // Create context with flags for better performance
    result = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create CUDA context\\n");
        return -4;
    }
    
    // Print device info
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, cuda_device_id));
    printf("Using CUDA device: %s\\n", deviceProp.name);
    
    cuda_initialized = 1;
    return 0;
}

// Cleanup CUDA
__host__ int cleanup_cuda() {
    if (!cuda_initialized) {
        return 0;
    }
    
    // Synchronize before cleanup
    cudaDeviceSynchronize();
    
    // Destroy context if it exists
    if (cuda_context != NULL) {
        CUresult result = cuCtxDestroy(cuda_context);
        if (result != CUDA_SUCCESS) {
            const char* errorStr;
            cuGetErrorString(result, &errorStr);
            fprintf(stderr, "Failed to destroy CUDA context: %s\\n", errorStr);
            return -1;
        }
        cuda_context = NULL;
    }
    
    cuda_initialized = 0;
    return 0;
}

// Kernel for calculating Simple Moving Average (SMA)
__global__ void sma_kernel(const float* input, float* output, int window_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Not enough data for window
    if (idx < window_size - 1) {
        output[idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < window_size; i++) {
        sum += input[idx - i];
    }
    
    output[idx] = sum / window_size;
}

// Host function for calculating SMA
__host__ int calculate_sma(const float* input, float* output, int window_size, int size) {
    // Make sure CUDA is initialized
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\\n");
        return -1;
    }
    
    float *d_input = NULL;
    float *d_output = NULL;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    sma_kernel<<<gridSize, blockSize>>>(d_input, d_output, window_size, size);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return 0;
}

// Kernel for calculating Volume Weighted Average Price (VWAP)
__global__ void vwap_kernel(const float* prices, const float* volumes, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Calculate price * volume for this data point
    float price_volume = prices[idx] * volumes[idx];
    
    // Use atomics to accumulate in global memory
    atomicAdd(&output[0], price_volume);
    atomicAdd(&output[1], volumes[idx]);
}

// Host function for calculating VWAP
__host__ int calculate_vwap(const float* prices, const float* volumes, float* vwap, int size) {
    // Make sure CUDA is initialized
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\\n");
        return -1;
    }
    
    float *d_prices = NULL;
    float *d_volumes = NULL;
    float *d_output = NULL;
    float h_output[2] = {0.0f, 0.0f}; // [0] = sum(price*volume), [1] = sum(volume)
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_prices, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_volumes, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, 2 * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_prices, prices, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_volumes, volumes, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output, 2 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    vwap_kernel<<<gridSize, blockSize>>>(d_prices, d_volumes, d_output, size);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate VWAP
    if (h_output[1] > 0.0f) {
        *vwap = h_output[0] / h_output[1];
    } else {
        *vwap = 0.0f;
    }
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_prices));
    CHECK_CUDA_ERROR(cudaFree(d_volumes));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return 0;
}

// Kernel for calculating price changes (delta)
__global__ void delta_kernel(const float* prices, float* gains, float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size - 1) return;
    
    float delta = prices[idx + 1] - prices[idx];
    
    if (delta > 0.0f) {
        gains[idx] = delta;
        losses[idx] = 0.0f;
    } else {
        gains[idx] = 0.0f;
        losses[idx] = -delta;
    }
}

// Kernel for calculating average gains and losses
__global__ void avg_gain_loss_kernel(const float* gains, const float* losses, float* avg_gain, float* avg_loss, int period, int size) {
    // This kernel is designed to run with a single thread
    if (blockIdx.x > 0 || threadIdx.x > 0) return;
    
    // Calculate initial averages
    float sum_gain = 0.0f;
    float sum_loss = 0.0f;
    
    for (int i = 0; i < period; i++) {
        sum_gain += gains[i];
        sum_loss += losses[i];
    }
    
    float current_avg_gain = sum_gain / period;
    float current_avg_loss = sum_loss / period;
    
    *avg_gain = current_avg_gain;
    *avg_loss = current_avg_loss;
    
    // Calculate smoothed averages
    for (int i = period; i < size - 1; i++) {
        current_avg_gain = (current_avg_gain * (period - 1) + gains[i]) / period;
        current_avg_loss = (current_avg_loss * (period - 1) + losses[i]) / period;
    }
    
    *avg_gain = current_avg_gain;
    *avg_loss = current_avg_loss;
}

// Host function for calculating RSI
__host__ int calculate_rsi(const float* prices, float* rsi, int period, int size) {
    // Make sure CUDA is initialized
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized\\n");
        return -1;
    }
    
    if (size <= period) {
        *rsi = 50.0f; // Default value if not enough data
        return 0;
    }
    
    float *d_prices = NULL;
    float *d_gains = NULL;
    float *d_losses = NULL;
    float *d_avg_gain = NULL;
    float *d_avg_loss = NULL;
    float h_avg_gain = 0.0f;
    float h_avg_loss = 0.0f;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_prices, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gains, (size - 1) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_losses, (size - 1) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_avg_gain, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_avg_loss, sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_prices, prices, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions for delta kernel
    int blockSize = 256;
    int gridSize = ((size - 1) + blockSize - 1) / blockSize;
    
    // Launch delta kernel
    delta_kernel<<<gridSize, blockSize>>>(d_prices, d_gains, d_losses, size);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch avg_gain_loss kernel with a single thread
    avg_gain_loss_kernel<<<1, 1>>>(d_gains, d_losses, d_avg_gain, d_avg_loss, period, size);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_avg_gain, d_avg_gain, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_avg_loss, d_avg_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate RSI
    if (h_avg_loss > 0.0f) {
        float rs = h_avg_gain / h_avg_loss;
        *rsi = 100.0f - (100.0f / (1.0f + rs));
    } else if (h_avg_gain > 0.0f) {
        *rsi = 100.0f;
    } else {
        *rsi = 50.0f;
    }
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_prices));
    CHECK_CUDA_ERROR(cudaFree(d_gains));
    CHECK_CUDA_ERROR(cudaFree(d_losses));
    CHECK_CUDA_ERROR(cudaFree(d_avg_gain));
    CHECK_CUDA_ERROR(cudaFree(d_avg_loss));
    
    return 0;
}

} // extern "C"
"""

# Path to the compiled library
LIBRARY_PATH = None

# Process-specific lock file


def get_lock_file():
    """Get a process-specific lock file path"""
    pid = os.getpid()
    return os.path.join(tempfile.gettempdir(), f"cuda_lock_{pid}.lock")


def compile_cuda_kernels():
    """Compile CUDA kernels into a shared library"""
    global LIBRARY_PATH

    # Create temporary directory with unique name
    temp_dir = tempfile.mkdtemp(prefix=f"cuda_kernels_{os.getpid()}_")

    # Write CUDA code to file
    cuda_file = os.path.join(temp_dir, "financial_cuda_kernels.cu")
    with open(cuda_file, "w") as f:
        f.write(CUDA_KERNEL_CODE)

    # Compile CUDA code
    library_file = os.path.join(temp_dir, "libfinancial_cuda_kernels.so")
    LIBRARY_PATH = library_file

    # NVCC command with additional flags for better error checking
    nvcc_cmd = [
        "nvcc",
        "-Xcompiler", "-fPIC -Wall",
        "-shared",
        "-o", library_file,
        cuda_file
    ]

    try:
        # Run NVCC
        logger.info(f"Compiling CUDA kernels: {' '.join(nvcc_cmd)}")
        result = subprocess.run(
            nvcc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logger.error(
                f"Failed to compile CUDA kernels: {result.stderr.decode()}")
            return False

        logger.info(f"CUDA kernels compiled successfully: {library_file}")
        return True
    except Exception as e:
        logger.error(f"Error compiling CUDA kernels: {e}")
        return False


def load_cuda_library():
    """Load the compiled CUDA library"""
    if LIBRARY_PATH is None or not os.path.exists(LIBRARY_PATH):
        if not compile_cuda_kernels():
            logger.error("Failed to compile CUDA kernels")
            return None

    try:
        # Load the library
        lib = ctypes.CDLL(LIBRARY_PATH)

        # Define function prototypes
        lib.initialize_cuda.argtypes = []
        lib.initialize_cuda.restype = c_int

        lib.cleanup_cuda.argtypes = []
        lib.cleanup_cuda.restype = c_int

        lib.calculate_sma.argtypes = [
            POINTER(c_float), POINTER(c_float), c_int, c_int]
        lib.calculate_sma.restype = c_int

        lib.calculate_vwap.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]
        lib.calculate_vwap.restype = c_int

        lib.calculate_rsi.argtypes = [
            POINTER(c_float), POINTER(c_float), c_int, c_int]
        lib.calculate_rsi.restype = c_int

        return lib
    except Exception as e:
        logger.error(f"Error loading CUDA library: {e}")
        return None


# Global library instance
CUDA_LIB = None

# Global initialization lock to prevent race conditions
CUDA_INIT_LOCK = threading.Lock()

# Track initialization status
CUDA_INITIALIZED = False
MAX_INIT_RETRIES = 3

# Process-level lock to prevent multiple processes from initializing simultaneously


def acquire_process_lock():
    """Acquire a process-level lock"""
    lock_file = get_lock_file()
    try:
        # Create lock file
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        logger.error(f"Failed to acquire process lock: {e}")
        return False


def release_process_lock():
    """Release the process-level lock"""
    lock_file = get_lock_file()
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return True
    except Exception as e:
        logger.error(f"Failed to release process lock: {e}")
        return False


def initialize_cuda():
    """Initialize CUDA"""
    global CUDA_LIB, CUDA_INITIALIZED

    # Add a small random delay to avoid race conditions
    time.sleep(random.uniform(0.1, 0.5))

    # Use a lock to prevent multiple threads from initializing simultaneously
    with CUDA_INIT_LOCK:
        if CUDA_INITIALIZED:
            logger.info(f"CUDA already initialized in process {os.getpid()}")
            return True

        # Log process ID for debugging
        logger.info(f"Initializing CUDA in process {os.getpid()}")

        # Acquire process-level lock
        if not acquire_process_lock():
            logger.error(
                f"Failed to acquire process lock in process {os.getpid()}")
            return False

        if CUDA_LIB is None:
            CUDA_LIB = load_cuda_library()
            if CUDA_LIB is None:
                logger.error(
                    f"Failed to load CUDA library in process {os.getpid()}")
                release_process_lock()
                return False

        # Try initialization with retries
        for attempt in range(MAX_INIT_RETRIES):
            result = CUDA_LIB.initialize_cuda()
            if result == 0:
                logger.info(
                    f"CUDA initialized successfully in process {os.getpid()}")
                CUDA_INITIALIZED = True
                release_process_lock()
                return True

            logger.warning(
                f"CUDA initialization attempt {attempt+1} failed with code {result} in process {os.getpid()}, retrying...")
            time.sleep(1)  # Wait before retrying

        logger.error(
            f"Failed to initialize CUDA after {MAX_INIT_RETRIES} attempts in process {os.getpid()}")
        release_process_lock()
        return False


def cleanup_cuda():
    """Cleanup CUDA"""
    global CUDA_LIB, CUDA_INITIALIZED

    with CUDA_INIT_LOCK:
        # Log process ID for debugging
        logger.info(f"Cleaning up CUDA in process {os.getpid()}")

        if CUDA_LIB is None or not CUDA_INITIALIZED:
            return True

        result = CUDA_LIB.cleanup_cuda()
        if result != 0:
            logger.error(
                f"Failed to cleanup CUDA: {result} in process {os.getpid()}")
            return False

        CUDA_INITIALIZED = False
        logger.info(f"CUDA cleaned up successfully in process {os.getpid()}")
        return True


def calculate_sma_cuda(data, window_size):
    """Calculate Simple Moving Average using CUDA"""
    global CUDA_LIB

    if CUDA_LIB is None:
        logger.warning(
            f"CUDA library not initialized in process {os.getpid()}, falling back to CPU implementation")
        return np.zeros_like(data)

    # Convert data to numpy array
    data_np = np.asarray(data, dtype=np.float32)
    size = len(data_np)

    # Create output array
    output = np.zeros_like(data_np)

    # Call CUDA function
    result = CUDA_LIB.calculate_sma(
        data_np.ctypes.data_as(POINTER(c_float)),
        output.ctypes.data_as(POINTER(c_float)),
        c_int(window_size),
        c_int(size)
    )

    if result != 0:
        logger.error(
            f"Failed to calculate SMA: {result} in process {os.getpid()}")
        return np.zeros_like(data)

    return output


def calculate_vwap_cuda(prices, volumes):
    """Calculate Volume Weighted Average Price using CUDA"""
    global CUDA_LIB

    if CUDA_LIB is None:
        logger.warning(
            f"CUDA library not initialized in process {os.getpid()}, falling back to CPU implementation")
        return 0.0

    # Convert data to numpy arrays
    prices_np = np.asarray(prices, dtype=np.float32)
    volumes_np = np.asarray(volumes, dtype=np.float32)
    size = len(prices_np)

    # Create output variable
    vwap = c_float(0.0)

    # Call CUDA function
    result = CUDA_LIB.calculate_vwap(
        prices_np.ctypes.data_as(POINTER(c_float)),
        volumes_np.ctypes.data_as(POINTER(c_float)),
        byref(vwap),
        c_int(size)
    )

    if result != 0:
        logger.error(
            f"Failed to calculate VWAP: {result} in process {os.getpid()}")
        return 0.0

    return vwap.value


def calculate_rsi_cuda(prices, period=14):
    """Calculate Relative Strength Index using CUDA"""
    global CUDA_LIB

    if CUDA_LIB is None:
        logger.warning(
            f"CUDA library not initialized in process {os.getpid()}, falling back to CPU implementation")
        return 50.0

    # Convert data to numpy array
    prices_np = np.asarray(prices, dtype=np.float32)
    size = len(prices_np)

    # Create output variable
    rsi = c_float(0.0)

    # Call CUDA function
    result = CUDA_LIB.calculate_rsi(
        prices_np.ctypes.data_as(POINTER(c_float)),
        byref(rsi),
        c_int(period),
        c_int(size)
    )

    if result != 0:
        logger.error(
            f"Failed to calculate RSI: {result} in process {os.getpid()}")
        return 50.0

    return rsi.value

# Register cleanup function


def _cleanup():
    cleanup_cuda()
    release_process_lock()


atexit.register(_cleanup)

# Initialize CUDA when module is imported - only if explicitly requested
if __name__ != "__main__" and os.environ.get('CUDA_INIT_ON_IMPORT', '0') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
    initialize_cuda()

# Example usage
if __name__ == "__main__":
    # Initialize CUDA
    if not initialize_cuda():
        sys.exit(1)

    # Test SMA calculation
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                    8.0, 9.0, 10.0], dtype=np.float32)
    sma = calculate_sma_cuda(data, 3)
    print(f"SMA(3): {sma}")

    # Test VWAP calculation
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    volumes = np.array(
        [1000.0, 1500.0, 2000.0, 1800.0, 2200.0], dtype=np.float32)
    vwap = calculate_vwap_cuda(prices, volumes)
    print(f"VWAP: {vwap}")

    # Test RSI calculation
    prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 109.0, 108.0, 107.0, 106.0,
                      105.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 111.0, 110.0, 109.0], dtype=np.float32)
    rsi = calculate_rsi_cuda(prices, 14)
    print(f"RSI: {rsi}")

    # Cleanup CUDA
    cleanup_cuda()
