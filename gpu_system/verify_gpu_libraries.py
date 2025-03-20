#!/usr/bin/env python3
"""
Verification script for TensorFlow, TensorRT, and CuPy with GPU support.
"""

import os
import sys
import subprocess
import platform


def print_section(title):
    """Print a section title with decorative formatting."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def run_command(cmd):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e.stderr}")
        return None


def check_tensorflow():
    """Check TensorFlow installation and GPU availability."""
    print_section("TensorFlow Verification")

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow can access {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")

            # Get GPU device details
            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"\nGPU {i} details:")
                for key, value in gpu_details.items():
                    print(f"  {key}: {value}")

            # Test a simple operation on GPU
            print("\nRunning a simple TensorFlow operation on GPU...")
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                y = tf.random.normal([1000, 1000])
                z = tf.matmul(x, y)
                result = z.numpy().mean()
            print(f"Matrix multiplication result (mean value): {result}")
            print("\nTensorFlow is working correctly with GPU support!")
        else:
            print("No GPU detected by TensorFlow!")
            print("TensorFlow is installed but running on CPU only.")
    except ImportError:
        print("TensorFlow is not installed!")
    except Exception as e:
        print(f"Error checking TensorFlow: {str(e)}")


def check_tensorrt():
    """Check TensorRT installation."""
    print_section("TensorRT Verification")

    try:
        # First check if TensorRT is available through TensorFlow
        import tensorflow as tf

        print("Checking TensorRT availability through TensorFlow...")
        trt_available = hasattr(tf.experimental, 'tensorrt')

        if trt_available:
            print("TensorRT is available through TensorFlow experimental API.")
            if hasattr(tf.experimental.tensorrt, '__version__'):
                print(
                    f"TensorRT version: {tf.experimental.tensorrt.__version__}")
        else:
            print("TensorRT is not available through TensorFlow experimental API.")

        # Try to import tensorrt directly
        try:
            import tensorrt as trt
            print(f"\nTensorRT standalone version: {trt.__version__}")
            print("TensorRT is installed correctly!")
        except ImportError:
            print("\nCould not import TensorRT directly.")

            # Check if the TensorRT libraries are installed
            tensorrt_libs = run_command(
                "find /usr -name 'libnvinfer.so*' 2>/dev/null")
            if tensorrt_libs:
                print("TensorRT libraries found on the system:")
                print(tensorrt_libs)
            else:
                print("No TensorRT libraries found on the system.")
    except Exception as e:
        print(f"Error checking TensorRT: {str(e)}")


def check_cupy():
    """Check CuPy installation and GPU functionality."""
    print_section("CuPy Verification")

    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")

        # Get CUDA version used by CuPy
        print(
            f"CUDA version used by CuPy: {cp.cuda.runtime.runtimeGetVersion()}")

        # Get device count and properties
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Number of GPU devices: {device_count}")

        if device_count > 0:
            for i in range(device_count):
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"\nGPU {i} properties:")
                print(f"  Name: {device_props['name'].decode('utf-8')}")
                print(
                    f"  Compute Capability: {device_props['major']}.{device_props['minor']}")
                print(
                    f"  Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")
                print(
                    f"  Multi Processors: {device_props['multiProcessorCount']}")

            # Test a simple operation on GPU
            print("\nRunning a simple CuPy operation on GPU...")
            x_gpu = cp.random.normal(size=(1000, 1000))
            y_gpu = cp.random.normal(size=(1000, 1000))
            z_gpu = cp.matmul(x_gpu, y_gpu)
            result = float(z_gpu.mean())
            print(f"Matrix multiplication result (mean value): {result}")
            print("\nCuPy is working correctly with GPU support!")
        else:
            print("No GPU detected by CuPy!")
    except ImportError:
        print("CuPy is not installed!")
    except Exception as e:
        print(f"Error checking CuPy: {str(e)}")


def check_nvidia_smi():
    """Check NVIDIA GPU status using nvidia-smi."""
    print_section("NVIDIA System Management Interface")

    nvidia_smi = run_command("nvidia-smi")
    if nvidia_smi:
        print(nvidia_smi)
    else:
        print("nvidia-smi command failed or not available.")


def main():
    """Main function to run all checks."""
    print_section("System Information")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    # Check NVIDIA driver and CUDA
    print("\nCUDA Environment:")
    for env_var in ['CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']:
        print(f"  {env_var}: {os.environ.get(env_var, 'Not set')}")

    # Run all checks
    check_nvidia_smi()
    check_tensorflow()
    check_tensorrt()
    check_cupy()

    print_section("Verification Complete")


if __name__ == "__main__":
    main()
