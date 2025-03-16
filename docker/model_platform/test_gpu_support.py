#!/usr/bin/env python3
"""
GPU Support Test Script for NVIDIA GH200 Grace Hopper Superchip
Tests both TensorFlow and PyTorch GPU support and provides detailed information.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime


def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error message: {e.stderr}")
        return None


def test_system_info():
    """Print system information."""
    print_section("SYSTEM INFORMATION")

    # Print date and time
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print OS information
    print("\nOS Information:")
    run_command("uname -a")

    # Print CPU information
    print("\nCPU Information:")
    print(run_command("lscpu | grep 'Model name\\|Architecture\\|CPU(s)\\|Thread'"))

    # Print memory information
    print("\nMemory Information:")
    print(run_command("free -h"))

    # Print environment variables
    print("\nRelevant Environment Variables:")
    for var in ["CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
                "TF_GPU_ALLOCATOR", "TF_XLA_FLAGS", "XLA_FLAGS",
                "PYTORCH_CUDA_ALLOC_CONF"]:
        print(f"{var}={os.environ.get(var, 'Not set')}")


def test_nvidia_smi():
    """Run nvidia-smi and print the output."""
    print_section("NVIDIA-SMI OUTPUT")

    # Check if nvidia-smi is available
    if run_command("which nvidia-smi"):
        print(run_command("nvidia-smi"))
        print("\nNVIDIA Driver Version:")
        print(run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader"))
        print("\nCUDA Version:")
        print(run_command("nvidia-smi --query-gpu=cuda_version --format=csv,noheader"))
    else:
        print("nvidia-smi not found. NVIDIA driver may not be installed.")


def test_tensorflow():
    """Test TensorFlow GPU support."""
    print_section("TENSORFLOW GPU SUPPORT")

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")

        # Check if TensorFlow can see GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs detected by TensorFlow: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")

        # Check if TensorFlow was built with CUDA support
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")

        # If GPUs are available, run a simple test
        if gpus:
            print("\nRunning TensorFlow GPU test...")

            # Configure memory growth
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu}")
                except:
                    print(f"Failed to set memory growth for {gpu}")

            # Create and run a simple model on GPU
            with tf.device('/GPU:0'):
                start_time = time.time()

                # Create large tensors to force GPU usage
                a = tf.random.normal([5000, 5000])
                b = tf.random.normal([5000, 5000])

                # Perform matrix multiplication
                c = tf.matmul(a, b)

                # Force execution
                _ = c[0, 0].numpy()

                end_time = time.time()

                print(
                    f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
        else:
            print("No GPUs available for TensorFlow test.")

    except ImportError:
        print("TensorFlow not installed.")
    except Exception as e:
        print(f"Error testing TensorFlow: {e}")


def test_pytorch():
    """Test PyTorch GPU support."""
    print_section("PYTORCH GPU SUPPORT")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        # Check if CUDA is available
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")

            # Print information about each device
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Capability: {torch.cuda.get_device_capability(i)}")
                print(
                    f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

            # Run a simple test
            print("\nRunning PyTorch GPU test...")

            start_time = time.time()

            # Create large tensors on GPU
            a = torch.randn(5000, 5000, device='cuda')
            b = torch.randn(5000, 5000, device='cuda')

            # Perform matrix multiplication
            c = torch.matmul(a, b)

            # Force execution
            _ = c[0, 0].item()

            end_time = time.time()

            print(
                f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
        else:
            print("CUDA not available for PyTorch test.")

    except ImportError:
        print("PyTorch not installed.")
    except Exception as e:
        print(f"Error testing PyTorch: {e}")


def main():
    """Run all tests."""
    print_section("GPU SUPPORT TEST SCRIPT")
    print("Testing GPU support for NVIDIA GH200 Grace Hopper Superchip")
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_system_info()
    test_nvidia_smi()
    test_tensorflow()
    test_pytorch()

    print_section("TEST COMPLETE")
    print(
        f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
