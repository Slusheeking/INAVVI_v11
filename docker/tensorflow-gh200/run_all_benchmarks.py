import tensorflow as tf
import numpy as np
import time
import os
import json
import argparse
from financial_cuda_ops import test_financial_cuda_ops
from optimized_trading_models import benchmark_models
from market_data_pipeline import benchmark_pipeline
from polygon_data_source import benchmark_polygon_data, process_polygon_data_with_cuda

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)

# Set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(
        f"Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")

    # Print GPU details
    for i, device in enumerate(physical_devices):
        details = tf.config.experimental.get_device_details(device)
        print(f"GPU {i} details: {details}")
else:
    print("No GPUs found")
    exit(1)


def run_matrix_multiplication_benchmark(sizes=[5000, 10000, 15000], num_runs=3):
    """
    Run matrix multiplication benchmark

    Args:
        sizes: List of matrix sizes
        num_runs: Number of runs for each size

    Returns:
        results: Benchmark results
    """
    print("\n" + "="*50)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("="*50)

    results = {}

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")

        cpu_times = []
        gpu_times = []
        gpu_xla_times = []

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

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
                cpu_times.append(cpu_time)

            # Clear memory
            tf.keras.backend.clear_session()

            # GPU test (without XLA)
            with tf.device('/GPU:0'):
                # Temporarily disable XLA
                old_jit = tf.config.optimizer.get_jit()
                tf.config.optimizer.set_jit(False)

                start_time = time.time()

                # Create matrices on GPU
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                # Perform matrix multiplication
                c = tf.matmul(a, b)

                # Force execution and measure time
                result = c.numpy()
                gpu_time = time.time() - start_time

                # Restore XLA setting
                tf.config.optimizer.set_jit(old_jit)

                print(f"GPU time (without XLA): {gpu_time:.2f} seconds")
                gpu_times.append(gpu_time)

            # Clear memory
            tf.keras.backend.clear_session()

            # GPU test (with XLA)
            with tf.device('/GPU:0'):
                start_time = time.time()

                # Create matrices on GPU
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                # Perform matrix multiplication with XLA
                @tf.function(jit_compile=True)
                def matmul_fn(x, y):
                    return tf.matmul(x, y)

                c = matmul_fn(a, b)

                # Force execution and measure time
                result = c.numpy()
                gpu_xla_time = time.time() - start_time

                print(f"GPU time (with XLA): {gpu_xla_time:.2f} seconds")
                gpu_xla_times.append(gpu_xla_time)

            # Clear memory
            tf.keras.backend.clear_session()

        # Calculate average times
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        avg_gpu_xla_time = sum(gpu_xla_times) / len(gpu_xla_times)

        # Calculate speedups
        cpu_gpu_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        cpu_gpu_xla_speedup = avg_cpu_time / \
            avg_gpu_xla_time if avg_gpu_xla_time > 0 else 0
        gpu_xla_speedup = avg_gpu_time / avg_gpu_xla_time if avg_gpu_xla_time > 0 else 0

        print(f"\nAverage CPU time: {avg_cpu_time:.2f} seconds")
        print(f"Average GPU time (without XLA): {avg_gpu_time:.2f} seconds")
        print(f"Average GPU time (with XLA): {avg_gpu_xla_time:.2f} seconds")
        print(f"CPU to GPU speedup: {cpu_gpu_speedup:.2f}x")
        print(f"CPU to GPU+XLA speedup: {cpu_gpu_xla_speedup:.2f}x")
        print(f"GPU to GPU+XLA speedup: {gpu_xla_speedup:.2f}x")

        results[size] = {
            'avg_cpu_time': avg_cpu_time,
            'avg_gpu_time': avg_gpu_time,
            'avg_gpu_xla_time': avg_gpu_xla_time,
            'cpu_gpu_speedup': cpu_gpu_speedup,
            'cpu_gpu_xla_speedup': cpu_gpu_xla_speedup,
            'gpu_xla_speedup': gpu_xla_speedup
        }

    return results


def run_mixed_precision_benchmark(sizes=[5000, 10000, 15000], num_runs=3):
    """
    Run mixed precision benchmark

    Args:
        sizes: List of matrix sizes
        num_runs: Number of runs for each size

    Returns:
        results: Benchmark results
    """
    print("\n" + "="*50)
    print("MIXED PRECISION BENCHMARK")
    print("="*50)

    results = {}

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")

        fp32_times = []
        fp16_times = []

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

            # FP32 test
            with tf.device('/GPU:0'):
                # Set policy to float32
                old_policy = tf.keras.mixed_precision.global_policy()
                tf.keras.mixed_precision.set_global_policy('float32')

                start_time = time.time()

                # Create matrices on GPU
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                # Perform matrix multiplication with XLA
                @tf.function(jit_compile=True)
                def matmul_fn(x, y):
                    return tf.matmul(x, y)

                c = matmul_fn(a, b)

                # Force execution and measure time
                result = c.numpy()
                fp32_time = time.time() - start_time

                print(f"FP32 time: {fp32_time:.2f} seconds")
                fp32_times.append(fp32_time)

                # Restore policy
                tf.keras.mixed_precision.set_global_policy(old_policy)

            # Clear memory
            tf.keras.backend.clear_session()

            # FP16 test
            with tf.device('/GPU:0'):
                # Set policy to mixed_float16
                old_policy = tf.keras.mixed_precision.global_policy()
                tf.keras.mixed_precision.set_global_policy('mixed_float16')

                start_time = time.time()

                # Create matrices on GPU
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                # Perform matrix multiplication with XLA
                @tf.function(jit_compile=True)
                def matmul_fn(x, y):
                    return tf.matmul(x, y)

                c = matmul_fn(a, b)

                # Force execution and measure time
                result = c.numpy()
                fp16_time = time.time() - start_time

                print(f"FP16 time: {fp16_time:.2f} seconds")
                fp16_times.append(fp16_time)

                # Restore policy
                tf.keras.mixed_precision.set_global_policy(old_policy)

            # Clear memory
            tf.keras.backend.clear_session()

        # Calculate average times
        avg_fp32_time = sum(fp32_times) / len(fp32_times)
        avg_fp16_time = sum(fp16_times) / len(fp16_times)

        # Calculate speedup
        fp32_fp16_speedup = avg_fp32_time / avg_fp16_time if avg_fp16_time > 0 else 0

        print(f"\nAverage FP32 time: {avg_fp32_time:.2f} seconds")
        print(f"Average FP16 time: {avg_fp16_time:.2f} seconds")
        print(f"FP32 to FP16 speedup: {fp32_fp16_speedup:.2f}x")

        results[size] = {
            'avg_fp32_time': avg_fp32_time,
            'avg_fp16_time': avg_fp16_time,
            'fp32_fp16_speedup': fp32_fp16_speedup
        }

    return results


def run_data_pipeline_benchmark(batch_sizes=[256, 512, 1024], num_runs=3):
    """
    Run data pipeline benchmark

    Args:
        batch_sizes: List of batch sizes
        num_runs: Number of runs for each batch size

    Returns:
        results: Benchmark results
    """
    print("\n" + "="*50)
    print("DATA PIPELINE BENCHMARK")
    print("="*50)

    results = {}

    # Create a large dataset
    num_samples = 1000000
    feature_dim = 50

    print(f"Generating dataset with {num_samples} samples...")
    data = np.random.random((num_samples, feature_dim)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        standard_times = []
        optimized_times = []

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

            # Standard pipeline
            start_time = time.time()

            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            dataset = dataset.batch(batch_size)

            # Iterate through the dataset
            for _ in dataset:
                pass

            standard_time = time.time() - start_time
            print(f"Standard pipeline time: {standard_time:.2f} seconds")
            standard_times.append(standard_time)

            # Optimized pipeline
            start_time = time.time()

            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            dataset = dataset.cache()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            # Enable parallelism
            options = tf.data.Options()
            options.experimental_optimization.map_parallelization = True
            options.experimental_optimization.parallel_batch = True
            dataset = dataset.with_options(options)

            # Iterate through the dataset
            for _ in dataset:
                pass

            optimized_time = time.time() - start_time
            print(f"Optimized pipeline time: {optimized_time:.2f} seconds")
            optimized_times.append(optimized_time)

        # Calculate average times
        avg_standard_time = sum(standard_times) / len(standard_times)
        avg_optimized_time = sum(optimized_times) / len(optimized_times)

        # Calculate speedup
        speedup = avg_standard_time / avg_optimized_time if avg_optimized_time > 0 else 0

        print(
            f"\nAverage standard pipeline time: {avg_standard_time:.2f} seconds")
        print(
            f"Average optimized pipeline time: {avg_optimized_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")

        results[batch_size] = {
            'avg_standard_time': avg_standard_time,
            'avg_optimized_time': avg_optimized_time,
            'speedup': speedup
        }

    return results


def run_all_benchmarks():
    """
    Run all benchmarks

    Returns:
        results: All benchmark results
    """
    results = {}

    # Matrix multiplication benchmark
    results['matrix_multiplication'] = run_matrix_multiplication_benchmark()

    # Mixed precision benchmark
    results['mixed_precision'] = run_mixed_precision_benchmark()

    # Data pipeline benchmark
    results['data_pipeline'] = run_data_pipeline_benchmark()

    # Financial CUDA ops benchmark
    results['financial_cuda_ops'] = test_financial_cuda_ops()

    # Trading models benchmark
    results['trading_models'] = benchmark_models()

    # Market data pipeline benchmark
    results['market_data_pipeline'] = benchmark_pipeline()

    # Polygon data benchmark
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results['polygon_data'] = benchmark_polygon_data(
        num_symbols=len(symbols), data_type="aggregates")

    # Process Polygon data with CUDA
    results['polygon_cuda'] = process_polygon_data_with_cuda(
        symbols, data_type="aggregates")

    return results


def save_results(results, filename='benchmark_results.json'):
    """
    Save benchmark results to a file

    Args:
        results: Benchmark results
        filename: Output filename
    """
    # Convert numpy values to Python types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    results = convert_numpy(results)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Run GH200 benchmarks')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for benchmark results')
    parser.add_argument('--matrix', action='store_true',
                        help='Run matrix multiplication benchmark')
    parser.add_argument('--precision', action='store_true',
                        help='Run mixed precision benchmark')
    parser.add_argument('--pipeline', action='store_true',
                        help='Run data pipeline benchmark')
    parser.add_argument('--financial', action='store_true',
                        help='Run financial CUDA ops benchmark')
    parser.add_argument('--models', action='store_true',
                        help='Run trading models benchmark')
    parser.add_argument('--market', action='store_true',
                        help='Run market data pipeline benchmark')
    parser.add_argument('--polygon', action='store_true',
                        help='Run Polygon.io data benchmark')
    parser.add_argument('--polygon-cuda', action='store_true',
                        help='Run Polygon.io data with CUDA benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmarks')

    args = parser.parse_args()

    results = {}

    if args.all or (not args.matrix and not args.precision and not args.pipeline and
                    not args.financial and not args.models and not args.market and
                    not args.polygon and not args.polygon_cuda):
        results = run_all_benchmarks()
    else:
        if args.matrix:
            results['matrix_multiplication'] = run_matrix_multiplication_benchmark()

        if args.precision:
            results['mixed_precision'] = run_mixed_precision_benchmark()

        if args.pipeline:
            results['data_pipeline'] = run_data_pipeline_benchmark()

        if args.financial:
            results['financial_cuda_ops'] = test_financial_cuda_ops()

        if args.models:
            results['trading_models'] = benchmark_models()

        if args.market:
            results['market_data_pipeline'] = benchmark_pipeline()

        if args.polygon:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            results['polygon_data'] = benchmark_polygon_data(
                num_symbols=len(symbols), data_type="aggregates")

        if args.polygon_cuda:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            results['polygon_cuda'] = process_polygon_data_with_cuda(
                symbols, data_type="aggregates")

    save_results(results, args.output)


if __name__ == "__main__":
    main()
