#!/usr/bin/env python3
"""
Benchmark for the Ultra-optimized Polygon.io Data Source
Specifically designed for NVIDIA GH200 Grace Hopper Superchip

This benchmark tests the performance of the ultra-optimized implementation with:
1. Custom CUDA Kernels
2. Shared Memory Parallelism
3. Zero-Copy Memory Architecture
4. Asynchronous Processing Pipeline
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import tensorflow as tf
import argparse

# Import implementations
from polygon_data_source_turbo_fixed import PolygonDataSourceTurboFixed
from polygon_data_source_ultra import PolygonDataSourceUltra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ultra_benchmark')

# Set environment variable to prevent automatic CUDA initialization on import
os.environ['CUDA_INIT_ON_IMPORT'] = '0'

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Memory growth configuration failed: {e}")


def benchmark_ultra(num_symbols=10, data_type="aggregates", timespan="minute"):
    """
    Benchmark the ultra-optimized Polygon.io data source
    """
    print("\n" + "="*50)
    print("ULTRA-OPTIMIZED POLYGON.IO BENCHMARK")
    print("="*50)

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN",
                   "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]
    symbols = all_symbols[:num_symbols]
    print(f"Using symbols: {symbols}")

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Create client with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = PolygonDataSourceUltra()
            break
        except Exception as e:
            logger.error(
                f"Failed to create client (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)  # Wait before retrying

    # Benchmark
    start_time = time.time()

    # Fetch data
    results = client.get_aggregates_batch(
        symbols, timespan=timespan, from_date=from_date, to_date=to_date)

    # Count records
    record_count = 0
    for symbol, df in results.items():
        record_count += len(df)

    elapsed = time.time() - start_time

    print(f"Processed {record_count} records for {len(symbols)} symbols")
    print(f"Total time: {elapsed:.4f} seconds")
    throughput = record_count / elapsed if elapsed > 0 else 0
    print(f"Throughput: {throughput:.2f} records/second")

    # Process with GPU
    start_time = time.time()
    processed = client.process_data_with_gpu(results)
    process_time = time.time() - start_time

    print(f"GPU processing time: {process_time:.4f} seconds")
    gpu_throughput = record_count / process_time if process_time > 0 else 0
    print(f"GPU throughput: {gpu_throughput:.2f} records/second")

    # Close client with proper error handling
    try:
        client.close()
    except Exception as e:
        logger.error(f"Error closing client: {e}")
        # Try to force cleanup
        try:
            from financial_cuda_kernels import cleanup_cuda
            cleanup_cuda()
        except:
            pass

    return {
        'symbols': symbols,
        'data_type': data_type,
        'timespan': timespan,
        'from_date': from_date,
        'to_date': to_date,
        'record_count': record_count,
        'fetch_elapsed': elapsed,
        'process_elapsed': process_time,
        'fetch_throughput': throughput,
        'process_throughput': gpu_throughput
    }


def compare_implementations(num_symbols=5):
    """
    Compare turbo-fixed and ultra implementations
    """
    print("\n" + "="*50)
    print("COMPARING IMPLEMENTATIONS")
    print("="*50)

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN",
                   "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]
    symbols = all_symbols[:num_symbols]
    print(f"Using symbols: {symbols}")

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Benchmark turbo-fixed implementation
    print("\nBenchmarking Turbo-Fixed Implementation...")
    turbo_client = PolygonDataSourceTurboFixed()

    start_time = time.time()
    turbo_results = turbo_client.get_aggregates_batch(
        symbols, timespan="minute", from_date=from_date, to_date=to_date)
    turbo_fetch_time = time.time() - start_time

    turbo_record_count = 0
    for symbol, df in turbo_results.items():
        turbo_record_count += len(df)

    start_time = time.time()
    turbo_processed = turbo_client.process_data_with_gpu(turbo_results)
    turbo_process_time = time.time() - start_time

    turbo_client.close()

    # Benchmark ultra implementation
    print("\nBenchmarking Ultra-Optimized Implementation...")
    ultra_client = PolygonDataSourceUltra()

    start_time = time.time()
    ultra_results = ultra_client.get_aggregates_batch(
        symbols, timespan="minute", from_date=from_date, to_date=to_date)
    ultra_fetch_time = time.time() - start_time

    ultra_record_count = 0
    for symbol, df in ultra_results.items():
        ultra_record_count += len(df)

    start_time = time.time()
    ultra_processed = ultra_client.process_data_with_gpu(ultra_results)
    ultra_process_time = time.time() - start_time

    ultra_client.close()

    # Calculate throughput
    turbo_fetch_throughput = turbo_record_count / \
        turbo_fetch_time if turbo_fetch_time > 0 else 0
    turbo_process_throughput = turbo_record_count / \
        turbo_process_time if turbo_process_time > 0 else 0

    ultra_fetch_throughput = ultra_record_count / \
        ultra_fetch_time if ultra_fetch_time > 0 else 0
    ultra_process_throughput = ultra_record_count / \
        ultra_process_time if ultra_process_time > 0 else 0

    # Print results
    print("\nResults:")
    print("-"*50)
    print(f"Turbo-Fixed Implementation:")
    print(f"  Fetch Time: {turbo_fetch_time:.4f} seconds")
    print(f"  Process Time: {turbo_process_time:.4f} seconds")
    print(f"  Fetch Throughput: {turbo_fetch_throughput:.2f} records/second")
    print(
        f"  Process Throughput: {turbo_process_throughput:.2f} records/second")
    print("-"*50)
    print(f"Ultra-Optimized Implementation:")
    print(f"  Fetch Time: {ultra_fetch_time:.4f} seconds")
    print(f"  Process Time: {ultra_process_time:.4f} seconds")
    print(f"  Fetch Throughput: {ultra_fetch_throughput:.2f} records/second")
    print(
        f"  Process Throughput: {ultra_process_throughput:.2f} records/second")
    print("-"*50)

    # Calculate speedup
    fetch_speedup = ultra_fetch_throughput / \
        turbo_fetch_throughput if turbo_fetch_throughput > 0 else 0
    process_speedup = ultra_process_throughput / \
        turbo_process_throughput if turbo_process_throughput > 0 else 0

    print(f"Speedup (Ultra vs Turbo-Fixed):")
    print(f"  Fetch Speedup: {fetch_speedup:.2f}x")
    print(f"  Process Speedup: {process_speedup:.2f}x")

    # Create comparison chart
    plt.figure(figsize=(12, 8))

    # Bar chart for fetch throughput
    plt.subplot(2, 1, 1)
    implementations = ['Turbo-Fixed', 'Ultra-Optimized']
    fetch_throughputs = [turbo_fetch_throughput, ultra_fetch_throughput]

    bars = plt.bar(implementations, fetch_throughputs)
    plt.title('Fetch Throughput Comparison')
    plt.ylabel('Records per second')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height):,}',
                 ha='center', va='bottom', rotation=0)

    # Bar chart for process throughput
    plt.subplot(2, 1, 2)
    process_throughputs = [turbo_process_throughput, ultra_process_throughput]

    bars = plt.bar(implementations, process_throughputs)
    plt.title('Process Throughput Comparison')
    plt.ylabel('Records per second')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height):,}',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('/app/ultra_comparison.png')

    return {
        'turbo_fetch_throughput': turbo_fetch_throughput,
        'turbo_process_throughput': turbo_process_throughput,
        'ultra_fetch_throughput': ultra_fetch_throughput,
        'ultra_process_throughput': ultra_process_throughput,
        'fetch_speedup': fetch_speedup,
        'process_speedup': process_speedup
    }


def run_scaling_test(max_symbols=10):
    """
    Test how the ultra implementation scales with increasing number of symbols
    """
    print("\n" + "="*50)
    print("ULTRA SCALING TEST")
    print("="*50)

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
                   "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO"]

    # Symbol counts to test
    symbol_counts = [1, 2, 5, 10]
    symbol_counts = [count for count in symbol_counts if count <= max_symbols]

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Prepare results
    results = []

    # Create client
    client = PolygonDataSourceUltra()

    for count in symbol_counts:
        print(f"\nRunning ultra benchmark with {count} symbols")

        symbols = all_symbols[:count]

        # Benchmark
        start_time = time.time()
        data = client.get_aggregates_batch(
            symbols, timespan="minute", from_date=from_date, to_date=to_date)

        # Count records
        record_count = 0
        for symbol, df in data.items():
            record_count += len(df)

        elapsed = time.time() - start_time
        throughput = record_count / elapsed if elapsed > 0 else 0

        # Process with GPU
        start_time = time.time()
        processed = client.process_data_with_gpu(data)
        process_time = time.time() - start_time
        process_throughput = record_count / process_time if process_time > 0 else 0

        # Add to results
        results.append({
            'symbol_count': count,
            'record_count': record_count,
            'fetch_time': elapsed,
            'process_time': process_time,
            'fetch_throughput': throughput,
            'process_throughput': process_throughput
        })

    # Close client
    client.close()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv('/app/ultra_scaling_test_results.csv', index=False)

    # Create scaling chart
    plt.figure(figsize=(12, 10))

    # Throughput vs symbol count
    plt.subplot(2, 1, 1)
    plt.plot(df['symbol_count'], df['fetch_throughput'],
             marker='o', linewidth=2, label='Fetch')
    plt.plot(df['symbol_count'], df['process_throughput'],
             marker='s', linewidth=2, label='Process')
    plt.title('Ultra Implementation Scaling: Throughput vs Symbol Count')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Rows per second')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add values next to points
    for i, row in df.iterrows():
        plt.annotate(f"{int(row['fetch_throughput']):,}",
                     (row['symbol_count'], row['fetch_throughput']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')
        plt.annotate(f"{int(row['process_throughput']):,}",
                     (row['symbol_count'], row['process_throughput']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')

    # Time vs symbol count
    plt.subplot(2, 1, 2)
    plt.plot(df['symbol_count'], df['fetch_time'],
             marker='o', linewidth=2, label='Fetch')
    plt.plot(df['symbol_count'], df['process_time'],
             marker='s', linewidth=2, label='Process')
    plt.title('Ultra Implementation Scaling: Time vs Symbol Count')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add values next to points
    for i, row in df.iterrows():
        plt.annotate(f"{row['fetch_time']:.2f}s",
                     (row['symbol_count'], row['fetch_time']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')
        plt.annotate(f"{row['process_time']:.2f}s",
                     (row['symbol_count'], row['process_time']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')

    plt.tight_layout()
    plt.savefig('/app/ultra_scaling_test.png')

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark Ultra-Optimized Polygon.io implementation')
    parser.add_argument('--test', choices=['ultra', 'compare', 'scaling', 'all'], default='ultra',
                        help='Test to run (default: ultra)')
    parser.add_argument('--symbols', type=int, default=5,
                        help='Number of symbols to use (default: 5)')

    args = parser.parse_args()

    if args.test == 'ultra' or args.test == 'all':
        benchmark_ultra(num_symbols=args.symbols)

    if args.test == 'compare' or args.test == 'all':
        compare_implementations(num_symbols=args.symbols)

    if args.test == 'scaling' or args.test == 'all':
        run_scaling_test(max_symbols=args.symbols)
