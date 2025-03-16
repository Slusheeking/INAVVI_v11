#!/usr/bin/env python3
"""
Fixed benchmark for the Turbo-charged Polygon.io Data Source
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import tensorflow as tf
import argparse

# Import the fixed implementation
from polygon_data_source_turbo_fixed import PolygonDataSourceTurboFixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('turbo_benchmark_fixed')

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


def benchmark_turbo_fixed(num_symbols=10, data_type="aggregates", timespan="minute"):
    """
    Benchmark the fixed turbo-charged Polygon.io data source
    """
    print("\n" + "="*50)
    print("FIXED TURBO-CHARGED POLYGON.IO BENCHMARK")
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
            client = PolygonDataSourceTurboFixed()
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
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark Fixed Turbo Polygon.io implementation')
    parser.add_argument('--symbols', type=int, default=5,
                        help='Number of symbols to use (default: 5)')

    args = parser.parse_args()

    benchmark_turbo_fixed(num_symbols=args.symbols)
