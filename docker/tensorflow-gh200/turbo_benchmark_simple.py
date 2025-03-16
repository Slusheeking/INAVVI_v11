#!/usr/bin/env python3
"""
Simple benchmark for the Turbo-charged Polygon.io Data Source
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import argparse

# Import only what we need
from polygon_data_source_turbo import PolygonDataSourceTurbo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('turbo_benchmark')

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Memory growth configuration failed: {e}")


def benchmark_turbo(num_symbols=10, data_type="aggregates", timespan="minute"):
    """
    Benchmark the turbo-charged Polygon.io data source
    """
    print("\n" + "="*50)
    print("TURBO-CHARGED POLYGON.IO BENCHMARK")
    print("="*50)

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN",
                   "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]
    symbols = all_symbols[:num_symbols]
    print(f"Using symbols: {symbols}")

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Create client
    client = PolygonDataSourceTurbo()

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
    print(f"Throughput: {record_count / elapsed:.2f} records/second")

    # Process with GPU
    start_time = time.time()
    processed = client.process_data_with_gpu(results)
    process_time = time.time() - start_time

    print(f"GPU processing time: {process_time:.4f} seconds")
    gpu_throughput = record_count / process_time if process_time > 0 else 0
    print(f"GPU throughput: {gpu_throughput:.2f} records/second")

    # Close client
    client.close()

    return {
        'symbols': symbols,
        'data_type': data_type,
        'timespan': timespan,
        'from_date': from_date,
        'to_date': to_date,
        'record_count': record_count,
        'fetch_elapsed': elapsed,
        'process_elapsed': process_time,
        'fetch_throughput': record_count / elapsed if elapsed > 0 else 0,
        'process_throughput': record_count / process_time if process_time > 0 else 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark Turbo Polygon.io implementation')
    parser.add_argument('--symbols', type=int, default=10,
                        help='Number of symbols to use (default: 10)')

    args = parser.parse_args()

    benchmark_turbo(num_symbols=args.symbols)
