#!/usr/bin/env python3
"""
Benchmark for the Turbo-charged Polygon.io Data Source
Compares performance against standard, GH200-optimized, and enhanced implementations
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
import argparse

# Import all implementations
from polygon_data_source import benchmark_polygon_data, process_polygon_data_with_cuda
from polygon_data_source_gh200 import benchmark_polygon_data_gh200, process_polygon_data_with_cuda_gh200
from polygon_data_source_enhanced import PolygonDataSource as EnhancedPolygonDataSource
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

    # Get top symbols by market cap
    client = PolygonDataSourceTurbo()

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
                   "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO",
                   "ADBE", "CRM", "AVGO", "ACN", "ORCL", "CMCSA", "PEP", "TMO", "ABT", "VZ",
                   "COST", "NKE", "MRK", "WMT", "LLY", "PFE", "T", "QCOM", "DHR", "NEE",
                   "TXN", "UPS", "PM", "MS", "RTX", "HON", "SBUX", "MDT", "BMY", "CVX"]

    symbols = all_symbols[:num_symbols]
    print(f"Using symbols: {symbols}")

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Benchmark turbo implementation
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
    print(f"GPU throughput: {record_count / process_time:.2f} records/second")

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
        'fetch_throughput': record_count / elapsed,
        'process_throughput': record_count / process_time
    }


def compare_all_implementations(num_symbols=5, runs=1):
    """
    Compare all implementations
    """
    print("\n" + "="*50)
    print("COMPARING ALL POLYGON.IO IMPLEMENTATIONS")
    print("="*50)

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN",
                   "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]
    symbols = all_symbols[:num_symbols]

    print(f"Using {num_symbols} symbols over {runs} runs")

    # Prepare results
    results = []

    for run in range(1, runs+1):
        print(f"\nRun {run}/{runs}")

        # Standard implementation
        print("\nRunning standard benchmark...")
        start_time = time.time()
        standard_result = benchmark_polygon_data(
            num_symbols=num_symbols, data_type="aggregates")
        standard_time = time.time() - start_time
        standard_rows = standard_result['record_count']
        standard_throughput = standard_result['throughput']

        # GH200 implementation
        print("\nRunning GH200-optimized benchmark...")
        start_time = time.time()
        gh200_result = benchmark_polygon_data_gh200(
            num_symbols=num_symbols, data_type="aggregates")
        gh200_time = time.time() - start_time
        gh200_rows = gh200_result['record_count']
        gh200_throughput = gh200_result['throughput']

        # Enhanced implementation
        print("\nRunning enhanced benchmark...")
        start_time = time.time()
        enhanced_client = EnhancedPolygonDataSource()
        enhanced_results = enhanced_client.get_aggregates_batch(
            symbols, timespan="minute", from_date=from_date, to_date=to_date)
        enhanced_rows = 0
        for symbol, df in enhanced_results.items():
            enhanced_rows += len(df)
        enhanced_time = time.time() - start_time
        enhanced_throughput = enhanced_rows / enhanced_time
        enhanced_client.close()

        # Turbo implementation
        print("\nRunning turbo-charged benchmark...")
        start_time = time.time()
        turbo_client = PolygonDataSourceTurbo()
        turbo_results = turbo_client.get_aggregates_batch(
            symbols, timespan="minute", from_date=from_date, to_date=to_date)
        turbo_rows = 0
        for symbol, df in turbo_results.items():
            turbo_rows += len(df)
        turbo_time = time.time() - start_time
        turbo_throughput = turbo_rows / turbo_time
        turbo_client.close()

        # Add to results
        results.append({
            'implementation': 'Standard',
            'elapsed_time': standard_time,
            'num_symbols': num_symbols,
            'total_rows': standard_rows,
            'rows_per_second': standard_throughput,
            'run': run
        })

        results.append({
            'implementation': 'GH200',
            'elapsed_time': gh200_time,
            'num_symbols': num_symbols,
            'total_rows': gh200_rows,
            'rows_per_second': gh200_throughput,
            'run': run
        })

        results.append({
            'implementation': 'Enhanced',
            'elapsed_time': enhanced_time,
            'num_symbols': num_symbols,
            'total_rows': enhanced_rows,
            'rows_per_second': enhanced_throughput,
            'run': run
        })

        results.append({
            'implementation': 'Turbo',
            'elapsed_time': turbo_time,
            'num_symbols': num_symbols,
            'total_rows': turbo_rows,
            'rows_per_second': turbo_throughput,
            'run': run
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Calculate averages
    avg_df = df.groupby('implementation').agg({
        'elapsed_time': 'mean',
        'total_rows': 'mean',
        'rows_per_second': 'mean'
    }).reset_index()

    # Save results
    df.to_csv('/app/turbo_comparison_aggregates_results.csv', index=False)

    # Create comparison chart
    plt.figure(figsize=(12, 8))

    # Bar chart for throughput
    plt.subplot(2, 1, 1)
    bars = plt.bar(avg_df['implementation'], avg_df['rows_per_second'])
    plt.title('Polygon.io Implementation Throughput Comparison')
    plt.ylabel('Rows per second')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height):,}',
                 ha='center', va='bottom', rotation=0)

    # Bar chart for elapsed time
    plt.subplot(2, 1, 2)
    bars = plt.bar(avg_df['implementation'], avg_df['elapsed_time'])
    plt.title('Polygon.io Implementation Elapsed Time Comparison')
    plt.ylabel('Seconds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}s',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('/app/turbo_comparison_aggregates.png')

    # Print summary
    print("\nPerformance Comparison Summary:")
    print("="*50)

    for _, row in avg_df.iterrows():
        impl = row['implementation']
        time = row['elapsed_time']
        rows = row['total_rows']
        throughput = row['rows_per_second']

        print(f"{impl} Implementation:")
        print(f"  Average Time: {time:.2f} seconds")
        print(f"  Average Rows: {int(rows)}")
        print(f"  Average Throughput: {throughput:.2f} rows/second")
        print("-"*50)

    # Calculate speedup factors
    standard_throughput = avg_df[avg_df['implementation']
                                 == 'Standard']['rows_per_second'].values[0]
    gh200_throughput = avg_df[avg_df['implementation']
                              == 'GH200']['rows_per_second'].values[0]
    enhanced_throughput = avg_df[avg_df['implementation']
                                 == 'Enhanced']['rows_per_second'].values[0]
    turbo_throughput = avg_df[avg_df['implementation']
                              == 'Turbo']['rows_per_second'].values[0]

    print("\nSpeedup Factors:")
    print(f"GH200 vs Standard: {gh200_throughput / standard_throughput:.2f}x")
    print(
        f"Enhanced vs Standard: {enhanced_throughput / standard_throughput:.2f}x")
    print(f"Turbo vs Standard: {turbo_throughput / standard_throughput:.2f}x")
    print(f"Enhanced vs GH200: {enhanced_throughput / gh200_throughput:.2f}x")
    print(f"Turbo vs GH200: {turbo_throughput / gh200_throughput:.2f}x")
    print(f"Turbo vs Enhanced: {turbo_throughput / enhanced_throughput:.2f}x")

    return df


def run_scaling_test(max_symbols=50):
    """
    Test how the turbo implementation scales with increasing number of symbols
    """
    print("\n" + "="*50)
    print("TURBO SCALING TEST")
    print("="*50)

    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
                   "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO",
                   "ADBE", "CRM", "AVGO", "ACN", "ORCL", "CMCSA", "PEP", "TMO", "ABT", "VZ",
                   "COST", "NKE", "MRK", "WMT", "LLY", "PFE", "T", "QCOM", "DHR", "NEE",
                   "TXN", "UPS", "PM", "MS", "RTX", "HON", "SBUX", "MDT", "BMY", "CVX"]

    # Symbol counts to test
    symbol_counts = [1, 2, 5, 10, 20, 50]
    symbol_counts = [count for count in symbol_counts if count <= max_symbols]

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Prepare results
    results = []

    for count in symbol_counts:
        print(f"\nRunning turbo benchmark with {count} symbols")

        symbols = all_symbols[:count]

        # Create client
        client = PolygonDataSourceTurbo()

        # Benchmark
        start_time = time.time()
        data = client.get_aggregates_batch(
            symbols, timespan="minute", from_date=from_date, to_date=to_date)

        # Count records
        record_count = 0
        for symbol, df in data.items():
            record_count += len(df)

        elapsed = time.time() - start_time
        throughput = record_count / elapsed

        # Add to results
        results.append({
            'implementation': 'Turbo',
            'elapsed_time': elapsed,
            'num_symbols': count,
            'total_rows': record_count,
            'rows_per_second': throughput,
            'symbol_count': count
        })

        # Close client
        client.close()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv('/app/turbo_scaling_test_aggregates_results.csv', index=False)

    # Create scaling chart
    plt.figure(figsize=(12, 10))

    # Throughput vs symbol count
    plt.subplot(2, 1, 1)
    plt.plot(df['symbol_count'], df['rows_per_second'],
             marker='o', linewidth=2)
    plt.title('Turbo Implementation Scaling: Throughput vs Symbol Count')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Rows per second')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add values next to points
    for i, row in df.iterrows():
        plt.annotate(f"{int(row['rows_per_second']):,}",
                     (row['symbol_count'], row['rows_per_second']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')

    # Time vs symbol count
    plt.subplot(2, 1, 2)
    plt.plot(df['symbol_count'], df['elapsed_time'], marker='o', linewidth=2)
    plt.title('Turbo Implementation Scaling: Time vs Symbol Count')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Seconds')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add values next to points
    for i, row in df.iterrows():
        plt.annotate(f"{row['elapsed_time']:.2f}s",
                     (row['symbol_count'], row['elapsed_time']),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left')

    plt.tight_layout()
    plt.savefig('/app/turbo_scaling_test_aggregates.png')

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark Polygon.io implementations')
    parser.add_argument('--test', choices=['turbo', 'compare', 'scaling', 'all'], default='all',
                        help='Test to run (default: all)')
    parser.add_argument('--symbols', type=int, default=10,
                        help='Number of symbols to use (default: 10)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for comparison (default: 1)')

    args = parser.parse_args()

    if args.test == 'turbo' or args.test == 'all':
        benchmark_turbo(num_symbols=args.symbols)

    if args.test == 'compare' or args.test == 'all':
        compare_all_implementations(
            num_symbols=min(args.symbols, 10), runs=args.runs)

    if args.test == 'scaling' or args.test == 'all':
        run_scaling_test(max_symbols=args.symbols)
