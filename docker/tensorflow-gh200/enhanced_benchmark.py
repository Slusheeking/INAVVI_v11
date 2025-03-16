#!/usr/bin/env python3
"""
Enhanced Benchmark for Polygon.io API with Redis Caching and Parallel Processing
Optimized for NVIDIA GH200 Grace Hopper Superchip

This script benchmarks the enhanced Polygon.io API implementation with:
- Redis caching
- Connection pooling
- Parallel processing
- GH200-specific optimizations
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf

# Import the different implementations
try:
    import polygon_data_source as ps
    import polygon_data_source_gh200 as pg
    import polygon_data_source_enhanced as pe
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_benchmark')

# Default symbols for benchmarking
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Default date range (last 30 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
DEFAULT_FROM_DATE = start_date.strftime("%Y-%m-%d")
DEFAULT_TO_DATE = end_date.strftime("%Y-%m-%d")

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Memory growth configuration failed: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Polygon.io API Benchmark')

    parser.add_argument('--symbols', type=str, default=','.join(DEFAULT_SYMBOLS),
                        help='Comma-separated list of ticker symbols')

    parser.add_argument('--data-type', type=str, default='aggregates',
                        choices=['aggregates', 'trades', 'quotes'],
                        help='Type of data to fetch')

    parser.add_argument('--from-date', type=str, default=DEFAULT_FROM_DATE,
                        help='Start date (YYYY-MM-DD)')

    parser.add_argument('--to-date', type=str, default=DEFAULT_TO_DATE,
                        help='End date (YYYY-MM-DD)')

    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs')

    parser.add_argument('--compare', action='store_true',
                        help='Compare with standard and GH200 implementations')

    parser.add_argument('--scaling-test', action='store_true',
                        help='Run scaling test with increasing number of symbols')

    parser.add_argument('--max-symbols', type=int, default=50,
                        help='Maximum number of symbols for scaling test')

    parser.add_argument('--large-scale', action='store_true',
                        help='Run large-scale benchmark with many symbols')

    parser.add_argument('--output-dir', type=str, default='/app',
                        help='Directory to save results')

    return parser.parse_args()


def run_standard_benchmark(symbols, data_type, from_date, to_date):
    """Run benchmark with standard implementation"""
    logger.info(f"Running standard benchmark with {len(symbols)} symbols")

    client = ps.PolygonDataSource()

    start_time = time.time()
    results = {}

    for symbol in symbols:
        if data_type == 'aggregates':
            df = client.get_aggregates(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_date=from_date,
                to_date=to_date
            )
            if df is not None and not df.empty:
                results[symbol] = df

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_rows = sum(len(df) for df in results.values()) if results else 0

    return {
        'implementation': 'Standard',
        'elapsed_time': elapsed_time,
        'num_symbols': len(symbols),
        'total_rows': total_rows,
        'rows_per_second': total_rows / elapsed_time if elapsed_time > 0 else 0
    }


def run_gh200_benchmark(symbols, data_type, from_date, to_date):
    """Run benchmark with GH200-optimized implementation"""
    logger.info(
        f"Running GH200-optimized benchmark with {len(symbols)} symbols")

    client = pg.PolygonDataSource()

    start_time = time.time()
    results = {}

    for symbol in symbols:
        if data_type == 'aggregates':
            df = client.get_aggregates(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_date=from_date,
                to_date=to_date
            )
            if df is not None and not df.empty:
                results[symbol] = df

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_rows = sum(len(df) for df in results.values()) if results else 0

    return {
        'implementation': 'GH200',
        'elapsed_time': elapsed_time,
        'num_symbols': len(symbols),
        'total_rows': total_rows,
        'rows_per_second': total_rows / elapsed_time if elapsed_time > 0 else 0
    }


def run_enhanced_benchmark(symbols, data_type, from_date, to_date):
    """Run benchmark with enhanced implementation"""
    logger.info(f"Running enhanced benchmark with {len(symbols)} symbols")

    client = pe.PolygonDataSource()

    start_time = time.time()

    # Use batch processing for enhanced implementation
    if data_type == 'aggregates':
        results = client.get_aggregates_batch(
            tickers=symbols,
            multiplier=1,
            timespan="minute",
            from_date=from_date,
            to_date=to_date
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_rows = sum(len(df) for df in results.values()) if results else 0

    return {
        'implementation': 'Enhanced',
        'elapsed_time': elapsed_time,
        'num_symbols': len(symbols),
        'total_rows': total_rows,
        'rows_per_second': total_rows / elapsed_time if elapsed_time > 0 else 0
    }


def run_scaling_test(max_symbols, data_type, from_date, to_date):
    """Run scaling test with increasing number of symbols"""
    logger.info(f"Running scaling test up to {max_symbols} symbols")

    # Generate a list of common stock symbols
    all_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PYPL", "ADBE",
        "INTC", "CSCO", "CMCSA", "PEP", "AVGO", "TXN", "QCOM", "TMUS", "AMGN", "SBUX",
        "INTU", "AMAT", "ISRG", "ADI", "MDLZ", "REGN", "VRTX", "BKNG", "GILD", "ADP",
        "FISV", "ATVI", "CSX", "CHTR", "MAR", "LRCX", "ADSK", "MU", "ILMN", "MNST",
        "KLAC", "ALGN", "IDXX", "WDAY", "PCAR", "CTAS", "MCHP", "PAYX", "EXC", "BIIB"
    ]

    # Ensure we don't exceed the available symbols
    max_symbols = min(max_symbols, len(all_symbols))

    results = []
    symbol_counts = [1, 2, 5, 10, 20, 50]
    symbol_counts = [s for s in symbol_counts if s <= max_symbols]

    # Add max_symbols if not already in the list
    if max_symbols not in symbol_counts:
        symbol_counts.append(max_symbols)

    for count in symbol_counts:
        symbols = all_symbols[:count]

        # Run enhanced benchmark
        enhanced_result = run_enhanced_benchmark(
            symbols, data_type, from_date, to_date)
        enhanced_result['symbol_count'] = count
        results.append(enhanced_result)

        # Sleep to avoid rate limiting
        time.sleep(1)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['symbol_count'], df['elapsed_time'],
             'o-', linewidth=2, markersize=8)
    plt.title('Scaling Performance: Time vs. Number of Symbols')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Time (seconds)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df['symbol_count'], df['rows_per_second'],
             'o-', linewidth=2, markersize=8, color='green')
    plt.title('Scaling Performance: Throughput vs. Number of Symbols')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Rows per Second')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"/app/enhanced_scaling_test_{data_type}.png")

    # Save results to CSV
    df.to_csv(
        f"/app/enhanced_scaling_test_{data_type}_results.csv", index=False)

    logger.info(
        f"Scaling test results saved to /app/enhanced_scaling_test_{data_type}_results.csv")
    logger.info(
        f"Scaling test chart saved to /app/enhanced_scaling_test_{data_type}.png")

    return df


def run_large_scale_benchmark(symbols, data_type, from_date, to_date):
    """Run large-scale benchmark with many symbols"""
    logger.info("Running large-scale benchmark")

    # Generate a list of common stock symbols
    all_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PYPL", "ADBE",
        "INTC", "CSCO", "CMCSA", "PEP", "AVGO", "TXN", "QCOM", "TMUS", "AMGN", "SBUX",
        "INTU", "AMAT", "ISRG", "ADI", "MDLZ", "REGN", "VRTX", "BKNG", "GILD", "ADP",
        "FISV", "ATVI", "CSX", "CHTR", "MAR", "LRCX", "ADSK", "MU", "ILMN", "MNST",
        "KLAC", "ALGN", "IDXX", "WDAY", "PCAR", "CTAS", "MCHP", "PAYX", "EXC", "BIIB"
    ]

    # Use all available symbols for large-scale test
    large_symbols = all_symbols

    # Run enhanced benchmark with all symbols
    result = run_enhanced_benchmark(
        large_symbols, data_type, from_date, to_date)

    logger.info(
        f"Large-scale benchmark completed with {len(large_symbols)} symbols")
    logger.info(f"Total time: {result['elapsed_time']:.2f} seconds")
    logger.info(f"Total rows: {result['total_rows']}")
    logger.info(f"Rows per second: {result['rows_per_second']:.2f}")

    return result


def compare_implementations(symbols, data_type, from_date, to_date, num_runs=3):
    """Compare standard, GH200, and enhanced implementations"""
    logger.info(
        f"Comparing implementations with {len(symbols)} symbols over {num_runs} runs")

    results = []

    for run in range(1, num_runs + 1):
        logger.info(f"Run {run}/{num_runs}")

        # Run standard benchmark
        standard_result = run_standard_benchmark(
            symbols, data_type, from_date, to_date)
        standard_result['run'] = run
        results.append(standard_result)

        # Run GH200 benchmark
        gh200_result = run_gh200_benchmark(
            symbols, data_type, from_date, to_date)
        gh200_result['run'] = run
        results.append(gh200_result)

        # Run enhanced benchmark
        enhanced_result = run_enhanced_benchmark(
            symbols, data_type, from_date, to_date)
        enhanced_result['run'] = run
        results.append(enhanced_result)

        # Sleep to avoid rate limiting
        time.sleep(1)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Calculate average results by implementation
    avg_results = df.groupby('implementation').agg({
        'elapsed_time': 'mean',
        'total_rows': 'mean',
        'rows_per_second': 'mean'
    }).reset_index()

    # Plot comparison
    plt.figure(figsize=(12, 10))

    # Plot elapsed time
    plt.subplot(2, 1, 1)
    bars = plt.bar(avg_results['implementation'], avg_results['elapsed_time'])
    plt.title('Average Elapsed Time by Implementation')
    plt.xlabel('Implementation')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y')

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s',
                 ha='center', va='bottom')

    # Plot rows per second
    plt.subplot(2, 1, 2)
    bars = plt.bar(avg_results['implementation'],
                   avg_results['rows_per_second'], color='green')
    plt.title('Average Throughput by Implementation')
    plt.xlabel('Implementation')
    plt.ylabel('Rows per Second')
    plt.grid(axis='y')

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"/app/enhanced_comparison_{data_type}.png")

    # Save results to CSV
    df.to_csv(f"/app/enhanced_comparison_{data_type}_results.csv", index=False)
    avg_results.to_csv(
        f"/app/enhanced_comparison_{data_type}_avg_results.csv", index=False)

    logger.info(
        f"Comparison results saved to /app/enhanced_comparison_{data_type}_results.csv")
    logger.info(
        f"Comparison chart saved to /app/enhanced_comparison_{data_type}.png")

    # Print summary
    print("\nPerformance Comparison Summary:")
    print("=" * 50)
    for _, row in avg_results.iterrows():
        print(f"{row['implementation']} Implementation:")
        print(f"  Average Time: {row['elapsed_time']:.2f} seconds")
        print(f"  Average Rows: {row['total_rows']:.0f}")
        print(
            f"  Average Throughput: {row['rows_per_second']:.2f} rows/second")
        print("-" * 50)

    # Calculate speedups
    if len(avg_results) >= 3:
        standard_time = avg_results.loc[avg_results['implementation']
                                        == 'Standard', 'elapsed_time'].values[0]
        gh200_time = avg_results.loc[avg_results['implementation']
                                     == 'GH200', 'elapsed_time'].values[0]
        enhanced_time = avg_results.loc[avg_results['implementation']
                                        == 'Enhanced', 'elapsed_time'].values[0]

        gh200_speedup = standard_time / gh200_time if gh200_time > 0 else 0
        enhanced_speedup = standard_time / enhanced_time if enhanced_time > 0 else 0
        enhanced_vs_gh200 = gh200_time / enhanced_time if enhanced_time > 0 else 0

        print("\nSpeedup Factors:")
        print(f"GH200 vs Standard: {gh200_speedup:.2f}x")
        print(f"Enhanced vs Standard: {enhanced_speedup:.2f}x")
        print(f"Enhanced vs GH200: {enhanced_vs_gh200:.2f}x")

    return df, avg_results


def main():
    """Main function"""
    args = parse_args()

    # Parse symbols
    symbols = args.symbols.split(',')

    logger.info(f"Starting benchmark with {len(symbols)} symbols")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Date range: {args.from_date} to {args.to_date}")

    if args.scaling_test:
        # Run scaling test
        run_scaling_test(args.max_symbols, args.data_type,
                         args.from_date, args.to_date)
    elif args.large_scale:
        # Run large-scale benchmark
        run_large_scale_benchmark(
            symbols, args.data_type, args.from_date, args.to_date)
    elif args.compare:
        # Run comparison
        compare_implementations(symbols, args.data_type,
                                args.from_date, args.to_date, args.num_runs)
    else:
        # Run enhanced benchmark only
        result = run_enhanced_benchmark(
            symbols, args.data_type, args.from_date, args.to_date)

        print("\nEnhanced Implementation Results:")
        print("=" * 50)
        print(f"Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print(f"Total Rows: {result['total_rows']}")
        print(f"Throughput: {result['rows_per_second']:.2f} rows/second")


if __name__ == "__main__":
    main()
