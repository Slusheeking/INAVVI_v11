from polygon_data_source_gh200 import benchmark_polygon_data_gh200, process_polygon_data_with_cuda_gh200
from polygon_data_source import benchmark_polygon_data, process_polygon_data_with_cuda
import tensorflow as tf
import numpy as np
import time
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib backend to Agg for non-interactive environments
plt.switch_backend('Agg')

# Import both original and optimized implementations

# Configure TensorFlow for GH200
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        # Enable memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(device, True)
    print(
        f"Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")

    # Get GPU details
    for i, device in enumerate(physical_devices):
        details = tf.config.experimental.get_device_details(device)
        print(f"GPU {i} details: {details}")
else:
    print("No GPUs found")


def run_benchmark(symbols, data_type="aggregates", timespan="minute", num_runs=3):
    """
    Run benchmark comparing original and GH200-optimized implementations

    Args:
        symbols: List of symbols to use
        data_type: Type of data (aggregates, trades, quotes, order_book)
        timespan: Timespan for aggregates
        num_runs: Number of benchmark runs

    Returns:
        results: Dictionary with benchmark results
    """
    print("\n" + "="*70)
    print(f"POLYGON.IO DATA BENCHMARK ON GH200 - {data_type.upper()} DATA")
    print("="*70)

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Results storage
    original_results = []
    gh200_results = []

    # Run multiple times to get average performance
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")

        # Run original benchmark
        print("\nRunning original implementation...")
        start_time = time.time()
        original_benchmark = benchmark_polygon_data(
            num_symbols=len(symbols),
            data_type=data_type,
            timespan=timespan
        )
        original_results.append(original_benchmark)

        # Run GH200-optimized benchmark
        print("\nRunning GH200-optimized implementation...")
        start_time = time.time()
        gh200_benchmark = benchmark_polygon_data_gh200(
            num_symbols=len(symbols),
            data_type=data_type,
            timespan=timespan
        )
        gh200_results.append(gh200_benchmark)

    # Calculate average results
    avg_original_throughput = sum(r['throughput']
                                  for r in original_results) / num_runs
    avg_gh200_throughput = sum(r['throughput']
                               for r in gh200_results) / num_runs
    avg_speedup = avg_gh200_throughput / \
        avg_original_throughput if avg_original_throughput > 0 else 0

    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Data type: {data_type}")
    print(f"Symbols: {symbols}")
    print(
        f"Average original throughput: {avg_original_throughput:.2f} records/second")
    print(
        f"Average GH200 throughput: {avg_gh200_throughput:.2f} records/second")
    print(f"Average speedup: {avg_speedup:.2f}x")

    return {
        'symbols': symbols,
        'data_type': data_type,
        'timespan': timespan,
        'original_results': original_results,
        'gh200_results': gh200_results,
        'avg_original_throughput': avg_original_throughput,
        'avg_gh200_throughput': avg_gh200_throughput,
        'avg_speedup': avg_speedup
    }


def run_cuda_benchmark(symbols, data_type="aggregates", timespan="minute"):
    """
    Run CUDA processing benchmark comparing original and GH200-optimized implementations

    Args:
        symbols: List of symbols to use
        data_type: Type of data (aggregates, trades, quotes, order_book)
        timespan: Timespan for aggregates

    Returns:
        results: Dictionary with benchmark results
    """
    print("\n" + "="*70)
    print(f"POLYGON.IO CUDA PROCESSING BENCHMARK - {data_type.upper()} DATA")
    print("="*70)

    # Run original CUDA processing
    print("\nRunning original CUDA processing...")
    start_time = time.time()
    original_results = process_polygon_data_with_cuda(
        symbols, data_type=data_type)
    original_time = time.time() - start_time

    # Run GH200-optimized CUDA processing
    print("\nRunning GH200-optimized CUDA processing...")
    start_time = time.time()
    gh200_results = process_polygon_data_with_cuda_gh200(
        symbols, data_type=data_type)
    gh200_time = time.time() - start_time

    # Calculate speedup for each symbol
    speedups = {}
    for symbol in gh200_results:
        if symbol in original_results:
            if 'indicators_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['indicators_time']
                gh200_time = gh200_results[symbol]['indicators_time']
                speedups[f"{symbol}_indicators"] = orig_time / \
                    gh200_time if gh200_time > 0 else 0

            if 'macd_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['macd_time']
                gh200_time = gh200_results[symbol]['macd_time']
                speedups[f"{symbol}_macd"] = orig_time / \
                    gh200_time if gh200_time > 0 else 0

            if 'order_book_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['order_book_time']
                gh200_time = gh200_results[symbol]['order_book_time']
                speedups[f"{symbol}_order_book"] = orig_time / \
                    gh200_time if gh200_time > 0 else 0

    # Calculate average speedup
    avg_speedup = sum(speedups.values()) / len(speedups) if speedups else 0

    print("\n" + "="*70)
    print("CUDA PROCESSING SUMMARY")
    print("="*70)
    print(f"Data type: {data_type}")
    print(f"Symbols: {symbols}")
    print(f"Original processing time: {original_time:.2f} seconds")
    print(f"GH200 processing time: {gh200_time:.2f} seconds")
    print(f"Overall speedup: {original_time / gh200_time:.2f}x")
    print(f"Average operation speedup: {avg_speedup:.2f}x")

    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbols': symbols,
        'data_type': data_type,
        'original_results': original_results,
        'gh200_results': gh200_results,
        'original_time': original_time,
        'gh200_time': gh200_time,
        'speedups': speedups,
        'avg_speedup': avg_speedup
    }


def plot_benchmark_results(benchmark_results, cuda_results):
    """
    Plot benchmark results

    Args:
        benchmark_results: Results from run_benchmark
        cuda_results: Results from run_cuda_benchmark
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Throughput comparison
    axs[0, 0].bar(['Original', 'GH200 Optimized'],
                  [benchmark_results['avg_original_throughput'],
                   benchmark_results['avg_gh200_throughput']],
                  color=['#1f77b4', '#ff7f0e'])
    axs[0, 0].set_title('Data Processing Throughput')
    axs[0, 0].set_ylabel('Records/second')
    axs[0, 0].text(0.5, 0.9, f"Speedup: {benchmark_results['avg_speedup']:.2f}x",
                   transform=axs[0, 0].transAxes, ha='center')

    # Plot 2: Processing time comparison
    axs[0, 1].bar(['Original', 'GH200 Optimized'],
                  [cuda_results['original_time'],
                   cuda_results['gh200_time']],
                  color=['#1f77b4', '#ff7f0e'])
    axs[0, 1].set_title('CUDA Processing Time')
    axs[0, 1].set_ylabel('Seconds')
    axs[0, 1].text(0.5, 0.9, f"Speedup: {cuda_results['original_time'] / cuda_results['gh200_time']:.2f}x",
                   transform=axs[0, 1].transAxes, ha='center')

    # Plot 3: Operation speedups
    operations = list(cuda_results['speedups'].keys())
    speedups = list(cuda_results['speedups'].values())

    if operations:
        # Sort by speedup value
        sorted_indices = np.argsort(speedups)
        sorted_operations = [operations[i] for i in sorted_indices]
        sorted_speedups = [speedups[i] for i in sorted_indices]

        axs[1, 0].barh(sorted_operations, sorted_speedups, color='#2ca02c')
        axs[1, 0].set_title('Operation Speedups')
        axs[1, 0].set_xlabel('Speedup (x)')
        axs[1, 0].axvline(x=1.0, color='r', linestyle='--')

    # Plot 4: Memory usage (if available)
    # This is a placeholder - in a real implementation, you would track memory usage
    axs[1, 1].text(0.5, 0.5, "Memory usage data not available",
                   transform=axs[1, 1].transAxes, ha='center')
    axs[1, 1].set_title('Memory Usage')

    # Add overall title
    plt.suptitle(f"Polygon.io Data Processing on GH200 - {benchmark_results['data_type'].upper()} Data",
                 fontsize=16)

    plt.suptitle(f"Benchmark run on {benchmark_results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
                 fontsize=10, y=0.01)
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"polygon_benchmark_{benchmark_results['data_type']}.png")
    print(
        f"Benchmark results saved to polygon_benchmark_{benchmark_results['data_type']}.png")

    # Also save results as CSV
    results_df = pd.DataFrame({
        'Metric': [
            'Original Throughput (records/s)',
            'GH200 Throughput (records/s)',
            'Throughput Speedup',
            'Original Processing Time (s)',
            'GH200 Processing Time (s)',
            'Processing Time Speedup',
            'Average Operation Speedup'
        ],
        'Value': [
            benchmark_results['avg_original_throughput'],
            benchmark_results['avg_gh200_throughput'],
            benchmark_results['avg_speedup'],
            cuda_results['original_time'],
            cuda_results['gh200_time'],
            cuda_results['original_time'] / cuda_results['gh200_time'],
            cuda_results['avg_speedup']
        ]
    })

    results_df.to_csv(
        f"polygon_benchmark_{benchmark_results['data_type']}_results.csv", index=False)
    print(
        f"Benchmark results saved to polygon_benchmark_{benchmark_results['data_type']}_results.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Polygon.io Data Benchmark on GH200')
    parser.add_argument('--symbols', type=str, default="AAPL,MSFT,GOOGL,AMZN,TSLA",
                        help='Comma-separated list of symbols to use')
    parser.add_argument('--data-type', type=str, default="aggregates",
                        choices=['aggregates', 'order_book'],
                        help='Type of data to benchmark')
    parser.add_argument('--timespan', type=str, default="minute",
                        choices=['minute', 'hour', 'day', 'week', 'month'],
                        help='Timespan for aggregates data')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting of results')
    parser.add_argument('--compare', action='store_true',
                        help='Run a comprehensive comparison of standard and GH200 implementations')

    args = parser.parse_args()

    # Parse symbols
    symbols = args.symbols.split(',')

    # Run benchmarks
    if args.compare:
        print("\n" + "="*70)
        print("COMPREHENSIVE BENCHMARK COMPARISON")
        print("="*70)
        print(
            f"Running comparison with {len(symbols)} symbols: {', '.join(symbols)}")
        print(
            f"Data type: {args.data_type}, Timespan: {args.timespan}, Runs: {args.num_runs}")

        benchmark_results = run_benchmark(
            symbols=symbols,
            data_type=args.data_type,
            timespan=args.timespan,
            num_runs=args.num_runs
        )

        cuda_results = run_cuda_benchmark(
            symbols=symbols,
            data_type=args.data_type,
            timespan=args.timespan
        )
    else:
        # Run standard benchmark
        benchmark_results = run_benchmark(
            symbols=symbols,
            data_type=args.data_type,
            timespan=args.timespan,
            num_runs=args.num_runs
        )

        cuda_results = run_cuda_benchmark(
            symbols=symbols,
            data_type=args.data_type,
            timespan=args.timespan
        )

    # Plot results
    if not args.no_plot:
        plot_benchmark_results(benchmark_results, cuda_results)

        # Print summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(
            f"Data throughput speedup: {benchmark_results['avg_speedup']:.2f}x")
        print(
            f"Processing time speedup: {cuda_results['original_time'] / cuda_results['gh200_time']:.2f}x")
        print(f"Average operation speedup: {cuda_results['avg_speedup']:.2f}x")
        print("\nResults saved to:")
        print(f"- polygon_benchmark_{args.data_type}.png")
        print(f"- polygon_benchmark_{args.data_type}_results.csv")


if __name__ == "__main__":
    main()
