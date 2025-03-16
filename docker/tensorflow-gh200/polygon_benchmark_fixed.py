import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# Import both implementations
from polygon_data_source import benchmark_polygon_data, process_polygon_data_with_cuda
from polygon_data_source_gh200_fixed import benchmark_polygon_data_gh200, process_polygon_data_with_cuda_gh200


def run_benchmark(symbols=None, data_type="aggregates", num_runs=1, compare=False):
    """
    Run benchmark for Polygon.io data processing
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Convert string to list if needed
    if isinstance(symbols, str):
        symbols = symbols.split(',')

    print("\n" + "="*70)
    print(f"POLYGON.IO DATA BENCHMARK ON GH200 - {data_type.upper()} DATA")
    print("="*70)

    original_throughputs = []
    gh200_throughputs = []

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")

        # Run original implementation
        print("\nRunning original implementation...")
        original_benchmark = benchmark_polygon_data(
            num_symbols=len(symbols), data_type=data_type)
        original_throughputs.append(original_benchmark['throughput'])

        # Run GH200-optimized implementation
        print("\nRunning GH200-optimized implementation...")
        gh200_benchmark = benchmark_polygon_data_gh200(
            num_symbols=len(symbols), data_type=data_type)
        gh200_throughputs.append(gh200_benchmark['throughput'])

    # Calculate average throughputs
    avg_original = sum(original_throughputs) / len(original_throughputs)
    avg_gh200 = sum(gh200_throughputs) / len(gh200_throughputs)
    avg_speedup = avg_gh200 / avg_original if avg_original > 0 else 0

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Data type: {data_type}")
    print(f"Symbols: {symbols}")
    print(f"Average original throughput: {avg_original:.2f} records/second")
    print(f"Average GH200 throughput: {avg_gh200:.2f} records/second")
    print(f"Average speedup: {avg_speedup:.2f}x")

    # Generate comparison chart if requested
    if compare and num_runs > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_runs+1), original_throughputs,
                 'b-o', label='Original')
        plt.plot(range(1, num_runs+1), gh200_throughputs, 'r-o', label='GH200')
        plt.xlabel('Run')
        plt.ylabel('Throughput (records/second)')
        plt.title(f'Polygon.io {data_type} Data Processing Throughput')
        plt.legend()
        plt.grid(True)

        # Save chart
        chart_path = f"polygon_benchmark_{data_type}.png"
        plt.savefig(chart_path)
        print(f"\nChart saved to {chart_path}")

        # Save results to CSV
        csv_path = f"polygon_benchmark_{data_type}_results.csv"
        with open(csv_path, 'w') as f:
            f.write("Run,Original,GH200,Speedup\n")
            for i in range(num_runs):
                speedup = gh200_throughputs[i] / \
                    original_throughputs[i] if original_throughputs[i] > 0 else 0
                f.write(
                    f"{i+1},{original_throughputs[i]:.2f},{gh200_throughputs[i]:.2f},{speedup:.2f}\n")
            f.write(
                f"Average,{avg_original:.2f},{avg_gh200:.2f},{avg_speedup:.2f}\n")
        print(f"Results saved to {csv_path}")

    return {
        'original_throughputs': original_throughputs,
        'gh200_throughputs': gh200_throughputs,
        'avg_original': avg_original,
        'avg_gh200': avg_gh200,
        'avg_speedup': avg_speedup
    }


def run_cuda_benchmark(symbols=None, data_type="aggregates"):
    """
    Run CUDA processing benchmark for Polygon.io data
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Convert string to list if needed
    if isinstance(symbols, str):
        symbols = symbols.split(',')

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

    # Calculate overall speedup
    speedup = original_time / gh200_time if gh200_time > 0 else 0

    # Print summary
    print("\n" + "="*70)
    print("CUDA PROCESSING SUMMARY")
    print("="*70)
    print(f"Data type: {data_type}")
    print(f"Symbols: {symbols}")
    print(f"Original processing time: {original_time:.2f} seconds")
    print(f"GH200 processing time: {gh200_time:.2f} seconds")
    print(f"Overall speedup: {speedup:.2f}x")

    # Compare detailed performance
    print("\n" + "="*70)
    print("DETAILED PERFORMANCE COMPARISON")
    print("="*70)

    for symbol in gh200_results:
        if symbol in original_results:
            print(f"\nSymbol: {symbol}")

            if 'indicators_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['indicators_time']
                gh200_time = gh200_results[symbol]['indicators_time']
                ind_speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  Technical indicators: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {ind_speedup:.2f}x)")

            if 'macd_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['macd_time']
                gh200_time = gh200_results[symbol]['macd_time']
                macd_speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  MACD calculation: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {macd_speedup:.2f}x)")

            if 'order_book_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['order_book_time']
                gh200_time = gh200_results[symbol]['order_book_time']
                ob_speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  Order book processing: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {ob_speedup:.2f}x)")

    return {
        'original_results': original_results,
        'gh200_results': gh200_results,
        'original_time': original_time,
        'gh200_time': gh200_time,
        'speedup': speedup
    }


def main():
    """
    Main function to run benchmarks
    """
    parser = argparse.ArgumentParser(description='Polygon.io Data Benchmark')
    parser.add_argument('--symbols', type=str, default="AAPL,MSFT,GOOGL,AMZN,TSLA",
                        help='Comma-separated list of symbols to benchmark')
    parser.add_argument('--data-type', type=str, default="aggregates",
                        choices=["aggregates", "order_book"],
                        help='Type of data to benchmark')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of benchmark runs')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison chart')

    args = parser.parse_args()

    # Run data loading benchmark
    benchmark_results = run_benchmark(
        symbols=args.symbols,
        data_type=args.data_type,
        num_runs=args.num_runs,
        compare=args.compare
    )

    # Run CUDA processing benchmark
    cuda_results = run_cuda_benchmark(
        symbols=args.symbols,
        data_type=args.data_type)

    return benchmark_results, cuda_results


if __name__ == "__main__":
    main()
