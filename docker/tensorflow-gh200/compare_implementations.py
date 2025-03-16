import tensorflow as tf
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta

# Import both implementations
from polygon_data_source import process_polygon_data_with_cuda as original_process
from polygon_data_source_gh200_fixed import process_polygon_data_with_cuda_gh200 as fixed_process


def run_comparison(symbols=None, data_type="aggregates"):
    """
    Run a comparison between the original and fixed implementations
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("\n" + "="*70)
    print("POLYGON.IO IMPLEMENTATION COMPARISON")
    print("="*70)

    # Run original implementation
    print("\nRunning original implementation...")
    try:
        start_time = time.time()
        original_results = original_process(symbols, data_type=data_type)
        original_time = time.time() - start_time
        print(
            f"Original implementation completed in {original_time:.2f} seconds")
        original_success = True
    except Exception as e:
        print(f"Original implementation failed with error: {e}")
        original_results = {}
        original_time = 0
        original_success = False

    # Run fixed implementation
    print("\nRunning fixed implementation...")
    try:
        start_time = time.time()
        fixed_results = fixed_process(symbols, data_type=data_type)
        fixed_time = time.time() - start_time
        print(f"Fixed implementation completed in {fixed_time:.2f} seconds")
        fixed_success = True
    except Exception as e:
        print(f"Fixed implementation failed with error: {e}")
        fixed_results = {}
        fixed_time = 0
        fixed_success = False

    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    if original_success and fixed_success:
        speedup = original_time / fixed_time if fixed_time > 0 else 0
        print(f"Overall speedup: {speedup:.2f}x")

        # Compare detailed metrics
        for symbol in fixed_results:
            if symbol in original_results:
                print(f"\nSymbol: {symbol}")

                if 'indicators_time' in fixed_results[symbol] and 'indicators_time' in original_results[symbol]:
                    orig_time = original_results[symbol]['indicators_time']
                    fixed_time = fixed_results[symbol]['indicators_time']
                    speedup = orig_time / fixed_time if fixed_time > 0 else 0
                    print(
                        f"  Technical indicators: {orig_time:.4f}s vs {fixed_time:.4f}s (Speedup: {speedup:.2f}x)")

                if 'macd_time' in fixed_results[symbol] and 'macd_time' in original_results[symbol]:
                    orig_time = original_results[symbol]['macd_time']
                    fixed_time = fixed_results[symbol]['macd_time']
                    speedup = orig_time / fixed_time if fixed_time > 0 else 0
                    print(
                        f"  MACD calculation: {orig_time:.4f}s vs {fixed_time:.4f}s (Speedup: {speedup:.2f}x)")

                if 'order_book_time' in fixed_results[symbol] and 'order_book_time' in original_results[symbol]:
                    orig_time = original_results[symbol]['order_book_time']
                    fixed_time = fixed_results[symbol]['order_book_time']
                    speedup = orig_time / fixed_time if fixed_time > 0 else 0
                    print(
                        f"  Order book processing: {orig_time:.4f}s vs {fixed_time:.4f}s (Speedup: {speedup:.2f}x)")
    elif fixed_success and not original_success:
        print("The fixed implementation works while the original implementation fails.")
        print("This is a significant improvement in reliability.")

        # Show fixed implementation metrics
        for symbol in fixed_results:
            print(f"\nSymbol: {symbol}")

            if 'indicators_time' in fixed_results[symbol]:
                print(
                    f"  Technical indicators: {fixed_results[symbol]['indicators_time']:.4f}s")

            if 'macd_time' in fixed_results[symbol]:
                print(
                    f"  MACD calculation: {fixed_results[symbol]['macd_time']:.4f}s")

            if 'order_book_time' in fixed_results[symbol]:
                print(
                    f"  Order book processing: {fixed_results[symbol]['order_book_time']:.4f}s")
    elif original_success and not fixed_success:
        print("The original implementation works while the fixed implementation fails.")
        print("Further debugging is needed for the fixed implementation.")
    else:
        print("Both implementations failed. Further debugging is needed.")


if __name__ == "__main__":
    # Parse command line arguments
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data_type = "aggregates"

    if len(sys.argv) > 1:
        symbols_arg = sys.argv[1]
        symbols = symbols_arg.split(',')

    if len(sys.argv) > 2:
        data_type = sys.argv[2]

    # Run comparison
    run_comparison(symbols, data_type)
