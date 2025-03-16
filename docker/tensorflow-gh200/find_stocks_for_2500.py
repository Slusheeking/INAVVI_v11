#!/usr/bin/env python3
"""
Find Stocks for $2,500 Position
This script demonstrates how many stocks need to be scanned to find
optimal candidates for a $2,500 position.
"""

from dynamic_stock_finder import DynamicStockFinder, StockScreeningCriteria
import os
import sys
import time
import argparse
import pandas as pd
import logging
import importlib.util

# Set Polygon API key - use a valid API key
# Check if API key is already set in environment, otherwise use placeholder
if 'POLYGON_API_KEY' not in os.environ:
    print("WARNING: POLYGON_API_KEY environment variable not set. Please set it before running this script.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_finder')

# Import tabulate
try:
    from tabulate import tabulate
except ImportError:
    print("Installing tabulate package...")
    os.system("pip install tabulate")
    from tabulate import tabulate

# Import dynamic_stock_finder module directly from the file
spec = importlib.util.spec_from_file_location(
    "dynamic_stock_finder", "docker/tensorflow-gh200/dynamic_stock_finder.py")
dynamic_stock_finder = importlib.util.module_from_spec(spec)
sys.modules["dynamic_stock_finder"] = dynamic_stock_finder
spec.loader.exec_module(dynamic_stock_finder)

# Import polygon_data_source_ultra directly
spec = importlib.util.spec_from_file_location(
    "polygon_data_source_ultra", "docker/tensorflow-gh200/polygon_data_source_ultra.py")
polygon_data_source_ultra = importlib.util.module_from_spec(spec)
sys.modules["polygon_data_source_ultra"] = polygon_data_source_ultra
spec.loader.exec_module(polygon_data_source_ultra)

# Set up the necessary references
dynamic_stock_finder.PolygonDataSourceUltra = polygon_data_source_ultra.PolygonDataSourceUltra

# Import the classes we need


def find_stocks_for_position(position_size=2500.0, universe_size=500, top_candidates=5, debug=False):
    """
    Find stocks for a specific position size

    Args:
        position_size: Position size in dollars
        universe_size: Number of stocks to scan
        top_candidates: Number of top candidates to return
        debug: Enable debug mode for more verbose output
    """
    # Print header with comprehensive information
    print("\n" + "="*80)
    print(
        f"FINDING STOCKS FOR ${position_size:.2f} POSITION WITH OPTIMAL SIZING FROM MASSIVE UNIVERSE")
    print("="*80)
    print(
        f"Scanning {universe_size} stocks to find {top_candidates} candidates...")
    print(
        f"Prioritizing stocks that fit well within the ${position_size:.2f} budget")
    print(f"Ideal range: 5-100 shares per position for optimal diversification")

    # Create stock finder
    finder = DynamicStockFinder(position_size=position_size)

    # Print API key info (masked for security)
    api_key = os.environ.get('POLYGON_API_KEY', '')
    if api_key:
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + \
            api_key[-4:] if len(api_key) > 8 else '****'
        print(f"Using Polygon API key: {masked_key}")
    else:
        print("WARNING: No Polygon API key found in environment variables")

    # Set debug mode if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Find stocks
        start_time = time.time()
        print(
            f"Starting massive scan of {universe_size} stocks across all market sectors and letters...")

        # Get stock universe first to verify we have stocks
        stocks = finder.stock_universe.get_stocks(limit=universe_size)
        print(f"Retrieved {len(stocks)} stocks from universe")
        if len(stocks) > 0:
            print(f"Sample tickers: {', '.join(stocks[:10])}...")

        # Try to find stocks with progressively smaller universe sizes if needed
        candidates = []
        current_size = universe_size

        while not candidates and current_size >= 20:
            print(
                f"Attempting to find stocks with universe size: {current_size}")
            candidates = finder.find_stocks(
                universe_size=current_size, top_candidates=top_candidates)

            if not candidates:
                # Reduce universe size by half and try again
                current_size = current_size // 2
                print(
                    f"No candidates found. Reducing universe size to {current_size} and trying again...")

                if current_size < 20:
                    break  # Stop if universe size gets too small
        elapsed = time.time() - start_time

        # Print results
        print("\n" + "="*80)
        print(
            f"TOP {len(candidates)} STOCK CANDIDATES FOR ${position_size:.2f} POSITION FROM MASSIVE UNIVERSE")
        print("="*80)

        if not candidates:
            print(
                "No candidates found after multiple attempts. Try the following:")
            print("1. Check if the Polygon API key is valid")
            print("2. Try with a smaller universe size (--universe-size 50)")
            print("3. Run with --debug flag for more detailed logging")
            print("4. Check network connectivity to Polygon.io API")
            print("5. Try with different stock tickers directly")
            print("6. Check if the API rate limits have been exceeded")
            return []

        # Create table data
        table_data = []
        for candidate in candidates:
            table_data.append([
                candidate["ticker"],
                f"${candidate['price']:.2f}",
                candidate["shares"],
                f"${candidate['position_value']:.2f}",
                f"{candidate['score']:.2f}",
                f"{candidate['rsi']:.2f}",
                f"{candidate['momentum']*100:.2f}%",
                f"{candidate['volatility']*100:.2f}%"
            ])

        # Print table
        headers = ["Ticker", "Price", "Shares", "Position",
                   "Score", "RSI", "Momentum", "Volatility"]
        print(tabulate(table_data, headers=headers,
              tablefmt="grid", floatfmt=".2f"))

        # Calculate statistics
        total_position = sum(candidate["position_value"]
                             for candidate in candidates)
        avg_position = total_position / len(candidates) if candidates else 0
        total_shares = sum(candidate["shares"] for candidate in candidates)
        avg_shares_per_position = total_shares / \
            len(candidates) if candidates else 0

        print("\nSummary:")
        print(f"Total position value: ${total_position:.2f}")
        print(f"Average position size: ${avg_position:.2f}")
        print(f"Total shares: {total_shares}")
        print(f"Average shares per position: {avg_shares_per_position:.1f}")
        print(f"Scanned {universe_size} stocks in {elapsed:.2f} seconds")
        print(f"Processing rate: {universe_size/elapsed:.2f} stocks/second")
        print(
            f"Position sizing optimized for ${position_size:.2f} budget with diverse stock selection")

        # Print letter distribution
        letter_counts = {}
        for candidate in candidates:
            first_letter = candidate["ticker"][0].upper()
            if first_letter not in letter_counts:
                letter_counts[first_letter] = 0
            letter_counts[first_letter] += 1

        print("\nLetter Distribution:")
        for letter in sorted(letter_counts.keys()):
            print(f"{letter}: {letter_counts[letter]} stocks")

        # Return candidates
        return candidates
    except Exception as e:
        print(f"Error finding stocks: {e}")
        return []
    finally:
        # Close resources
        finder.close()


def run_universe_size_comparison(debug=False):
    """Run comparison of different universe sizes"""
    print("\n" + "="*80)
    print("UNIVERSE SIZE COMPARISON FOR $2,500 POSITION")
    print("="*80)

    universe_sizes = [500, 1000, 2000, 3000, 5000]
    results = []

    # Set debug mode if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    for size in universe_sizes:
        print(f"\nTesting universe size: {size}")

        # Create stock finder
        finder = DynamicStockFinder(position_size=2500.0)

        try:
            # Find stocks
            start_time = time.time()
            candidates = finder.find_stocks(
                universe_size=size, top_candidates=5)
            elapsed = time.time() - start_time

            # Calculate statistics
            avg_score = sum(
                candidate["score"] for candidate in candidates) / len(candidates) if candidates else 0
            max_score = max(candidate["score"]
                            for candidate in candidates) if candidates else 0
            total_shares = sum(candidate["shares"]
                               for candidate in candidates) if candidates else 0

            # Add to results
            results.append({
                "universe_size": size,
                "candidates_found": len(candidates),
                "processing_time": elapsed,
                "stocks_per_second": size / elapsed,
                "avg_score": avg_score,
                "max_score": max_score,
                "total_shares": total_shares
            })

            print(
                f"Found {len(candidates)} candidates in {elapsed:.2f} seconds")
            print(f"Average score: {avg_score:.2f}")
            print(f"Maximum score: {max_score:.2f}")
            print(f"Total shares: {total_shares}")
        except Exception as e:
            print(f"Error testing universe size {size}: {e}")
            results.append({
                "universe_size": size,
                "candidates_found": 0,
                "processing_time": 0,
                "stocks_per_second": 0,
                "avg_score": 0,
                "max_score": 0,
                "total_shares": 0
            })
        finally:
            # Close resources
            finder.close()

    if not results:
        print("No results to display.")
        return []

    # Print comparison table
    print("\n" + "="*80)
    print("UNIVERSE SIZE COMPARISON RESULTS")
    print("="*80)

    # Create table data
    table_data = []
    for result in results:
        table_data.append([
            result["universe_size"],
            result["candidates_found"],
            f"{result['processing_time']:.2f}s",
            f"{result['stocks_per_second']:.2f}",
            f"{result['avg_score']:.2f}",
            f"{result['max_score']:.2f}",
            result["total_shares"]
        ])

    # Print table
    headers = ["Universe Size", "Candidates", "Time",
               "Stocks/Second", "Avg Score", "Max Score", "Total Shares"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Find optimal universe size
    optimal_size = results[0]["universe_size"]
    max_score_value = results[0]["max_score"]

    for result in results:
        if result["max_score"] > max_score_value:
            max_score_value = result["max_score"]
            optimal_size = result["universe_size"]

    print(f"\nOptimal universe size: {optimal_size}")
    print(f"Maximum score: {max_score_value:.2f}")

    # Return results
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Find Stocks for $2,500 Position from a massive diverse universe with optimal position sizing")
    parser.add_argument("--universe-size", type=int, default=50000,
                        help="Number of stocks to scan (default: 50000)")
    parser.add_argument("--position-size", type=float, default=2500.0,
                        help="Position size in dollars (default: 2500.0)")
    parser.add_argument("--top-candidates", type=int, default=50,
                        help="Number of top candidates to return (default: 50)")
    parser.add_argument("--compare", action="store_true",
                        help="Run universe size comparison")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for more verbose output")

    args = parser.parse_args()

    if args.compare:
        run_universe_size_comparison(debug=args.debug)
    else:
        find_stocks_for_position(
            position_size=args.position_size,
            universe_size=args.universe_size,
            top_candidates=args.top_candidates,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
