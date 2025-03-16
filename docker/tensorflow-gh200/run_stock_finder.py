#!/usr/bin/env python3
"""
Run Dynamic Stock Finder
This script demonstrates the dynamic stock finder's capabilities
by scanning through different universe sizes to find optimal stocks
for a $2,500 position.
"""

import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dynamic_stock_finder import DynamicStockFinder, StockScreeningCriteria


def run_universe_size_test(position_size=2500.0, max_universe=5000, step=1000):
    """
    Test how many stocks we need to scan to find optimal candidates

    Args:
        position_size: Position size in dollars
        max_universe: Maximum universe size to test
        step: Step size for universe size
    """
    print("\n" + "="*80)
    print(f"UNIVERSE SIZE TEST FOR ${position_size:.2f} POSITION")
    print("="*80)

    # Create stock finder
    finder = DynamicStockFinder(position_size=position_size)

    # Test different universe sizes
    results = []
    universe_sizes = list(range(step, max_universe + step, step))

    for universe_size in universe_sizes:
        print(f"\nTesting universe size: {universe_size}")

        # Find stocks
        start_time = time.time()
        candidates = finder.find_stocks(
            universe_size=universe_size, top_candidates=10)
        elapsed = time.time() - start_time

        # Calculate statistics
        avg_score = sum(
            candidate["score"] for candidate in candidates) / len(candidates) if candidates else 0
        max_score = max(candidate["score"]
                        for candidate in candidates) if candidates else 0

        # Add to results
        results.append({
            "universe_size": universe_size,
            "candidates_found": len(candidates),
            "processing_time": elapsed,
            "stocks_per_second": universe_size / elapsed,
            "avg_score": avg_score,
            "max_score": max_score
        })

        print(f"Found {len(candidates)} candidates in {elapsed:.2f} seconds")
        print(f"Processing rate: {universe_size/elapsed:.2f} stocks/second")
        print(f"Average score: {avg_score:.2f}")
        print(f"Maximum score: {max_score:.2f}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv("universe_size_test_results.csv", index=False)

    # Create charts
    plt.figure(figsize=(15, 10))

    # Score vs Universe Size
    plt.subplot(2, 2, 1)
    plt.plot(df["universe_size"], df["avg_score"],
             marker='o', label="Average Score")
    plt.plot(df["universe_size"], df["max_score"],
             marker='s', label="Maximum Score")
    plt.title("Score vs Universe Size")
    plt.xlabel("Universe Size (# of stocks)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Processing Time vs Universe Size
    plt.subplot(2, 2, 2)
    plt.plot(df["universe_size"], df["processing_time"], marker='o')
    plt.title("Processing Time vs Universe Size")
    plt.xlabel("Universe Size (# of stocks)")
    plt.ylabel("Processing Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Candidates Found vs Universe Size
    plt.subplot(2, 2, 3)
    plt.plot(df["universe_size"], df["candidates_found"], marker='o')
    plt.title("Candidates Found vs Universe Size")
    plt.xlabel("Universe Size (# of stocks)")
    plt.ylabel("Candidates Found")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Processing Rate vs Universe Size
    plt.subplot(2, 2, 4)
    plt.plot(df["universe_size"], df["stocks_per_second"], marker='o')
    plt.title("Processing Rate vs Universe Size")
    plt.xlabel("Universe Size (# of stocks)")
    plt.ylabel("Stocks per Second")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("universe_size_test_results.png")

    # Print summary
    print("\n" + "="*80)
    print("UNIVERSE SIZE TEST SUMMARY")
    print("="*80)
    print(
        f"Optimal universe size: {df.loc[df['max_score'].idxmax(), 'universe_size']}")
    print(f"Maximum score: {df['max_score'].max():.2f}")
    print(
        f"Average processing rate: {df['stocks_per_second'].mean():.2f} stocks/second")

    # Close resources
    finder.close()

    return df


def run_criteria_sensitivity_test(position_size=2500.0, universe_size=2000):
    """
    Test sensitivity of different screening criteria

    Args:
        position_size: Position size in dollars
        universe_size: Universe size to test
    """
    print("\n" + "="*80)
    print(f"CRITERIA SENSITIVITY TEST FOR ${position_size:.2f} POSITION")
    print("="*80)

    # Create stock finder
    finder = DynamicStockFinder(position_size=position_size)

    # Define criteria variations to test
    criteria_variations = [
        {"name": "Default", "changes": {}},
        {"name": "Low Price", "changes": {"min_price": 1.0, "max_price": 50.0}},
        {"name": "High Price", "changes": {"min_price": 50.0, "max_price": 1000.0}},
        {"name": "High Volume", "changes": {"min_avg_volume": 1000000}},
        {"name": "Low Volatility", "changes": {
            "min_volatility": 0.005, "max_volatility": 0.02}},
        {"name": "High Volatility", "changes": {
            "min_volatility": 0.03, "max_volatility": 0.10}},
        {"name": "Oversold", "changes": {
            "rsi_lower_bound": 20, "rsi_upper_bound": 40}},
        {"name": "Overbought", "changes": {
            "rsi_lower_bound": 60, "rsi_upper_bound": 80}},
        {"name": "High Momentum", "changes": {"min_momentum": 0.05}}
    ]

    # Test each criteria variation
    results = []

    for variation in criteria_variations:
        print(f"\nTesting criteria variation: {variation['name']}")

        # Create custom criteria
        criteria = StockScreeningCriteria()
        for key, value in variation["changes"].items():
            setattr(criteria, key, value)

        # Update finder criteria
        finder.criteria = criteria

        # Find stocks
        start_time = time.time()
        candidates = finder.find_stocks(
            universe_size=universe_size, top_candidates=10)
        elapsed = time.time() - start_time

        # Calculate statistics
        avg_score = sum(
            candidate["score"] for candidate in candidates) / len(candidates) if candidates else 0
        max_score = max(candidate["score"]
                        for candidate in candidates) if candidates else 0
        avg_price = sum(
            candidate["price"] for candidate in candidates) / len(candidates) if candidates else 0
        avg_shares = sum(
            candidate["shares"] for candidate in candidates) / len(candidates) if candidates else 0

        # Add to results
        results.append({
            "criteria": variation["name"],
            "candidates_found": len(candidates),
            "processing_time": elapsed,
            "avg_score": avg_score,
            "max_score": max_score,
            "avg_price": avg_price,
            "avg_shares": avg_shares
        })

        print(f"Found {len(candidates)} candidates in {elapsed:.2f} seconds")
        print(f"Average score: {avg_score:.2f}")
        print(f"Maximum score: {max_score:.2f}")
        print(f"Average price: ${avg_price:.2f}")
        print(f"Average shares: {avg_shares:.2f}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv("criteria_sensitivity_test_results.csv", index=False)

    # Create charts
    plt.figure(figsize=(15, 10))

    # Score by Criteria
    plt.subplot(2, 2, 1)
    plt.bar(df["criteria"], df["avg_score"], alpha=0.7, label="Average Score")
    plt.bar(df["criteria"], df["max_score"], alpha=0.5, label="Maximum Score")
    plt.title("Score by Criteria")
    plt.xlabel("Criteria")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Candidates Found by Criteria
    plt.subplot(2, 2, 2)
    plt.bar(df["criteria"], df["candidates_found"])
    plt.title("Candidates Found by Criteria")
    plt.xlabel("Criteria")
    plt.ylabel("Candidates Found")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Average Price by Criteria
    plt.subplot(2, 2, 3)
    plt.bar(df["criteria"], df["avg_price"])
    plt.title("Average Price by Criteria")
    plt.xlabel("Criteria")
    plt.ylabel("Average Price ($)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Average Shares by Criteria
    plt.subplot(2, 2, 4)
    plt.bar(df["criteria"], df["avg_shares"])
    plt.title("Average Shares by Criteria")
    plt.xlabel("Criteria")
    plt.ylabel("Average Shares")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("criteria_sensitivity_test_results.png")

    # Print summary
    print("\n" + "="*80)
    print("CRITERIA SENSITIVITY TEST SUMMARY")
    print("="*80)
    print(f"Best criteria: {df.loc[df['max_score'].idxmax(), 'criteria']}")
    print(f"Maximum score: {df['max_score'].max():.2f}")

    # Close resources
    finder.close()

    return df


def run_position_size_test(min_position=1000.0, max_position=10000.0, step=1000.0, universe_size=2000):
    """
    Test different position sizes

    Args:
        min_position: Minimum position size in dollars
        max_position: Maximum position size in dollars
        step: Step size for position size
        universe_size: Universe size to test
    """
    print("\n" + "="*80)
    print(f"POSITION SIZE TEST")
    print("="*80)

    # Test different position sizes
    results = []
    position_sizes = [p for p in range(
        int(min_position), int(max_position) + int(step), int(step))]

    for position_size in position_sizes:
        print(f"\nTesting position size: ${position_size:.2f}")

        # Create stock finder
        finder = DynamicStockFinder(position_size=position_size)

        # Find stocks
        start_time = time.time()
        candidates = finder.find_stocks(
            universe_size=universe_size, top_candidates=10)
        elapsed = time.time() - start_time

        # Calculate statistics
        avg_score = sum(
            candidate["score"] for candidate in candidates) / len(candidates) if candidates else 0
        max_score = max(candidate["score"]
                        for candidate in candidates) if candidates else 0
        avg_price = sum(
            candidate["price"] for candidate in candidates) / len(candidates) if candidates else 0
        avg_shares = sum(
            candidate["shares"] for candidate in candidates) / len(candidates) if candidates else 0
        total_shares = sum(candidate["shares"] for candidate in candidates)

        # Add to results
        results.append({
            "position_size": position_size,
            "candidates_found": len(candidates),
            "processing_time": elapsed,
            "avg_score": avg_score,
            "max_score": max_score,
            "avg_price": avg_price,
            "avg_shares": avg_shares,
            "total_shares": total_shares
        })

        print(f"Found {len(candidates)} candidates in {elapsed:.2f} seconds")
        print(f"Average score: {avg_score:.2f}")
        print(f"Maximum score: {max_score:.2f}")
        print(f"Average price: ${avg_price:.2f}")
        print(f"Average shares: {avg_shares:.2f}")
        print(f"Total shares: {total_shares}")

        # Close resources
        finder.close()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv("position_size_test_results.csv", index=False)

    # Create charts
    plt.figure(figsize=(15, 10))

    # Total Shares vs Position Size
    plt.subplot(2, 2, 1)
    plt.plot(df["position_size"], df["total_shares"], marker='o')
    plt.title("Total Shares vs Position Size")
    plt.xlabel("Position Size ($)")
    plt.ylabel("Total Shares")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Average Shares vs Position Size
    plt.subplot(2, 2, 2)
    plt.plot(df["position_size"], df["avg_shares"], marker='o')
    plt.title("Average Shares vs Position Size")
    plt.xlabel("Position Size ($)")
    plt.ylabel("Average Shares")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Score vs Position Size
    plt.subplot(2, 2, 3)
    plt.plot(df["position_size"], df["avg_score"],
             marker='o', label="Average Score")
    plt.plot(df["position_size"], df["max_score"],
             marker='s', label="Maximum Score")
    plt.title("Score vs Position Size")
    plt.xlabel("Position Size ($)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Average Price vs Position Size
    plt.subplot(2, 2, 4)
    plt.plot(df["position_size"], df["avg_price"], marker='o')
    plt.title("Average Price vs Position Size")
    plt.xlabel("Position Size ($)")
    plt.ylabel("Average Price ($)")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("position_size_test_results.png")

    # Print summary
    print("\n" + "="*80)
    print("POSITION SIZE TEST SUMMARY")
    print("="*80)
    print(
        f"Optimal position size: ${df.loc[df['max_score'].idxmax(), 'position_size']:.2f}")
    print(f"Maximum score: {df['max_score'].max():.2f}")
    print(
        f"Total shares at optimal position: {df.loc[df['max_score'].idxmax(), 'total_shares']}")

    return df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run Dynamic Stock Finder Tests")
    parser.add_argument("--test", choices=["universe", "criteria", "position", "all"], default="all",
                        help="Test to run (default: all)")
    parser.add_argument("--position-size", type=float, default=2500.0,
                        help="Position size in dollars (default: 2500.0)")
    parser.add_argument("--universe-size", type=int, default=2000,
                        help="Universe size to test (default: 2000)")

    args = parser.parse_args()

    if args.test == "universe" or args.test == "all":
        run_universe_size_test(
            position_size=args.position_size, max_universe=args.universe_size)

    if args.test == "criteria" or args.test == "all":
        run_criteria_sensitivity_test(
            position_size=args.position_size, universe_size=args.universe_size)

    if args.test == "position" or args.test == "all":
        run_position_size_test(universe_size=args.universe_size)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
