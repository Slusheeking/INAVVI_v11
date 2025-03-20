#!/usr/bin/env python3
"""
Peak Detection Monitor

This module provides real-time monitoring of price peaks and troughs,
identifying potential trading opportunities based on pattern recognition.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import json
import redis
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('peak_detection_monitor')

# Constants for peak detection
MIN_PEAK_HEIGHT = 0.01  # Minimum height for a peak/trough as percentage
MIN_PEAK_DISTANCE = 5   # Minimum distance between peaks in data points
MIN_PEAK_PROMINENCE = 0.005  # Minimum prominence for peak detection


class PeakDetectionMonitor:
    """Monitor for detecting price peaks and troughs in stock data"""

    def __init__(self, redis_client: redis.Redis = None):
        """Initialize the peak detection monitor"""
        self.redis_client = redis_client
        if not self.redis_client:
            redis_host = os.environ.get('REDIS_HOST', 'localhost')
            redis_port = int(os.environ.get('REDIS_PORT', 6379))
            redis_password = os.environ.get('REDIS_PASSWORD', '')
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )

        # Cache for peak data
        self.peak_cache = {}
        self.running = True
        self.tasks = []

    async def start(self):
        """Start the peak detection monitor"""
        logger.info("Starting peak detection monitor")
        self.running = True

        # Get list of monitored tickers
        monitored_tickers = await self.get_monitored_tickers()

        # Start monitoring task for each ticker
        for ticker in monitored_tickers:
            task = asyncio.create_task(self.monitor_ticker(ticker))
            self.tasks.append(task)

        logger.info(f"Monitoring peaks for {len(monitored_tickers)} tickers")

    async def stop(self):
        """Stop the peak detection monitor"""
        logger.info("Stopping peak detection monitor")
        self.running = False

        # Cancel all running tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []

    async def get_monitored_tickers(self) -> List[str]:
        """Get list of tickers to monitor"""
        try:
            # Try to get tickers from Redis
            tickers_json = self.redis_client.get("monitored_tickers")
            if tickers_json:
                return json.loads(tickers_json)

            # Return default list if no Redis entry
            default_symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK"]
            return default_symbols
        except Exception as e:
            logger.error(f"Error getting monitored tickers: {e}")
            # Return default list on error
            return ["SPY", "QQQ", "IWM", "DIA", "XLK"]

    async def monitor_ticker(self, ticker: str):
        """
        Monitor a specific ticker for peak patterns

        Args:
            ticker: Stock ticker symbol
        """
        logger.info(f"Starting to monitor peaks for {ticker}")

        while self.running:
            try:
                # Get price data
                price_data = await self.get_ticker_price_data(ticker)
                if not price_data or len(price_data) < 20:
                    await asyncio.sleep(15)
                    continue

                # Detect peaks and troughs
                peaks, troughs = self.detect_peaks_and_troughs(price_data)

                # Analyze patterns
                patterns = self.analyze_patterns(
                    ticker, price_data, peaks, troughs)

                # Store results
                if patterns:
                    self.store_patterns(ticker, patterns)

            except Exception as e:
                logger.error(f"Error monitoring peaks for {ticker}: {e}")

            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

    async def get_ticker_price_data(self, ticker: str) -> List[float]:
        """
        Get historical price data for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of closing prices
        """
        try:
            # Try to get from Redis
            key = f"stock:{ticker}:candles:minute"
            candles = self.redis_client.hgetall(key)

            if not candles:
                logger.warning(f"No price data found for {ticker}")
                return []

            # Convert to list of prices
            prices = []
            for timestamp, candle_json in sorted(candles.items()):
                candle = json.loads(candle_json)
                prices.append(candle['close'])

            # Determine if this is the test dataset (ticker with 200 data points)
            is_test_data = (ticker.startswith('SPY') and len(prices) == 200)
            if is_test_data:
                logger.info(
                    f"Using test data for {ticker} with {len(prices)} data points")

            return prices
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return []

    def detect_peaks_and_troughs(self, price_data: List[float]) -> Tuple[List[int], List[int]]:
        """
        Detect peaks and troughs in price data

        Args:
            price_data: List of prices

        Returns:
            Tuple of (peak indices, trough indices)
        """
        try:
            prices = np.array(price_data)
            price_mean = np.mean(prices)
            price_std = np.std(prices)

            # Normalize peaks for detection
            normalized_heights = np.abs(
                prices - price_mean) / price_std if price_std > 0 else np.abs(prices - price_mean)

            # Find peaks
            peak_indices = []
            trough_indices = []

            # Simple peak detection with prominence
            for i in range(1, len(prices) - 1):
                # Peak detection
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    prominence = min(
                        prices[i] - prices[i-1], prices[i] - prices[i+1])
                    if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                        # Check if far enough from previous peak
                        if not peak_indices or i - peak_indices[-1] >= MIN_PEAK_DISTANCE:
                            peak_indices.append(i)

                # Trough detection
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    prominence = min(
                        prices[i-1] - prices[i], prices[i+1] - prices[i])
                    if prominence / prices[i] >= MIN_PEAK_PROMINENCE:
                        # Check if far enough from previous trough
                        if not trough_indices or i - trough_indices[-1] >= MIN_PEAK_DISTANCE:
                            trough_indices.append(i)

            return peak_indices, trough_indices
        except Exception as e:
            logger.error(f"Error detecting peaks: {e}")
            return [], []

    def analyze_patterns(self, ticker: str, price_data: List[float],
                         peaks: List[int], troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Analyze patterns from peaks and troughs

        Args:
            ticker: Stock ticker symbol
            price_data: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of detected patterns
        """
        patterns = []

        if not peaks or not troughs or not price_data:
            return patterns

        try:
            # Find double top pattern
            double_tops = self.find_double_tops(price_data, peaks, troughs)
            patterns.extend(double_tops)

            # Find double bottom pattern
            double_bottoms = self.find_double_bottoms(
                price_data, peaks, troughs)
            patterns.extend(double_bottoms)

            # Find head and shoulders pattern
            head_shoulders = self.find_head_and_shoulders(
                price_data, peaks, troughs)
            patterns.extend(head_shoulders)

            # Add ticker to patterns
            for pattern in patterns:
                pattern['ticker'] = ticker
                pattern['timestamp'] = int(time.time())

            return patterns
        except Exception as e:
            logger.error(f"Error analyzing patterns for {ticker}: {e}")
            return []

    def find_double_tops(self, prices: List[float], peaks: List[int],
                         troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find double top patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of double top patterns
        """
        patterns = []

        if len(peaks) < 2:
            return patterns

        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak1_price = prices[peak1_idx]

            for j in range(i + 1, len(peaks)):
                peak2_idx = peaks[j]
                peak2_price = prices[peak2_idx]

                # Check if peaks are at similar price levels (within 2%)
                price_diff_pct = abs(peak1_price - peak2_price) / peak1_price
                if price_diff_pct > 0.02:
                    continue

                # Check for trough in between
                trough_between = False
                trough_idx = None
                min_price = float('inf')

                for t in troughs:
                    if peak1_idx < t < peak2_idx:
                        trough_between = True
                        if prices[t] < min_price:
                            min_price = prices[t]
                            trough_idx = t

                if not trough_between:
                    continue

                # Calculate confirmation level (neckline)
                neckline = prices[trough_idx]

                # Check if the pattern has enough height (at least 3%)
                height_pct = (peak1_price - neckline) / neckline
                if height_pct < 0.03:
                    continue

                patterns.append({
                    'pattern': 'double_top',
                    'peak1_idx': peak1_idx,
                    'peak1_price': peak1_price,
                    'peak2_idx': peak2_idx,
                    'peak2_price': peak2_price,
                    'trough_idx': trough_idx,
                    'trough_price': prices[trough_idx],
                    'neckline': neckline,
                    'confidence': 0.7 * (1 - price_diff_pct * 10) + 0.3 * min(1, height_pct * 10)
                })

        return patterns

    def find_double_bottoms(self, prices: List[float], peaks: List[int],
                            troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find double bottom patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of double bottom patterns
        """
        patterns = []

        if len(troughs) < 2:
            return patterns

        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough1_price = prices[trough1_idx]

            for j in range(i + 1, len(troughs)):
                trough2_idx = troughs[j]
                trough2_price = prices[trough2_idx]

                # Check if troughs are at similar price levels (within 2%)
                price_diff_pct = abs(
                    trough1_price - trough2_price) / trough1_price
                if price_diff_pct > 0.02:
                    continue

                # Check for peak in between
                peak_between = False
                peak_idx = None
                max_price = float('-inf')

                for p in peaks:
                    if trough1_idx < p < trough2_idx:
                        peak_between = True
                        if prices[p] > max_price:
                            max_price = prices[p]
                            peak_idx = p

                if not peak_between:
                    continue

                # Calculate confirmation level (resistance line)
                resistance = prices[peak_idx]

                # Check if the pattern has enough height (at least 3%)
                height_pct = (resistance - trough1_price) / trough1_price
                if height_pct < 0.03:
                    continue

                patterns.append({
                    'pattern': 'double_bottom',
                    'trough1_idx': trough1_idx,
                    'trough1_price': trough1_price,
                    'trough2_idx': trough2_idx,
                    'trough2_price': trough2_price,
                    'peak_idx': peak_idx,
                    'peak_price': prices[peak_idx],
                    'resistance': resistance,
                    'confidence': 0.7 * (1 - price_diff_pct * 10) + 0.3 * min(1, height_pct * 10)
                })

        return patterns

    def find_head_and_shoulders(self, prices: List[float], peaks: List[int],
                                troughs: List[int]) -> List[Dict[str, Any]]:
        """
        Find head and shoulders patterns

        Args:
            prices: List of prices
            peaks: Indices of price peaks
            troughs: Indices of price troughs

        Returns:
            List of head and shoulders patterns
        """
        patterns = []

        if len(peaks) < 3 or len(troughs) < 2:
            return patterns

        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            left_shoulder_price = prices[left_shoulder_idx]

            for j in range(i + 1, len(peaks) - 1):
                head_idx = peaks[j]
                head_price = prices[head_idx]

                # Head should be higher than left shoulder
                if head_price <= left_shoulder_price:
                    continue

                for k in range(j + 1, len(peaks)):
                    right_shoulder_idx = peaks[k]
                    right_shoulder_price = prices[right_shoulder_idx]

                    # Right shoulder should be similar to left shoulder
                    shoulder_diff_pct = abs(
                        left_shoulder_price - right_shoulder_price) / left_shoulder_price
                    if shoulder_diff_pct > 0.05:
                        continue

                    # Check for troughs between shoulders and head
                    left_trough_idx = None
                    right_trough_idx = None
                    left_trough_price = float('inf')
                    right_trough_price = float('inf')

                    for t in troughs:
                        if left_shoulder_idx < t < head_idx and prices[t] < left_trough_price:
                            left_trough_idx = t
                            left_trough_price = prices[t]
                        elif head_idx < t < right_shoulder_idx and prices[t] < right_trough_price:
                            right_trough_idx = t
                            right_trough_price = prices[t]

                    if left_trough_idx is None or right_trough_idx is None:
                        continue

                    # Calculate neckline as approximate line between troughs
                    neckline = (prices[left_trough_idx] +
                                prices[right_trough_idx]) / 2

                    # Pattern height
                    height_pct = (head_price - neckline) / neckline
                    if height_pct < 0.03:
                        continue

                    patterns.append({
                        'pattern': 'head_and_shoulders',
                        'left_shoulder_idx': left_shoulder_idx,
                        'left_shoulder_price': left_shoulder_price,
                        'head_idx': head_idx,
                        'head_price': head_price,
                        'right_shoulder_idx': right_shoulder_idx,
                        'right_shoulder_price': right_shoulder_price,
                        'left_trough_idx': left_trough_idx,
                        'left_trough_price': prices[left_trough_idx],
                        'right_trough_idx': right_trough_idx,
                        'right_trough_price': prices[right_trough_idx],
                        'neckline': neckline,
                        'confidence': 0.4 * (1 - shoulder_diff_pct * 5) + 0.6 * min(1, height_pct * 10)
                    })

        return patterns

    def store_patterns(self, ticker: str, patterns: List[Dict[str, Any]]):
        """
        Store detected patterns in Redis

        Args:
            ticker: Stock ticker symbol
            patterns: List of detected patterns
        """
        if not patterns:
            return

        try:
            # Store in Redis
            now = int(time.time())
            key = f"patterns:{ticker}"

            pipeline = self.redis_client.pipeline()
            for pattern in patterns:
                pattern_id = f"{pattern['pattern']}:{now}:{pattern['confidence']}"
                pipeline.hset(key, pattern_id, json.dumps(pattern))

            # Set expiration for patterns (1 day)
            pipeline.expire(key, 86400)
            pipeline.execute()

            # Log detection
            logger.info(
                f"Detected {len(patterns)} patterns for {ticker}: " +
                ", ".join(
                    [f"{p['pattern']} ({p['confidence']:.2f})" for p in patterns])
            )
        except Exception as e:
            logger.error(f"Error storing patterns for {ticker}: {e}")
