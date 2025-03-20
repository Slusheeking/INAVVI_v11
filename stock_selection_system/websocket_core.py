#!/usr/bin/env python3
"""
WebSocket Core Module
This module consolidates the functionality from:
- websocket_handlers.py: Handlers for different types of WebSocket messages
- real_time_metrics.py: Calculations and checks for real-time metrics
- subscription_manager.py: Management of WebSocket subscriptions

This consolidation reduces code fragmentation and makes the flow clearer
while maintaining the same functionality.
"""

# First import sys and os to set up the Python path
from collections import deque
import numpy as np
import pytz
import asyncio
import logging
import datetime
import json
import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Print the Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Standard library imports

# Import ml_system.technical_indicators
try:
    from ml_system.technical_indicators import (
        calculate_macd,
        calculate_bollinger_bands,
        calculate_adx,
        calculate_obv
    )
    print("Successfully imported from ml_system.technical_indicators")
except ImportError as e:
    print(f"Import error: {e}")
    # Try to find the ml_system module
    print("\nLooking for ml_system module...")
    for path in sys.path:
        if os.path.exists(os.path.join(path, 'ml_system')):
            print(f"Found ml_system in {path}")
    if __name__ == "__main__":
        print("This module is meant to be imported, not run directly.")
        print("For testing, import it from another module.")
        sys.exit(1)

# Handle package imports differently based on how the module is being used
if __name__ == "__main__":
    # When run directly, use absolute imports
    from stock_selection_system.gpu_optimized_polygon_websocket_client import (
        subscribe_to_trades,
        subscribe_to_quotes,
        subscribe_to_minute_aggs,
        subscribe_to_second_aggs
    )
else:
    # When imported as part of a package, use relative imports
    from .gpu_optimized_polygon_websocket_client import (
        subscribe_to_trades,
        subscribe_to_quotes,
        subscribe_to_minute_aggs,
        subscribe_to_second_aggs
    )

# Configure logging
logger = logging.getLogger('websocket_core')

# Add this at the end of the file
if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("For testing, import it from another module.")
    sys.exit(0)


class WebSocketCore:
    """
    Consolidated WebSocket functionality for the WebSocketEnhancedStockSelection class.

    This class combines the functionality from:
    - WebSocketHandlers: Processing different types of WebSocket messages
    - RealTimeMetrics: Calculating and checking real-time metrics
    - SubscriptionManager: Managing WebSocket subscriptions
    """

    #
    # WebSocket Message Handlers (from websocket_handlers.py)
    #

    @staticmethod
    async def handle_trade_message(self, message):
        """Handle trade messages from WebSocket"""
        try:
            ticker = message.get('sym')
            if not ticker:
                return

            price = message.get('p', 0)
            size = message.get('s', 0)
            timestamp = message.get('t', 0)

            # Skip if missing essential data
            if not price or not size or not timestamp:
                return

            # Process trade data
            async with self.trade_lock:
                # Initialize data structures if needed
                if ticker not in self.real_time_data['trades']:
                    self.real_time_data['trades'][ticker] = []
                    self.data_windows['trades'][ticker] = deque(
                        maxlen=self.window_sizes['volume_spike'])

                # Add to real-time data
                trade_data = {
                    'price': price,
                    'size': size,
                    'timestamp': timestamp,
                    'datetime': datetime.datetime.fromtimestamp(timestamp / 1000.0)
                }

                self.real_time_data['trades'][ticker].append(trade_data)
                self.data_windows['trades'][ticker].append(trade_data)

                # Keep only the last 100 trades to avoid memory issues
                if len(self.real_time_data['trades'][ticker]) > 100:
                    self.real_time_data['trades'][ticker] = self.real_time_data['trades'][ticker][-100:]

                # Update last price in Redis
                self.redis.hset(f"stock:{ticker}:last_trade", mapping={
                    "price": price,
                    "size": size,
                    "timestamp": timestamp
                })

                # Update last price
                self.redis.hset(f"stock:{ticker}:last_price", mapping={
                    "price": price,
                    "timestamp": timestamp
                })

                # Check for volume spike
                await self._check_volume_spike(ticker)

                # Check for price jump
                await self._check_price_jump(ticker, price)

                # Update day trading positions if needed
                if ticker in self.day_trading_candidates:
                    await self._update_day_trading_position(ticker, price)

        except Exception as e:
            logger.error(
                f"Error handling trade message for {ticker}: {str(e)}")

    @staticmethod
    async def handle_quote_message(self, message):
        """Handle quote messages from WebSocket"""
        try:
            ticker = message.get('sym')
            if not ticker:
                return

            bid_price = message.get('bp', 0)
            bid_size = message.get('bs', 0)
            ask_price = message.get('ap', 0)
            ask_size = message.get('as', 0)
            timestamp = message.get('t', 0)

            # Skip if missing essential data
            if not bid_price or not ask_price or not timestamp:
                return

            # Calculate mid price and spread
            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0

            # Process quote data
            async with self.quote_lock:
                # Initialize data structures if needed
                if ticker not in self.real_time_data['quotes']:
                    self.real_time_data['quotes'][ticker] = []
                    self.data_windows['quotes'][ticker] = deque(
                        maxlen=self.window_sizes['spread_change'])

                # Add to real-time data
                quote_data = {
                    'bid_price': bid_price,
                    'bid_size': bid_size,
                    'ask_price': ask_price,
                    'ask_size': ask_size,
                    'mid_price': mid_price,
                    'spread': spread,
                    'spread_percent': spread_percent,
                    'timestamp': timestamp,
                    'datetime': datetime.datetime.fromtimestamp(timestamp / 1000.0)
                }

                self.real_time_data['quotes'][ticker].append(quote_data)
                self.data_windows['quotes'][ticker].append(quote_data)

                # Keep only the last 100 quotes to avoid memory issues
                if len(self.real_time_data['quotes'][ticker]) > 100:
                    self.real_time_data['quotes'][ticker] = self.real_time_data['quotes'][ticker][-100:]

                # Update last quote in Redis
                self.redis.hset(f"stock:{ticker}:last_quote", mapping={
                    "bid_price": bid_price,
                    "bid_size": bid_size,
                    "ask_price": ask_price,
                    "ask_size": ask_size,
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_percent": spread_percent,
                    "timestamp": timestamp
                })

                # Check for spread changes
                await self._check_spread_change(ticker)

        except Exception as e:
            logger.error(
                f"Error handling quote message for {ticker}: {str(e)}")

    @staticmethod
    async def handle_minute_agg_message(self, message):
        """Handle minute aggregate messages from WebSocket"""
        try:
            ticker = message.get('sym')
            if not ticker:
                return

            open_price = message.get('o', 0)
            high_price = message.get('h', 0)
            low_price = message.get('l', 0)
            close_price = message.get('c', 0)
            volume = message.get('v', 0)
            timestamp = message.get('s', 0)  # Start timestamp

            # Skip if missing essential data
            if not open_price or not close_price or not timestamp:
                return

            # Process minute aggregate data
            async with self.agg_lock:
                # Initialize data structures if needed
                if ticker not in self.real_time_data['minute_aggs']:
                    self.real_time_data['minute_aggs'][ticker] = []
                    self.data_windows['minute_aggs'][ticker] = deque(
                        maxlen=self.window_sizes['momentum_shift'])

                # Add to real-time data
                agg_data = {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'timestamp': timestamp,
                    'datetime': datetime.datetime.fromtimestamp(timestamp / 1000.0)
                }

                self.real_time_data['minute_aggs'][ticker].append(agg_data)
                self.data_windows['minute_aggs'][ticker].append(agg_data)

                # Keep only the last 100 minute aggregates to avoid memory issues
                if len(self.real_time_data['minute_aggs'][ticker]) > 100:
                    self.real_time_data['minute_aggs'][ticker] = self.real_time_data['minute_aggs'][ticker][-100:]

                # Update latest candle in Redis
                self.redis.hset(f"stock:{ticker}:latest_candle", mapping={
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "timestamp": timestamp
                })

                # Store in candles hash
                candle_key = f"stock:{ticker}:candles:minute"
                candle_data = {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "timestamp": timestamp
                }
                self.redis.hset(candle_key, timestamp, json.dumps(candle_data))

                # Check for momentum shifts
                await self._check_momentum_shift(ticker)

        except Exception as e:
            logger.error(
                f"Error handling minute aggregate message for {ticker}: {str(e)}")

    @staticmethod
    async def handle_second_agg_message(self, message):
        """Handle second aggregate messages from WebSocket"""
        try:
            ticker = message.get('sym')
            if not ticker:
                return

            open_price = message.get('o', 0)
            high_price = message.get('h', 0)
            low_price = message.get('l', 0)
            close_price = message.get('c', 0)
            volume = message.get('v', 0)
            timestamp = message.get('s', 0)  # Start timestamp

            # Skip if missing essential data
            if not open_price or not close_price or not timestamp:
                return

            # Process second aggregate data
            async with self.agg_lock:
                # Initialize data structures if needed
                if ticker not in self.real_time_data['second_aggs']:
                    self.real_time_data['second_aggs'][ticker] = []

                # Add to real-time data
                agg_data = {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'timestamp': timestamp,
                    'datetime': datetime.datetime.fromtimestamp(timestamp / 1000.0)
                }

                self.real_time_data['second_aggs'][ticker].append(agg_data)

                # Keep only the last 60 second aggregates to avoid memory issues
                if len(self.real_time_data['second_aggs'][ticker]) > 60:
                    self.real_time_data['second_aggs'][ticker] = self.real_time_data['second_aggs'][ticker][-60:]

                # Store in candles hash
                candle_key = f"stock:{ticker}:candles:second"
                candle_data = {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "timestamp": timestamp
                }
                self.redis.hset(candle_key, timestamp, json.dumps(candle_data))

        except Exception as e:
            logger.error(
                f"Error handling second aggregate message for {ticker}: {str(e)}")

    #
    # Real-Time Metrics (from real_time_metrics.py)
    #

    @staticmethod
    async def check_volume_spike(self, ticker):
        """Check for volume spikes in real-time trade data"""
        try:
            if ticker not in self.data_windows['trades'] or len(self.data_windows['trades'][ticker]) < 5:
                return

            # Get recent trades
            recent_trades = list(self.data_windows['trades'][ticker])

            # Calculate average volume of previous trades
            if len(recent_trades) < 10:
                return

            # Get the most recent trade
            latest_trade = recent_trades[-1]

            # Calculate average volume of previous 9 trades
            prev_trades = recent_trades[-10:-1]
            avg_volume = sum(t['size'] for t in prev_trades) / len(prev_trades)

            # Check for volume spike
            if avg_volume > 0 and latest_trade['size'] > avg_volume * self.thresholds['volume_spike']:
                # Add to volume spikes set
                self.real_time_metrics['volume_spikes'].add(ticker)

                # Log the volume spike
                logger.info(
                    f"Volume spike detected for {ticker}: {latest_trade['size']} vs avg {avg_volume:.2f}")

                # Store in Redis
                self.redis.hset(f"stock:{ticker}:alerts", mapping={
                    "volume_spike": json.dumps({
                        "current_volume": latest_trade['size'],
                        "avg_volume": avg_volume,
                        "ratio": latest_trade['size'] / avg_volume,
                        "timestamp": latest_trade['timestamp'],
                        "datetime": datetime.datetime.fromtimestamp(latest_trade['timestamp'] / 1000.0).isoformat()
                    })
                })

                # Publish alert
                self.redis.publish("stock_alerts", json.dumps({
                    "type": "volume_spike",
                    "ticker": ticker,
                    "current_volume": latest_trade['size'],
                    "avg_volume": avg_volume,
                    "ratio": latest_trade['size'] / avg_volume,
                    "timestamp": latest_trade['timestamp']
                }))

        except Exception as e:
            logger.error(f"Error checking volume spike for {ticker}: {str(e)}")

    @staticmethod
    async def check_price_jump(self, ticker, current_price):
        """Check for significant price jumps in real-time trade data"""
        try:
            if ticker not in self.data_windows['trades'] or len(self.data_windows['trades'][ticker]) < 5:
                return

            # Get recent trades
            recent_trades = list(self.data_windows['trades'][ticker])

            # Need at least 2 trades to calculate price change
            if len(recent_trades) < 2:
                return

            # Get the previous trade
            prev_trade = recent_trades[-2]
            prev_price = prev_trade['price']

            # Calculate price change percentage
            if prev_price > 0:
                price_change_pct = abs(
                    (current_price - prev_price) / prev_price * 100)

                # Check for price jump
                if price_change_pct > self.thresholds['price_jump']:
                    # Add to price jumps set
                    self.real_time_metrics['price_jumps'].add(ticker)

                    # Log the price jump
                    direction = "up" if current_price > prev_price else "down"
                    logger.info(
                        f"Price jump detected for {ticker}: {direction} {price_change_pct:.2f}% from {prev_price:.2f} to {current_price:.2f}")

                    # Store in Redis
                    self.redis.hset(f"stock:{ticker}:alerts", mapping={
                        "price_jump": json.dumps({
                            "current_price": current_price,
                            "prev_price": prev_price,
                            "change_pct": price_change_pct,
                            "direction": direction,
                            "timestamp": recent_trades[-1]['timestamp'],
                            "datetime": datetime.datetime.fromtimestamp(recent_trades[-1]['timestamp'] / 1000.0).isoformat()
                        })
                    })

                    # Publish alert
                    self.redis.publish("stock_alerts", json.dumps({
                        "type": "price_jump",
                        "ticker": ticker,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "change_pct": price_change_pct,
                        "direction": direction,
                        "timestamp": recent_trades[-1]['timestamp']
                    }))

        except Exception as e:
            logger.error(f"Error checking price jump for {ticker}: {str(e)}")

    @staticmethod
    async def check_spread_change(self, ticker):
        """Check for significant spread changes in real-time quote data"""
        try:
            if ticker not in self.data_windows['quotes'] or len(self.data_windows['quotes'][ticker]) < 5:
                return

            # Get recent quotes
            recent_quotes = list(self.data_windows['quotes'][ticker])

            # Need at least 5 quotes to calculate spread change
            if len(recent_quotes) < 5:
                return

            # Get the most recent quote
            latest_quote = recent_quotes[-1]

            # Calculate average spread of previous 4 quotes
            prev_quotes = recent_quotes[-5:-1]
            avg_spread_pct = sum(q['spread_percent']
                                 for q in prev_quotes) / len(prev_quotes)

            # Check for spread change
            if avg_spread_pct > 0:
                spread_change_pct = abs(
                    (latest_quote['spread_percent'] - avg_spread_pct) / avg_spread_pct * 100)

                # Check for significant spread change
                if spread_change_pct > self.thresholds['spread_change']:
                    # Add to spread changes set
                    self.real_time_metrics['spread_changes'].add(ticker)

                    # Log the spread change
                    direction = "widened" if latest_quote['spread_percent'] > avg_spread_pct else "narrowed"
                    logger.info(
                        f"Spread change detected for {ticker}: {direction} {spread_change_pct:.2f}% from {avg_spread_pct:.4f}% to {latest_quote['spread_percent']:.4f}%")

                    # Store in Redis
                    self.redis.hset(f"stock:{ticker}:alerts", mapping={
                        "spread_change": json.dumps({
                            "current_spread_pct": latest_quote['spread_percent'],
                            "avg_spread_pct": avg_spread_pct,
                            "change_pct": spread_change_pct,
                            "direction": direction,
                            "timestamp": latest_quote['timestamp'],
                            "datetime": datetime.datetime.fromtimestamp(latest_quote['timestamp'] / 1000.0).isoformat()
                        })
                    })

                    # Publish alert
                    self.redis.publish("stock_alerts", json.dumps({
                        "type": "spread_change",
                        "ticker": ticker,
                        "current_spread_pct": latest_quote['spread_percent'],
                        "avg_spread_pct": avg_spread_pct,
                        "change_pct": spread_change_pct,
                        "direction": direction,
                        "timestamp": latest_quote['timestamp']
                    }))

        except Exception as e:
            logger.error(
                f"Error checking spread change for {ticker}: {str(e)}")

    @staticmethod
    async def check_momentum_shift(self, ticker):
        """Check for momentum shifts in real-time minute aggregate data"""
        try:
            if ticker not in self.data_windows['minute_aggs'] or len(self.data_windows['minute_aggs'][ticker]) < self.window_sizes['momentum_shift']:
                return

            # Get recent minute aggregates
            recent_aggs = list(self.data_windows['minute_aggs'][ticker])

            # Need at least 5 candles to calculate momentum shift
            if len(recent_aggs) < 5:
                return

            # Calculate price change over the last 5 minutes
            start_price = recent_aggs[0]['close']
            end_price = recent_aggs[-1]['close']

            # Calculate momentum (price change percentage)
            if start_price > 0:
                momentum = (end_price - start_price) / start_price * 100

                # Check for significant momentum shift
                if abs(momentum) > self.thresholds['momentum_shift']:
                    # Add to momentum shifts set
                    self.real_time_metrics['momentum_shifts'].add(ticker)

                    # Log the momentum shift
                    direction = "positive" if momentum > 0 else "negative"
                    logger.info(
                        f"Momentum shift detected for {ticker}: {direction} {momentum:.2f}% over the last 5 minutes")

                    # Store in Redis
                    self.redis.hset(f"stock:{ticker}:alerts", mapping={
                        "momentum_shift": json.dumps({
                            "start_price": start_price,
                            "end_price": end_price,
                            "momentum": momentum,
                            "direction": direction,
                            "timestamp": recent_aggs[-1]['timestamp'],
                            "datetime": datetime.datetime.fromtimestamp(recent_aggs[-1]['timestamp'] / 1000.0).isoformat()
                        })
                    })

                    # Publish alert
                    self.redis.publish("stock_alerts", json.dumps({
                        "type": "momentum_shift",
                        "ticker": ticker,
                        "start_price": start_price,
                        "end_price": end_price,
                        "momentum": momentum,
                        "direction": direction,
                        "timestamp": recent_aggs[-1]['timestamp']
                    }))

        except Exception as e:
            logger.error(
                f"Error checking momentum shift for {ticker}: {str(e)}")

    @staticmethod
    async def calculate_real_time_metrics(self, ticker):
        """Calculate real-time metrics for a ticker"""
        try:
            # Skip if not enough data
            if (ticker not in self.real_time_data['trades'] or
                ticker not in self.real_time_data['quotes'] or
                    ticker not in self.real_time_data['minute_aggs']):
                return

            # Get the latest data
            trades = self.real_time_data['trades'][ticker]
            quotes = self.real_time_data['quotes'][ticker]
            minute_aggs = self.real_time_data['minute_aggs'][ticker]

            if not trades or not quotes or not minute_aggs:
                return

            # Calculate real-time VWAP
            total_volume = sum(trade['size'] for trade in trades)
            total_price_volume = sum(
                trade['price'] * trade['size'] for trade in trades)
            vwap = total_price_volume / total_volume if total_volume > 0 else 0

            # Calculate real-time volatility (high-low range as percentage of price)
            if minute_aggs:
                high = max(candle['high'] for candle in minute_aggs)
                low = min(candle['low'] for candle in minute_aggs)
                open_price = minute_aggs[0]['open']
                volatility = (high - low) / open_price * \
                    100 if open_price > 0 else 0
            else:
                volatility = 0

            # Calculate real-time spread
            latest_quote = quotes[-1]
            spread = latest_quote['spread']
            spread_percent = latest_quote['spread_percent']

            # Calculate real-time momentum (1-minute price change)
            if len(minute_aggs) >= 2:
                prev_close = minute_aggs[-2]['close']
                current_close = minute_aggs[-1]['close']
                momentum = (current_close - prev_close) / \
                    prev_close * 100 if prev_close > 0 else 0
            else:
                momentum = 0
                current_close = minute_aggs[-1]['close'] if minute_aggs else 0

            # Calculate additional technical indicators if we have enough data
            additional_metrics = {}

            if len(minute_aggs) >= 26:  # Minimum required for MACD
                try:
                    # Extract price arrays for technical indicators
                    close_prices = np.array(
                        [candle['close'] for candle in minute_aggs])
                    high_prices = np.array([candle['high']
                                           for candle in minute_aggs])
                    low_prices = np.array([candle['low']
                                          for candle in minute_aggs])
                    volumes = np.array([candle['volume']
                                       for candle in minute_aggs])

                    # Calculate MACD
                    macd, macd_signal, macd_hist = calculate_macd(
                        close_prices, fast_period=12, slow_period=26, signal_period=9)

                    # Calculate Bollinger Bands
                    upper, middle, lower = calculate_bollinger_bands(
                        close_prices, period=20, num_std=2)

                    # Calculate ADX
                    adx = calculate_adx(
                        high_prices, low_prices, close_prices, period=14)

                    # Calculate OBV
                    obv = calculate_obv(close_prices, volumes)

                    # Store the most recent values
                    macd_val = macd[-1] if not np.isnan(macd[-1]) else 0
                    macd_signal_val = macd_signal[-1] if not np.isnan(
                        macd_signal[-1]) else 0
                    macd_hist_val = macd_hist[-1] if not np.isnan(
                        macd_hist[-1]) else 0

                    bb_upper_val = upper[-1] if not np.isnan(
                        upper[-1]) else current_close * 1.1
                    bb_middle_val = middle[-1] if not np.isnan(
                        middle[-1]) else current_close
                    bb_lower_val = lower[-1] if not np.isnan(
                        lower[-1]) else current_close * 0.9

                    adx_val = adx[-1] if not np.isnan(adx[-1]) else 0
                    obv_val = obv[-1] if not np.isnan(obv[-1]) else 0

                    # Calculate OBV change
                    if len(obv) >= 5 and obv[-5] != 0:
                        obv_change_val = (obv[-1] - obv[-5]) / abs(obv[-5])
                    else:
                        obv_change_val = 0

                    additional_metrics = {
                        "macd": macd_val,
                        "macd_signal": macd_signal_val,
                        "macd_hist": macd_hist_val,
                        "bb_upper": bb_upper_val,
                        "bb_middle": bb_middle_val,
                        "bb_lower": bb_lower_val,
                        "adx": adx_val,
                        "obv": obv_val,
                        "obv_change": obv_change_val
                    }
                except Exception as e:
                    logger.warning(
                        f"Error calculating advanced indicators for {ticker}: {str(e)}")

            # Store real-time metrics in Redis
            metrics_data = {
                "vwap": vwap,
                "volatility": volatility,
                "spread": spread,
                "spread_percent": spread_percent,
                "momentum": momentum,
                "last_update": datetime.datetime.now().isoformat()
            }

            # Add additional metrics if available
            if additional_metrics:
                metrics_data.update(additional_metrics)

            self.redis.hset(f"stock:{ticker}:real_time", mapping=metrics_data)

            # Calculate real-time score
            volume_score = min(100, total_volume /
                               10000) if total_volume > 0 else 0
            volatility_score = min(
                100, volatility * 10) if volatility > 0 else 0
            spread_score = max(0, 100 - spread_percent *
                               20) if spread_percent > 0 else 100
            momentum_score = min(100, abs(momentum) *
                                 10) if momentum != 0 else 0

            # Additional scores from technical indicators
            macd_score = 0
            bb_score = 0
            adx_score = 0
            obv_score = 0

            if additional_metrics:
                # MACD crossover score (positive histogram is bullish)
                macd_score = min(
                    additional_metrics["macd_hist"] * 20, 10) if additional_metrics["macd_hist"] > 0 else 0

                # Bollinger Band position (middle of the band is safer)
                bb_position = (current_close - additional_metrics["bb_lower"]) / (additional_metrics["bb_upper"] -
                                                                                  additional_metrics["bb_lower"]) if additional_metrics["bb_upper"] > additional_metrics["bb_lower"] else 0.5
                # Highest score when price is in the middle
                bb_score = 10 - abs(bb_position - 0.5) * 20

                # ADX score (trend strength, higher is better up to a point)
                adx_score = min(additional_metrics["adx"] / 5, 10)

                # Volume trend score (OBV change, positive is better)
                obv_score = min(
                    additional_metrics["obv_change"] * 50, 10) if additional_metrics.get("obv_change", 0) > 0 else 0

            # Weighted score
            weights = self.config['weights']
            total_score = (volume_score * weights['volume'] + volatility_score *
                           weights['volatility'] + spread_score * 0.1 + momentum_score * weights['momentum'])

            # Add technical indicator scores if available (with small weights to not overpower the primary metrics)
            if additional_metrics:
                tech_weight = 0.05  # Small weight for technical indicators
                total_score += (max(0, macd_score) * tech_weight + max(0, bb_score) * tech_weight +
                                max(0, adx_score) * tech_weight + max(0, obv_score) * tech_weight)

            # Store real-time score in Redis
            self.redis.hset(f"stock:{ticker}:real_time", mapping={
                "volume_score": volume_score,
                "volatility_score": volatility_score,
                "spread_score": spread_score,
                "momentum_score": momentum_score,
                "macd_score": max(0, macd_score) if additional_metrics else 0,
                "bb_score": max(0, bb_score) if additional_metrics else 0,
                "adx_score": max(0, adx_score) if additional_metrics else 0,
                "obv_score": max(0, obv_score) if additional_metrics else 0,
                "total_score": total_score
            })

            # Update real-time ranking
            self.redis.zadd("real_time:rankings", {ticker: total_score})

        except Exception as e:
            logger.error(
                f"Error calculating real-time metrics for {ticker}: {str(e)}")

    #
    # Subscription Management (from subscription_manager.py)
    #

    @staticmethod
    async def subscription_manager(self):
        """Task to manage WebSocket subscriptions based on active watchlist"""
        logger.info("Starting subscription manager task")

        while self.running:
            try:
                # Check if WebSocket client is available
                if not self.polygon_ws or not self.polygon_ws.connected:
                    await asyncio.sleep(5)
                    continue

                # Get current watchlist and focused list
                watchlist_data = self.redis.zrevrange(
                    "watchlist:active", 0, -1)
                focused_data = self.redis.zrevrange("watchlist:focused", 0, -1)
                day_trading_data = self.redis.zrevrange(
                    "day_trading:active", 0, -1)

                # Convert to sets of strings
                watchlist = {item.decode(
                    'utf-8') if isinstance(item, bytes) else item for item in watchlist_data}
                focused = {item.decode(
                    'utf-8') if isinstance(item, bytes) else item for item in focused_data}
                day_trading = {item.decode(
                    'utf-8') if isinstance(item, bytes) else item for item in day_trading_data}

                # Combine all lists
                all_tickers = watchlist.union(focused).union(day_trading)

                # Determine subscription changes
                for channel_type in ['trades', 'quotes', 'minute_aggs', 'second_aggs']:
                    current_subs = self.active_subscriptions[channel_type]

                    # Determine which tickers to subscribe to
                    if channel_type in ['trades', 'quotes']:
                        # Subscribe to all tickers for trades and quotes
                        target_subs = all_tickers
                    elif channel_type == 'minute_aggs':
                        # Subscribe to all tickers for minute aggregates
                        target_subs = all_tickers
                    elif channel_type == 'second_aggs':
                        # Subscribe only to focused and day trading tickers for second aggregates
                        target_subs = focused.union(day_trading)

                    # Calculate changes
                    to_add = target_subs - current_subs
                    to_remove = current_subs - target_subs

                    # Apply changes
                    if to_add:
                        if channel_type == 'trades':
                            subscribe_to_trades(self.polygon_ws, list(to_add))
                        elif channel_type == 'quotes':
                            subscribe_to_quotes(self.polygon_ws, list(to_add))
                        elif channel_type == 'minute_aggs':
                            subscribe_to_minute_aggs(
                                self.polygon_ws, list(to_add))
                        elif channel_type == 'second_aggs':
                            subscribe_to_second_aggs(
                                self.polygon_ws, list(to_add))

                        logger.info(
                            f"Subscribed to {channel_type} for {len(to_add)} new tickers")

                        # Update active subscriptions
                        self.active_subscriptions[channel_type].update(to_add)

                    if to_remove:
                        # Create channel strings for unsubscribe
                        if channel_type == 'trades':
                            channels = [f"T.{ticker}" for ticker in to_remove]
                        elif channel_type == 'quotes':
                            channels = [f"Q.{ticker}" for ticker in to_remove]
                        elif channel_type == 'minute_aggs':
                            channels = [f"AM.{ticker}" for ticker in to_remove]
                        elif channel_type == 'second_aggs':
                            channels = [f"A.{ticker}" for ticker in to_remove]

                        # Unsubscribe
                        self.polygon_ws.unsubscribe(channels)
                        logger.info(
                            f"Unsubscribed from {channel_type} for {len(to_remove)} tickers")

                        # Update active subscriptions
                        self.active_subscriptions[channel_type] -= to_remove

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                logger.info("Subscription manager task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subscription manager: {str(e)}")
                await asyncio.sleep(30)

    @staticmethod
    async def real_time_stock_selector(self):
        """Task to select stocks based on real-time data"""
        logger.info("Starting real-time stock selector task")

        while self.running:
            try:
                # Get current time (Eastern)
                now = datetime.datetime.now(pytz.timezone('US/Eastern'))

                # Only run during market hours
                if 9 <= now.hour < 16 and now.weekday() < 5:  # 9 AM to 4 PM ET, weekdays only
                    # Get real-time rankings
                    rankings = self.redis.zrevrange(
                        "real_time:rankings", 0, 49, withscores=True)

                    if rankings:
                        # Convert to list of tuples
                        ranked_tickers = [(item[0].decode('utf-8') if isinstance(item[0], bytes) else item[0], item[1])
                                          for item in rankings]

                        # Update real-time watchlist
                        pipeline = self.redis.pipeline()
                        pipeline.delete("watchlist:real_time")

                        for ticker, score in ranked_tickers:
                            pipeline.zadd("watchlist:real_time",
                                          {ticker: score})

                        # Store last update time
                        now_str = datetime.datetime.now().isoformat()
                        pipeline.set(
                            "watchlist:real_time:last_update", now_str)

                        pipeline.execute()

                        logger.info(
                            f"Updated real-time watchlist with {len(ranked_tickers)} stocks")

                        # Check for new day trading opportunities
                        await self._check_real_time_opportunities(ranked_tickers)

                    # Wait before next update
                    await asyncio.sleep(60)  # Update every minute
                else:
                    # Outside market hours, check less frequently
                    await asyncio.sleep(300)  # 5 minutes

            except asyncio.CancelledError:
                logger.info("Real-time stock selector task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in real-time stock selector: {str(e)}")
                await asyncio.sleep(60)

    @staticmethod
    async def real_time_metric_calculator(self):
        """Task to periodically calculate real-time metrics"""
        logger.info("Starting real-time metric calculator task")

        while self.running:
            try:
                # Process each ticker with real-time data
                tickers_to_process = set()

                # Collect tickers from all data types
                for data_type in ['trades', 'quotes', 'minute_aggs', 'second_aggs']:
                    tickers_to_process.update(
                        self.real_time_data[data_type].keys())

                # Process each ticker
                for ticker in tickers_to_process:
                    # Skip if not enough data
                    if (ticker not in self.real_time_data['trades'] or
                        ticker not in self.real_time_data['quotes'] or
                            ticker not in self.real_time_data['minute_aggs']):
                        continue

                    # Calculate real-time metrics
                    await self._calculate_real_time_metrics(ticker)

                # Wait before next calculation
                await asyncio.sleep(5)  # Calculate every 5 seconds

            except asyncio.CancelledError:
                logger.info("Real-time metric calculator task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in real-time metric calculator: {str(e)}")
                await asyncio.sleep(5)
