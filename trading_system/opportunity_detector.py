#!/usr/bin/env python3
"""
Opportunity Detector Module
Contains methods for detecting trading opportunities and managing positions.
"""

import json
import datetime
import logging
import time
import pytz

# Configure logging
logger = logging.getLogger('opportunity_detector')


class OpportunityDetector:
    """Trading opportunity detection and position management for the WebSocketEnhancedStockSelection class"""

    @staticmethod
    async def check_real_time_opportunities(self, ranked_tickers):
        """Check for new day trading opportunities based on real-time data"""
        try:
            # Get current day trading positions
            active_positions = self.redis.zrange(
                "day_trading:active", 0, -1, withscores=True)
            active_tickers = {pos[0].decode('utf-8') if isinstance(pos[0], bytes) else pos[0]
                              for pos in active_positions}

            # Get current time (Eastern)
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))

            # Only look for new opportunities during certain times
            if not (9 <= now.hour < 15) or now.weekday() >= 5:  # 9 AM to 3 PM ET, weekdays only
                return

            # Check if we have capacity for new positions
            max_positions = self.config['day_trading']['max_positions']
            if len(active_tickers) >= max_positions:
                return

            # Get available capital
            max_total_position = self.config['day_trading']['max_total_position']
            used_capital = 0

            for ticker in active_tickers:
                position_data = self.redis.hgetall(
                    f"day_trading:position:{ticker}")
                if position_data:
                    position_value = float(position_data.get(
                        b'position_value', b'0').decode('utf-8'))
                    used_capital += position_value

            available_capital = max_total_position - used_capital

            # If we have less than $500 available, don't open new positions
            if available_capital < 500:
                return

            # Look for new opportunities
            for ticker, score in ranked_tickers:
                # Skip if already in active positions
                if ticker in active_tickers:
                    continue

                # Check if ticker has real-time data
                if (ticker not in self.real_time_data['trades'] or
                    ticker not in self.real_time_data['quotes'] or
                        ticker not in self.real_time_data['minute_aggs']):
                    continue

                # Check if ticker has real-time alerts
                has_volume_spike = ticker in self.real_time_metrics['volume_spikes']
                has_price_jump = ticker in self.real_time_metrics['price_jumps']
                has_momentum_shift = ticker in self.real_time_metrics['momentum_shifts']

                # Only consider tickers with at least one alert
                if not (has_volume_spike or has_price_jump or has_momentum_shift):
                    continue

                # Get current price
                latest_trades = self.real_time_data['trades'][ticker]
                if not latest_trades:
                    continue

                current_price = latest_trades[-1]['price']

                # Skip if price is outside our range
                if current_price < self.config['min_price'] or current_price > self.config['max_price']:
                    continue

                # Calculate position size
                max_position_size = min(
                    available_capital,
                    self.config['day_trading']['max_total_position'] *
                    (self.config['day_trading']['max_position_percent'] / 100)
                )

                # Calculate shares
                shares = int(max_position_size / current_price)

                # Skip if not enough shares
                if shares < 10:
                    continue

                # Calculate position value
                position_value = shares * current_price

                # Calculate stop loss and target prices
                stop_loss_percent = self.config['day_trading']['stop_loss_percent']
                target_profit_percent = self.config['day_trading']['target_profit_percent']

                stop_price = current_price * (1 - stop_loss_percent / 100)
                target_price = current_price * \
                    (1 + target_profit_percent / 100)

                # Calculate risk/reward ratio
                risk = current_price - stop_price
                reward = target_price - current_price
                risk_reward = reward / risk if risk > 0 else 0

                # Skip if risk/reward is too low
                if risk_reward < 2.0:
                    continue

                # We have a valid opportunity - add to day trading candidates
                logger.info(
                    f"Real-time opportunity detected for {ticker} at ${current_price:.2f}")

                # Add to day trading active list
                self.redis.zadd("day_trading:active", {ticker: score})

                # Send entry signal to execution system
                self._send_entry_signal(
                    ticker, shares, current_price, stop_price, target_price)

                # Store position details
                position_data = {
                    'price': str(current_price),
                    'shares': str(shares),
                    'position_value': str(position_value),
                    'stop_price': str(stop_price),
                    'target_price': str(target_price),
                    'risk_reward': str(risk_reward),
                    'score': str(score),
                    'entry_time': datetime.datetime.now().isoformat(),
                    'status': 'open',
                    'entry_reason': 'real_time_alert'
                }

                self.redis.hset(
                    f"day_trading:position:{ticker}", mapping=position_data)

                # Update local state
                self.day_trading_candidates.add(ticker)

                # Publish new position alert
                self.redis.publish("position_updates", json.dumps({
                    "type": "new_position",
                    "ticker": ticker,
                    "price": current_price,
                    "shares": shares,
                    "position_value": position_value,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "risk_reward": risk_reward,
                    "entry_reason": "real_time_alert"
                }))

                # Update available capital
                available_capital -= position_value

                # Stop if we've reached max positions or used all available capital
                if len(self.day_trading_candidates) >= max_positions or available_capital < 500:
                    break

        except Exception as e:
            logger.error(f"Error checking real-time opportunities: {str(e)}")

    @staticmethod
    async def update_day_trading_position(self, ticker, current_price):
        """Update day trading position based on real-time price data"""
        try:
            # Check if we have an active position for this ticker
            position_data = self.redis.hgetall(
                f"day_trading:position:{ticker}")
            if not position_data:
                return

            # Convert bytes to strings
            position = {k.decode('utf-8') if isinstance(k, bytes) else k:
                        v.decode('utf-8') if isinstance(v, bytes) else v
                        for k, v in position_data.items()}

            # Check for exit conditions
            entry_price = float(position.get('price', 0))
            stop_price = float(position.get('stop_price', 0))
            target_price = float(position.get('target_price', 0))

            # Skip if we don't have valid prices
            if not entry_price or not stop_price or not target_price:
                return

            # Check if price hit target or stop
            if current_price >= target_price:
                # Target reached - exit position
                logger.info(
                    f"Target price reached for {ticker}: {current_price:.2f} >= {target_price:.2f}")

                # Update position status
                self.redis.hset(f"day_trading:position:{ticker}", mapping={
                    "status": "closed",
                    "exit_price": str(current_price),
                    "exit_reason": "target_reached",
                    "exit_time": datetime.datetime.now().isoformat(),
                    "profit_pct": str((current_price - entry_price) / entry_price * 100)
                })

                # Remove from active positions
                self.redis.zrem("day_trading:active", ticker)

                # Add to closed positions
                self.redis.zadd("day_trading:closed", {ticker: time.time()})

                # Publish position update
                self.redis.publish("position_updates", json.dumps({
                    "type": "position_closed",
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "reason": "target_reached",
                    "profit_pct": (current_price - entry_price) / entry_price * 100
                }))

            elif current_price <= stop_price:
                # Stop loss hit - exit position
                logger.info(
                    f"Stop loss hit for {ticker}: {current_price:.2f} <= {stop_price:.2f}")

                # Update position status
                self.redis.hset(f"day_trading:position:{ticker}", mapping={
                    "status": "closed",
                    "exit_price": str(current_price),
                    "exit_reason": "stop_loss",
                    "exit_time": datetime.datetime.now().isoformat(),
                    "profit_pct": str((current_price - entry_price) / entry_price * 100)
                })

                # Remove from active positions
                self.redis.zrem("day_trading:active", ticker)

                # Add to closed positions
                self.redis.zadd("day_trading:closed", {ticker: time.time()})

                # Publish position update
                self.redis.publish("position_updates", json.dumps({
                    "type": "position_closed",
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "reason": "stop_loss",
                    "profit_pct": (current_price - entry_price) / entry_price * 100
                }))

            else:
                # Update current position metrics
                current_profit_pct = (
                    current_price - entry_price) / entry_price * 100

                # Update position status
                self.redis.hset(f"day_trading:position:{ticker}", mapping={
                    "current_price": str(current_price),
                    "current_profit_pct": str(current_profit_pct),
                    "last_update": datetime.datetime.now().isoformat()
                })

        except Exception as e:
            logger.error(
                f"Error updating day trading position for {ticker}: {str(e)}")

    @staticmethod
    def send_entry_signal(self, ticker, shares, entry_price, stop_price, target_price):
        """Send entry signal to execution system via Redis"""
        try:
            # Create signal
            signal = {
                'ticker': ticker,
                'direction': 'long',  # We only do long positions for day trading
                'signal_score': 80,   # High confidence for day trading signals
                'confidence': 0.8,
                'position_size': int(shares),
                'stop_loss': float(stop_price),
                'price_target': float(target_price),
                'signal_source': 'websocket_enhanced_stock_selection',
                'timestamp': datetime.datetime.now().timestamp()
            }

            # Publish to execution system
            self.redis.publish("execution:new_signal", json.dumps(signal))

            logger.info(
                f"Sent entry signal for {ticker}: {shares} shares at ~${entry_price}")
            return True
        except Exception as e:
            logger.error(f"Error sending entry signal for {ticker}: {str(e)}")
            return False
