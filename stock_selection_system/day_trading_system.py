#!/usr/bin/env python3
"""
Day Trading System Module
This module contains functions for day trading operations.
"""

from trading_system.execution_system import ExecutionSystem
import ml_system.market_data_helpers as mdh
from gpu_system.gpu_utils import log_memory_usage
from typing import Dict, List, Set, Any, Optional
import logging
import json
import asyncio
import datetime
import pytz
import pandas as pd
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our utility modules

# Configure logging
logger = logging.getLogger('gpu_stock_selection')


async def calculate_intraday_profit_potential(self, ticker):
    """Calculate potential intraday profit for a ticker based on historical patterns"""
    try:
        # Get historical intraday data for the past 10 trading days
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ten_days_ago = (datetime.datetime.now() -
                        datetime.timedelta(days=14)).strftime("%Y-%m-%d")

        # Get minute aggregates
        aggs = await self.polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_date=ten_days_ago,
            to_date=today
        )

        if not isinstance(aggs, pd.DataFrame) or aggs.empty:
            return {
                'average_range_percent': 0,
                'profit_probability': 0,
                'dollar_profit_potential': 0
            }

        # Group by day
        aggs['date'] = pd.to_datetime(aggs['timestamp']).dt.date

        # Calculate daily stats
        daily_stats = []
        for date, group in aggs.groupby('date'):
            if len(group) < 30:  # Skip days with little data
                continue

            day_open = group['open'].iloc[0]
            day_high = group['high'].max()
            day_low = group['low'].min()
            day_close = group['close'].iloc[-1]

            # Calculate range and movement
            day_range_percent = (day_high - day_low) / \
                day_open * 100 if day_open > 0 else 0
            day_move_percent = (day_close - day_open) / \
                day_open * 100 if day_open > 0 else 0

            daily_stats.append({
                'date': date,
                'range_percent': day_range_percent,
                'move_percent': day_move_percent,
                'positive_close': day_close > day_open
            })

        if not daily_stats:
            return {
                'average_range_percent': 0,
                'profit_probability': 0,
                'dollar_profit_potential': 0
            }

        # Calculate average range and probability of positive close
        df_stats = pd.DataFrame(daily_stats)
        average_range = df_stats['range_percent'].mean()
        profit_probability = df_stats['positive_close'].mean() * 100

        # Get current price
        current_price = await mdh.get_current_price(self.redis, self.polygon_api, ticker)
        if not current_price:
            return {
                'average_range_percent': average_range,
                'profit_probability': profit_probability,
                'dollar_profit_potential': 0
            }

        # Calculate dollar profit potential based on target percentage
        target_profit_percent = self.config['day_trading']['target_profit_percent']
        dollar_profit_potential = current_price * \
            (target_profit_percent / 100)

        # Adjust by probability
        adjusted_dollar_potential = dollar_profit_potential * \
            (profit_probability / 100)

        return {
            'average_range_percent': average_range,
            'profit_probability': profit_probability,
            'dollar_profit_potential': adjusted_dollar_potential,
            'target_profit_percent': target_profit_percent
        }

    except Exception as e:
        logger.error(
            f"Error calculating profit potential for {ticker}: {str(e)}")
        return {
            'average_range_percent': 0,
            'profit_probability': 0,
            'dollar_profit_potential': 0
        }


async def update_day_trading_candidates(self):
    """Update the list of day trading candidates"""
    logger.info("Updating day trading candidates with $5000 position limit")
    log_memory_usage("before_update_day_trading")

    candidates = []
    try:
        # Get current focused watchlist
        watchlist_data = self.redis.zrevrange(
            "watchlist:focused", 0, -1, withscores=True)

        if not watchlist_data:
            # Fall back to active watchlist if focused is empty
            watchlist_data = self.redis.zrevrange(
                "watchlist:active", 0, -1, withscores=True)

        if not watchlist_data:
            logger.warning(
                "No stocks in watchlist for day trading selection")
            return []

        # Convert to list of tuples
        watchlist = [(item[0].decode('utf-8') if isinstance(item[0], bytes) else item[0], item[1])
                     for item in watchlist_data]

        # Calculate day trading metrics for each stock
        for ticker, base_score in watchlist:
            # Get current price
            current_price = await mdh.get_current_price(self.redis, self.polygon_api, ticker)
            if not current_price or current_price <= 0:
                continue

            # Skip if price is outside our range
            if current_price < self.config['min_price'] or current_price > self.config['max_price']:
                continue

            # Calculate profit potential
            profit_potential = await self.calculate_intraday_profit_potential(ticker)

            # Calculate optimal position size
            max_position = min(
                # Max 25% in one stock
                self.config['day_trading']['max_total_position'] * 0.25,
                5000  # Hard limit of $5000 per position
            )

            # Calculate optimal shares
            optimal_shares = int(max_position / current_price)

            # Skip if we can't buy at least 10 shares
            if optimal_shares < 10:
                continue

            # Calculate actual position value
            position_value = optimal_shares * current_price

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

            # Calculate day trading score
            day_trading_score = self._calculate_day_trading_score(
                base_score,
                profit_potential,
                risk_reward,
                optimal_shares
            )

            # Add to candidates
            candidates.append({
                'ticker': ticker,
                'price': current_price,
                'optimal_shares': optimal_shares,
                'max_position': position_value,
                'stop_price': stop_price,
                'target_price': target_price,
                'risk_reward': risk_reward,
                'score': day_trading_score
            })

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Take top candidates up to max positions
        max_positions = self.config['day_trading']['max_positions']
        candidates = candidates[:min(max_positions, len(candidates))]

        if not candidates:
            logger.warning("No suitable day trading candidates found")
            return []

        # Store candidates in Redis
        pipeline = self.redis.pipeline()
        pipeline.delete("day_trading:active")

        for candidate in candidates:
            pipeline.zadd("day_trading:active", {
                          candidate['ticker']: candidate['score']})

            # Store detailed data
            position_data = {
                'price': str(candidate['price']),
                'shares': str(candidate['optimal_shares']),
                'position_value': str(candidate['max_position']),
                'stop_price': str(candidate['stop_price']),
                'target_price': str(candidate['target_price']),
                'risk_reward': str(candidate['risk_reward']),
                'score': str(candidate['score']),
                'timestamp': datetime.datetime.now().isoformat()
            }

            self.redis.hset(
                f"day_trading:position:{candidate['ticker']}", mapping=position_data)

        # Store last update time
        now = datetime.datetime.now().isoformat()
        pipeline.set("day_trading:active:last_update", now)

        pipeline.execute()

        # Update local state
        self.day_trading_candidates = set(
            [c['ticker'] for c in candidates])

        logger.info(
            f"Day trading candidates updated with {len(candidates)} stocks")

        # Calculate total position
        total_position = sum(c['max_position'] for c in candidates)
        logger.info(
            f"Total day trading position: ${total_position:.2f} (Limit: $5,000)")

        # Log the candidates
        for candidate in candidates:
            logger.info(f"Day Trading Candidate: {candidate['ticker']}, "
                        f"Price: ${candidate['price']:.2f}, "
                        f"Shares: {candidate['optimal_shares']}, "
                        f"Position: ${candidate['max_position']:.2f}, "
                        f"Target: ${candidate['target_price']:.2f}, "
                        f"Stop: ${candidate['stop_price']:.2f}, "
                        f"Score: {candidate['score']:.2f}")

        log_memory_usage("after_update_day_trading")

        return candidates

    except Exception as e:
        logger.error(
            f"Error updating day trading candidates: {str(e)}", exc_info=True)
        return []


def _calculate_day_trading_score(self, base_score, profit_potential, risk_reward, shares):
    """Calculate day trading score based on multiple factors"""
    try:
        # Start with base score from watchlist
        score = base_score

        # Adjust based on profit potential
        if profit_potential:
            avg_range = profit_potential.get('average_range_percent', 0)
            probability = profit_potential.get('profit_probability', 0)

            # Boost score for stocks with higher average range
            if avg_range >= 3.0:
                score *= 1.3
            elif avg_range >= 2.0:
                score *= 1.2
            elif avg_range >= 1.0:
                score *= 1.1

            # Boost score for stocks with higher profit probability
            if probability >= 70:
                score *= 1.3
            elif probability >= 60:
                score *= 1.2
            elif probability >= 50:
                score *= 1.1

        # Adjust based on risk/reward ratio
        if risk_reward >= 5.0:
            score *= 1.5
        elif risk_reward >= 4.0:
            score *= 1.4
        elif risk_reward >= 3.0:
            score *= 1.3
        elif risk_reward >= 2.0:
            score *= 1.2

        # Adjust based on number of shares (prefer 20-100 shares)
        if 20 <= shares <= 100:
            score *= 1.2
        elif 10 <= shares < 20 or 100 < shares <= 200:
            score *= 1.1

        return score

    except Exception as e:
        logger.error(f"Error calculating day trading score: {str(e)}")
        return base_score


async def _day_trading_update_task(self):
    """
    Task to periodically update day trading candidates
    """
    logger.info("Starting day trading update task")

    # Regular updates during trading hours
    while self.running:
        try:
            # Get current time (Eastern)
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))

            # Check if market is open
            market_open = now.replace(
                hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(
                hour=16, minute=0, second=0, microsecond=0)

            if market_open <= now < market_close and now.weekday() < 5:  # Weekdays only
                # Update frequency increases near market open and close
                if now < market_open.replace(hour=10):  # First 30 minutes
                    update_interval = 300  # 5 minutes
                # Last 30 minutes
                elif now > market_close.replace(hour=15, minute=30):
                    update_interval = 300  # 5 minutes
                else:
                    update_interval = 900  # 15 minutes

                # Update day trading candidates
                await self.update_day_trading_candidates()

                # Wait for next update
                await asyncio.sleep(update_interval)
            else:
                # Outside market hours, check less frequently
                await asyncio.sleep(1800)  # 30 minutes

        except asyncio.CancelledError:
            logger.info("Day trading update task cancelled")
            break
        except Exception as e:
            logger.error(
                f"Error in day trading update task: {str(e)}", exc_info=True)
            await asyncio.sleep(60)

    def _send_entry_signal(self, ticker, shares, entry_price, stop_price, target_price):
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
                'signal_source': 'day_trading_system',
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

    def _send_exit_signal(self, ticker, shares, entry_price):
        """Send exit signal to execution system via Redis"""
        try:
            # Create signal
            signal = {
                'ticker': ticker,
                'direction': 'close',  # Special direction to close position
                'signal_score': 90,    # High priority for closing positions
                'confidence': 0.9,
                'position_size': int(shares),
                'signal_source': 'day_trading_system_close',
                'timestamp': datetime.datetime.now().timestamp()
            }

            # Publish to execution system
            self.redis.publish("execution:new_signal", json.dumps(signal))

            logger.info(
                f"Sent exit signal for {ticker}: {shares} shares, entry: ${entry_price}")
            return True
        except Exception as e:
            logger.error(f"Error sending exit signal for {ticker}: {str(e)}")
            return False


async def close_all_day_trading_positions(self):
    """
    Close all day trading positions at end of day (no overnight holds)
    """
    logger.info(
        "Closing all day trading positions - no overnight holds policy")

    try:
        # Get active positions
        active_positions = self.redis.zrange(
            "day_trading:active", 0, -1, withscores=True)

        if not active_positions:
            logger.info("No active day trading positions to close")
            return

        # Convert to list of tickers
        positions = [pos[0].decode('utf-8') if isinstance(pos[0], bytes) else pos[0]
                     for pos in active_positions]

        logger.info(f"Closing {len(positions)} day trading positions")

        # Here you would integrate with your trading execution system
        # For now, just log the closure intent
        for ticker in positions:
            position_data = self.redis.hgetall(
                f"day_trading:position:{ticker}")
            if position_data:
                shares = position_data.get(b'shares', b'0').decode('utf-8')
                price = position_data.get(b'price', b'0').decode('utf-8')

                logger.info(
                    f"Closing position: {ticker}, Shares: {shares}, Entry: ${price}")

                # Send signal to execution system
                self._send_exit_signal(ticker, shares, price)

        # Clear active positions
        self.redis.delete("day_trading:active")
        self.day_trading_candidates = set()

        logger.info(
            "All day trading positions closed - ready for next trading day")

    except Exception as e:
        logger.error(
            f"Error closing day trading positions: {str(e)}", exc_info=True)


async def _market_close_monitor(self):
    """
    Monitor for market close to ensure all positions are closed
    """
    logger.info("Starting market close monitor")

    while self.running:
        try:
            # Get current time (Eastern)
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))

            # Check if approaching market close (15 minutes before)
            market_close = now.replace(
                hour=16, minute=0, second=0, microsecond=0)
            time_to_close = (market_close - now).total_seconds()

            # If within 15 minutes of close and we have positions, close them
            if 0 < time_to_close <= 900 and now.weekday() < 5:  # Weekdays only
                # Check if we have active positions
                active_positions = self.redis.zrange(
                    "day_trading:active", 0, -1)
                if active_positions:
                    logger.info(
                        f"Market closing in {time_to_close/60:.1f} minutes, closing all positions")
                    await self.close_all_day_trading_positions()

            # Check every minute near close, otherwise every 5 minutes
            if time_to_close <= 900:
                await asyncio.sleep(60)  # 1 minute
            else:
                await asyncio.sleep(300)  # 5 minutes

        except asyncio.CancelledError:
            logger.info("Market close monitor task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in market close monitor: {str(e)}")
            await asyncio.sleep(60)
