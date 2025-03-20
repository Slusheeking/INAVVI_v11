#!/usr/bin/env python3
"""
Execution System

This module provides a production-ready trading execution system that handles:
1. Signal execution
2. Position management
3. Risk control
4. Order routing and monitoring
5. Performance tracking

The system integrates with Alpaca for order execution and position management.
"""

import logging
import time
import json
import asyncio
import threading
import queue
import datetime
import pytz
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('execution_system')


class ExecutionSystem:
    """Production-ready trading execution system"""

    def __init__(self, redis_client, alpaca_client):
        self.redis = redis_client
        self.alpaca = alpaca_client

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Processing queues
        self.signal_queue = queue.Queue(maxsize=100)
        self.execution_queue = queue.Queue(maxsize=100)
        self.monitoring_queue = queue.Queue(maxsize=100)

        # Configuration
        self.config = {
            'max_positions': 5,
            'max_exposure': 5000.0,
            'max_loss_per_trade': 0.005,  # 0.5% of account
            'take_profit_default': 0.03,   # 3% target
            'stop_loss_default': 0.01,     # 1% stop
            'position_timeout': 14400,     # 4 hours max hold time
            'market_hours_only': True,
            'default_order_type': 'limit',
            'limit_price_offset': 0.001,   # 0.1% from current price
            'enable_trailing_stops': True,
            'trailing_stop_percent': 0.005,  # 0.5% trailing stop
            'slippage_tolerance': 0.002    # 0.2% max slippage
        }

        # Internal state
        self.running = False
        self.threads = []
        self.active_positions = {}
        self.pending_orders = {}
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_exposure': 0.0
        }

        logger.info("Execution System initialized")

    def start(self):
        """Start the execution system"""
        if self.running:
            logger.warning("Execution system already running")
            return

        self.running = True
        logger.info("Starting execution system")

        # Initialize state
        self._initialize_state()

        # Start worker threads
        self.threads.append(threading.Thread(
            target=self._signal_listener_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._execution_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._position_monitor_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._order_status_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._market_data_worker, daemon=True))

        for thread in self.threads:
            thread.start()

        logger.info("Execution system started")

    def stop(self):
        """Stop the execution system"""
        if not self.running:
            logger.warning("Execution system not running")
            return

        logger.info("Stopping execution system")
        self.running = False

        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)

        # Shutdown thread pool
        self.executor.shutdown(wait=False)

        logger.info("Execution system stopped")

    def _initialize_state(self):
        """Initialize system state"""
        try:
            logger.info("Initializing execution system state")

            # Clear Redis state
            self.redis.delete("execution:daily_stats")

            # Get current positions from Alpaca
            positions = self.alpaca.list_positions()

            if positions:
                logger.warning(
                    f"Found {len(positions)} existing positions, closing them")

                # Close all existing positions
                self.alpaca.close_all_positions()

                # Wait for positions to close
                time.sleep(5.0)

            # Reset daily stats
            self._reset_daily_stats()

            # Reset active positions and pending orders
            self.active_positions = {}
            self.pending_orders = {}

            # Store in Redis
            self.redis.delete("positions:active")
            self.redis.delete("orders:pending")

            logger.info("Execution system state initialized")

        except Exception as e:
            logger.error(
                f"Error initializing execution system state: {str(e)}", exc_info=True)

    def _reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_exposure': 0.0,
            'start_time': time.time(),
            'last_update': time.time()
        }

        # Store in Redis
        self.redis.hmset("execution:daily_stats", self.daily_stats)

    def _signal_listener_worker(self):
        """Worker thread to listen for new trading signals"""
        logger.info("Starting signal listener worker")

        pubsub = self.redis.pubsub()
        pubsub.subscribe("execution:new_signal")

        while self.running:
            try:
                # Get new message with timeout
                message = pubsub.get_message(timeout=1.0)

                if message and message['type'] == 'message':
                    # Extract data
                    data = message['data']

                    if isinstance(data, bytes):
                        data = data.decode('utf-8')

                    # Parse signal
                    try:
                        signal = json.loads(data)

                        # Validate signal
                        if self._validate_signal(signal):
                            # Add to signal queue
                            self.signal_queue.put(signal, block=False)
                            logger.info(
                                f"Received valid signal for {signal['ticker']}")
                        else:
                            logger.warning(
                                f"Received invalid signal: {signal}")

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON signal data")

            except queue.Full:
                logger.warning("Signal queue is full, dropping signal")
            except Exception as e:
                logger.error(f"Error in signal listener: {str(e)}")
                time.sleep(1.0)

    def _execution_worker(self):
        """Worker thread for signal execution"""
        logger.info("Starting execution worker")

        while self.running:
            try:
                # Get signal from queue with timeout
                signal = self.signal_queue.get(timeout=1.0)

                # Check if market is open
                if not self._check_market_status(signal):
                    logger.info(
                        f"Market closed, skipping signal for {signal['ticker']}")
                    self.signal_queue.task_done()
                    continue

                # Check if we can take new positions
                if not self._check_position_limits(signal):
                    logger.info(
                        f"Position limits reached, skipping signal for {signal['ticker']}")
                    self.signal_queue.task_done()
                    continue

                # Execute signal
                self._execute_signal(signal)

                # Mark task as done
                self.signal_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in execution worker: {str(e)}")
                time.sleep(1.0)

    def _position_monitor_worker(self):
        """Worker thread for position monitoring"""
        logger.info("Starting position monitor worker")

        while self.running:
            try:
                # Get active positions
                positions = self.active_positions.copy()

                # Check each position
                for ticker, position in positions.items():
                    # Skip if no longer active
                    if ticker not in self.active_positions:
                        continue

                    # Get current price
                    current_price = self._get_current_price(ticker)
                    if not current_price:
                        continue

                    # Update position stats
                    self._update_position_stats(ticker, current_price)

                    # Check for exits
                    if self._check_exit_conditions(ticker, current_price):
                        # Exit position
                        self._exit_position(ticker, "signal", current_price)

                # Update daily stats
                self._update_daily_stats()

                # Sleep for a bit
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in position monitor: {str(e)}")
                time.sleep(1.0)

    def _order_status_worker(self):
        """Worker thread for order status monitoring"""
        logger.info("Starting order status worker")

        while self.running:
            try:
                # Get pending orders
                orders = self.pending_orders.copy()

                # Check each order
                for order_id, order in orders.items():
                    # Skip if no longer pending
                    if order_id not in self.pending_orders:
                        continue

                    # Check order status
                    status = self._check_order_status(order_id)

                    if status == "filled":
                        # Handle filled order
                        self._handle_filled_order(order_id, order)
                    elif status == "canceled" or status == "expired":
                        # Handle canceled order
                        self._handle_canceled_order(order_id, order)
                    elif status == "rejected":
                        # Handle rejected order
                        self._handle_rejected_order(order_id, order)

                # Sleep for a bit
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in order status worker: {str(e)}")
                time.sleep(1.0)

    def _market_data_worker(self):
        """Worker thread for market data monitoring"""
        logger.info("Starting market data worker")

        while self.running:
            try:
                # Check market hours
                is_market_open = self._is_market_open()

                # Store in Redis
                self.redis.set("market:is_open",
                               "1" if is_market_open else "0")

                # If market is about to close, exit all positions
                if not is_market_open and self.active_positions and self._is_near_market_close():
                    logger.info("Market closing soon, exiting all positions")
                    self._exit_all_positions("market_close")

                # Sleep for a bit (check every minute)
                time.sleep(60.0)

            except Exception as e:
                logger.error(f"Error in market data worker: {str(e)}")
                time.sleep(60.0)

    def _validate_signal(self, signal):
        """Validate trading signal"""
        required_fields = ['ticker', 'direction', 'signal_score', 'confidence']
        return all(field in signal for field in required_fields)

    def _check_market_status(self, signal):
        """Check if market is open for trading"""
        if not self.config['market_hours_only']:
            return True

        return self._is_market_open()

    def _is_market_open(self):
        """Check if market is currently open"""
        try:
            # Get Alpaca clock
            clock = self.alpaca.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")

            # Fallback to time-based check
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            is_weekday = now.weekday() < 5
            is_market_hours = 9 <= now.hour < 16 or (
                now.hour == 16 and now.minute == 0)

            return is_weekday and is_market_hours

    def _is_near_market_close(self):
        """Check if market is about to close"""
        try:
            # Get Alpaca clock
            clock = self.alpaca.get_clock()
            if not clock.is_open:
                return False

            # Check if within 5 minutes of close
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            close_time = clock.next_close.astimezone(
                pytz.timezone('US/Eastern'))

            time_to_close = (close_time - now).total_seconds()
            return 0 < time_to_close <= 300  # Within 5 minutes

        except Exception as e:
            logger.error(f"Error checking market close: {str(e)}")

            # Fallback to time-based check
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            is_near_close = now.hour == 15 and now.minute >= 55

            return is_near_close

    def _check_position_limits(self, signal):
        """Check if we can take new positions"""
        # Check number of positions
        if len(self.active_positions) >= self.config['max_positions']:
            return False

        # Check if already in this position
        if signal['ticker'] in self.active_positions:
            return False

        # Check total exposure
        account = self._get_account_info()
        if not account:
            return False

        current_exposure = sum(pos['current_value']
                               for pos in self.active_positions.values())
        max_exposure = min(self.config['max_exposure'], account['equity'])

        if current_exposure >= max_exposure:
            return False

        return True

    def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            logger.info(f"Executing signal for {signal['ticker']}")

            # Get current price
            ticker = signal['ticker']
            current_price = self._get_current_price(ticker)

            if not current_price:
                logger.error(f"Failed to get current price for {ticker}")
                return

            # Calculate position size
            position_size = self._calculate_position_size(
                signal, current_price)

            if position_size <= 0:
                logger.warning(f"Invalid position size for {ticker}")
                return

            # Calculate order parameters
            params = self._calculate_order_parameters(
                signal, current_price, position_size)

            # Submit order
            order_id = self._submit_order(params)

            if not order_id:
                logger.error(f"Failed to submit order for {ticker}")
                return

            # Track pending order
            self.pending_orders[order_id] = {
                'ticker': ticker,
                'direction': signal['direction'],
                'quantity': position_size,
                'limit_price': params['limit_price'],
                'submitted_at': time.time(),
                'signal': signal
            }

            # Store in Redis
            self.redis.hset(f"orders:pending", order_id,
                            json.dumps(self.pending_orders[order_id]))

            logger.info(
                f"Submitted order {order_id} for {ticker}: {position_size} shares at {params['limit_price']}")

        except Exception as e:
            logger.error(
                f"Error executing signal for {signal['ticker']}: {str(e)}")

    def _calculate_position_size(self, signal, current_price):
        """Calculate position size based on risk parameters"""
        try:
            # Get account info
            account = self._get_account_info()
            if not account:
                return 0

            # Calculate max position value
            max_exposure = min(self.config['max_exposure'], account['equity'])
            current_exposure = sum(pos['current_value']
                                   for pos in self.active_positions.values())
            available_capital = max_exposure - current_exposure

            # Limit to 20% of max exposure per position
            max_position_value = min(available_capital, max_exposure * 0.2)

            # Use signal position size if provided
            if 'position_size' in signal and signal['position_size'] > 0:
                # Convert to dollars
                position_value = signal['position_size'] * current_price
                # Cap at max position value
                position_value = min(position_value, max_position_value)
            else:
                # Use risk-based sizing
                stop_loss = signal.get(
                    'stop_loss', current_price * (1 - self.config['stop_loss_default']))
                risk_amount = account['equity'] * \
                    self.config['max_loss_per_trade']
                price_risk = abs(current_price - stop_loss)

                if price_risk > 0:
                    position_value = risk_amount / (price_risk / current_price)
                else:
                    position_value = max_position_value

                # Cap at max position value
                position_value = min(position_value, max_position_value)

            # Calculate shares
            shares = int(position_value / current_price)

            # Ensure minimum size
            if shares < 1:
                shares = 0

            return shares

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def _calculate_order_parameters(self, signal, current_price, position_size):
        """Calculate order parameters"""
        # Determine order type
        order_type = self.config['default_order_type']

        # Calculate limit price
        if order_type == 'limit':
            # Add offset for buys, subtract for sells
            if signal['direction'] == 'long':
                limit_price = current_price * \
                    (1 + self.config['limit_price_offset'])
            else:
                limit_price = current_price * \
                    (1 - self.config['limit_price_offset'])
        else:
            limit_price = current_price

        # Calculate stop loss
        if 'stop_loss' in signal and signal['stop_loss'] > 0:
            stop_loss = signal['stop_loss']
        else:
            if signal['direction'] == 'long':
                stop_loss = current_price * \
                    (1 - self.config['stop_loss_default'])
            else:
                stop_loss = current_price * \
                    (1 + self.config['stop_loss_default'])

        # Calculate take profit
        if 'price_target' in signal and signal['price_target'] > 0:
            take_profit = signal['price_target']
        else:
            if signal['direction'] == 'long':
                take_profit = current_price * \
                    (1 + self.config['take_profit_default'])
            else:
                take_profit = current_price * \
                    (1 - self.config['take_profit_default'])

        # Create parameters
        params = {
            'ticker': signal['ticker'],
            'quantity': position_size,
            'side': 'buy' if signal['direction'] == 'long' else 'sell',
            'order_type': order_type,
            'time_in_force': 'day',
            'limit_price': limit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        return params

    def _submit_order(self, params):
        """Submit order to Alpaca"""
        try:
            # Create order
            if params['order_type'] == 'limit':
                order = self.alpaca.submit_order(
                    symbol=params['ticker'],
                    qty=params['quantity'],
                    side=params['side'],
                    type=params['order_type'],
                    time_in_force=params['time_in_force'],
                    limit_price=params['limit_price']
                )
            else:
                order = self.alpaca.submit_order(
                    symbol=params['ticker'],
                    qty=params['quantity'],
                    side=params['side'],
                    type=params['order_type'],
                    time_in_force=params['time_in_force']
                )

            return order.id

        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return None

    def _check_order_status(self, order_id):
        """Check status of an order"""
        try:
            order = self.alpaca.get_order(order_id)
            return order.status
        except Exception as e:
            logger.error(
                f"Error checking order status for {order_id}: {str(e)}")
            return "unknown"

    def _handle_filled_order(self, order_id, order_info):
        """Handle a filled order"""
        try:
            # Get order details from Alpaca
            order = self.alpaca.get_order(order_id)

            # Extract details
            ticker = order_info['ticker']
            filled_price = float(order.filled_avg_price)
            filled_qty = int(order.filled_qty)
            side = order.side

            # Create position record
            position = {
                'ticker': ticker,
                'entry_price': filled_price,
                'quantity': filled_qty,
                'direction': order_info['direction'],
                'current_price': filled_price,
                'current_value': filled_price * filled_qty,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'high_price': filled_price,
                'low_price': filled_price,
                'stop_loss': order_info.get('stop_loss', 0),
                'take_profit': order_info.get('take_profit', 0),
                'entry_time': time.time(),
                'last_update': time.time(),
                'order_id': order_id,
                'signal': order_info.get('signal', {})
            }

            # Save to active positions
            self.active_positions[ticker] = position

            # Store in Redis
            self.redis.hset("positions:active",
                            f"{ticker}:{order_id}", json.dumps(position))

            # Remove from pending orders
            self.pending_orders.pop(order_id, None)
            self.redis.hdel("orders:pending", order_id)

            # Update daily stats
            self.daily_stats['trades_executed'] += 1
            self.daily_stats['current_exposure'] += position['current_value']
            self.redis.hmset("execution:daily_stats", self.daily_stats)

            # Submit stop loss and take profit orders if configured
            if self.config['enable_trailing_stops']:
                self._submit_trailing_stop(ticker, position)

            logger.info(
                f"Order filled for {ticker}: {filled_qty} shares at {filled_price}")

        except Exception as e:
            logger.error(f"Error handling filled order {order_id}: {str(e)}")

    def _handle_canceled_order(self, order_id, order_info):
        """Handle a canceled order"""
        try:
            # Remove from pending orders
            self.pending_orders.pop(order_id, None)
            self.redis.hdel("orders:pending", order_id)

            logger.info(
                f"Order canceled for {order_info['ticker']}: {order_id}")

        except Exception as e:
            logger.error(f"Error handling canceled order {order_id}: {str(e)}")

    def _handle_rejected_order(self, order_id, order_info):
        """Handle a rejected order"""
        try:
            # Remove from pending orders
            self.pending_orders.pop(order_id, None)
            self.redis.hdel("orders:pending", order_id)

            logger.warning(
                f"Order rejected for {order_info['ticker']}: {order_id}")

        except Exception as e:
            logger.error(f"Error handling rejected order {order_id}: {str(e)}")

    def _get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            # Try Redis first
            price_data = self.redis.hgetall(f"stock:{ticker}:last_trade")

            if price_data and b'price' in price_data:
                return float(price_data[b'price'])

            # Fallback to Alpaca
            last_trade = self.alpaca.get_latest_trade(ticker)
            return float(last_trade.price)

        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return None

    def _update_position_stats(self, ticker, current_price):
        """Update statistics for a position"""
        try:
            position = self.active_positions[ticker]

            # Update current price and value
            position['current_price'] = current_price
            position['current_value'] = current_price * position['quantity']

            # Calculate P&L
            entry_value = position['entry_price'] * position['quantity']
            current_value = position['current_value']

            if position['direction'] == 'long':
                position['unrealized_pnl'] = current_value - entry_value
                position['unrealized_pnl_pct'] = (
                    current_price / position['entry_price'] - 1) * 100
            else:
                position['unrealized_pnl'] = entry_value - current_value
                position['unrealized_pnl_pct'] = (
                    position['entry_price'] / current_price - 1) * 100

            # Update high/low prices
            position['high_price'] = max(position['high_price'], current_price)
            position['low_price'] = min(position['low_price'], current_price)

            # Update last update time
            position['last_update'] = time.time()

            # Update trailing stop if enabled
            if self.config['enable_trailing_stops']:
                self._update_trailing_stop(ticker, position)

            # Store updated position in Redis
            self.redis.hset(
                "positions:active",
                f"{ticker}:{position['order_id']}",
                json.dumps(position)
            )

        except Exception as e:
            logger.error(
                f"Error updating position stats for {ticker}: {str(e)}")

    def _check_exit_conditions(self, ticker, current_price):
        """Check if position should be exited"""
        try:
            position = self.active_positions[ticker]

            # Check stop loss
            if position['stop_loss'] > 0:
                if (position['direction'] == 'long' and current_price <= position['stop_loss']) or \
                   (position['direction'] == 'short' and current_price >= position['stop_loss']):
                    logger.info(
                        f"Stop loss triggered for {ticker} at {current_price}")
                    return True

            # Check take profit
            if position['take_profit'] > 0:
                if (position['direction'] == 'long' and current_price >= position['take_profit']) or \
                   (position['direction'] == 'short' and current_price <= position['take_profit']):
                    logger.info(
                        f"Take profit triggered for {ticker} at {current_price}")
                    return True

            # Check time-based exit
            position_age = time.time() - position['entry_time']
            if position_age >= self.config['position_timeout']:
                logger.info(
                    f"Time-based exit triggered for {ticker} after {position_age/3600:.1f} hours")
                return True

            # Check for extreme profit (let winners run)
            if position['unrealized_pnl_pct'] > 10.0:  # 10% profit
                # Continue holding
                return False

            # Check for prolonged sideways movement
            # Would need more sophisticated logic based on price action

            return False

        except Exception as e:
            logger.error(
                f"Error checking exit conditions for {ticker}: {str(e)}")
            return False

    def _exit_position(self, ticker, reason, price=None):
        """Exit a position"""
        try:
            position = self.active_positions[ticker]

            # Calculate exit parameters
            quantity = position['quantity']
            side = 'sell' if position['direction'] == 'long' else 'buy'

            # Submit order
            order = self.alpaca.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )

            # Calculate realized P&L (estimated)
            entry_value = position['entry_price'] * quantity
            exit_value = price * quantity if price else entry_value

            if position['direction'] == 'long':
                realized_pnl = exit_value - entry_value
                realized_pnl_pct = (
                    price / position['entry_price'] - 1) * 100 if price else 0
            else:
                realized_pnl = entry_value - exit_value
                realized_pnl_pct = (
                    position['entry_price'] / price - 1) * 100 if price else 0

            # Update daily stats
            self.daily_stats['total_pnl'] += realized_pnl
            if realized_pnl > 0:
                self.daily_stats['profitable_trades'] += 1
            self.daily_stats['current_exposure'] -= position['current_value']

            # Store trade in Redis
            trade_id = str(uuid.uuid4())
            trade = {
                'ticker': ticker,
                'entry_price': position['entry_price'],
                'exit_price': price if price else 0,
                'quantity': quantity,
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'exit_time': time.time(),
                'realized_pnl': realized_pnl,
                'realized_pnl_pct': realized_pnl_pct,
                'exit_reason': reason,
                'order_id': position['order_id'],
                'exit_order_id': order.id
            }

            self.redis.hset("trades:history", trade_id, json.dumps(trade))

            # Cancel any related orders (stop loss, take profit)
            self._cancel_position_orders(ticker)

            # Remove from active positions
            self.active_positions.pop(ticker, None)
            self.redis.hdel("positions:active",
                            f"{ticker}:{position['order_id']}")

            # Update Redis stats
            self.redis.hmset("execution:daily_stats", self.daily_stats)

            logger.info(
                f"Exited position for {ticker}: {quantity} shares at ~{price}, PnL: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")

            return True

        except Exception as e:
            logger.error(f"Error exiting position for {ticker}: {str(e)}")
            return False

    def _exit_all_positions(self, reason):
        """Exit all active positions"""
        try:
            # Copy positions to avoid modification during iteration
            positions = self.active_positions.copy()

            for ticker in positions:
                # Get current price
                price = self._get_current_price(ticker)

                # Exit position
                self._exit_position(ticker, reason, price)

            return True

        except Exception as e:
            logger.error(f"Error exiting all positions: {str(e)}")
            return False

    def _submit_trailing_stop(self, ticker, position):
        """Submit a trailing stop order"""
        try:
            # Calculate trail percent
            trail_percent = self.config['trailing_stop_percent']

            # Create order
            order = self.alpaca.submit_order(
                symbol=ticker,
                qty=position['quantity'],
                side='sell' if position['direction'] == 'long' else 'buy',
                type='trailing_stop',
                trail_percent=trail_percent * 100,  # Alpaca needs percentage in whole numbers
                time_in_force='gtc'
            )

            # Store order ID
            position['stop_order_id'] = order.id

            logger.info(
                f"Submitted trailing stop for {ticker}: {trail_percent:.1f}% trail")

            return True

        except Exception as e:
            logger.error(
                f"Error submitting trailing stop for {ticker}: {str(e)}")
            return False

    def _update_trailing_stop(self, ticker, position):
        """Update trailing stop based on price movement"""
        try:
            # Skip if trailing stop not enabled
            if 'stop_order_id' not in position:
                return False

            # No need to manually update for Alpaca trailing stops
            return True

        except Exception as e:
            logger.error(
                f"Error updating trailing stop for {ticker}: {str(e)}")
            return False

    def _cancel_position_orders(self, ticker):
        """Cancel all orders related to a position"""
        try:
            position = self.active_positions.get(ticker)
            if not position:
                return False

            # Cancel stop loss order if exists
            if 'stop_order_id' in position:
                try:
                    self.alpaca.cancel_order(position['stop_order_id'])
                except Exception:
                    pass

            # Cancel take profit order if exists
            if 'take_profit_order_id' in position:
                try:
                    self.alpaca.cancel_order(position['take_profit_order_id'])
                except Exception:
                    pass

            return True

        except Exception as e:
            logger.error(
                f"Error canceling position orders for {ticker}: {str(e)}")
            return False

    def _update_daily_stats(self):
        """Update daily statistics"""
        try:
            # Update exposure
            total_exposure = sum(pos['current_value']
                                 for pos in self.active_positions.values())
            self.daily_stats['current_exposure'] = total_exposure

            # Calculate current drawdown
            equity = self._get_account_equity()
            if equity:
                drawdown = (self.daily_stats.get('peak_equity', equity) -
                            equity) / self.daily_stats.get('peak_equity', equity)
                self.daily_stats['max_drawdown'] = max(
                    self.daily_stats.get('max_drawdown', 0), drawdown)

                # Update peak equity
                if equity > self.daily_stats.get('peak_equity', 0):
                    self.daily_stats['peak_equity'] = equity

            # Update last update time
            self.daily_stats['last_update'] = time.time()

            # Store in Redis (only if changed)
            self.redis.hmset("execution:daily_stats", self.daily_stats)

        except Exception as e:
            logger.error(f"Error updating daily stats: {str(e)}")

    def _get_account_info(self):
        """Get account information from Alpaca"""
        try:
            account = self.alpaca.get_account()

            return {
                'id': account.id,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'pdt_rule': account.pattern_day_trader
            }

        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def _get_account_equity(self):
        """Get current account equity"""
        try:
            account = self.alpaca.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting account equity: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    import os
    import redis
    import alpaca_trade_api as tradeapi

    # Create Redis client
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6380)),
        db=int(os.environ.get('REDIS_DB', 0))
    )

    # Create Alpaca client
    alpaca_client = tradeapi.REST(
        key_id=os.environ.get('ALPACA_API_KEY', ''),
        secret_key=os.environ.get('ALPACA_API_SECRET', ''),
        base_url=os.environ.get(
            'ALPACA_API_URL', 'https://paper-api.alpaca.markets')
    )

    # Create execution system
    execution_system = ExecutionSystem(redis_client, alpaca_client)

    # Start system
    execution_system.start()

    try:
        # Run for a while
        print("Execution system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        execution_system.stop()
