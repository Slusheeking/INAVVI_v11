#!/usr/bin/env python3
"""
GPU-Optimized Polygon.io WebSocket Client

This module provides an optimized client for interacting with the Polygon.io WebSocket API,
designed specifically for high-performance trading systems using GPU acceleration.
"""

import os
import json
import logging
import asyncio
import signal
import time
import threading
import sys
import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, List, Optional, Union, Any, Callable
import websockets
from websockets.exceptions import ConnectionClosed
from datetime import datetime
import redis
import random
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_polygon_websocket_client')

# Environment variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
MAX_RECONNECT_ATTEMPTS = int(os.environ.get(
    'POLYGON_MAX_RECONNECT_ATTEMPTS', 10))
RECONNECT_DELAY = float(os.environ.get('POLYGON_RECONNECT_DELAY', 2.0))
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379)
                 )  # Changed from 6380 to 6379
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_USERNAME = os.environ.get(
    'REDIS_USERNAME', 'default')  # Added Redis username
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')  # Added Redis password
REDIS_SSL = os.environ.get('REDIS_SSL', 'false').lower(
) == 'true'  # Added Redis SSL option
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
# Number of messages to buffer before batch processing
BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE', 1000))


class GPUPolygonWebSocketClient:
    """GPU-Optimized client for the Polygon.io WebSocket API"""

    def __init__(self, api_key=POLYGON_API_KEY, redis_client=None, use_gpu=USE_GPU,
                 max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
                 reconnect_delay=RECONNECT_DELAY, buffer_size=BUFFER_SIZE):
        """
        Initialize GPU-Optimized Polygon WebSocket client

        Args:
            api_key: API key for authentication
            redis_client: Optional Redis client for data storage
            use_gpu: Whether to use GPU acceleration
            max_reconnect_attempts: Maximum number of reconnection attempts
            reconnect_delay: Initial delay between reconnection attempts (will use exponential backoff)
            buffer_size: Number of messages to buffer before batch processing
        """
        self.api_key = api_key
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.use_gpu = use_gpu
        self.buffer_size = buffer_size
        self.redis_client = redis_client

        # Verify API key is provided
        if not self.api_key:
            logger.warning(
                "No API key provided. Set the POLYGON_API_KEY environment variable.")
            logger.error(
                "API key is required for Polygon.io WebSocket connection")
        else:
            logger.info(
                f"Initialized Polygon WebSocket client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}")

        # WebSocket connection
        self.ws = None
        self.ws_url = f"wss://socket.polygon.io/stocks"

        # Subscription tracking
        self.subscriptions = set()

        # Message handlers
        self.message_handlers = {}

        # Connection status
        self.connected = False
        self.reconnect_count = 0
        self.last_heartbeat = 0

        # Control flags
        self.running = False
        self.heartbeat_task = None
        self.event_loop = None
        self.main_task = None
        self.background_thread = None

        # Message buffers for batch processing
        self.trade_buffer = []
        self.quote_buffer = []
        self.agg_buffer = []
        self.buffer_lock = threading.Lock()

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Initialize GPU if available and requested
        self.gpu_initialized = False
        if self.use_gpu:
            # Initialize CuPy - no fallback to CPU
            self._initialize_gpu()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop()

    def _initialize_gpu(self):
        """Initialize GPU for data processing"""
        try:
            import cupy as cp

            # Get device count
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                # Find GH200 device if available
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    if "GH200" in device_props["name"].decode():
                        cp.cuda.Device(i).use()
                        logger.info(f"Using GH200 GPU at index {i}")
                        break

                # Use unified memory for better performance
                cp.cuda.set_allocator(cp.cuda.MemoryPool(
                    cp.cuda.malloc_managed).malloc)
                logger.info("CuPy configured with unified memory")
                self.gpu_initialized = True
            else:
                logger.warning("No GPU devices found")
                self.use_gpu = False
                self.gpu_initialized = False
        except ImportError:
            logger.warning("CuPy not available. GPU acceleration disabled")
            self.use_gpu = False
            self.gpu_initialized = False
        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.use_gpu = False
            self.gpu_initialized = False

    async def _connect(self):
        """Establish WebSocket connection"""
        if not self.api_key:
            logger.error("Cannot connect: No API key provided")
            return False

        try:
            logger.info(f"Connecting to {self.ws_url}...")
            # Connect with API key in the URL path
            connection_url = f"wss://socket.polygon.io/stocks?apiKey={self.api_key}"
            self.ws = await websockets.connect(connection_url)
            logger.info("WebSocket connection established")

            # Authenticate
            await self._authenticate()

            # Wait for auth response
            response = await self.ws.recv()
            response_data = json.loads(response)

            if isinstance(response_data, list) and response_data and response_data[0].get('status') == 'connected':
                logger.info("Authentication successful")
                self.connected = True
                self.reconnect_count = 0
                self.last_heartbeat = time.time()

                # Resubscribe to previous subscriptions
                if self.subscriptions:
                    await self._resubscribe()

                return True
            else:
                logger.error(f"Authentication failed: {response_data}")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False

    async def _authenticate(self):
        """Authenticate with the Polygon WebSocket API"""
        try:
            # Send authentication message
            auth_message = {"action": "auth", "params": self.api_key}
            await self.ws.send(json.dumps(auth_message))
            logger.info("Sent authentication request")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    async def _resubscribe(self):
        """Resubscribe to all previous subscriptions"""
        if not self.subscriptions:
            return

        logger.info(f"Resubscribing to {len(self.subscriptions)} channels...")

        # Group subscriptions by cluster (stocks, options, forex, crypto)
        subscriptions_by_cluster = {}
        for sub in self.subscriptions:
            parts = sub.split(".")
            if len(parts) >= 2:
                cluster = parts[0]
                if cluster not in subscriptions_by_cluster:
                    subscriptions_by_cluster[cluster] = []
                subscriptions_by_cluster[cluster].append(sub)

        # Subscribe to each cluster
        for cluster, subs in subscriptions_by_cluster.items():
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subs)
            }

            try:
                await self.ws.send(json.dumps(subscribe_message))
                logger.info(f"Resubscribed to {len(subs)} {cluster} channels")
            except Exception as e:
                logger.error(f"Error resubscribing to {cluster} channels: {e}")

    async def _heartbeat(self):
        """Send heartbeat messages and monitor connection health"""
        while self.running and self.connected:
            try:
                # Check if we've received a heartbeat recently
                if time.time() - self.last_heartbeat > 30:  # No heartbeat for 30 seconds
                    logger.warning(
                        "No heartbeat received for 30 seconds, reconnecting...")
                    self.connected = False
                    await self._reconnect()
                    continue

                # Sleep for a while
                await asyncio.sleep(15)

            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(5)

    async def _reconnect(self):
        """Attempt to reconnect to the WebSocket"""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error(
                f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.running = False
            return False

        self.reconnect_count += 1
        delay = self.reconnect_delay * \
            (2 ** (self.reconnect_count - 1))  # Exponential backoff

        logger.info(
            f"Attempting to reconnect (attempt {self.reconnect_count}/{self.max_reconnect_attempts}) in {delay:.2f} seconds...")

        try:
            # Close existing connection if any
            if self.ws:
                await self.ws.close()

            # Wait before reconnecting
            await asyncio.sleep(delay)

            # Attempt to reconnect
            return await self._connect()

        except Exception as e:
            logger.error(f"Reconnection error: {e}")
            return False

    async def _listen(self):
        """Listen for WebSocket messages"""
        while self.running:
            try:
                if not self.connected:
                    success = await self._connect()
                    if not success:
                        await self._reconnect()
                        continue

                # Receive message
                message = await self.ws.recv()

                # Update heartbeat timestamp
                self.last_heartbeat = time.time()

                # Parse message
                try:
                    data = json.loads(message)

                    # Handle message based on type
                    if isinstance(data, list):
                        for item in data:
                            await self._process_message(item)
                    else:
                        await self._process_message(data)

                except json.JSONDecodeError:
                    logger.warning(
                        f"Received invalid JSON: {message[:100]}...")

            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                await self._reconnect()

            except asyncio.CancelledError:
                logger.info("Listen task cancelled")
                break

            except Exception as e:
                logger.error(f"Error in listen task: {e}")
                self.connected = False
                await asyncio.sleep(1)

    async def _process_message(self, message):
        """Process a WebSocket message"""
        # Check for status messages
        if 'status' in message:
            status = message.get('status')

            if status == 'connected':
                logger.info("Connected to Polygon WebSocket")

            elif status == 'auth_success':
                logger.info("Authentication successful")

            elif status == 'success':
                if message.get('message') == 'authenticated':
                    logger.info("Authentication successful")
                else:
                    action = message.get('action')
                    if action == 'subscribe':
                        logger.info(
                            f"Successfully subscribed to: {message.get('params', '')}")
                    elif action == 'unsubscribe':
                        logger.info(
                            f"Successfully unsubscribed from: {message.get('params', '')}")

            elif status == 'error':
                logger.error(
                    f"Error from Polygon WebSocket: {message.get('message', '')}")

            return

        # Handle data messages
        event_type = message.get('ev')
        if not event_type:
            logger.warning(f"Received message without event type: {message}")
            return

        # Buffer messages for batch processing
        with self.buffer_lock:
            if event_type == 'T':  # Trade
                self.trade_buffer.append(message)
                if len(self.trade_buffer) >= self.buffer_size:
                    # Process trade buffer
                    trades_to_process = self.trade_buffer.copy()
                    self.trade_buffer = []
                    self.thread_pool.submit(
                        self._process_trade_batch, trades_to_process)

            elif event_type == 'Q':  # Quote
                self.quote_buffer.append(message)
                if len(self.quote_buffer) >= self.buffer_size:
                    # Process quote buffer
                    quotes_to_process = self.quote_buffer.copy()
                    self.quote_buffer = []
                    self.thread_pool.submit(
                        self._process_quote_batch, quotes_to_process)

            elif event_type in ['AM', 'A']:  # Aggregates
                self.agg_buffer.append(message)
                if len(self.agg_buffer) >= self.buffer_size:
                    # Process aggregate buffer
                    aggs_to_process = self.agg_buffer.copy()
                    self.agg_buffer = []
                    self.thread_pool.submit(
                        self._process_agg_batch, aggs_to_process)

        # Call appropriate handler
        if event_type in self.message_handlers:
            for handler in self.message_handlers[event_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(
                        f"Error in message handler for {event_type}: {e}")

    def _process_trade_batch(self, trades):
        """Process a batch of trade messages with GPU acceleration"""
        if not trades:
            return

        try:
            # Group trades by ticker
            trades_by_ticker = {}
            for trade in trades:
                ticker = trade.get('sym')
                if ticker not in trades_by_ticker:
                    trades_by_ticker[ticker] = []
                trades_by_ticker[ticker].append(trade)

            # Process each ticker's trades
            for ticker, ticker_trades in trades_by_ticker.items():
                # Extract trade data
                prices = [t.get('p', 0) for t in ticker_trades]
                sizes = [t.get('s', 0) for t in ticker_trades]
                timestamps = [t.get('t', 0) for t in ticker_trades]

                # Use GPU for calculations if available
                if self.use_gpu and self.gpu_initialized:
                    # Convert to CuPy arrays
                    cp_prices = cp.array(prices, dtype=cp.float32)
                    cp_sizes = cp.array(sizes, dtype=cp.float32)

                    # Calculate VWAP
                    price_volume = cp_prices * cp_sizes
                    total_price_volume = cp.sum(price_volume)
                    total_volume = cp.sum(cp_sizes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0

                    # Calculate statistics
                    avg_price = cp.mean(cp_prices)
                    max_price = cp.max(cp_prices)
                    min_price = cp.min(cp_prices)

                    # Convert back to numpy
                    vwap = float(cp.asnumpy(vwap))
                    avg_price = float(cp.asnumpy(avg_price))
                    max_price = float(cp.asnumpy(max_price))
                    min_price = float(cp.asnumpy(min_price))
                else:
                    # CPU calculations
                    price_volume = np.array(prices) * np.array(sizes)
                    total_price_volume = np.sum(price_volume)
                    total_volume = np.sum(sizes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0

                    avg_price = np.mean(prices)
                    max_price = np.max(prices)
                    min_price = np.min(prices)

                # Get latest trade
                latest_trade = max(ticker_trades, key=lambda t: t.get('t', 0))
                latest_price = latest_trade.get('p', 0)
                latest_size = latest_trade.get('s', 0)
                latest_timestamp = latest_trade.get('t', 0)

                # Store in Redis if available
                if self.redis_client:
                    # Update last trade
                    self.redis_client.hmset(f"stock:{ticker}:last_trade", {
                        "price": latest_price,
                        "size": latest_size,
                        "timestamp": latest_timestamp,
                        "exchange": latest_trade.get('x', '')
                    })

                    # Update last price
                    self.redis_client.hmset(f"stock:{ticker}:last_price", {
                        "price": latest_price,
                        "timestamp": latest_timestamp
                    })

                    # Store trade statistics
                    self.redis_client.hmset(f"stock:{ticker}:trade_stats", {
                        "vwap": vwap,
                        "avg_price": avg_price,
                        "max_price": max_price,
                        "min_price": min_price,
                        "total_volume": int(total_volume),
                        "trade_count": len(ticker_trades),
                        "last_update": datetime.now().isoformat()
                    })

                    # Publish update to subscribers
                    self.redis_client.publish(f"price_update:{ticker}", json.dumps({
                        "type": "trade",
                        "ticker": ticker,
                        "price": latest_price,
                        "size": latest_size,
                        "timestamp": latest_timestamp
                    }))

        except Exception as e:
            logger.error(f"Error processing trade batch: {e}")

    def _process_quote_batch(self, quotes):
        """Process a batch of quote messages with GPU acceleration"""
        if not quotes:
            return

        try:
            # Group quotes by ticker
            quotes_by_ticker = {}
            for quote in quotes:
                ticker = quote.get('sym')
                if ticker not in quotes_by_ticker:
                    quotes_by_ticker[ticker] = []
                quotes_by_ticker[ticker].append(quote)

            # Process each ticker's quotes
            for ticker, ticker_quotes in quotes_by_ticker.items():
                # Extract quote data
                bid_prices = [q.get('bp', 0) for q in ticker_quotes]
                bid_sizes = [q.get('bs', 0) for q in ticker_quotes]
                ask_prices = [q.get('ap', 0) for q in ticker_quotes]
                ask_sizes = [q.get('as', 0) for q in ticker_quotes]
                timestamps = [q.get('t', 0) for q in ticker_quotes]

                # Use GPU for calculations if available
                if self.use_gpu and self.gpu_initialized:
                    # Convert to CuPy arrays
                    cp_bid_prices = cp.array(bid_prices, dtype=cp.float32)
                    cp_ask_prices = cp.array(ask_prices, dtype=cp.float32)

                    # Calculate mid prices
                    cp_mid_prices = (cp_bid_prices + cp_ask_prices) / 2

                    # Calculate spreads
                    cp_spreads = cp_ask_prices - cp_bid_prices

                    # Calculate statistics
                    avg_bid = cp.mean(cp_bid_prices)
                    avg_ask = cp.mean(cp_ask_prices)
                    avg_mid = cp.mean(cp_mid_prices)
                    avg_spread = cp.mean(cp_spreads)

                    # Convert back to numpy
                    avg_bid = float(cp.asnumpy(avg_bid))
                    avg_ask = float(cp.asnumpy(avg_ask))
                    avg_mid = float(cp.asnumpy(avg_mid))
                    avg_spread = float(cp.asnumpy(avg_spread))
                else:
                    # CPU calculations
                    mid_prices = [(b + a) / 2 for b,
                                  a in zip(bid_prices, ask_prices)]
                    spreads = [a - b for a, b in zip(ask_prices, bid_prices)]

                    avg_bid = np.mean(bid_prices)
                    avg_ask = np.mean(ask_prices)
                    avg_mid = np.mean(mid_prices)
                    avg_spread = np.mean(spreads)

                # Get latest quote
                latest_quote = max(ticker_quotes, key=lambda q: q.get('t', 0))
                latest_bid = latest_quote.get('bp', 0)
                latest_ask = latest_quote.get('ap', 0)
                latest_bid_size = latest_quote.get('bs', 0)
                latest_ask_size = latest_quote.get('as', 0)
                latest_timestamp = latest_quote.get('t', 0)
                latest_mid = (latest_bid + latest_ask) / \
                    2 if latest_bid > 0 and latest_ask > 0 else 0
                latest_spread = latest_ask - latest_bid if latest_bid > 0 and latest_ask > 0 else 0

                # Store in Redis if available
                if self.redis_client:
                    # Update last quote
                    self.redis_client.hmset(f"stock:{ticker}:last_quote", {
                        "bid_price": latest_bid,
                        "bid_size": latest_bid_size,
                        "ask_price": latest_ask,
                        "ask_size": latest_ask_size,
                        "mid_price": latest_mid,
                        "spread": latest_spread,
                        "timestamp": latest_timestamp
                    })

                    # Update last price
                    self.redis_client.hmset(f"stock:{ticker}:last_price", {
                        "bid": latest_bid,
                        "ask": latest_ask,
                        "mid": latest_mid,
                        "timestamp": latest_timestamp
                    })

                    # Publish update to subscribers
                    self.redis_client.publish(f"price_update:{ticker}", json.dumps({
                        "type": "quote",
                        "ticker": ticker,
                        "bid": latest_bid,
                        "ask": latest_ask,
                        "mid": latest_mid,
                        "timestamp": latest_timestamp
                    }))

        except Exception as e:
            logger.error(f"Error processing quote batch: {e}")

    def _process_agg_batch(self, aggs):
        """Process a batch of aggregate messages with GPU acceleration"""
        if not aggs:
            return

        try:
            # Group aggregates by ticker
            aggs_by_ticker = {}
            for agg in aggs:
                ticker = agg.get('sym')
                if ticker not in aggs_by_ticker:
                    aggs_by_ticker[ticker] = []
                aggs_by_ticker[ticker].append(agg)

            # Process each ticker's aggregates
            for ticker, ticker_aggs in aggs_by_ticker.items():
                # Extract aggregate data
                opens = [a.get('o', 0) for a in ticker_aggs]
                highs = [a.get('h', 0) for a in ticker_aggs]
                lows = [a.get('l', 0) for a in ticker_aggs]
                closes = [a.get('c', 0) for a in ticker_aggs]
                volumes = [a.get('v', 0) for a in ticker_aggs]
                timestamps = [a.get('s', 0)
                              for a in ticker_aggs]  # Start timestamps

                # Get latest aggregate
                latest_agg = max(ticker_aggs, key=lambda a: a.get('s', 0))
                latest_open = latest_agg.get('o', 0)
                latest_high = latest_agg.get('h', 0)
                latest_low = latest_agg.get('l', 0)
                latest_close = latest_agg.get('c', 0)
                latest_volume = latest_agg.get('v', 0)
                latest_timestamp = latest_agg.get('s', 0)

                # Store in Redis if available
                if self.redis_client:
                    # Store latest candle
                    self.redis_client.hmset(f"stock:{ticker}:latest_candle", {
                        "open": latest_open,
                        "high": latest_high,
                        "low": latest_low,
                        "close": latest_close,
                        "volume": latest_volume,
                        "timestamp": latest_timestamp
                    })

                    # Store in candles hash
                    timespan = "minute" if latest_agg.get(
                        'ev') == 'AM' else "second"
                    candle_key = f"stock:{ticker}:candles:{timespan}"

                    # Add each candle to the hash
                    for agg in ticker_aggs:
                        timestamp = agg.get('s', 0)
                        candle_data = {
                            "open": agg.get('o', 0),
                            "high": agg.get('h', 0),
                            "low": agg.get('l', 0),
                            "close": agg.get('c', 0),
                            "volume": agg.get('v', 0),
                            "timestamp": timestamp
                        }
                        self.redis_client.hset(
                            candle_key, timestamp, json.dumps(candle_data))

                    # Publish update to subscribers
                    self.redis_client.publish(f"candle_update:{ticker}", json.dumps({
                        "type": "candle",
                        "ticker": ticker,
                        "timespan": timespan,
                        "open": latest_open,
                        "high": latest_high,
                        "low": latest_low,
                        "close": latest_close,
                        "volume": latest_volume,
                        "timestamp": latest_timestamp
                    }))

        except Exception as e:
            logger.error(f"Error processing aggregate batch: {e}")

    def start(self):
        """Start the WebSocket client in a background thread"""
        if self.running:
            logger.warning("WebSocket client is already running")
            return

        self.running = True

        def run_event_loop():
            """Run the event loop in a background thread"""
            try:
                # Create new event loop for this thread
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)

                # Create tasks
                self.main_task = self.event_loop.create_task(self._listen())
                self.heartbeat_task = self.event_loop.create_task(
                    self._heartbeat())

                # Run the event loop
                self.event_loop.run_until_complete(asyncio.gather(
                    self.main_task,
                    self.heartbeat_task
                ))

            except Exception as e:
                logger.error(f"Error in WebSocket thread: {e}")

            finally:
                # Clean up
                if self.event_loop and not self.event_loop.is_closed():
                    self.event_loop.close()
                logger.info("WebSocket thread stopped")

        # Start background thread
        self.background_thread = threading.Thread(target=run_event_loop)
        self.background_thread.daemon = True
        self.background_thread.start()

        logger.info("WebSocket client started")

    def stop(self):
        """Stop the WebSocket client"""
        if not self.running:
            logger.warning("WebSocket client is not running")
            return

        logger.info("Stopping WebSocket client...")
        self.running = False

        # Cancel tasks
        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._cancel_tasks(), self.event_loop)

        # Wait for background thread to stop
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)

        # Process remaining buffered messages
        with self.buffer_lock:
            if self.trade_buffer:
                self._process_trade_batch(self.trade_buffer)
                self.trade_buffer = []

            if self.quote_buffer:
                self._process_quote_batch(self.quote_buffer)
                self.quote_buffer = []

            if self.agg_buffer:
                self._process_agg_batch(self.agg_buffer)
                self.agg_buffer = []

        # Clean up GPU resources
        if self.use_gpu and self.gpu_initialized:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Error clearing CuPy memory pool: {e}")

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

        logger.info("WebSocket client stopped")

    async def _cancel_tasks(self):
        """Cancel all tasks"""
        if self.main_task:
            self.main_task.cancel()

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.ws:
            await self.ws.close()

        self.connected = False

    def subscribe(self, channels):
        """
        Subscribe to WebSocket channels

        Args:
            channels: Channel or list of channels to subscribe to
        """
        if isinstance(channels, str):
            channels = [channels]

        # Add to subscription set
        new_channels = []
        for channel in channels:
            if channel not in self.subscriptions:
                self.subscriptions.add(channel)
                new_channels.append(channel)

        if not new_channels:
            logger.info("No new channels to subscribe to")
            return

        # Send subscription message if connected
        if self.connected and self.event_loop and not self.event_loop.is_closed():
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(new_channels)
            }

            async def send_subscribe():
                try:
                    await self.ws.send(json.dumps(subscribe_message))
                    logger.info(
                        f"Subscribed to channels: {', '.join(new_channels)}")
                except Exception as e:
                    logger.error(f"Error subscribing to channels: {e}")

            asyncio.run_coroutine_threadsafe(send_subscribe(), self.event_loop)
        else:
            logger.info(
                f"Added channels to subscription list (will subscribe when connected): {', '.join(new_channels)}")

    def unsubscribe(self, channels):
        """
        Unsubscribe from WebSocket channels

        Args:
            channels: Channel or list of channels to unsubscribe from
        """
        if isinstance(channels, str):
            channels = [channels]

        # Remove from subscription set
        removed_channels = []
        for channel in channels:
            if channel in self.subscriptions:
                self.subscriptions.remove(channel)
                removed_channels.append(channel)

        if not removed_channels:
            logger.info("No channels to unsubscribe from")
            return

        # Send unsubscription message if connected
        if self.connected and self.event_loop and not self.event_loop.is_closed():
            unsubscribe_message = {
                "action": "unsubscribe",
                "params": ",".join(removed_channels)
            }

            async def send_unsubscribe():
                try:
                    await self.ws.send(json.dumps(unsubscribe_message))
                    logger.info(
                        f"Unsubscribed from channels: {', '.join(removed_channels)}")
                except Exception as e:
                    logger.error(f"Error unsubscribing from channels: {e}")

            asyncio.run_coroutine_threadsafe(
                send_unsubscribe(), self.event_loop)
        else:
            logger.info(
                f"Removed channels from subscription list: {', '.join(removed_channels)}")

    def add_message_handler(self, event_type, handler):
        """
        Add a message handler for a specific event type

        Args:
            event_type: Event type to handle (e.g., 'T' for trades, 'Q' for quotes)
            handler: Callback function to handle the message
        """
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []

        self.message_handlers[event_type].append(handler)
        logger.info(f"Added message handler for event type: {event_type}")

    def remove_message_handler(self, event_type, handler):
        """
        Remove a message handler for a specific event type

        Args:
            event_type: Event type to handle (e.g., 'T' for trades, 'Q' for quotes)
            handler: Callback function to remove
        """
        if event_type in self.message_handlers:
            if handler in self.message_handlers[event_type]:
                self.message_handlers[event_type].remove(handler)
                logger.info(
                    f"Removed message handler for event type: {event_type}")

            if not self.message_handlers[event_type]:
                del self.message_handlers[event_type]


# Helper functions for common subscriptions

def subscribe_to_trades(client, tickers):
    """
    Subscribe to trade events for specific tickers

    Args:
        client: GPUPolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"T.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_quotes(client, tickers):
    """
    Subscribe to quote events for specific tickers

    Args:
        client: GPUPolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"Q.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_minute_aggs(client, tickers):
    """
    Subscribe to minute aggregates for specific tickers

    Args:
        client: GPUPolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"AM.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_second_aggs(client, tickers):
    """
    Subscribe to second aggregates for specific tickers

    Args:
        client: GPUPolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"A.{ticker}" for ticker in tickers]
    client.subscribe(channels)


# Example usage
if __name__ == "__main__":
    # Create client
    api_key = os.environ.get('POLYGON_API_KEY', '')

    if not api_key:
        logger.error(
            "API key is required. Please set the POLYGON_API_KEY environment variable.")
        logger.info("Example: export POLYGON_API_KEY=your_api_key_here")
        logger.info("You can get an API key from https://polygon.io/")
        sys.exit(1)

    client = GPUPolygonWebSocketClient(api_key=api_key)

    # Define message handlers
    def trade_handler(message):
        ticker = message.get('sym')
        price = message.get('p')
        size = message.get('s')
        timestamp = message.get('t')

        if ticker and price and size and timestamp:
            dt = datetime.fromtimestamp(timestamp / 1000.0)
            print(f"[{dt}] TRADE: {ticker} - Price: ${price:.2f}, Size: {size}")

    def quote_handler(message):
        ticker = message.get('sym')
        bid_price = message.get('bp')
        bid_size = message.get('bs')
        ask_price = message.get('ap')
        ask_size = message.get('as')
        timestamp = message.get('t')

        if ticker and bid_price and ask_price and timestamp:
            dt = datetime.fromtimestamp(timestamp / 1000.0)
            print(
                f"[{dt}] QUOTE: {ticker} - Bid: ${bid_price:.2f} x {bid_size}, Ask: ${ask_price:.2f} x {ask_size}")

    def agg_handler(message):
        ticker = message.get('sym')
        open_price = message.get('o')
        high_price = message.get('h')
        low_price = message.get('l')
        close_price = message.get('c')
        volume = message.get('v')
        timestamp = message.get('s')  # Start timestamp

        if ticker and close_price and timestamp:
            dt = datetime.fromtimestamp(timestamp / 1000.0)
            print(f"[{dt}] AGG: {ticker} - O: ${open_price:.2f}, H: ${high_price:.2f}, L: ${low_price:.2f}, C: ${close_price:.2f}, Vol: {volume}")

    # Add message handlers
    client.add_message_handler('T', trade_handler)
    client.add_message_handler('Q', quote_handler)
    client.add_message_handler('AM', agg_handler)

    # Start client
    client.start()

    try:
        # Subscribe to some channels
        tickers = os.environ.get('DEFAULT_TICKERS', 'SPY,QQQ,IWM').split(',')
        subscribe_to_trades(client, tickers)
        subscribe_to_quotes(client, tickers)
        subscribe_to_minute_aggs(client, tickers)

        # Keep the main thread running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Stop client
        client.stop()
