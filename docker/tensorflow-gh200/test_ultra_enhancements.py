#!/usr/bin/env python3
"""
Test script for enhanced polygon_data_source_ultra.py implementation
This script demonstrates the memory monitoring and resource management improvements
"""

import os
import time
import logging
import argparse
import json
import uuid
import random
import asyncio
import requests
import websockets
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient as EnhancedUnusualWhalesClient
from polygon_data_source_ultra import PolygonDataSourceUltra, log_memory_usage, monitor_worker_processes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_ultra_enhancements')

# API Keys
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf')
# Set the Unusual Whales API key directly to ensure it's available
UNUSUAL_WHALES_API_KEY = '4ad71b9e-7ace-4f24-bdfc-532ace219a18'

class PolygonWebSocketClient:
    """Polygon WebSocket client for real-time market data"""
    
    def __init__(self, api_key=POLYGON_API_KEY):
        """Initialize WebSocket client"""
        self.api_key = api_key
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.subscribed_channels = set()
        self.messages = []
        self.running = True
        
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            self.ws = await websockets.connect("wss://socket.polygon.io/stocks")
            self.connected = True
            logger.info("Connected to Polygon WebSocket")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
            
    async def authenticate(self):
        """Authenticate with API key"""
        if not self.connected:
            logger.error("Cannot authenticate: not connected")
            return False
            
        auth_message = {"action": "auth", "params": self.api_key}
        await self.ws.send(json.dumps(auth_message))
        
        # Wait for auth response
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=5)
            data = json.loads(response)
            
            if isinstance(data, list) and len(data) > 0 and "status" in data[0] and data[0]["status"] == "auth_success":
                self.authenticated = True
                logger.info("Authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {data}")
                return False
        except asyncio.TimeoutError:
            logger.error("Authentication timed out")
            return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
        
    async def subscribe(self, channels):
        """Subscribe to channels"""
        if not self.authenticated:
            logger.error("Cannot subscribe: not authenticated")
            return False
            
        sub_message = {"action": "subscribe", "params": ",".join(channels)}
        await self.ws.send(json.dumps(sub_message))
        self.subscribed_channels.update(channels)
        logger.info(f"Subscribed to channels: {channels}")
        return True
        
    async def listen(self, duration=10):
        """Listen for messages for a specified duration"""
        if not self.authenticated:
            logger.error("Cannot listen: not authenticated")
            return []
            
        logger.info(f"Listening for messages for {duration} seconds...")
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1)
                data = json.loads(message)
                self.messages.append(data)
                logger.debug(f"Received message: {data}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
                
        logger.info(f"Received {len(self.messages)} messages")
        return self.messages
        
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            self.authenticated = False
            logger.info("WebSocket connection closed")

    def generate_mock_data(self, symbols, num_messages=50):
        """Generate mock WebSocket data for testing during non-market hours"""
        logger.info(f"Generating {num_messages} mock messages for {len(symbols)} symbols")
        mock_messages = []
        
        for _ in range(num_messages):
            symbol = random.choice(symbols)
            
            # Generate a random message type
            message_type = random.choice(["T", "Q", "A"])
            
            if message_type == "T":  # Trade
                message = {
                    "ev": "T",
                    "sym": symbol,
                    "p": round(random.uniform(50, 500), 2),  # Price
                    "s": random.randint(1, 1000),  # Size
                    "t": int(time.time() * 1000),  # Timestamp
                    "c": [0],  # Conditions
                    "i": str(uuid.uuid4())  # Trade ID
                }
            elif message_type == "Q":  # Quote
                price = round(random.uniform(50, 500), 2)
                message = {
                    "ev": "Q",
                    "sym": symbol,
                    "bp": price - random.uniform(0.01, 0.1),  # Bid price
                    "bs": random.randint(1, 1000),  # Bid size
                    "ap": price + random.uniform(0.01, 0.1),  # Ask price
                    "as": random.randint(1, 1000),  # Ask size
                    "t": int(time.time() * 1000),  # Timestamp
                    "c": [0],  # Conditions
                    "i": str(uuid.uuid4())  # Quote ID
                }
            else:  # Aggregate
                price = round(random.uniform(50, 500), 2)
                message = {
                    "ev": "A",
                    "sym": symbol,
                    "v": random.randint(1, 10000),  # Volume
                    "o": price - random.uniform(0.5, 2),  # Open
                    "c": price,  # Close
                    "h": price + random.uniform(0.1, 1),  # High
                    "l": price - random.uniform(0.1, 1),  # Low
                    "a": random.randint(1, 100),  # Average
                    "s": int(time.time() * 1000),  # Start timestamp
                    "e": int(time.time() * 1000) + 60000  # End timestamp
                }
                
            mock_messages.append(message)
            
        self.messages = mock_messages
        return mock_messages

def test_basic_functionality(symbols=None, timespan="day"):
    """Test basic functionality of the enhanced implementation"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    logger.info(f"Testing basic functionality with symbols: {symbols}")
    
    # Create client
    client = PolygonDataSourceUltra()
    
    # Log initial memory state
    log_memory_usage("initial_state")
    
    # Fetch data
    logger.info("Fetching data...")
    start_time = time.time()
    results = client.get_aggregates_batch(symbols, timespan=timespan)
    fetch_time = time.time() - start_time
    
    # Log memory after fetch
    log_memory_usage("after_fetch")
    
    # Process results
    record_count = 0
    for symbol, df in results.items():
        record_count += len(df)
        logger.info(f"{symbol}: {len(df)} data points")
    
    # Process with GPU
    logger.info("Processing with GPU...")
    start_time = time.time()
    processed = client.process_data_with_gpu(results)
    process_time = time.time() - start_time
    
    # Log memory after processing
    log_memory_usage("after_processing")
    
    # Print results
    logger.info("\nProcessed Results:")
    for symbol, df in processed.items():
        if df is not None:
            logger.info(f"{symbol}:")
            logger.info(df)
    
    logger.info(f"\nFetch time: {fetch_time:.4f} seconds")
    logger.info(f"Process time: {process_time:.4f} seconds")
    logger.info(f"Total time: {fetch_time + process_time:.4f} seconds")
    logger.info(f"Processed {record_count} records for {len(symbols)} symbols")
    logger.info(f"Throughput: {record_count / (fetch_time + process_time):.2f} records/second")
    
    # Check worker processes
    monitor_worker_processes()
    
    # Close client
    logger.info("Closing client...")
    client.close()
    
    # Log final memory state
    log_memory_usage("final_state")
    
    return {
        'symbols': symbols,
        'record_count': record_count,
        'fetch_time': fetch_time,
        'process_time': process_time,
        'total_time': fetch_time + process_time
    }

def test_stress_test(num_symbols=20, timespan="day"):
    """Run a stress test with a larger number of symbols"""
    # Use a predefined list of symbols for consistency
    all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
                  "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO",
                  "VZ", "ADBE", "CRM", "CMCSA", "PFE", "KO", "PEP", "ABT", "MRK", "WMT"]
    
    symbols = all_symbols[:min(num_symbols, len(all_symbols))]
    
    logger.info(f"Running stress test with {len(symbols)} symbols")
    
    # Create client
    client = PolygonDataSourceUltra()
    
    # Log initial memory state
    log_memory_usage("stress_test_initial")
    
    # Fetch data
    logger.info("Fetching data...")
    start_time = time.time()
    results = client.get_aggregates_batch(symbols, timespan=timespan)
    fetch_time = time.time() - start_time
    
    # Log memory after fetch
    log_memory_usage("stress_test_after_fetch")
    
    # Process results
    record_count = 0
    for symbol, df in results.items():
        record_count += len(df)
    
    # Process with GPU
    logger.info("Processing with GPU...")
    start_time = time.time()
    processed = client.process_data_with_gpu(results)
    process_time = time.time() - start_time
    
    # Log memory after processing
    log_memory_usage("stress_test_after_processing")
    
    logger.info(f"Processed {record_count} records for {len(symbols)} symbols")
    logger.info(f"Fetch time: {fetch_time:.4f} seconds")
    logger.info(f"Process time: {process_time:.4f} seconds")
    logger.info(f"Total time: {fetch_time + process_time:.4f} seconds")
    logger.info(f"Throughput: {record_count / (fetch_time + process_time):.2f} records/second")
    
    # Check worker processes
    monitor_worker_processes()
    
    # Close client
    logger.info("Closing client...")
    client.close()
    
    # Log final memory state
    log_memory_usage("stress_test_final")
    
    return {
        'symbols': symbols,
        'record_count': record_count,
        'fetch_time': fetch_time,
        'process_time': process_time,
        'total_time': fetch_time + process_time
    }

def test_memory_leak_detection(iterations=5, symbols=None):
    """Test for memory leaks by running multiple iterations"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]
    
    logger.info(f"Testing for memory leaks with {iterations} iterations")
    
    memory_usage = []
    
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")
        
        # Create client
        client = PolygonDataSourceUltra()
        
        # Log initial memory state
        log_memory_usage(f"iteration_{i+1}_initial")
        
        # Fetch and process data
        results = client.get_aggregates_batch(symbols, timespan="day")
        processed = client.process_data_with_gpu(results)
        
        # Log memory after processing
        log_memory_usage(f"iteration_{i+1}_after_processing")
        
        # Close client
        client.close()
        
        # Log memory after closing
        log_memory_usage(f"iteration_{i+1}_after_close")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Wait a bit to allow for cleanup
        time.sleep(2)
    
    logger.info("Memory leak test completed")
    
    return memory_usage

async def test_polygon_websocket(symbols=None, duration=10, use_mock=True):
    """Test Polygon WebSocket functionality"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    logger.info(f"Testing Polygon WebSocket with symbols: {symbols}")
    
    # Create WebSocket client
    client = PolygonWebSocketClient()
    
    if use_mock:
        # Generate mock data for testing during non-market hours
        mock_data = client.generate_mock_data(symbols, num_messages=50)
        logger.info(f"Generated {len(mock_data)} mock messages")
        
        # Print sample of mock data
        if mock_data:
            logger.info(f"Sample mock message: {mock_data[0]}")
            
        return {
            'symbols': symbols,
            'message_count': len(mock_data),
            'mock': True
        }
    else:
        # Connect to WebSocket
        connected = await client.connect()
        if not connected:
            logger.error("Failed to connect to WebSocket")
            return {'error': 'Connection failed'}
        
        try:
            # Authenticate
            authenticated = await client.authenticate()
            if not authenticated:
                logger.error("Failed to authenticate with WebSocket")
                return {'error': 'Authentication failed'}
            
            # Subscribe to channels
            channels = []
            for symbol in symbols:
                channels.extend([f"T.{symbol}", f"Q.{symbol}", f"A.{symbol}"])
            
            subscribed = await client.subscribe(channels)
            if not subscribed:
                logger.error("Failed to subscribe to channels")
                return {'error': 'Subscription failed'}
            
            # Listen for messages
            messages = await client.listen(duration)
            
            # Close connection
            await client.close()
            
            return {
                'symbols': symbols,
                'message_count': len(messages),
                'mock': False
            }
        except Exception as e:
            logger.error(f"Error in WebSocket test: {e}")
            await client.close()
            return {'error': str(e)}

def test_unusual_whales(limit=10, test_new_endpoints=True):
    """Test Unusual Whales API functionality"""
    logger.info(f"Testing Unusual Whales API with limit: {limit}")
    
    # Create Unusual Whales client
    # Explicitly set the API key to ensure it's used
    client = EnhancedUnusualWhalesClient(
        api_key=UNUSUAL_WHALES_API_KEY
    )
    
    results = {}
    
    # Test options flow (previously unusual options)
    start_time = time.time()
    options_flow = client.get_options_flow(limit=limit)
    fetch_time = time.time() - start_time
    
    logger.info(f"Using Unusual Whales API key: {UNUSUAL_WHALES_API_KEY}")
    # Process results
    options_count = len(options_flow)
    logger.info(f"Retrieved {options_count} options flow entries in {fetch_time:.4f} seconds")
    
    # Print sample data
    if options_count > 0:
        logger.info(f"Sample options flow: {options_flow[0]}")
    
    results['options_flow'] = options_flow
    
    # Test new endpoints if requested
    if test_new_endpoints:
        # Test flow alerts
        logger.info("Testing flow alerts endpoint...")
        ticker = "AAPL"
        flow_alerts = client.get_flow_alerts(ticker=ticker, limit=5)
        logger.info(f"Retrieved {len(flow_alerts)} flow alerts for {ticker}")
        if flow_alerts:
            logger.info(f"Sample flow alert: {flow_alerts[0]}")
        results['flow_alerts'] = flow_alerts
        
        # Test dark pool data
        logger.info("Testing dark pool data endpoints...")
        recent_darkpool = client.get_darkpool_recent(limit=5)
        logger.info(f"Retrieved {len(recent_darkpool)} recent dark pool trades")
        if recent_darkpool:
            logger.info(f"Sample dark pool trade: {recent_darkpool[0]}")
        results['darkpool_recent'] = recent_darkpool
        
        ticker_darkpool = client.get_darkpool_ticker(ticker=ticker, limit=5)
        logger.info(f"Retrieved {len(ticker_darkpool)} dark pool trades for {ticker}")
        if ticker_darkpool:
            logger.info(f"Sample {ticker} dark pool trade: {ticker_darkpool[0]}")
        results['darkpool_ticker'] = ticker_darkpool
        
        # Test insider trading
        logger.info("Testing insider trading endpoints...")
        transactions = client.get_insider_transactions(limit=5)
        logger.info(f"Retrieved {len(transactions)} insider transactions")
        if transactions:
            logger.info(f"Sample insider transaction: {transactions[0]}")
        results['insider_transactions'] = transactions
        
        # Test alerts
        logger.info("Testing alerts endpoints...")
        alerts = client.get_alerts(limit=5)
        logger.info(f"Retrieved {len(alerts)} alerts")
        if alerts:
            logger.info(f"Sample alert: {alerts[0]}")
        results['alerts'] = alerts
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test enhanced polygon_data_source_ultra.py")
    parser.add_argument("--test", choices=["basic", "stress", "memory", "all"], default="basic",
                        help="Test to run (default: basic)")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,AMZN,TSLA",
                        help="Comma-separated list of symbols to test")
    parser.add_argument("--num-symbols", type=int, default=20,
                        help="Number of symbols for stress test")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations for memory leak test")
    parser.add_argument("--websocket", action="store_true",
                        help="Test Polygon WebSocket functionality")
    parser.add_argument("--unusual-whales", action="store_true",
                        help="Test Unusual Whales API functionality")
    parser.add_argument("--use-mock", action="store_true", default=True,
                        help="Use mock data for testing during non-market hours")
    parser.add_argument("--test-new-endpoints", action="store_true", default=True,
                        help="Test new Unusual Whales API endpoints")
    parser.add_argument("--timespan", type=str, default="day", choices=["minute", "hour", "day", "week", "month", "quarter", "year"],
                        help="Timespan for data (default: day)")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else None
    
    # Run tests
    if args.test == "basic" or args.test == "all":
        test_basic_functionality(symbols, args.timespan)
    
    if args.test == "stress" or args.test == "all":
        test_stress_test(args.num_symbols, args.timespan)
    
    if args.test == "memory" or args.test == "all":
        test_memory_leak_detection(args.iterations, symbols[:3] if symbols else None)
        
    if args.websocket or args.test == "all":
        # Run WebSocket test asynchronously
        asyncio.run(test_polygon_websocket(
            symbols=symbols[:5] if symbols else None,
            duration=10,
            use_mock=args.use_mock
        ))
    
    if args.unusual_whales or args.test == "all":
        test_unusual_whales(limit=10, test_new_endpoints=args.test_new_endpoints)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()