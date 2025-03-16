import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import json
import requests
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
from financial_cuda_ops import process_order_book, calculate_technical_indicators, normalize_tick_data, fused_financial_ops

# Polygon API configuration
API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
BASE_URL = "https://api.polygon.io"

# Enable XLA JIT compilation for better performance on GH200
tf.config.optimizer.set_jit(True)

# Configure TensorFlow for GH200
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        # Enable memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(device, True)
    print(
        f"Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")

    # Get GPU details
    for i, device in enumerate(physical_devices):
        details = tf.config.experimental.get_device_details(device)
        print(f"GPU {i} details: {details}")

    # Set TensorFlow to use the GPU
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

# Set CuPy to use unified memory for better performance on GH200
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


class PolygonDataSource:
    """
    Data source for Polygon.io API
    """

    def __init__(self, api_key=API_KEY, max_retries=3, retry_delay=1.0):
        self.api_key = api_key
        self.session = requests.Session()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create a cache for API responses
        self.cache = {}

    def _make_request(self, url, params):
        """
        Make a request to the Polygon API with retry logic
        """
        # Check cache first
        cache_key = f"{url}_{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Make request with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    # Cache the response
                    self.cache[cache_key] = response.json()
                    return self.cache[cache_key]
                elif response.status_code == 429:  # Rate limit
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    break
            except Exception as e:
                print(f"Request error: {e}")
                time.sleep(self.retry_delay * (attempt + 1))

        return None

    def get_tickers(self, market="stocks", limit=100):
        """
        Get list of tickers
        """
        url = f"{BASE_URL}/v3/reference/tickers"
        params = {
            "market": market,
            "active": True,
            "limit": limit,
            "apiKey": self.api_key
        }

        data = self._make_request(url, params)
        if data and "results" in data:
            return [ticker["ticker"] for ticker in data["results"]]
        else:
            print(f"Error fetching tickers")
            return []

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                       from_date=None, to_date=None, limit=10000):
        """
        Get aggregated data for a ticker
        """
        # Set default dates if not provided
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)
                         ).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": True,
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key
        }

        data = self._make_request(url, params)
        if data and "results" in data and data["results"]:
            # Use pandas for efficient data processing
            df = pd.DataFrame(data["results"])
            # Rename columns
            df = df.rename(columns={
                "v": "volume",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "t": "timestamp",
                "n": "transactions"
            })
            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        else:
            print(f"No data found for {ticker}")
            return pd.DataFrame()

    def get_trades(self, ticker, date=None, limit=50000):
        """
        Get trades for a ticker
        """
        # Set default date if not provided
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        url = f"{BASE_URL}/v3/trades/{ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": limit,
            "apiKey": self.api_key
        }

        all_trades = []
        next_url = url

        # Paginate through results
        while next_url and len(all_trades) < limit:
            data = self._make_request(next_url, params)
            if data and "results" in data and data["results"]:
                all_trades.extend(data["results"])

                # Check for next page
                if "next_url" in data:
                    next_url = data["next_url"]
                    # Next URL already has other params
                    params = {"apiKey": self.api_key}
                else:
                    break
            else:
                break

        if all_trades:
            df = pd.DataFrame(all_trades)
            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(
                df["participant_timestamp"], unit="ns")
            return df
        else:
            print(f"No trades found for {ticker} on {date}")
            return pd.DataFrame()

    def get_quotes(self, ticker, date=None, limit=50000):
        """
        Get quotes for a ticker
        """
        # Set default date if not provided
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        url = f"{BASE_URL}/v3/quotes/{ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": limit,
            "apiKey": self.api_key
        }

        all_quotes = []
        next_url = url

        # Paginate through results
        while next_url and len(all_quotes) < limit:
            data = self._make_request(next_url, params)
            if data and "results" in data and data["results"]:
                all_quotes.extend(data["results"])

                # Check for next page
                if "next_url" in data:
                    next_url = data["next_url"]
                    # Next URL already has other params
                    params = {"apiKey": self.api_key}
                else:
                    break
            else:
                break

        if all_quotes:
            df = pd.DataFrame(all_quotes)
            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ns")
            return df
        else:
            print(f"No quotes found for {ticker} on {date}")
            return pd.DataFrame()

    def get_order_book(self, ticker, date=None, limit=10000):
        """
        Simulate order book from quotes
        """
        quotes_df = self.get_quotes(ticker, date, limit)
        if quotes_df.empty:
            return None

        # Create order book snapshots
        snapshots = []

        # Group quotes by minute
        quotes_df["minute"] = quotes_df["datetime"].dt.floor("1min")
        grouped = quotes_df.groupby("minute")

        for minute, group in grouped:
            # Get last 10 unique bid and ask prices
            bids = group.sort_values(
                "bid_price", ascending=False).drop_duplicates("bid_price").head(10)
            asks = group.sort_values(
                "ask_price").drop_duplicates("ask_price").head(10)

            if len(bids) < 10 or len(asks) < 10:
                continue

            # Create snapshot
            snapshot = {
                "timestamp": minute.timestamp(),
                "bids": bids["bid_price"].values,
                "asks": asks["ask_price"].values,
                "bid_sizes": bids["bid_size"].values,
                "ask_sizes": asks["ask_size"].values
            }

            snapshots.append(snapshot)

        return snapshots


class PolygonMarketDataSource:
    """
    Market data source using Polygon.io API optimized for GH200
    """

    def __init__(self, symbols, batch_size=256, data_type="aggregates",
                 timespan="minute", from_date=None, to_date=None):
        self.symbols = symbols
        self.batch_size = batch_size
        self.data_type = data_type  # aggregates, trades, quotes, order_book
        self.timespan = timespan
        self.from_date = from_date
        self.to_date = to_date

        self.polygon = PolygonDataSource()
        self._stop_event = threading.Event()
        self._data_queue = queue.Queue(maxsize=100)

        # Cache for data
        self.data_cache = {}

        # Pre-allocate GPU memory for batches
        self._initialize_gpu_memory()

    def _initialize_gpu_memory(self):
        """
        Pre-allocate GPU memory for batches to avoid fragmentation
        """
        # Allocate memory for different data types
        if self.data_type == "aggregates":
            # Pre-allocate memory for prices, volumes, timestamps
            self._gpu_prices = cp.zeros((self.batch_size,), dtype=cp.float32)
            self._gpu_volumes = cp.zeros((self.batch_size,), dtype=cp.float32)
            self._gpu_timestamps = cp.zeros(
                (self.batch_size,), dtype=cp.float64)
        elif self.data_type == "order_book":
            # Pre-allocate memory for order book data
            self._gpu_bids = cp.zeros((self.batch_size, 10), dtype=cp.float32)
            self._gpu_asks = cp.zeros((self.batch_size, 10), dtype=cp.float32)
            self._gpu_bid_sizes = cp.zeros(
                (self.batch_size, 10), dtype=cp.float32)
            self._gpu_ask_sizes = cp.zeros(
                (self.batch_size, 10), dtype=cp.float32)

    def start(self):
        """
        Start the data source
        """
        self._stop_event.clear()

        # Fetch data for all symbols
        print(
            f"Fetching {self.data_type} data for {len(self.symbols)} symbols...")

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=min(10, len(self.symbols))) as executor:
            futures = []
            for symbol in self.symbols:
                futures.append(executor.submit(self._fetch_data, symbol))

            for future in futures:
                future.result()

        # Start processing thread
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """
        Stop the data source
        """
        self._stop_event.set()
        if hasattr(self, '_thread'):
            self._thread.join(timeout=5.0)

    def get_batch(self, timeout=1.0):
        """
        Get a batch of data
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _fetch_data(self, symbol):
        """
        Fetch data for a symbol
        """
        if self.data_type == "aggregates":
            df = self.polygon.get_aggregates(
                symbol,
                timespan=self.timespan,
                from_date=self.from_date,
                to_date=self.to_date
            )
            if not df.empty:
                self.data_cache[symbol] = df
                print(f"Fetched {len(df)} {self.timespan} bars for {symbol}")

        elif self.data_type == "trades":
            df = self.polygon.get_trades(symbol, date=self.from_date)
            if not df.empty:
                self.data_cache[symbol] = df
                print(f"Fetched {len(df)} trades for {symbol}")

        elif self.data_type == "quotes":
            df = self.polygon.get_quotes(symbol, date=self.from_date)
            if not df.empty:
                self.data_cache[symbol] = df
                print(f"Fetched {len(df)} quotes for {symbol}")

        elif self.data_type == "order_book":
            snapshots = self.polygon.get_order_book(
                symbol, date=self.from_date)
            if snapshots:
                self.data_cache[symbol] = snapshots
                print(
                    f"Created {len(snapshots)} order book snapshots for {symbol}")

    def _run(self):
        """
        Run the data source with GH200 optimizations
        """
        # Process data in batches
        batch_prices = []
        batch_volumes = []
        batch_timestamps = []
        batch_symbols = []

        batch_bids = []
        batch_asks = []
        batch_bid_sizes = []
        batch_ask_sizes = []

        # Iterate through cached data
        for symbol, data in self.data_cache.items():
            if self.data_type == "aggregates":
                # Process OHLCV data
                # Convert to numpy arrays for faster processing
                if isinstance(data, pd.DataFrame):
                    prices = data["close"].values
                    volumes = data["volume"].values
                    timestamps = data["timestamp"].values

                    # Process in chunks to avoid memory issues
                    chunk_size = min(len(prices), self.batch_size)
                    for i in range(0, len(prices), chunk_size):
                        end_idx = min(i + chunk_size, len(prices))
                        chunk_prices = prices[i:end_idx]
                        chunk_volumes = volumes[i:end_idx]
                        chunk_timestamps = timestamps[i:end_idx]

                        # Add to batch
                        batch_prices.extend(chunk_prices)
                        batch_volumes.extend(chunk_volumes)
                        batch_timestamps.extend(chunk_timestamps)
                        batch_symbols.extend([symbol] * len(chunk_prices))

                        # If batch is full, process it
                        if len(batch_prices) >= self.batch_size:
                            self._process_aggregate_batch(
                                batch_prices, batch_volumes, batch_timestamps, batch_symbols)

                            # Reset batch
                            batch_prices = []
                            batch_volumes = []
                            batch_timestamps = []
                            batch_symbols = []

                        # Check if stopped
                        if self._stop_event.is_set():
                            return

            elif self.data_type == "order_book":
                # Process order book data
                for snapshot in data:
                    batch_bids.append(snapshot["bids"])
                    batch_asks.append(snapshot["asks"])
                    batch_bid_sizes.append(snapshot["bid_sizes"])
                    batch_ask_sizes.append(snapshot["ask_sizes"])
                    batch_symbols.append(symbol)

                    # If batch is full, process it
                    if len(batch_bids) >= self.batch_size:
                        self._process_order_book_batch(
                            batch_bids, batch_asks, batch_bid_sizes, batch_ask_sizes, batch_symbols)

                        # Reset batch
                        batch_bids = []
                        batch_asks = []
                        batch_bid_sizes = []
                        batch_ask_sizes = []
                        batch_symbols = []

                    # Check if stopped
                    if self._stop_event.is_set():
                        return

        # Put any remaining data in the queue
        if self.data_type == "aggregates" and batch_prices:
            self._process_aggregate_batch(
                batch_prices, batch_volumes, batch_timestamps, batch_symbols)

        elif self.data_type == "order_book" and batch_bids:
            self._process_order_book_batch(
                batch_bids, batch_asks, batch_bid_sizes, batch_ask_sizes, batch_symbols)

        # Signal that we're done
        self._data_queue.put(None)

    def _process_aggregate_batch(self, prices, volumes, timestamps, symbols):
        """
        Process a batch of aggregate data with GH200 optimizations
        """
        # Convert to numpy arrays
        prices_np = np.array(prices, dtype=np.float32)
        volumes_np = np.array(volumes, dtype=np.float32)
        timestamps_np = np.array(timestamps, dtype=np.float64)

        # Create batch dictionary
        batch = {
            'prices': prices_np,
            'volumes': volumes_np,
            'timestamps': timestamps_np,
            'symbols': symbols
        }

        try:
            self._data_queue.put(batch, timeout=0.1)
        except queue.Full:
            # If queue is full, discard the batch
            pass

    def _process_order_book_batch(self, bids, asks, bid_sizes, ask_sizes, symbols):
        """
        Process a batch of order book data with GH200 optimizations
        """
        # Convert to numpy arrays
        bids_np = np.array(bids, dtype=np.float32)
        asks_np = np.array(asks, dtype=np.float32)
        bid_sizes_np = np.array(bid_sizes, dtype=np.float32)
        ask_sizes_np = np.array(ask_sizes, dtype=np.float32)

        # Create batch dictionary
        batch = {
            'bids': bids_np,
            'asks': asks_np,
            'bid_sizes': bid_sizes_np,
            'ask_sizes': ask_sizes_np,
            'symbols': symbols
        }

        try:
            self._data_queue.put(batch, timeout=0.1)
        except queue.Full:
            # If queue is full, discard the batch
            pass


def create_polygon_dataset(symbols, data_type="aggregates", timespan="minute",
                           from_date=None, to_date=None, batch_size=256):
    """
    Create a TensorFlow dataset from Polygon.io data optimized for GH200
    """
    # Set default dates if not provided
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")

    # Create data source
    data_source = PolygonMarketDataSource(
        symbols,
        batch_size=batch_size,
        data_type=data_type,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )

    # Start data source
    data_source.start()

    # Create dataset
    def generator():
        while True:
            batch = data_source.get_batch()
            if batch is None:
                break

            if data_type == "aggregates":
                # Yield entire batches for better performance
                yield batch
            elif data_type == "order_book":
                # Yield entire batches for better performance
                yield batch

    # Create output signatures based on data type
    if data_type == "aggregates":
        output_signature = {
            'prices': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'volumes': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'timestamps': tf.TensorSpec(shape=(None,), dtype=tf.float64),
            'symbols': tf.TensorSpec(shape=(None,), dtype=tf.string)
        }
    elif data_type == "order_book":
        output_signature = {
            'bids': tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            'asks': tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            'bid_sizes': tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            'ask_sizes': tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
            'symbols': tf.TensorSpec(shape=(None,), dtype=tf.string)
        }
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Create dataset with optimizations
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Apply GH200-specific optimizations
    options = tf.data.Options()
    options.experimental_optimization.noop_elimination = True
    # Use TF 2.15.0 compatible options
    options.experimental_optimization.parallel_batch = True
    options.experimental_threading.max_intra_op_parallelism = 8
    options.experimental_threading.private_threadpool_size = 16
    dataset = dataset.with_options(options)

    # Apply additional optimizations
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def benchmark_polygon_data_gh200(num_symbols=10, data_type="aggregates", timespan="minute"):
    """
    Benchmark Polygon.io data processing on GH200
    """
    print("\n" + "="*50)
    print("POLYGON.IO DATA BENCHMARK ON GH200")
    print("="*50)

    # Get top symbols by market cap
    polygon = PolygonDataSource()
    all_symbols = polygon.get_tickers(limit=100)
    symbols = all_symbols[:num_symbols]

    print(f"Using symbols: {symbols}")

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Create dataset
    start_time = time.time()

    dataset = create_polygon_dataset(
        symbols,
        data_type=data_type,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )

    # Process dataset
    batch_count = 0
    record_count = 0

    # Use TensorFlow's optimized iteration
    for batch in dataset:
        batch_count += 1
        if data_type == "aggregates":
            record_count += len(batch['symbols'])
        elif data_type == "order_book":
            record_count += len(batch['symbols'])

    elapsed = time.time() - start_time

    print(f"Processed {record_count} records in {batch_count} batches")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Throughput: {record_count / elapsed:.2f} records/second")

    return {
        'symbols': symbols,
        'data_type': data_type,
        'timespan': timespan,
        'from_date': from_date,
        'to_date': to_date,
        'batch_count': batch_count,
        'record_count': record_count,
        'elapsed': elapsed,
        'throughput': record_count / elapsed
    }


def process_polygon_data_with_cuda_gh200(symbols, data_type="aggregates", timespan="minute"):
    """
    Process Polygon.io data with CUDA kernels optimized for GH200
    """
    print("\n" + "="*50)
    print("POLYGON.IO DATA PROCESSING WITH CUDA ON GH200")
    print("="*50)

    # Set date range (last 30 days)
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Create dataset
    dataset = create_polygon_dataset(
        symbols,
        data_type=data_type,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )

    results = {}

    if data_type == "aggregates":
        # Process price data with technical indicators
        for symbol in symbols:
            # Get data for this symbol
            symbol_data = []
            for batch in dataset:
                mask = tf.equal(batch['symbols'], symbol)
                if tf.reduce_any(mask):
                    prices = tf.boolean_mask(batch['prices'], mask)
                    symbol_data.append(prices)

            if not symbol_data:
                continue

            # Concatenate all batches
            prices = tf.concat(symbol_data, axis=0)

            # Reshape for technical indicators
            prices = tf.reshape(prices, (1, -1))

            # Calculate technical indicators - without XLA compilation
            # This avoids the EagerPyFunc error
            start_time = time.time()
            # Convert to numpy and use the CuPy-based function directly
            prices_np = prices.numpy().astype(np.float32)
            # Call the function directly with numpy array
            ema, rsi, upper_band, lower_band = calculate_technical_indicators(
                prices_np)
            indicators_time = time.time() - start_time

            # Calculate MACD with XLA compilation
            @tf.function(jit_compile=True)
            def calculate_macd(prices):
                return fused_financial_ops(prices)

            start_time = time.time()
            macd, signal, histogram = calculate_macd(prices)
            macd_time = time.time() - start_time

            results[symbol] = {
                'data_length': prices.shape[1],
                'indicators_time': indicators_time,
                'macd_time': macd_time
            }

            print(f"{symbol}: Processed {prices.shape[1]} price points")
            print(f"  Technical indicators: {indicators_time:.4f} seconds")
            print(f"  MACD calculation: {macd_time:.4f} seconds")
            print(
                f"  Processing rate: {prices.shape[1] / (indicators_time + macd_time):.2f} points/second")

    elif data_type == "order_book":
        # Process order book data
        for symbol in symbols:
            # Get data for this symbol
            symbol_data = {
                'bids': [],
                'asks': [],
                'bid_sizes': [],
                'ask_sizes': []
            }

            for batch in dataset:
                mask = tf.equal(batch['symbols'], symbol)
                if tf.reduce_any(mask):
                    symbol_data['bids'].append(
                        tf.boolean_mask(batch['bids'], mask))
                    symbol_data['asks'].append(
                        tf.boolean_mask(batch['asks'], mask))
                    symbol_data['bid_sizes'].append(
                        tf.boolean_mask(batch['bid_sizes'], mask))
                    symbol_data['ask_sizes'].append(
                        tf.boolean_mask(batch['ask_sizes'], mask))

            if not symbol_data['bids']:
                continue

            # Concatenate all batches
            bids = tf.concat(symbol_data['bids'], axis=0)
            asks = tf.concat(symbol_data['asks'], axis=0)
            bid_sizes = tf.concat(symbol_data['bid_sizes'], axis=0)
            ask_sizes = tf.concat(symbol_data['ask_sizes'], axis=0)

            # Process order book with XLA compilation
            @tf.function(jit_compile=True)
            def process_ob(bids, asks, bid_sizes, ask_sizes):
                return process_order_book(bids, asks, bid_sizes, ask_sizes)

            start_time = time.time()
            liquidity, imbalance, spread = process_ob(
                bids, asks, bid_sizes, ask_sizes)
            order_book_time = time.time() - start_time

            results[symbol] = {
                'data_length': bids.shape[0],
                'order_book_time': order_book_time
            }

            print(f"{symbol}: Processed {bids.shape[0]} order book snapshots")
            print(f"  Order book processing: {order_book_time:.4f} seconds")
            print(
                f"  Processing rate: {bids.shape[0] / order_book_time:.2f} snapshots/second")

    return results


def compare_performance(original_results, gh200_results):
    """
    Compare performance between original and GH200-optimized implementations
    """
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)

    for symbol in gh200_results:
        if symbol in original_results:
            print(f"\nSymbol: {symbol}")

            if 'indicators_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['indicators_time']
                gh200_time = gh200_results[symbol]['indicators_time']
                speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  Technical indicators: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {speedup:.2f}x)")

            if 'macd_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['macd_time']
                gh200_time = gh200_results[symbol]['macd_time']
                speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  MACD calculation: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {speedup:.2f}x)")

            if 'order_book_time' in gh200_results[symbol]:
                orig_time = original_results[symbol]['order_book_time']
                gh200_time = gh200_results[symbol]['order_book_time']
                speedup = orig_time / gh200_time if gh200_time > 0 else 0
                print(
                    f"  Order book processing: {orig_time:.4f}s vs {gh200_time:.4f}s (Speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Benchmark aggregates data
    print("\nRunning original benchmark...")
    from polygon_data_source import benchmark_polygon_data, process_polygon_data_with_cuda
    original_benchmark = benchmark_polygon_data(
        num_symbols=len(symbols), data_type="aggregates")
    original_results = process_polygon_data_with_cuda(
        symbols, data_type="aggregates")

    print("\nRunning GH200-optimized benchmark...")
    gh200_benchmark = benchmark_polygon_data_gh200(
        num_symbols=len(symbols), data_type="aggregates")
    gh200_results = process_polygon_data_with_cuda_gh200(
        symbols, data_type="aggregates")

    # Compare performance
    print("\nBenchmark comparison:")
    print(
        f"Original throughput: {original_benchmark['throughput']:.2f} records/second")
    print(
        f"GH200 throughput: {gh200_benchmark['throughput']:.2f} records/second")
    print(
        f"Speedup: {gh200_benchmark['throughput'] / original_benchmark['throughput']:.2f}x")

    # Compare detailed performance
    compare_performance(original_results, gh200_results)
