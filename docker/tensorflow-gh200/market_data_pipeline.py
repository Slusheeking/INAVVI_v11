import tensorflow as tf
import numpy as np
import time
import threading
import queue
import os
import json
from concurrent.futures import ThreadPoolExecutor
from financial_cuda_ops import process_order_book, calculate_technical_indicators, normalize_tick_data, fused_financial_ops

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)

# Set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


class MarketDataSource:
    """
    Base class for market data sources
    """

    def __init__(self, symbols, batch_size=256):
        self.symbols = symbols
        self.batch_size = batch_size
        self._stop_event = threading.Event()
        self._data_queue = queue.Queue(maxsize=100)

    def start(self):
        """
        Start the data source
        """
        self._stop_event.clear()
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

        Args:
            timeout: Timeout in seconds

        Returns:
            batch: Batch of data or None if timeout
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        """
        Run the data source (to be implemented by subclasses)
        """
        raise NotImplementedError("Subclasses must implement _run")


class SimulatedMarketDataSource(MarketDataSource):
    """
    Simulated market data source for testing
    """

    def __init__(self, symbols, batch_size=256, tick_interval=0.001):
        super(SimulatedMarketDataSource, self).__init__(symbols, batch_size)
        self.tick_interval = tick_interval

        # Initialize prices for each symbol
        self.prices = {symbol: 100.0 for symbol in symbols}
        self.volatilities = {symbol: np.random.uniform(
            0.01, 0.05) for symbol in symbols}

    def _run(self):
        """
        Run the simulated data source
        """
        batch_prices = []
        batch_volumes = []
        batch_timestamps = []
        batch_symbols = []

        while not self._stop_event.is_set():
            # Generate a tick for each symbol
            for symbol in self.symbols:
                # Update price with random walk
                price_change = np.random.normal(0, self.volatilities[symbol])
                self.prices[symbol] *= (1 + price_change)

                # Generate random volume
                volume = np.random.exponential(1000)

                # Add to batch
                batch_prices.append(self.prices[symbol])
                batch_volumes.append(volume)
                batch_timestamps.append(time.time())
                batch_symbols.append(symbol)

                # If batch is full, put it in the queue
                if len(batch_prices) >= self.batch_size:
                    batch = {
                        'prices': np.array(batch_prices, dtype=np.float32),
                        'volumes': np.array(batch_volumes, dtype=np.float32),
                        'timestamps': np.array(batch_timestamps, dtype=np.float64),
                        'symbols': batch_symbols
                    }

                    try:
                        self._data_queue.put(batch, timeout=0.1)
                    except queue.Full:
                        # If queue is full, discard the batch
                        pass

                    # Reset batch
                    batch_prices = []
                    batch_volumes = []
                    batch_timestamps = []
                    batch_symbols = []

            # Sleep for tick interval
            time.sleep(self.tick_interval)


class OrderBookDataSource(MarketDataSource):
    """
    Order book data source
    """

    def __init__(self, symbols, batch_size=256, depth=10, tick_interval=0.001):
        super(OrderBookDataSource, self).__init__(symbols, batch_size)
        self.depth = depth
        self.tick_interval = tick_interval

        # Initialize order books for each symbol
        self.order_books = {}
        for symbol in symbols:
            mid_price = 100.0
            spread = np.random.uniform(0.01, 0.1)

            # Initialize bids and asks
            bids = np.array([mid_price - spread/2 - i *
                            0.01 for i in range(depth)], dtype=np.float32)
            asks = np.array([mid_price + spread/2 + i *
                            0.01 for i in range(depth)], dtype=np.float32)

            # Initialize sizes
            bid_sizes = np.random.exponential(
                100, size=depth).astype(np.float32)
            ask_sizes = np.random.exponential(
                100, size=depth).astype(np.float32)

            self.order_books[symbol] = {
                'bids': bids,
                'asks': asks,
                'bid_sizes': bid_sizes,
                'ask_sizes': ask_sizes,
                'mid_price': mid_price,
                'spread': spread
            }

    def _run(self):
        """
        Run the order book data source
        """
        batch_bids = []
        batch_asks = []
        batch_bid_sizes = []
        batch_ask_sizes = []
        batch_symbols = []

        while not self._stop_event.is_set():
            # Generate an order book update for each symbol
            for symbol in self.symbols:
                # Get current order book
                order_book = self.order_books[symbol]

                # Update mid price with random walk
                price_change = np.random.normal(0, 0.01)
                order_book['mid_price'] *= (1 + price_change)

                # Update spread with mean reversion
                target_spread = np.random.uniform(0.01, 0.1)
                order_book['spread'] = 0.9 * \
                    order_book['spread'] + 0.1 * target_spread

                # Update bids and asks
                mid_price = order_book['mid_price']
                spread = order_book['spread']

                bids = np.array(
                    [mid_price - spread/2 - i*0.01 for i in range(self.depth)], dtype=np.float32)
                asks = np.array(
                    [mid_price + spread/2 + i*0.01 for i in range(self.depth)], dtype=np.float32)

                # Update sizes with some randomness
                bid_sizes = order_book['bid_sizes'] * \
                    np.random.uniform(0.9, 1.1, size=self.depth)
                ask_sizes = order_book['ask_sizes'] * \
                    np.random.uniform(0.9, 1.1, size=self.depth)

                # Store updated order book
                order_book['bids'] = bids
                order_book['asks'] = asks
                order_book['bid_sizes'] = bid_sizes
                order_book['ask_sizes'] = ask_sizes

                # Add to batch
                batch_bids.append(bids)
                batch_asks.append(asks)
                batch_bid_sizes.append(bid_sizes)
                batch_ask_sizes.append(ask_sizes)
                batch_symbols.append(symbol)

                # If batch is full, put it in the queue
                if len(batch_bids) >= self.batch_size:
                    batch = {
                        'bids': np.array(batch_bids, dtype=np.float32),
                        'asks': np.array(batch_asks, dtype=np.float32),
                        'bid_sizes': np.array(batch_bid_sizes, dtype=np.float32),
                        'ask_sizes': np.array(batch_ask_sizes, dtype=np.float32),
                        'symbols': batch_symbols
                    }

                    try:
                        self._data_queue.put(batch, timeout=0.1)
                    except queue.Full:
                        # If queue is full, discard the batch
                        pass

                    # Reset batch
                    batch_bids = []
                    batch_asks = []
                    batch_bid_sizes = []
                    batch_ask_sizes = []
                    batch_symbols = []

            # Sleep for tick interval
            time.sleep(self.tick_interval)


class MarketDataProcessor:
    """
    Process market data in real-time
    """

    def __init__(self, window_size=100, feature_dim=10):
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Initialize feature buffers for each symbol
        self.feature_buffers = {}

        # Initialize normalization parameters
        self.normalization_params = {}

    def process_tick_data(self, batch):
        """
        Process a batch of tick data

        Args:
            batch: Batch of tick data

        Returns:
            features: Extracted features
        """
        prices = batch['prices']
        volumes = batch['volumes']
        timestamps = batch['timestamps']
        symbols = batch['symbols']

        # Process each symbol
        results = {}
        for i, symbol in enumerate(symbols):
            # Get price and volume
            price = prices[i]
            volume = volumes[i]
            timestamp = timestamps[i]

            # Initialize buffer if needed
            if symbol not in self.feature_buffers:
                self.feature_buffers[symbol] = {
                    'prices': np.zeros(self.window_size, dtype=np.float32),
                    'volumes': np.zeros(self.window_size, dtype=np.float32),
                    'features': np.zeros((self.window_size, self.feature_dim), dtype=np.float32),
                    'position': 0,
                    'filled': False
                }

            # Update buffer
            buffer = self.feature_buffers[symbol]
            position = buffer['position']

            buffer['prices'][position] = price
            buffer['volumes'][position] = volume

            # Extract features
            if buffer['filled'] or position >= self.window_size - 1:
                # Get price window
                if buffer['filled']:
                    price_window = buffer['prices']
                    volume_window = buffer['volumes']
                else:
                    price_window = buffer['prices'][:position+1]
                    volume_window = buffer['volumes'][:position+1]

                # Convert to TensorFlow tensors
                price_tensor = tf.convert_to_tensor(
                    price_window, dtype=tf.float32)
                price_tensor = tf.expand_dims(
                    price_tensor, axis=0)  # Add batch dimension

                # Calculate technical indicators
                ema, rsi, upper_band, lower_band = calculate_technical_indicators(
                    price_tensor)

                # Calculate MACD
                macd, signal, histogram = fused_financial_ops(price_tensor)

                # Extract features
                features = np.zeros(self.feature_dim, dtype=np.float32)

                # Basic features
                features[0] = price
                features[1] = volume

                # Technical indicators
                if buffer['filled'] or position >= 14:
                    features[2] = rsi[0, -1].numpy()  # Latest RSI

                # MACD
                if buffer['filled'] or position >= 30:
                    features[3] = macd[0, -1].numpy()  # Latest MACD
                    features[4] = signal[0, -1].numpy()  # Latest signal
                    features[5] = histogram[0, -1].numpy()  # Latest histogram

                # Bollinger Bands
                if buffer['filled'] or position >= 20:
                    # Latest upper band
                    features[6] = upper_band[0, -1].numpy()
                    # Latest lower band
                    features[7] = lower_band[0, -1].numpy()

                    # Relative position within bands
                    band_width = features[6] - features[7]
                    if band_width > 0:
                        features[8] = (price - features[7]) / \
                            band_width  # 0 to 1

                # Store features
                buffer['features'][position] = features

                # Return latest features
                results[symbol] = features

            # Update position
            buffer['position'] = (position + 1) % self.window_size
            if position + 1 >= self.window_size:
                buffer['filled'] = True

        return results

    def process_order_book(self, batch):
        """
        Process a batch of order book data

        Args:
            batch: Batch of order book data

        Returns:
            features: Extracted features
        """
        bids = batch['bids']
        asks = batch['asks']
        bid_sizes = batch['bid_sizes']
        ask_sizes = batch['ask_sizes']
        symbols = batch['symbols']

        # Convert to TensorFlow tensors
        bids_tensor = tf.convert_to_tensor(bids, dtype=tf.float32)
        asks_tensor = tf.convert_to_tensor(asks, dtype=tf.float32)
        bid_sizes_tensor = tf.convert_to_tensor(bid_sizes, dtype=tf.float32)
        ask_sizes_tensor = tf.convert_to_tensor(ask_sizes, dtype=tf.float32)

        # Process order book
        liquidity, imbalance, spread = process_order_book(
            bids_tensor, asks_tensor, bid_sizes_tensor, ask_sizes_tensor
        )

        # Process each symbol
        results = {}
        for i, symbol in enumerate(symbols):
            # Extract features
            features = np.zeros(self.feature_dim, dtype=np.float32)

            # Order book features
            features[0] = spread[i].numpy()  # Spread
            features[1] = liquidity[i].numpy()  # Liquidity
            features[2] = imbalance[i].numpy()  # Imbalance

            # Calculate mid price
            mid_price = (bids[i, 0] + asks[i, 0]) / 2
            features[3] = mid_price

            # Calculate weighted mid price
            total_size = np.sum(bid_sizes[i]) + np.sum(ask_sizes[i])
            if total_size > 0:
                weighted_mid = (
                    np.sum(bids[i] * bid_sizes[i]) + np.sum(asks[i] * ask_sizes[i])) / total_size
                features[4] = weighted_mid

            # Store features
            results[symbol] = features

        return results


class RealTimeFeaturePipeline:
    """
    Real-time feature pipeline for market data
    """

    def __init__(self, symbols, window_size=100, feature_dim=10, batch_size=256):
        self.symbols = symbols
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.batch_size = batch_size

        # Create data sources
        self.tick_data_source = SimulatedMarketDataSource(symbols, batch_size)
        self.order_book_source = OrderBookDataSource(symbols, batch_size)

        # Create data processor
        self.processor = MarketDataProcessor(window_size, feature_dim)

        # Create output queue
        self.feature_queue = queue.Queue(maxsize=100)

        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Create stop event
        self._stop_event = threading.Event()

    def start(self):
        """
        Start the pipeline
        """
        self._stop_event.clear()

        # Start data sources
        self.tick_data_source.start()
        self.order_book_source.start()

        # Start processing threads
        self.tick_future = self.executor.submit(self._process_tick_data)
        self.order_book_future = self.executor.submit(self._process_order_book)

    def stop(self):
        """
        Stop the pipeline
        """
        self._stop_event.set()

        # Stop data sources
        self.tick_data_source.stop()
        self.order_book_source.stop()

        # Wait for processing threads to finish
        self.executor.shutdown(wait=True)

    def get_features(self, timeout=1.0):
        """
        Get features from the pipeline

        Args:
            timeout: Timeout in seconds

        Returns:
            features: Dictionary of features by symbol
        """
        try:
            return self.feature_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _process_tick_data(self):
        """
        Process tick data
        """
        while not self._stop_event.is_set():
            # Get batch from source
            batch = self.tick_data_source.get_batch()
            if batch is None:
                continue

            # Process batch
            features = self.processor.process_tick_data(batch)

            # Put features in queue
            try:
                self.feature_queue.put(('tick', features), timeout=0.1)
            except queue.Full:
                # If queue is full, discard the features
                pass

    def _process_order_book(self):
        """
        Process order book data
        """
        while not self._stop_event.is_set():
            # Get batch from source
            batch = self.order_book_source.get_batch()
            if batch is None:
                continue

            # Process batch
            features = self.processor.process_order_book(batch)

            # Put features in queue
            try:
                self.feature_queue.put(('order_book', features), timeout=0.1)
            except queue.Full:
                # If queue is full, discard the features
                pass


def create_tf_dataset_from_pipeline(pipeline, batch_size=32, prefetch_size=10):
    """
    Create a TensorFlow dataset from a real-time pipeline

    Args:
        pipeline: Real-time feature pipeline
        batch_size: Batch size
        prefetch_size: Prefetch size

    Returns:
        dataset: TensorFlow dataset
    """
    def generator():
        while True:
            features = pipeline.get_features()
            if features is None:
                continue

            data_type, feature_dict = features

            for symbol, feature_vector in feature_dict.items():
                yield feature_vector, symbol

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )

    # Apply optimizations
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset


def benchmark_pipeline(num_symbols=10, duration=10.0):
    """
    Benchmark the real-time pipeline

    Args:
        num_symbols: Number of symbols
        duration: Duration in seconds

    Returns:
        results: Benchmark results
    """
    print("Benchmarking real-time market data pipeline...")

    # Create symbols
    symbols = [f"SYMBOL_{i}" for i in range(num_symbols)]

    # Create pipeline
    pipeline = RealTimeFeaturePipeline(symbols)

    # Start pipeline
    pipeline.start()

    # Process features for the specified duration
    start_time = time.time()
    tick_count = 0
    order_book_count = 0

    try:
        while time.time() - start_time < duration:
            features = pipeline.get_features(timeout=0.1)
            if features is not None:
                data_type, feature_dict = features
                if data_type == 'tick':
                    tick_count += 1
                else:
                    order_book_count += 1
    finally:
        # Stop pipeline
        pipeline.stop()

    # Calculate throughput
    elapsed = time.time() - start_time
    tick_throughput = tick_count / elapsed
    order_book_throughput = order_book_count / elapsed
    total_throughput = (tick_count + order_book_count) / elapsed

    print(
        f"Processed {tick_count} tick batches and {order_book_count} order book batches in {elapsed:.2f} seconds")
    print(f"Tick throughput: {tick_throughput:.2f} batches/second")
    print(f"Order book throughput: {order_book_throughput:.2f} batches/second")
    print(f"Total throughput: {total_throughput:.2f} batches/second")

    return {
        'tick_count': tick_count,
        'order_book_count': order_book_count,
        'elapsed': elapsed,
        'tick_throughput': tick_throughput,
        'order_book_throughput': order_book_throughput,
        'total_throughput': total_throughput
    }


if __name__ == "__main__":
    benchmark_pipeline()
