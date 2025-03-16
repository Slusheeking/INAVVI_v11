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
from financial_cuda_ops import process_order_book, calculate_technical_indicators, normalize_tick_data, fused_financial_ops

# Polygon API configuration
API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
BASE_URL = "https://api.polygon.io"


class PolygonDataSource:
    """
    Data source for Polygon.io API
    """

    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def get_tickers(self, market="stocks", limit=100):
        """
        Get list of tickers

        Args:
            market: Market type (stocks, crypto, forex)
            limit: Maximum number of tickers to return

        Returns:
            tickers: List of tickers
        """
        url = f"{BASE_URL}/v3/reference/tickers"
        params = {
            "market": market,
            "active": True,
            "limit": limit,
            "apiKey": self.api_key
        }

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [ticker["ticker"] for ticker in data["results"]]
        else:
            print(f"Error fetching tickers: {response.status_code}")
            return []

    def get_aggregates(self, ticker, multiplier=1, timespan="minute",
                       from_date=None, to_date=None, limit=10000):
        """
        Get aggregated data for a ticker

        Args:
            ticker: Ticker symbol
            multiplier: Multiplier for timespan
            timespan: Timespan (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            df: Pandas DataFrame with aggregated data
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

        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
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
        else:
            print(
                f"Error fetching aggregates for {ticker}: {response.status_code}")
            return pd.DataFrame()

    def get_trades(self, ticker, date=None, limit=50000):
        """
        Get trades for a ticker

        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            df: Pandas DataFrame with trades
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
            response = self.session.get(next_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
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
            else:
                print(
                    f"Error fetching trades for {ticker}: {response.status_code}")
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

        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            df: Pandas DataFrame with quotes
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
            response = self.session.get(next_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
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
            else:
                print(
                    f"Error fetching quotes for {ticker}: {response.status_code}")
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

        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            order_book: Dictionary with order book data
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
    Market data source using Polygon.io API
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

    def start(self):
        """
        Start the data source
        """
        self._stop_event.clear()

        # Fetch data for all symbols
        print(
            f"Fetching {self.data_type} data for {len(self.symbols)} symbols...")

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

        Args:
            timeout: Timeout in seconds

        Returns:
            batch: Batch of data or None if timeout
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _fetch_data(self, symbol):
        """
        Fetch data for a symbol

        Args:
            symbol: Symbol to fetch data for
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
        Run the data source
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
                for _, row in data.iterrows():
                    batch_prices.append(row["close"])
                    batch_volumes.append(row["volume"])
                    batch_timestamps.append(row["timestamp"])
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

                        # Check if stopped
                        if self._stop_event.is_set():
                            return

        # Put any remaining data in the queue
        if self.data_type == "aggregates" and batch_prices:
            batch = {
                'prices': np.array(batch_prices, dtype=np.float32),
                'volumes': np.array(batch_volumes, dtype=np.float32),
                'timestamps': np.array(batch_timestamps, dtype=np.float64),
                'symbols': batch_symbols
            }

            try:
                self._data_queue.put(batch, timeout=0.1)
            except queue.Full:
                pass

        elif self.data_type == "order_book" and batch_bids:
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
                pass

        # Signal that we're done
        self._data_queue.put(None)


def create_polygon_dataset(symbols, data_type="aggregates", timespan="minute",
                           from_date=None, to_date=None, batch_size=256):
    """
    Create a TensorFlow dataset from Polygon.io data

    Args:
        symbols: List of symbols
        data_type: Type of data (aggregates, trades, quotes, order_book)
        timespan: Timespan for aggregates
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        batch_size: Batch size

    Returns:
        dataset: TensorFlow dataset
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
                for i in range(len(batch['symbols'])):
                    yield {
                        'price': batch['prices'][i],
                        'volume': batch['volumes'][i],
                        'timestamp': batch['timestamps'][i],
                        'symbol': batch['symbols'][i]
                    }
            elif data_type == "order_book":
                for i in range(len(batch['symbols'])):
                    yield {
                        'bids': batch['bids'][i],
                        'asks': batch['asks'][i],
                        'bid_sizes': batch['bid_sizes'][i],
                        'ask_sizes': batch['ask_sizes'][i],
                        'symbol': batch['symbols'][i]
                    }

    if data_type == "aggregates":
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                'price': tf.TensorSpec(shape=(), dtype=tf.float32),
                'volume': tf.TensorSpec(shape=(), dtype=tf.float32),
                'timestamp': tf.TensorSpec(shape=(), dtype=tf.float64),
                'symbol': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
    elif data_type == "order_book":
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                'bids': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                'asks': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                'bid_sizes': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                'ask_sizes': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                'symbol': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )

    # Apply optimizations
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def benchmark_polygon_data(num_symbols=10, data_type="aggregates", timespan="minute"):
    """
    Benchmark Polygon.io data processing

    Args:
        num_symbols: Number of symbols to use
        data_type: Type of data (aggregates, trades, quotes, order_book)
        timespan: Timespan for aggregates

    Returns:
        results: Benchmark results
    """
    print("\n" + "="*50)
    print("POLYGON.IO DATA BENCHMARK")
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

    for batch in dataset:
        batch_count += 1
        record_count += len(batch['symbol'])

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


def process_polygon_data_with_cuda(symbols, data_type="aggregates", timespan="minute"):
    """
    Process Polygon.io data with CUDA kernels

    Args:
        symbols: List of symbols to use
        data_type: Type of data (aggregates, trades, quotes, order_book)
        timespan: Timespan for aggregates

    Returns:
        results: Processing results
    """
    print("\n" + "="*50)
    print("POLYGON.IO DATA PROCESSING WITH CUDA")
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
                mask = tf.equal(batch['symbol'], symbol)
                if tf.reduce_any(mask):
                    prices = tf.boolean_mask(batch['price'], mask)
                    symbol_data.append(prices)

            if not symbol_data:
                continue

            # Concatenate all batches
            prices = tf.concat(symbol_data, axis=0)

            # Reshape for technical indicators
            prices = tf.reshape(prices, (1, -1))

            # Calculate technical indicators
            start_time = time.time()
            ema, rsi, upper_band, lower_band = calculate_technical_indicators(
                prices)
            indicators_time = time.time() - start_time

            # Calculate MACD
            start_time = time.time()
            macd, signal, histogram = fused_financial_ops(prices)
            macd_time = time.time() - start_time

            results[symbol] = {
                'data_length': prices.shape[1],
                'indicators_time': indicators_time,
                'macd_time': macd_time
            }

            print(f"{symbol}: Processed {prices.shape[1]} price points")
            print(f"  Technical indicators: {indicators_time:.4f} seconds")
            print(f"  MACD calculation: {macd_time:.4f} seconds")

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
                mask = tf.equal(batch['symbol'], symbol)
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

            # Process order book
            start_time = time.time()
            liquidity, imbalance, spread = process_order_book(
                bids, asks, bid_sizes, ask_sizes)
            order_book_time = time.time() - start_time

            results[symbol] = {
                'data_length': bids.shape[0],
                'order_book_time': order_book_time
            }

            print(f"{symbol}: Processed {bids.shape[0]} order book snapshots")
            print(f"  Order book processing: {order_book_time:.4f} seconds")

    return results


if __name__ == "__main__":
    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Benchmark aggregates data
    benchmark_polygon_data(num_symbols=len(symbols), data_type="aggregates")

    # Process data with CUDA
    process_polygon_data_with_cuda(symbols, data_type="aggregates")
