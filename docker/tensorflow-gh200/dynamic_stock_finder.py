#!/usr/bin/env python3
"""
Dynamic Stock Finder for NVIDIA GH200
Efficiently scans thousands of stocks to find optimal candidates for trading positions
Leverages GPU acceleration for parallel screening and analysis
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import cupy as cp
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import queue
import json
import requests
import random
from polygon_data_source_ultra import PolygonDataSourceUltra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dynamic_stock_finder')

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Memory growth configuration failed: {e}")

# Constants
DEFAULT_POSITION_SIZE = 5000.0  # Default total position size in dollars
DEFAULT_UNIVERSE_SIZE = 10000   # Default number of stocks to scan
DEFAULT_TOP_CANDIDATES = 500    # Default number of top candidates to return
DEFAULT_BATCH_SIZE = 100        # Reduced batch size for more reliable processing
DEFAULT_HISTORY_DAYS = 90       # Default number of days of historical data to analyze
DEFAULT_WORKER_COUNT = mp.cpu_count()  # Default number of worker processes


class StockScreeningCriteria:
    """Stock screening criteria for finding optimal candidates"""

    def __init__(self):
        """Initialize screening criteria with default values"""
        # Price range criteria
        self.min_price = 0.99  # Minimum price set to $0.99 as per requirement
        self.max_price = 5000.0  # Further increased to include more high-priced stocks

        # Volume criteria
        self.min_avg_volume = 10000  # Further lowered to include more stocks

        # Volatility criteria
        # Lowered from 0.01 (1%) to 0.5% daily volatility
        self.min_volatility = 0.001  # Further lowered to include more stocks
        # Increased from 0.05 (5%) to 10% daily volatility
        self.max_volatility = 0.50  # Further increased to 50% to include more volatile stocks

        # Technical indicators
        self.rsi_lower_bound = 20  # Lowered from 30 to include more oversold stocks
        self.rsi_upper_bound = 80  # Increased from 70 to include more overbought stocks

        # Momentum criteria
        # Lowered from 0.02 (2%) to 0% to include stocks with neutral momentum
        self.min_momentum = -0.05  # Allow slightly negative momentum

        # Fundamental criteria (if available)
        self.min_market_cap = 10000000  # Lowered to $10M market cap
        self.max_pe_ratio = 50

    def to_dict(self):
        """Convert criteria to dictionary"""
        return {
            "price_range": {
                "min": self.min_price,
                "max": self.max_price
            },
            "volume": {
                "min_avg": self.min_avg_volume
            },
            "volatility": {
                "min": self.min_volatility,
                "max": self.max_volatility
            },
            "technical": {
                "rsi_lower": self.rsi_lower_bound,
                "rsi_upper": self.rsi_upper_bound
            },
            "momentum": {
                "min": self.min_momentum
            },
            "fundamental": {
                "min_market_cap": self.min_market_cap,
                "max_pe_ratio": self.max_pe_ratio
            }
        }

    @classmethod
    def from_dict(cls, criteria_dict):
        """Create criteria from dictionary"""
        criteria = cls()

        if "price_range" in criteria_dict:
            criteria.min_price = criteria_dict["price_range"].get(
                "min", criteria.min_price)
            criteria.max_price = criteria_dict["price_range"].get(
                "max", criteria.max_price)

        if "volume" in criteria_dict:
            criteria.min_avg_volume = criteria_dict["volume"].get(
                "min_avg", criteria.min_avg_volume)

        if "volatility" in criteria_dict:
            criteria.min_volatility = criteria_dict["volatility"].get(
                "min", criteria.min_volatility)
            criteria.max_volatility = criteria_dict["volatility"].get(
                "max", criteria.max_volatility)

        if "technical" in criteria_dict:
            criteria.rsi_lower_bound = criteria_dict["technical"].get(
                "rsi_lower", criteria.rsi_lower_bound)
            criteria.rsi_upper_bound = criteria_dict["technical"].get(
                "rsi_upper", criteria.rsi_upper_bound)

        if "momentum" in criteria_dict:
            criteria.min_momentum = criteria_dict["momentum"].get(
                "min", criteria.min_momentum)

        if "fundamental" in criteria_dict:
            criteria.min_market_cap = criteria_dict["fundamental"].get(
                "min_market_cap", criteria.min_market_cap)
            criteria.max_pe_ratio = criteria_dict["fundamental"].get(
                "max_pe_ratio", criteria.max_pe_ratio)

        return criteria


class StockUniverse:
    """Stock universe manager for retrieving and caching stock lists"""

    def __init__(self, cache_dir="./data/cache"):
        """Initialize stock universe manager"""
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "stock_universe.json")
        self.last_update = None
        self.stocks = []

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Load cached stock universe if available
        self._load_cache()

    def _load_cache(self):
        """Load stock universe from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.stocks = data.get("stocks", [])
                    self.last_update = data.get("last_update")

                    # Check if cache is still valid (less than 1 day old)
                    if self.last_update:
                        last_update_time = datetime.fromisoformat(
                            self.last_update)
                        if (datetime.now() - last_update_time).days < 1:
                            logger.info(
                                f"Loaded {len(self.stocks)} stocks from cache")
                            return
            except Exception as e:
                logger.warning(
                    f"Failed to load stock universe from cache: {e}")

        # Cache is invalid or doesn't exist, fetch fresh data
        self._fetch_stock_universe()

    def _fetch_stock_universe(self):
        """Fetch stock universe from Polygon.io"""
        try:
            # Use Polygon.io API to fetch active tickers
            url = "https://api.polygon.io/v3/reference/tickers"
            api_key = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
            params = {
                "market": "stocks",
                "active": "true",
                "limit": 1000,
                "sort": "ticker",
                "apiKey": api_key
            }

            all_stocks = []
            next_url = url
            max_retries = 5  # Increased retries

            # Paginate through results up to the limit
            while next_url and len(all_stocks) < DEFAULT_UNIVERSE_SIZE:
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        response = requests.get(
                            next_url, params=params, timeout=60)  # Increased timeout
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get("results", [])
                            all_stocks.extend(results)

                            # Get next page URL
                            next_url = data.get("next_url")
                            if next_url:
                                # Remove API key from params since it's in the URL
                                params = {}
                            success = True
                        elif response.status_code == 403:
                            logger.error(f"API key unauthorized: {api_key}")
                            break
                        else:
                            logger.warning(
                                f"Failed to fetch stock universe (attempt {retry_count+1}/{max_retries}): {response.status_code}")
                            retry_count += 1
                            time.sleep(2 ** retry_count)  # Exponential backoff
                    except Exception as e:
                        logger.warning(
                            f"Error during API request (attempt {retry_count+1}/{max_retries}): {e}")
                        retry_count += 1
                        time.sleep(2 ** retry_count)  # Exponential backoff

                if not success:
                    logger.error(
                        f"Failed to fetch stock universe after {max_retries} retries")
                    break

            # Extract ticker symbols and filter out ETFs, ADRs, and other non-standard stocks
            filtered_stocks = []
            for stock in all_stocks:
                ticker = stock["ticker"]
                # Skip tickers with special characters (but allow some numbers)
                if len(ticker) > 5 or not all(c.isalnum() for c in ticker):
                    continue
                # Skip common ETF prefixes
                if ticker.startswith(('SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'XLF', 'XLE')):
                    continue
                # Skip if type is not CS (Common Stock)
                if stock.get("type") not in ["CS", None]:
                    continue
                # Skip if market is not stocks
                if stock.get("market") != "stocks":
                    continue
                filtered_stocks.append(ticker)

            # If we have too few stocks, add some reliable defaults
            if len(filtered_stocks) < 50:
                default_stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX",
                                  "INTC", "AMD", "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM",
                                  "DIS", "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "KO", "PEP"]
                for stock in default_stocks:
                    if stock not in filtered_stocks:
                        filtered_stocks.append(stock)
                logger.info(
                    f"Added {len(default_stocks)} default stocks to universe")

            self.stocks = filtered_stocks
            self.last_update = datetime.now().isoformat()

            # Save to cache
            self._save_cache()

            logger.info(f"Fetched {len(self.stocks)} stocks from Polygon.io")
        except Exception as e:
            logger.error(f"Error fetching stock universe: {e}")
            # Use a default list of major stocks if fetch fails
            self.stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX",
                           "INTC", "AMD", "JPM", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM",
                           "DIS", "CSCO", "VZ", "ADBE", "CRM", "CMCSA", "PFE", "KO", "PEP"]

    def _save_cache(self):
        """Save stock universe to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    "stocks": self.stocks,
                    "last_update": self.last_update
                }, f)
            logger.info(f"Saved {len(self.stocks)} stocks to cache")
        except Exception as e:
            logger.warning(f"Failed to save stock universe to cache: {e}")

    def get_stocks(self, limit=None):
        """Get stock universe"""

        # Ensure we get stocks from across the entire alphabet
        if limit and limit < len(self.stocks):
            # Group stocks by first letter
            stocks_by_letter = {}
            for stock in self.stocks:
                first_letter = stock[0].upper()
                if first_letter not in stocks_by_letter:
                    stocks_by_letter[first_letter] = []
                stocks_by_letter[first_letter].append(stock)

            # Get all available letters
            letters = sorted(stocks_by_letter.keys())

            # Ensure we have a good distribution across the alphabet
            # Prioritize letters beyond A and B to ensure diversity

            # First, allocate a minimum number from each letter group
            min_per_letter = max(1, limit // (len(letters) * 2))

            # Allocate more stocks to letters beyond A and B
            remaining_limit = limit - (min_per_letter * len(letters))
            stocks_per_letter = {}

            for letter in letters:
                # Base allocation
                stocks_per_letter[letter] = min_per_letter
                # Bonus allocation for letters beyond 'B'
                if letter > 'B' and remaining_limit > 0:
                    # Cap bonus at 5 per letter
                    bonus = min(remaining_limit // max(1, len(letters) - 2), 5)
                    stocks_per_letter[letter] += bonus
                    remaining_limit -= bonus

            # Select stocks from each letter group based on calculated allocations
            selected_stocks = []
            for letter in letters:
                letter_stocks = stocks_by_letter[letter]
                allocation = stocks_per_letter[letter]
                if len(letter_stocks) > allocation:
                    selected = random.sample(letter_stocks, allocation)
                else:
                    selected = letter_stocks
                selected_stocks.extend(selected)

            # If we haven't reached the limit, add more stocks randomly
            if len(selected_stocks) < limit:
                # Get all stocks not already selected
                remaining_stocks = [
                    s for s in self.stocks if s not in selected_stocks]
                # Add random stocks until we reach the limit
                if remaining_stocks:
                    additional = random.sample(
                        remaining_stocks,
                        min(limit - len(selected_stocks), len(remaining_stocks))
                    )
                    selected_stocks.extend(additional)

            # Shuffle the final selection to avoid any bias
            random.shuffle(selected_stocks)

            # Trim to the exact limit
            return selected_stocks[:limit]

        # If no limit or limit >= len(self.stocks), return all stocks (shuffled)
        shuffled = list(self.stocks)  # Create a copy
        random.shuffle(shuffled)
        return shuffled

    def refresh(self):
        """Force refresh of stock universe"""
        self._fetch_stock_universe()
        return self.stocks


class DynamicStockFinder:
    """Dynamic stock finder for identifying optimal trading candidates"""

    def __init__(self, position_size=DEFAULT_POSITION_SIZE, num_workers=None):
        """Initialize dynamic stock finder"""
        self.position_size = position_size
        # Limit workers to a smaller number to avoid system overload
        self.num_workers = num_workers if num_workers is not None else min(
            DEFAULT_WORKER_COUNT, 8)  # Cap at 8 workers to avoid system overload
        self.stock_universe = StockUniverse()
        self.data_source = None  # Initialize on demand to avoid unnecessary resource allocation
        self.criteria = StockScreeningCriteria()

        # Initialize processing pipeline
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.workers = []
        self.running = mp.Value('b', True)

        logger.info(
            f"Dynamic stock finder initialized with position size ${position_size}")

    def start_workers(self):
        """Start worker processes"""
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_process,
                args=(i, self.input_queue, self.output_queue,
                      self.running, self.criteria, self.position_size)
            )
            worker.daemon = False  # Changed from True to False to allow child processes
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.num_workers} worker processes")

    def stop_workers(self):
        """Stop worker processes"""
        self.running.value = False

        for worker in self.workers:
            # Increased timeout to allow for proper cleanup
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()

        self.workers = []
        logger.info("Stopped all worker processes")

    @staticmethod
    def _worker_process(worker_id, input_queue, output_queue, running, criteria, position_size=DEFAULT_POSITION_SIZE):
        """Worker process for analyzing stocks"""
        logger.info(f"Worker {worker_id} started")

        # Initialize data source for this worker
        data_source = PolygonDataSourceUltra(
            max_pool_size=30,  # Increase connection pool size
            max_retries=5,     # Increase retry attempts
            use_daemon=False   # Ensure child processes are not daemon processes
        )

        # Configure CuPy for this worker
        try:
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(
                cp.cuda.malloc_managed).malloc)
            logger.info(
                f"Worker {worker_id} configured CuPy with unified memory")
        except Exception as e:
            logger.warning(
                f"Failed to configure CuPy in worker {worker_id}: {e}")

        while running.value:
            try:
                # Get batch of stocks from input queue with timeout
                try:
                    batch = input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process each stock in the batch
                results = []
                for ticker in batch:
                    try:
                        # Fetch stock data
                        df = data_source.get_aggregates(
                            ticker=ticker,
                            timespan="day",
                            from_date=(datetime.now() -
                                       timedelta(days=DEFAULT_HISTORY_DAYS)).strftime("%Y-%m-%d"),
                            to_date=datetime.now().strftime("%Y-%m-%d")
                        )

                        if df.empty:
                            continue

                        # Calculate metrics
                        # Skip stocks with insufficient data
                        # Require at least 70% of the requested history
                        if len(df) < 10:  # Reduced from 70% of DEFAULT_HISTORY_DAYS to just 10 days
                            continue      # This allows more stocks to be considered

                        metrics = DynamicStockFinder._calculate_metrics(df)

                        # Apply screening criteria
                        if DynamicStockFinder._apply_criteria(metrics, criteria):
                            # Calculate score
                            score = DynamicStockFinder._calculate_score(
                                metrics)

                            # Calculate how many shares can be purchased with the position size
                            price = metrics["last_price"]
                            shares = int(position_size / price)

                            # Calculate position value
                            position_value = shares * price

                            # Calculate how well this stock fits our position size
                            # We want stocks where we can buy a reasonable number of shares
                            # Not too few (expensive stocks) and not too many (penny stocks)
                            fit_score = 0

                            # Ideal range: 5-100 shares for a position
                            if 5 <= shares <= 100:
                                fit_score = 10  # Perfect fit
                            elif 2 <= shares < 5 or 100 < shares <= 200:
                                fit_score = 8   # Good fit
                            elif 1 == shares or 200 < shares <= 500:
                                fit_score = 5   # Acceptable fit
                            else:
                                # Poor fit (too few or too many shares)
                                fit_score = 2

                            # Boost score based on position fit (up to 20% boost)
                            score = score * (1 + (fit_score / 50))

                            # Add to results
                            results.append({
                                "ticker": ticker,
                                "price": metrics["last_price"],
                                "volume": metrics["avg_volume"],
                                "volatility": metrics["volatility"],
                                "rsi": metrics["rsi"],
                                "momentum": metrics["momentum"],
                                "score": score,
                                "shares": shares
                            })
                    except Exception as e:
                        logger.warning(f"Error processing {ticker}: {e}")

                # Send results to output queue
                output_queue.put(results)
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")

        # Clean up resources
        try:
            data_source.close()
            cp.get_default_memory_pool().free_all_blocks()
            logger.info(f"Worker {worker_id} cleaned up resources")
        except Exception as e:
            logger.warning(
                f"Error cleaning up resources in worker {worker_id}: {e}")

        logger.info(f"Worker {worker_id} stopped")

    @staticmethod
    def _calculate_ema(data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([np.nan] * len(data))

        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[period-1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    @staticmethod
    def _calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(data) < slow_period + signal_period:
            return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

        # Calculate fast and slow EMAs
        fast_ema = DynamicStockFinder._calculate_ema(data, fast_period)
        slow_ema = DynamicStockFinder._calculate_ema(data, slow_period)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        signal_line = DynamicStockFinder._calculate_ema(
            macd_line, signal_period)

        # Calculate histogram (MACD line - signal line)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _calculate_bollinger_bands(data, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        if len(data) < period:
            return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

        # Calculate rolling mean (middle band)
        rolling_mean = np.array([np.nan] * len(data))
        for i in range(period-1, len(data)):
            rolling_mean[i] = np.mean(data[i-period+1:i+1])

        # Calculate rolling standard deviation
        rolling_std = np.array([np.nan] * len(data))
        for i in range(period-1, len(data)):
            rolling_std[i] = np.std(data[i-period+1:i+1])

        # Calculate upper and lower bands
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return upper_band, rolling_mean, lower_band

    @staticmethod
    def _calculate_adx(high, low, close, period=14):
        """Calculate Average Directional Index (ADX)"""
        if len(close) < period + 1:
            return np.array([np.nan] * len(close))

        # Calculate True Range (TR)
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        # Calculate +DM and -DM
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))

        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0

        # Calculate smoothed TR, +DM, and -DM
        smoothed_tr = np.zeros(len(close))
        smoothed_plus_dm = np.zeros(len(close))
        smoothed_minus_dm = np.zeros(len(close))

        # Initialize with simple averages
        smoothed_tr[period] = np.sum(tr[1:period+1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

        # Calculate smoothed values
        for i in range(period+1, len(close)):
            smoothed_tr[i] = smoothed_tr[i-1] - \
                (smoothed_tr[i-1] / period) + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - \
                (smoothed_plus_dm[i-1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - \
                (smoothed_minus_dm[i-1] / period) + minus_dm[i]

        # Calculate +DI and -DI
        # Add small epsilon (1e-8) to prevent division by zero
        plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-8))
        minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-8))

        # Calculate DX and ADX
        # Add small epsilon (1e-8) to prevent division by zero
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

        # Calculate ADX (smoothed DX)
        adx = np.zeros(len(close))
        adx[2*period-1] = np.mean(dx[period:2*period])

        for i in range(2*period, len(close)):
            adx[i] = ((adx[i-1] * (period-1)) + dx[i]) / period

        return adx

    @staticmethod
    def _calculate_obv(close, volume):
        """Calculate On-Balance Volume (OBV)"""
        obv = np.zeros(len(close))

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        return obv

    @staticmethod
    def _calculate_metrics(df):
        """Calculate metrics for a stock"""
        # Extract basic metrics
        try:
            last_price = df['close'].iloc[-1]
            avg_volume = df['volume'].mean()

            # Average volume over last 5 days - handle case where we have less than 5 days
            if len(df) >= 5:
                recent_volume = df['volume'].iloc[-5:].mean()
            else:
                recent_volume = avg_volume  # Use overall average if we don't have 5 days

            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()

            # Calculate RSI
            delta = df['close'].diff().dropna()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss

            avg_gain = gain.rolling(window=14).mean().dropna()
            avg_loss = loss.rolling(window=14).mean().dropna()

            if len(avg_gain) > 0 and len(avg_loss) > 0:
                rs = avg_gain.iloc[-1] / \
                    avg_loss.iloc[-1] if avg_loss.iloc[-1] > 0 else 100
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50

            # Calculate momentum (10-day return)
            if len(df) >= 10:
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-10]) - 1
            else:
                # For shorter periods, use the longest period available
                if len(df) >= 3:  # Need at least 3 days for a meaningful trend
                    momentum = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
                else:
                    momentum = 0

            # Calculate additional technical indicators if we have enough data
            additional_metrics = {}

            if len(df) >= 26:  # Minimum required for MACD
                # Convert to numpy arrays
                close_np = df['close'].values
                high_np = df['high'].values
                low_np = df['low'].values
                volume_np = df['volume'].values

                # MACD (Moving Average Convergence Divergence)
                macd, macd_signal, macd_hist = DynamicStockFinder._calculate_macd(
                    close_np, fast_period=12, slow_period=26, signal_period=9)

                # Bollinger Bands
                upper, middle, lower = DynamicStockFinder._calculate_bollinger_bands(
                    close_np, period=20, num_std=2)

                # ADX (Average Directional Index) - trend strength
                adx = DynamicStockFinder._calculate_adx(
                    high_np, low_np, close_np, period=14)

                # OBV (On-Balance Volume) - volume momentum
                obv = DynamicStockFinder._calculate_obv(close_np, volume_np)

                # Store the most recent values - with additional error checking
                try:
                    macd_val = macd[-1] if len(
                        macd) > 0 and not np.isnan(macd[-1]) else 0
                    macd_signal_val = macd_signal[-1] if len(
                        macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
                    macd_hist_val = macd_hist[-1] if len(
                        macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0

                    bb_upper_val = upper[-1] if len(upper) > 0 and not np.isnan(
                        upper[-1]) else last_price * 1.1
                    bb_middle_val = middle[-1] if len(
                        middle) > 0 and not np.isnan(middle[-1]) else last_price
                    bb_lower_val = lower[-1] if len(lower) > 0 and not np.isnan(
                        lower[-1]) else last_price * 0.9

                    adx_val = adx[-1] if len(
                        adx) > 0 and not np.isnan(adx[-1]) else 0
                    obv_val = obv[-1] if len(
                        obv) > 0 and not np.isnan(obv[-1]) else 0

                    # Calculate OBV change with careful index checking
                    if len(obv) >= 5 and obv[-5] != 0 and not np.isnan(obv[-5]):
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
                except IndexError as e:
                    # Handle any remaining index errors
                    logger.warning(f"Index error in technical indicators: {e}")
                    additional_metrics = {
                        "macd": 0, "macd_signal": 0, "macd_hist": 0,
                        "bb_upper": last_price * 1.1, "bb_middle": last_price, "bb_lower": last_price * 0.9,
                        "adx": 0, "obv": 0, "obv_change": 0
                    }
            else:
                additional_metrics = {
                    "macd": 0, "macd_signal": 0, "macd_hist": 0,
                    "bb_upper": last_price * 1.1, "bb_middle": last_price, "bb_lower": last_price * 0.9,
                    "adx": 0, "obv": 0, "obv_change": 0
                }

            return {
                "last_price": last_price,
                "avg_volume": avg_volume,
                "recent_volume": recent_volume,
                "volatility": volatility,
                "rsi": rsi,
                "momentum": momentum,
                **additional_metrics
            }
        except Exception as e:
            # Catch any other errors that might occur during metric calculation
            logger.warning(f"Error calculating metrics: {e}")
            raise  # Re-raise to be caught by the calling function

    @staticmethod
    def _apply_criteria(metrics, criteria):
        """Apply screening criteria to metrics"""
        try:
            # CRITICAL CRITERIA - must pass these
            # Price must be positive and not too high
            if metrics["last_price"] <= 0 or metrics["last_price"] > criteria.max_price:
                return False

            # Must have some trading volume
            if metrics["avg_volume"] < 1000:  # Very minimal volume requirement
                return False

            # SCORING CRITERIA - count how many of these are met
            criteria_met = 0
            total_criteria = 4  # Total number of scoring criteria checks

            # Price in range
            if metrics["last_price"] >= criteria.min_price:
                criteria_met += 1

            # Volume criteria
            if metrics["avg_volume"] >= criteria.min_avg_volume * 0.25:  # Reduced by 75%
                criteria_met += 1

            # Pass if at least 1 criterion is met (very lenient)
            return criteria_met >= 1
        except Exception as e:
            logger.warning(f"Error applying criteria: {e}")
            return True  # Be lenient on errors

    @staticmethod
    def _calculate_score(metrics):
        """Calculate score for a stock based on metrics"""
        # Simple scoring model
        score = 0

        # Base metrics scoring (40% of total)
        # Volume score (higher is better, but recent volume should be consistent or higher)
        volume_score = min(metrics["avg_volume"] / 1000000, 10)
        volume_trend = metrics["recent_volume"] / \
            metrics["avg_volume"] if metrics["avg_volume"] > 0 else 1
        # Boost if recent volume is higher
        volume_score = volume_score * (1 + max(0, volume_trend - 1))
        score += volume_score * 0.1

        # Volatility score (moderate is better)
        volatility_score = 10 - abs(metrics["volatility"] - 0.02) * 100
        score += max(0, volatility_score) * 0.1

        # RSI score (closer to 50 is better)
        rsi_score = 10 - abs(metrics["rsi"] - 50) / 5
        score += max(0, rsi_score) * 0.1

        # Momentum score (higher is better)
        momentum_score = min(metrics["momentum"] * 100, 10)
        score += max(0, momentum_score) * 0.1

        # Penalize negative momentum significantly
        if metrics["momentum"] < 0:
            # Apply a penalty proportional to the negative momentum
            # The more negative the momentum, the larger the penalty
            # 30% weight to the penalty
            momentum_penalty = abs(metrics["momentum"] * 100) * 0.3
            score -= momentum_penalty

        # Advanced metrics scoring (60% of total)
        if "macd" in metrics:
            # MACD crossover score (positive histogram is bullish)
            macd_score = min(metrics["macd_hist"] * 20,
                             10) if metrics["macd_hist"] > 0 else 0
            score += max(0, macd_score) * 0.15

            # Bollinger Band position (middle of the band is safer)
            bb_position = (metrics["last_price"] - metrics["bb_lower"]) / (metrics["bb_upper"] -
                                                                           metrics["bb_lower"]) if metrics["bb_upper"] > metrics["bb_lower"] else 0.5
            # Highest score when price is in the middle
            bb_score = 10 - abs(bb_position - 0.5) * 20
            score += max(0, bb_score) * 0.15

            # ADX score (trend strength, higher is better up to a point)
            adx_score = min(metrics["adx"] / 5, 10)
            score += max(0, adx_score) * 0.15

            # Volume trend score (OBV change, positive is better)
            obv_score = min(metrics["obv_change"] * 50,
                            10) if metrics["obv_change"] > 0 else 0
            score += max(0, obv_score) * 0.15

        return score

    def find_stocks(self, universe_size=DEFAULT_UNIVERSE_SIZE, top_candidates=DEFAULT_TOP_CANDIDATES):
        """Find top stock candidates based on criteria"""
        logger.info(
            f"Finding top candidates from universe of {universe_size} stocks")

        # Initialize data source if not already initialized
        if self.data_source is None:
            self.data_source = PolygonDataSourceUltra(
                max_pool_size=30, max_retries=5,
                use_daemon=False)  # Ensure child processes are not daemon processes

        # Get stock universe with the specified size limit
        stocks = self.stock_universe.get_stocks(limit=universe_size)
        if len(stocks) == 0:
            logger.warning(
                "Empty stock universe, refreshing and using default stocks...")
            stocks = self.stock_universe.refresh()[:universe_size]

        # Start worker processes
        self.start_workers()

        try:
            # Split stocks into batches
            # Optimize batch size based on number of workers
            batch_size = min(50, max(
                10, len(stocks) // (self.num_workers * 2)))  # Smaller batches for more reliability
            batches = [stocks[i:i+batch_size]
                       for i in range(0, len(stocks), batch_size)]

            # Submit batches to workers
            for batch in batches:
                self.input_queue.put(batch)

            # Collect results
            all_results = []
            for _ in range(len(batches)):
                try:
                    results = self.output_queue.get(
                        timeout=120)  # Increased timeout
                    all_results.extend(results)
                except queue.Empty:
                    logger.warning("Timeout waiting for results")
                    break
            # Sort results by score
            all_results.sort(key=lambda x: x["score"], reverse=True)

            # Filter out stocks where we can't buy at least 1 share with our position size
            all_results = [
                r for r in all_results if self.position_size / r["price"] >= 1]

            # Take top candidates
            top_results = all_results[:min(top_candidates, len(all_results))]

            # If we have no results, log this clearly
            if not top_results:
                logger.warning("No stock candidates found after screening!")
                logger.info(
                    f"Processed {len(batches)} batches with {len(stocks)} total stocks")
                return []

            # Define constants for position sizing
            # Allow up to half of total position size per stock
            # Reduced from /2 to /5 to allow more stocks
            MAX_SINGLE_POSITION = self.position_size / \
                3  # Increased to allow larger positions
            MIN_POSITION_SIZE = 150.0  # Minimum position size

            # Sort results by score (highest first)
            top_results.sort(key=lambda x: x["score"], reverse=True)

            # Filter out stocks where we can't buy at least 1 share with our position size
            top_results = [
                r for r in top_results if self.position_size / r["price"] >= 1]

            # Ensure we have enough candidates
            if len(top_results) < top_candidates:
                logger.warning(
                    f"Only found {len(top_results)} valid candidates that fit within position size")
                top_candidates = max(1, len(top_results))

            # Initialize all positions to zero
            for result in top_results:
                result["position_value"] = 0.0
                result["shares"] = 0

            # Allocate positions based on a dynamic approach with varied sizes
            remaining_funds = self.position_size

            # Calculate total score of all candidates to use for weighted allocation
            total_score = sum(
                result["score"] for result in top_results[:min(top_candidates, len(top_results))])

            # Create a varied allocation pattern based on scores and some randomization
            for i, result in enumerate(top_results[:min(top_candidates, len(top_results))]):
                # Stop if we've allocated all funds
                if remaining_funds <= 0:
                    break

                # Calculate a base position size proportional to the stock's score
                # This creates a natural variation where higher scoring stocks get larger positions
                score_weight = result["score"] / \
                    total_score if total_score > 0 else 1.0 / len(top_results)

                # Add some randomization to create more variation (±20%)
                # Random factor between 0.8 and 1.2
                variation_factor = 0.8 + (0.4 * random.random())

                # Calculate position value with variation
                base_position = self.position_size * score_weight
                position_value = base_position * variation_factor

                # Ensure position is within bounds
                position_value = min(max(position_value, MIN_POSITION_SIZE),
                                     min(MAX_SINGLE_POSITION, remaining_funds))

                # Skip if position would be too small
                if position_value < MIN_POSITION_SIZE:
                    continue

                # Update position value and shares
                # Round to nearest $25 for cleaner position sizes
                result["position_value"] = round(position_value / 25) * 25

                # Calculate shares and ensure at least 1 share
                shares = int(position_value / result["price"])
                result["shares"] = max(1, shares)
                result["position_value"] = result["shares"] * result["price"]
                remaining_funds -= result["position_value"]

            # Phase 2: If we still have funds left, distribute to next best stocks
            while remaining_funds > MIN_POSITION_SIZE:
                # Get stocks that haven't been allocated yet (beyond the top 10)
                unallocated = [
                    r for r in top_results if r["position_value"] == 0]

                for result in unallocated:
                    # Stop if remaining funds are too small
                    if remaining_funds < MIN_POSITION_SIZE:
                        break

                    # For remaining stocks, create varied position sizes
                    # Use a random factor to create diversity
                    # Random factor between 0.7 and 1.3
                    variation_factor = 0.7 + (0.6 * random.random())
                    position_value = MIN_POSITION_SIZE * variation_factor

                    # Ensure position is within bounds
                    position_value = min(position_value, remaining_funds)

                    # Update position value and shares
                    result["position_value"] = round(
                        position_value / 25) * 25  # Round to nearest $25

                    # Calculate shares and ensure at least 1 share
                    shares = int(result["position_value"] / result["price"])
                    result["shares"] = max(1, shares)
                    result["position_value"] = result["shares"] * \
                        result["price"]

                    # Update remaining funds
                    remaining_funds -= position_value

                # If we've gone through all unallocated stocks and still have funds,
                # break to avoid infinite loop
                if not unallocated:
                    break

            # Final filter: remove any stocks with 0 shares
            top_results = [r for r in top_results if r["shares"] > 0]

            logger.info(f"Found {len(top_results)} stock candidates")
            return top_results
        finally:
            # Always stop worker processes to clean up resources
            self.stop_workers()

    def close(self):
        """Close resources"""
        self.data_source.close()
        logger.info("Dynamic stock finder closed")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Stock Finder")
    parser.add_argument("--position-size", type=float, default=DEFAULT_POSITION_SIZE,
                        help=f"Total position size in dollars (default: {DEFAULT_POSITION_SIZE})")
    parser.add_argument("--universe-size", type=int, default=DEFAULT_UNIVERSE_SIZE,
                        help=f"Number of stocks to scan (default: {DEFAULT_UNIVERSE_SIZE})")
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_CANDIDATES,
                        help=f"Number of top candidates to return (default: {DEFAULT_TOP_CANDIDATES})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT,
                        help=f"Number of worker processes (default: {DEFAULT_WORKER_COUNT})")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})")

    args = parser.parse_args()

    # Create stock finder
    finder = DynamicStockFinder(
        position_size=args.position_size, num_workers=args.workers)

    try:
        # Find stocks
        start_time = time.time()
        results = finder.find_stocks(
            universe_size=args.universe_size, top_candidates=args.top_candidates)
        elapsed = time.time() - start_time

        # Print results
        print("\n" + "="*100)
        print(
            f"TOP {len(results)} STOCK CANDIDATES FOR ${args.position_size:.2f} POSITION")
        print("="*80)

        print(f"{'Ticker':<8} {'Price':<10} {'Shares':<8} {'Position':<12} {'Score':<8} {'RSI':<8} {'Momentum':<10} {'Volatility':<10} {'MACD':<10}")
        print("-"*80)

        # Show top 20 results regardless of position value
        for result in results[:min(20, len(results))]:
            print(f"{result['ticker']:<8} ${result['price']:<9.2f} {result['shares']:<8} ${result['position_value']:<11.2f} {result['score']:<8.2f} {result['rsi']:<8.2f} {result['momentum']*100:<9.2f}% {result['volatility']*100:<9.2f}%")

        print("-"*80)
        print(f"Scanned {args.universe_size} stocks in {elapsed:.2f} seconds")
        print(f"Historical data period: {DEFAULT_HISTORY_DAYS} days")
        print(f"Technical indicators: RSI, MACD, Bollinger Bands, ADX, OBV (custom implementations)")
        print(
            f"Processing rate: {args.universe_size/elapsed:.2f} stocks/second")

        # Calculate statistics (only for positions with value > 0)
        active_positions = [r for r in results if r["position_value"] > 0]
        total_position = sum(result["position_value"]
                             for result in active_positions)
        avg_position = total_position / \
            len(active_positions) if active_positions else 0

        print(f"Total position value: ${total_position:.2f}")
        print(f"Average position size: ${avg_position:.2f}")
        print(f"Number of positions: {len(active_positions)}")

    finally:
        # Close resources
        finder.close()


if __name__ == "__main__":
    main()
