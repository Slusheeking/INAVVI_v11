"""
Multi-Timeframe Data Collector

This module provides a collector for gathering market data across multiple timeframes in parallel.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient

# Import pandas and numpy directly
import pandas as pd
import numpy as np

# Import logging utility
from src.utils.logging import get_logger

# Set up logger
logger = get_logger("data_acquisition.collectors.multi_timeframe_data_collector")


class MultiTimeframeDataCollector:
    """Collects market data for multiple timeframes in parallel."""

    def __init__(
        self,
        polygon_client: PolygonClient,
        unusual_whales_client: UnusualWhalesClient,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the multi-timeframe data collector.

        Args:
            polygon_client: Polygon.io API client
            unusual_whales_client: Unusual Whales API client
            config: Configuration dictionary with options like max_threads
        """
        self.polygon_client = polygon_client
        self.unusual_whales_client = unusual_whales_client
        self.config = config or {}

        # Set up thread pool for parallel data collection
        self.max_threads = self.config.get("max_threads", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)

        # Default timeframes if not specified
        self.default_timeframes = ["1m", "5m", "15m", "1h", "1d"]

        logger.info(
            f"Initialized MultiTimeframeDataCollector with {self.max_threads} threads"
        )

    def collect_multi_timeframe_data(
        self,
        tickers: List[str],
        timeframes: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        adjusted: bool = True,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Collect market data for multiple tickers across multiple timeframes.

        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes (e.g., ['1m', '5m', '15m', '1h', '1d'])
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Nested dictionary mapping tickers to timeframes to DataFrames
        """
        logger.info(f"Collecting multi-timeframe data for {len(tickers)} tickers")

        # Use default timeframes if not specified
        timeframes = timeframes or self.default_timeframes

        # Set default dates if not specified
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            # Default to 30 days of data
            if isinstance(end_date, (datetime, date)):
                start_date = end_date - timedelta(days=30)
            else:
                # Parse end_date string and subtract 30 days using the appropriate library
                try:
                    end_date_obj = pd.to_datetime(end_date).date()
                except Exception:
                    logger.warning("Error parsing date. Using pandas fallback.")
                    end_date_obj = pd.to_datetime(end_date).date()
                start_date = (end_date_obj - timedelta(days=30)).isoformat()

        # Initialize results dictionary
        results = {ticker: {} for ticker in tickers}

        # Collect data for each timeframe in parallel
        futures = {}

        for timeframe in timeframes:
            future = self.executor.submit(
                self.collect_price_data,
                tickers,
                timeframe,
                start_date,
                end_date,
                adjusted,
            )
            futures[future] = timeframe

        # Process results as they complete
        for future in as_completed(futures):
            timeframe = futures[future]
            try:
                timeframe_data = future.result()

                # Add data to results dictionary
                for ticker, df in timeframe_data.items():
                    if ticker in results:
                        results[ticker][timeframe] = df

                logger.info(f"Completed collection for {timeframe} timeframe")
            except Exception as e:
                logger.error(f"Error collecting data for {timeframe} timeframe: {e}")
                logger.debug(traceback.format_exc())

        # Process the multi-timeframe data to ensure consistency
        processed_results = self._process_timeframe_data(results)

        logger.info(
            f"Completed multi-timeframe data collection for {len(tickers)} tickers"
        )
        return processed_results

    def collect_price_data(
        self,
        tickers: List[str],
        timeframe: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect price data for multiple tickers at a specific timeframe.

        Args:
            tickers: List of ticker symbols
            timeframe: Timeframe for data collection (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Dictionary mapping tickers to DataFrames with price data
        """
        logger.info(f"Collecting {timeframe} price data for {len(tickers)} tickers")

        # Parse timeframe
        if timeframe.endswith("m"):
            multiplier = int(timeframe[:-1])
            timespan = "minute"
        elif timeframe.endswith("h"):
            multiplier = int(timeframe[:-1])
            timespan = "hour"
        elif timeframe.endswith("d"):
            multiplier = int(timeframe[:-1])
            timespan = "day"
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Collect data for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(
                self._collect_ticker_price_data,
                ticker,
                multiplier,
                timespan,
                start_date,
                end_date,
                adjusted,
            )
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    # Add timeframe column
                    df["timeframe"] = timeframe
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting {timeframe} data for {ticker}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"Completed {timeframe} price data collection for {len(results)} tickers"
        )
        return results

    def collect_quote_data(
        self, tickers: List[str], date_to_collect: Union[str, datetime, date]
    ) -> dict[str, pd.DataFrame]:
        """
        Collect quote data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect quotes for

        Returns:
            Dictionary mapping tickers to DataFrames with quote data
        """
        logger.info(
            f"Collecting quote data for {len(tickers)} tickers on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect quotes for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(
                self._collect_ticker_quote_data, ticker, date_str
            )
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting quotes for {ticker}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed quote data collection for {len(results)} tickers")
        return results

    def collect_trade_data(
        self, tickers: List[str], date_to_collect: Union[str, datetime, date]
    ) -> dict[str, pd.DataFrame]:
        """
        Collect trade data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect trades for

        Returns:
            Dictionary mapping tickers to DataFrames with trade data
        """
        logger.info(
            f"Collecting trade data for {len(tickers)} tickers on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect trades for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(
                self._collect_ticker_trade_data, ticker, date_str
            )
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting trades for {ticker}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed trade data collection for {len(results)} tickers")
        return results

    def collect_options_data(
        self, tickers: List[str], days_back: int = 1, limit: int = 1000
    ) -> dict[str, pd.DataFrame]:
        """
        Collect options flow data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            days_back: Number of days to look back
            limit: Maximum number of results per ticker

        Returns:
            Dictionary mapping tickers to DataFrames with options data
        """
        logger.info(f"Collecting options data for {len(tickers)} tickers")

        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        # Collect options data for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(
                self._collect_ticker_options_data, ticker, start_date, end_date, limit
            )
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting options data for {ticker}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed options data collection for {len(results)} tickers")
        return results

    def _collect_ticker_price_data(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool,
    ) -> pd.DataFrame:
        """
        Collect price data for a single ticker.

        Args:
            ticker: Ticker symbol
            multiplier: Timespan multiplier
            timespan: Timespan unit
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits and dividends

        Returns:
            DataFrame with price data
        """
        try:
            logger.debug(f"Collecting {timespan} data for {ticker}")

            # Get aggregates from Polygon API
            df = self.polygon_client.get_aggregates(
                # Use named parameters to avoid confusion with parameter order
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date,
                adjusted=adjusted
            )

            if df.empty:
                logger.warning(f"No {timespan} data found for {ticker}")
                return df

            logger.debug(f"Collected {len(df)} {timespan} bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error collecting {timespan} data for {ticker}: {e}")
            logger.debug(traceback.format_exc())
            try:
                return pd.DataFrame()
            except Exception:
                return pd.DataFrame()

    def _collect_ticker_quote_data(self, ticker: str, date_str: str) -> pd.DataFrame:
        """
        Collect quote data for a single ticker.

        Args:
            ticker: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with quote data
        """
        try:
            logger.debug(f"Collecting quotes for {ticker} on {date_str}")

            # Get quotes from Polygon API
            df = self.polygon_client.get_quotes(ticker, date_str)

            if df.empty:
                logger.warning(f"No quotes found for {ticker} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} quotes for {ticker} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting quotes for {ticker} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def _collect_ticker_trade_data(self, ticker: str, date_str: str) -> pd.DataFrame:
        """
        Collect trade data for a single ticker.

        Args:
            ticker: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with trade data
        """
        try:
            logger.debug(f"Collecting trades for {ticker} on {date_str}")

            # Get trades from Polygon API
            df = self.polygon_client.get_trades(ticker, date_str)

            if df.empty:
                logger.warning(f"No trades found for {ticker} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} trades for {ticker} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting trades for {ticker} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def _collect_ticker_options_data(
        self, ticker: str, start_date: date, end_date: date, limit: int
    ) -> pd.DataFrame:
        """
        Collect options data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            limit: Maximum number of results

        Returns:
            DataFrame with options data
        """
        try:
            logger.debug(f"Collecting options data for {ticker}")

            # Get historical flow from Unusual Whales API
            historical_flow = self.unusual_whales_client.get_historical_flow(
                limit=limit, from_date=start_date, to_date=end_date, ticker=ticker
            )

            # Get live flow from Unusual Whales API
            live_flow = self.unusual_whales_client.get_options_flow(
                limit=limit, ticker=ticker
            )

            # Combine historical and live flow
            flow_data = []
            if historical_flow:
                flow_data.extend(historical_flow)
            if live_flow:
                flow_data.extend(live_flow)

            # Remove duplicates
            seen_ids = set()
            unique_flow_data = []

            for item in flow_data:
                item_id = item.get("id")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_flow_data.append(item)

            # Convert to DataFrame
            if unique_flow_data:
                try:
                    # Use the client's flow_to_dataframe method which should handle RAPIDS/pandas appropriately
                    df = self.unusual_whales_client.flow_to_dataframe(unique_flow_data)
                    logger.debug(f"Converted options flow to DataFrame for {ticker}")
                except Exception as e:
                    logger.warning(f"Error converting flow data: {e}. Using pandas fallback.")
                    df = pd.DataFrame(unique_flow_data)
                logger.debug(f"Collected {len(df)} options flow records for {ticker}")
                return df
            else:
                logger.warning(f"No options flow found for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting options data for {ticker}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def _process_timeframe_data(
        self, data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Process multi-timeframe data to ensure consistency across timeframes.

        Args:
            data: Nested dictionary mapping tickers to timeframes to DataFrames

        Returns:
            Processed multi-timeframe data
        """
        logger.info("Processing multi-timeframe data")

        processed_data = {}

        for ticker, timeframe_data in data.items():
            processed_data[ticker] = {}

            # Skip if no data for this ticker
            if not timeframe_data:
                continue

            # Get available timeframes for this ticker
            timeframes = list(timeframe_data.keys())

            # Process each timeframe
            for timeframe in timeframes:
                df = timeframe_data[timeframe]

                # Skip if empty DataFrame
                if df.empty:
                    continue

                # Ensure timestamp column is datetime
                if "timestamp" in df.columns:
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                    except Exception as e:
                        logger.warning(f"Error converting timestamp: {e}. Using pandas fallback.")
                        import pandas as pandas_pd
                        # Convert to pandas if needed
                        if hasattr(df, 'to_pandas'):
                            df = df.to_pandas()
                        df["timestamp"] = pandas_pd.to_datetime(df["timestamp"])

                # Sort by timestamp
                if "timestamp" in df.columns:
                    try:
                        df = df.sort_values("timestamp")
                    except Exception as e:
                        logger.warning(f"Error sorting by timestamp: {e}. Using pandas fallback.")
                        import pandas as pandas_pd
                        # Convert to pandas if needed
                        if hasattr(df, 'to_pandas'):
                            df = df.to_pandas()
                        df = df.sort_values("timestamp")

                # Add timeframe column if not present
                if "timeframe" not in df.columns:
                    df["timeframe"] = timeframe

                # Store processed DataFrame
                processed_data[ticker][timeframe] = df

            # Generate cross-timeframe features if multiple timeframes are available
            if len(timeframes) > 1:
                processed_data[ticker] = self._generate_cross_timeframe_features(
                    processed_data[ticker]
                )

        logger.info("Completed multi-timeframe data processing")
        return processed_data

    def _generate_cross_timeframe_features(
        self, timeframe_data: Dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Generate cross-timeframe features.

        Args:
            timeframe_data: Dictionary mapping timeframes to DataFrames

        Returns:
            Dictionary with added cross-timeframe features
        """
        # Skip if less than 2 timeframes
        if len(timeframe_data) < 2:
            return timeframe_data

        # Get available timeframes
        timeframes = list(timeframe_data.keys())

        # Sort timeframes by granularity (smallest to largest)
        timeframe_minutes = {}
        for tf in timeframes:
            if tf.endswith("m"):
                timeframe_minutes[tf] = int(tf[:-1])
            elif tf.endswith("h"):
                timeframe_minutes[tf] = int(tf[:-1]) * 60
            elif tf.endswith("d"):
                timeframe_minutes[tf] = int(tf[:-1]) * 60 * 24

        sorted_timeframes = sorted(
            timeframes, key=lambda tf: timeframe_minutes.get(tf, 0)
        )

        logger.info(f"Generating cross-timeframe features for {len(sorted_timeframes)} timeframes")

        # Process each timeframe pair (smaller to larger)
        for i, tf1 in enumerate(sorted_timeframes):
            df1 = timeframe_data[tf1]

            for tf2 in sorted_timeframes[i + 1 :]:
                df2 = timeframe_data[tf2]

                # Skip if either DataFrame is empty or doesn't have required columns
                if df1.empty or df2.empty:
                    logger.debug(f"Skipping {tf1}-{tf2} pair: empty DataFrame")
                    continue
                
                if "close" not in df1.columns or "close" not in df2.columns:
                    logger.debug(f"Skipping {tf1}-{tf2} pair: missing 'close' column")
                    continue

                if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
                    logger.debug(f"Skipping {tf1}-{tf2} pair: missing 'timestamp' column")
                    continue

                logger.debug(f"Processing timeframe pair: {tf1}-{tf2}")

                # Create feature column names
                ratio_col = f"close_ratio_{tf1}_{tf2}"
                
                # Create a copy of df2 with renamed timestamp column for merging
                df2_copy = df2.copy()
                df2_copy.rename(columns={
                    "timestamp": "larger_timestamp",
                    "close": "larger_close",
                    "volume": "larger_volume"
                }, inplace=True)
                
                try:
                    # Use numpy for processing
                    smaller_timestamps = df1["timestamp"].values
                    smaller_closes = df1["close"].values
                    larger_timestamps = df2_copy["larger_timestamp"].values
                    larger_closes = df2_copy["larger_close"].values
                    
                    # Initialize arrays for the new features
                    close_ratios = np.full(len(df1), np.nan)
                    
                    # For each row in the smaller timeframe
                    for j, ts in enumerate(smaller_timestamps):
                        # Find the most recent bar in the larger timeframe
                        mask = larger_timestamps <= ts
                        if np.any(mask):
                            idx = np.where(mask)[0][-1]  # Get the last (most recent) matching index
                            close_ratios[j] = smaller_closes[j] / larger_closes[idx]
                    
                    logger.debug(f"Calculated cross-timeframe features using numpy for {tf1}-{tf2}")
                except Exception as e:
                    logger.warning(f"Error calculating cross-timeframe features: {e}. Using pandas fallback.")
                    # Fallback to a simpler approach if the above fails
                    import numpy as numpy_np
                    # Convert to pandas if needed
                    if hasattr(df1, 'to_pandas'):
                        df1 = df1.to_pandas()
                    if hasattr(df2_copy, 'to_pandas'):
                        df2_copy = df2_copy.to_pandas()
                    
                    # Initialize the ratio column with NaN
                    df1[ratio_col] = numpy_np.nan
                    return timeframe_data

                # Add the features to the smaller timeframe DataFrame
                df1[ratio_col] = close_ratios

                # Update the DataFrame in the dictionary
                timeframe_data[tf1] = df1

        return timeframe_data

    def _handle_collection_error(
        self, data_type: str, ticker: str, error: Exception
    ) -> None:
        """
        Handle collection errors.

        Args:
            data_type: Type of data being collected
            ticker: Ticker symbol
            error: Exception that occurred
        """
        logger.error(f"Error collecting {data_type} for {ticker}: {error}")
        logger.debug(traceback.format_exc())
