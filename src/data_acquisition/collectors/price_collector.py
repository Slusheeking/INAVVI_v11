"""
Price Data Collector

This module provides a collector for gathering price data (OHLCV) from various sources.
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
from src.trading_strategy.alpaca.alpaca_client import AlpacaClient
from src.data_acquisition.api.polygon_client import PolygonClient

# Import pandas and numpy directly
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


class PriceCollector:
    """Collects price data (OHLCV) from various sources."""

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the price collector.

        Args:
            polygon_client: Polygon.io API client
            alpaca_client: Alpaca API client
            config: Configuration dictionary with options like max_threads
        """
        self.polygon = polygon_client or PolygonClient()
        self.alpaca = alpaca_client or AlpacaClient()
        self.config = config or {}

        # Set up thread pool for parallel data collection
        self.max_threads = self.config.get("max_threads", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)

        logger.info(f"Initialized PriceCollector with {self.max_threads} threads")

    def collect_stock_aggs(
        self,
        tickers: List[str],
        multiplier: int,
        timespan: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect stock aggregates (OHLCV) for multiple tickers.

        Args:
            tickers: List of ticker symbols
            multiplier: Timespan multiplier (e.g., 1, 5, 15)
            timespan: Timespan unit (e.g., 'minute', 'hour', 'day')
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Dictionary mapping tickers to DataFrames with price data
        """
        logger.info(f"Collecting {timespan} price data for {len(tickers)} tickers")

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
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting {timespan} data for {ticker}: {str(e)}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"Completed {timespan} price data collection for {len(results)} tickers"
        )
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
            # Use named parameters to avoid confusion with parameter order
            df = self.polygon.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date,
                adjusted=str(adjusted).lower()
            )

            if df.empty:
                logger.warning(f"No {timespan} data found for {ticker}")
                return df

            logger.debug(f"Collected {len(df)} {timespan} bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error collecting {timespan} data for {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_daily_bars(
        self,
        tickers: List[str],
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect daily bars for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Dictionary mapping tickers to DataFrames with daily price data
        """
        return self.collect_stock_aggs(
            tickers, 1, "day", start_date, end_date, adjusted
        )

    def collect_intraday_bars(
        self,
        tickers: List[str],
        interval_minutes: int,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect intraday bars for multiple tickers.

        Args:
            tickers: List of ticker symbols
            interval_minutes: Interval in minutes (e.g., 1, 5, 15)
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Dictionary mapping tickers to DataFrames with intraday price data
        """
        return self.collect_stock_aggs(
            tickers, interval_minutes, "minute", start_date, end_date, adjusted
        )

    def collect_hourly_bars(
        self,
        tickers: List[str],
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect hourly bars for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            adjusted: Whether to adjust for splits and dividends

        Returns:
            Dictionary mapping tickers to DataFrames with hourly price data
        """
        return self.collect_stock_aggs(
            tickers, 1, "hour", start_date, end_date, adjusted
        )

    def collect_from_alpaca(
        self,
        tickers: List[str],
        timeframe: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjustment: str = "all",
    ) -> dict[str, pd.DataFrame]:
        """
        Collect price data from Alpaca.

        Args:
            tickers: List of ticker symbols
            timeframe: Timeframe (e.g., '1Min', '5Min', '1Hour', '1Day')
            start_date: Start date for data collection
            end_date: End date for data collection
            adjustment: Adjustment type ('raw', 'split', 'dividend', 'all')

        Returns:
            Dictionary mapping tickers to DataFrames with price data
        """
        logger.info(
            f"Collecting {timeframe} price data from Alpaca for {len(tickers)} tickers"
        )

        # Collect data for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(
                self._collect_ticker_from_alpaca,
                ticker,
                timeframe,
                start_date,
                end_date,
                adjustment,
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
                logger.error(f"Error collecting {timeframe} data from Alpaca for {ticker}: {str(e)}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"Completed {timeframe} price data collection from Alpaca for {len(results)} tickers"
        )
        return results

    def _collect_ticker_from_alpaca(
        self,
        ticker: str,
        timeframe: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjustment: str,
    ) -> pd.DataFrame:
        """
        Collect price data from Alpaca for a single ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            adjustment: Adjustment type

        Returns:
            DataFrame with price data
        """
        try:
            logger.debug(f"Collecting {timeframe} data from Alpaca for {ticker}")

            # Get bars from Alpaca API
            df = self.alpaca.get_bars(
                ticker, timeframe, start_date, end_date, adjustment
            )

            if df.empty:
                logger.warning(f"No {timeframe} data found from Alpaca for {ticker}")
                return df

            logger.debug(
                f"Collected {len(df)} {timeframe} bars from Alpaca for {ticker}"
            )
            return df

        except Exception as e:
            logger.error(f"Error collecting {timeframe} data from Alpaca for {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
