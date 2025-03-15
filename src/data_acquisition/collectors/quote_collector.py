"""
Quote Data Collector

This module provides a collector for gathering quote data from various sources.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

# Import pandas and numpy directly
import pandas as pd

from src.trading_strategy.alpaca.alpaca_client import AlpacaClient
from src.data_acquisition.api.polygon_client import (
    PolygonClient
)

# Import logging utility
from src.utils.logging import get_logger

# Set up logger
logger = get_logger("data_acquisition.collectors.quote_collector")


class QuoteCollector:
    """Collects quote data from various sources."""

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the quote collector.

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

        logger.info(f"Initialized QuoteCollector with {self.max_threads} threads")

    def collect_quotes(
        self, tickers: List[str], date_to_collect: Union[str, datetime, date]
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect quotes for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect quotes for

        Returns:
            Dictionary mapping tickers to DataFrames with quote data
        """
        logger.info(
            f"Collecting quotes for {len(tickers)} tickers on {date_to_collect}"
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
            future = self.executor.submit(self._collect_ticker_quotes, ticker, date_str)
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

    def _collect_ticker_quotes(self, ticker: str, date_str: str) -> pd.DataFrame:
        """
        Collect quotes for a single ticker.

        Args:
            ticker: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with quote data
        """
        try:
            logger.debug(f"Collecting quotes for {ticker} on {date_str}")

            # Get quotes from Polygon API
            df = self.polygon.get_quotes(ticker, date_str)

            if df.empty:
                logger.warning(f"No quotes found for {ticker} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} quotes for {ticker} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting quotes for {ticker} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_nbbo(
        self, tickers: List[str], date_to_collect: Union[str, datetime, date]
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect National Best Bid and Offer (NBBO) quotes for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect NBBO quotes for

        Returns:
            Dictionary mapping tickers to DataFrames with NBBO quote data
        """
        logger.info(
            f"Collecting NBBO quotes for {len(tickers)} tickers on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect NBBO quotes for each ticker in parallel
        futures = {}
        results = {}

        for ticker in tickers:
            future = self.executor.submit(self._collect_ticker_nbbo, ticker, date_str)
            futures[future] = ticker

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                logger.error(f"Error collecting NBBO quotes for {ticker}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed NBBO quote data collection for {len(results)} tickers")
        return results

    def _collect_ticker_nbbo(self, ticker: str, date_str: str) -> pd.DataFrame:
        """
        Collect NBBO quotes for a single ticker.

        Args:
            ticker: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with NBBO quote data
        """
        try:
            logger.debug(f"Collecting NBBO quotes for {ticker} on {date_str}")

            # Get NBBO quotes from Polygon API
            # Note: This is a placeholder - the actual implementation would depend on
            # the specific API method available in the PolygonClient
            df = self.polygon.get_quotes(ticker, date_str, nbbo_only=True)

            if df.empty:
                logger.warning(f"No NBBO quotes found for {ticker} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} NBBO quotes for {ticker} on {date_str}")
            return df

        except Exception as e:
            logger.error(
                f"Error collecting NBBO quotes for {ticker} on {date_str}: {e}"
            )
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
