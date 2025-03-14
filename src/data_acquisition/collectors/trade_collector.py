"""
Trade Data Collector

This module provides a collector for gathering trade data from various sources.
Note: Trade data is collected from Polygon.io as Alpaca does not provide trade data in the free tier.
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

# Import pandas and numpy directly
import pandas as pd

from src.data_acquisition.api.polygon_client import (
    PolygonClient,
)


logger = logging.getLogger(__name__)


class TradeCollector:
    """Collects trade data from Polygon.io."""

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trade collector.

        Args:
            polygon_client: Polygon.io API client
            config: Configuration dictionary with options like max_threads
        """
        self.polygon = polygon_client or PolygonClient()
        self.config = config or {}

        # Set up thread pool for parallel data collection
        self.max_threads = self.config.get("max_threads", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)

        logger.info(f"Initialized TradeCollector with {self.max_threads} threads")

    def collect_trades(
        self, tickers: List[str], date_to_collect: Union[str, datetime, date]
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect trades for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect trades for

        Returns:
            Dictionary mapping tickers to DataFrames with trade data
        """
        logger.info(
            f"Collecting trades for {len(tickers)} tickers on {date_to_collect}"
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
            future = self.executor.submit(self._collect_ticker_trades, ticker, date_str)
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

    def _collect_ticker_trades(self, ticker: str, date_str: str) -> pd.DataFrame:
        """
        Collect trades for a single ticker.

        Args:
            ticker: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with trade data
        """
        try:
            logger.debug(f"Collecting trades for {ticker} on {date_str}")

            # Get trades from Polygon API
            df = self.polygon.get_trades(ticker, date_str)

            if df.empty:
                logger.warning(f"No trades found for {ticker} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} trades for {ticker} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting trades for {ticker} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_large_trades(
        self,
        tickers: List[str],
        date_to_collect: Union[str, datetime, date],
        min_size: int = 10000,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect large trades for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect trades for
            min_size: Minimum trade size to consider as a large trade

        Returns:
            Dictionary mapping tickers to DataFrames with large trade data
        """
        logger.info(
            f"Collecting large trades (min size: {min_size}) for {len(tickers)} tickers on {date_to_collect}"
        )

        # First collect all trades
        all_trades = self.collect_trades(tickers, date_to_collect)

        # Filter for large trades
        large_trades = {}

        for ticker, df in all_trades.items():
            if not df.empty:
                # Filter trades by size
                try:
                    large_df = df[df["size"] >= min_size].copy()
                except Exception:
                    logger.warning(f"Error filtering large trades for {ticker}. Skipping.")
                    continue

                if not large_df.empty:
                    large_trades[ticker] = large_df
                    logger.debug(
                        f"Found {len(large_df)} large trades for {ticker} on {date_to_collect}"
                    )
                else:
                    logger.debug(
                        f"No large trades found for {ticker} on {date_to_collect}"
                    )

        logger.info(
            f"Completed large trade data collection for {len(large_trades)} tickers"
        )
        return large_trades

    def collect_block_trades(
        self,
        tickers: List[str],
        date_to_collect: Union[str, datetime, date],
        min_value: float = 1000000.0,  # $1M minimum
    ) -> dict[str, pd.DataFrame]:
        """
        Collect block trades for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect trades for
            min_value: Minimum trade value in dollars

        Returns:
            Dictionary mapping tickers to DataFrames with block trade data
        """
        logger.info(
            f"Collecting block trades (min value: ${min_value:,.2f}) for {len(tickers)} tickers on {date_to_collect}"
        )

        # First collect all trades
        all_trades = self.collect_trades(tickers, date_to_collect)

        # Filter for block trades
        block_trades = {}

        for ticker, df in all_trades.items():
            if not df.empty:
                # Calculate trade value (price * size)
                if "price" in df.columns and "size" in df.columns:
                    try:
                        df["trade_value"] = df["price"] * df["size"]
                    except Exception:
                        logger.warning(f"Error calculating trade values for {ticker}. Skipping.")
                        continue

                    # Filter trades by value
                    block_df = df[df["trade_value"] >= min_value].copy()

                    if not block_df.empty:
                        block_trades[ticker] = block_df
                        logger.debug(
                            f"Found {len(block_df)} block trades for {ticker} on {date_to_collect}"
                        )
                    else:
                        logger.debug(
                            f"No block trades found for {ticker} on {date_to_collect}"
                        )
                else:
                    logger.warning(f"Missing price or size columns for {ticker}")

        logger.info(
            f"Completed block trade data collection for {len(block_trades)} tickers"
        )
        return block_trades

    def collect_unusual_trades(
        self,
        tickers: List[str],
        date_to_collect: Union[str, datetime, date],
        z_score_threshold: float = 3.0,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect unusual trades (outliers in terms of size) for multiple tickers.

        Args:
            tickers: List of ticker symbols
            date_to_collect: Date to collect trades for
            z_score_threshold: Z-score threshold for considering a trade as unusual

        Returns:
            Dictionary mapping tickers to DataFrames with unusual trade data
        """
        logger.info(
            f"Collecting unusual trades (z-score > {z_score_threshold}) for {len(tickers)} tickers on {date_to_collect}"
        )

        # First collect all trades
        all_trades = self.collect_trades(tickers, date_to_collect)

        # Filter for unusual trades
        unusual_trades = {}

        for ticker, df in all_trades.items():
            if (
                not df.empty and len(df) > 10
            ):  # Need enough data to calculate meaningful z-scores
                # Calculate z-scores for trade sizes
                try:
                    mean_size = df["size"].mean()
                    std_size = df["size"].std()
                except Exception:
                    logger.warning(f"Error calculating statistics for {ticker}. Skipping.")
                    continue

                if std_size > 0:  # Avoid division by zero
                    df["size_z_score"] = (df["size"] - mean_size) / std_size
                    unusual_df = df[df["size_z_score"] >= z_score_threshold].copy()

                    if not unusual_df.empty:
                        unusual_trades[ticker] = unusual_df
                        logger.debug(
                            f"Found {len(unusual_df)} unusual trades for {ticker} on {date_to_collect}"
                        )
                    else:
                        logger.debug(
                            f"No unusual trades found for {ticker} on {date_to_collect}"
                        )
                else:
                    logger.debug(
                        f"Standard deviation of trade sizes is zero for {ticker}"
                    )

        logger.info(
            f"Completed unusual trade data collection for {len(unusual_trades)} tickers"
        )
        return unusual_trades
