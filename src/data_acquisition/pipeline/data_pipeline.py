"""
Data Acquisition Pipeline

This module provides the main pipeline for acquiring market data from various sources
and storing it in the database.
"""

import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Use AlpacaAPIClient from utils.api instead of importing from trading_strategy
from src.utils.api import AlpacaAPIClient
from src.data_acquisition.api.polygon_client import (
    PolygonClient,
)
from src.data_acquisition.api.unusual_whales_client import (
    UnusualWhalesClient,
)
from src.data_acquisition.collectors.multi_timeframe_data_collector import (
    MultiTimeframeDataCollector,
)
from src.data_acquisition.storage.timescale_storage import (
    TimescaleStorage,
)
# Removed unused imports: from src.utils.time.market_hours import get_market_status, MarketStatus

# Import logging utility
from src.utils.logging import get_logger

# Set up logger
logger = get_logger("data_acquisition.pipeline.data_pipeline")


class DataPipeline:
    """Main pipeline for data acquisition."""

    def __init__(
        self,
        polygon_client: Optional[PolygonClient]= None,
        alpaca_client: Optional[AlpacaAPIClient] = None,
        unusual_whales_client: Optional[UnusualWhalesClient] = None,
        multi_timeframe_collector: Optional[MultiTimeframeDataCollector] = None,
        storage: Optional[TimescaleStorage] = None,
        max_workers: int = 10,
    ):
        """
        Initialize the data pipeline.

        Args:
            polygon_client: Polygon.io API client
            alpaca_client: Alpaca API client
            unusual_whales_client: Unusual Whales API client
            multi_timeframe_collector: Multi-timeframe data collector
            storage: TimescaleDB storage
            max_workers: Maximum number of worker threads
        """
        self.polygon = polygon_client or PolygonClient()
        self.unusual_whales = unusual_whales_client or UnusualWhalesClient()
        # Make Alpaca client optional - only initialize if explicitly provided
        self.alpaca = alpaca_client  # If None, it will stay None

        # Initialize multi-timeframe collector if not provided
        self.multi_timeframe_collector = (
            multi_timeframe_collector
            or MultiTimeframeDataCollector(
                polygon_client=self.polygon, unusual_whales_client=self.unusual_whales
            )
        )
        self.storage = storage or TimescaleStorage()
        self.max_workers = max_workers

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Queue for storing results
        self.result_queue = queue.Queue()

        # Flags for controlling the pipeline
        self.running = False
        self.stop_requested = False

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize ticker symbols to a standard format.

        Args:
            symbol: Ticker symbol

        Returns:
            Normalized symbol
        """
        if not symbol:
            return ""

        # Convert to uppercase and strip whitespace
        normalized = symbol.upper().strip()

        # Handle special cases
        if "/" in normalized:  # Crypto pairs like BTC/USD
            normalized = normalized.replace("/", "")
            if not normalized.startswith("X:"):
                normalized = f"X:{normalized}"

        # Handle options symbols (convert to OCC format if needed)
        if " " in normalized and any(
            x in normalized for x in ["C", "P", "CALL", "PUT"]
        ):
            parts = normalized.split()
            if len(parts) >= 3:
                underlying = parts[0]
                expiry_str = parts[1]
                strike_type = parts[2]

                # This is a simplified conversion - a real implementation would need more logic
                # to handle the full OCC format conversion
                normalized = f"{underlying}_{expiry_str}_{strike_type}"

        return normalized

    def collect_multi_timeframe_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
        store: bool = True,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect multi-timeframe data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            timeframes: List of timeframes (e.g., ['1m', '5m', '15m', '1h', '1d'])
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits
            store: Whether to store the data

        Returns:
            Nested dictionary mapping symbols to timeframes to DataFrames
        """
        logger.info(f"Collecting multi-timeframe data for {len(symbols)} symbols")

        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        # Use the multi-timeframe collector to collect data
        results = self.multi_timeframe_collector.collect_multi_timeframe_data(
            symbols, timeframes, start_date, end_date, adjusted
        )

        # Store data if requested
        if store:
            self._store_multi_timeframe_data(results)


        logger.info(f"Collected multi-timeframe data for {len(results)} symbols")
        return results

    def store_multi_timeframe_data(
        self, data: Dict[str, Dict[str, pd.DataFrame]], store: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Store multi-timeframe data.

        Args:
            data: Nested dictionary mapping symbols to timeframes to DataFrames
            store: Whether to store the data

        Returns:
            The input data
        """
        logger.info(
            f"Storing multi-timeframe data for {len(data)} symbols"
        )

        # Store data if requested
        if store:
            self._store_multi_timeframe_data(data)

        return data

    def _store_multi_timeframe_data(
        self, data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> None:
        """
        Store multi-timeframe data in the database.

        Args:
            data: Nested dictionary mapping symbols to timeframes to DataFrames
        """
        stored_count = 0

        for symbol, timeframe_data in data.items():
            for timeframe, df in timeframe_data.items():
                if not df.empty:
                    try:
                        # Store data
                        self.storage.store_stock_aggs(df)
                        stored_count += len(df)
                        logger.debug(f"Stored {len(df)} {timeframe} bars for {symbol}")
                    except Exception as e:
                        logger.error(
                            f"Error storing {timeframe} data for {symbol}: {e}"
                        )
                        logger.debug(traceback.format_exc())

        logger.info(f"Stored {stored_count} bars of multi-timeframe data")


    def collect_stock_aggs(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
        store: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect stock aggregates for multiple symbols.

        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits
            store: Whether to store the data

        Returns:
            Dictionary mapping symbols to DataFrames with aggregates
        """
        logger.info(f"Collecting stock aggregates for {len(symbols)} symbols")

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

        # Collect data for each symbol in parallel
        futures = {}
        results = {}

        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        for symbol in symbols:
            future = self.executor.submit(
                self._collect_stock_aggs_for_symbol,
                symbol,
                multiplier,
                timespan,
                start_date,
                end_date,
                adjusted,
            )
            futures[future] = symbol

        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        # Store in TimescaleDB
                        self.storage.store_stock_aggs(df)

            except Exception as e:
                logger.error(f"Error collecting stock aggregates for {symbol}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Collected stock aggregates for {len(results)} symbols")
        return results

    def _collect_stock_aggs_for_symbol(
        self,
        symbol: str,
        multiplier: int,
        timespan: str, 
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool, 
    ) -> pd.DataFrame:
        """
        Collect stock aggregates for a single symbol.

        Args:
            symbol: Ticker symbol
            multiplier: Timespan multiplier
            timespan: Timespan unit
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits

        Returns:
            DataFrame with aggregates
        """
        try:
            logger.debug(f"Collecting {timespan} aggregates for {symbol}")
            df = self.polygon.get_aggregates(
                symbol, multiplier, timespan, start_date, end_date, adjusted
            )

            if df.empty:
                logger.warning(f"No {timespan} aggregates found for {symbol}")
                return df

            logger.debug(f"Collected {len(df)} {timespan} aggregates for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error collecting {timespan} aggregates for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_quotes(
        self,
        symbols: List[str],
        date_to_collect: Union[str, datetime, date],
        store: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect quotes for multiple symbols.

        Args:
            symbols: List of ticker symbols
            date_to_collect: Date to collect quotes for
            store: Whether to store the data

        Returns:
            Dictionary mapping symbols to DataFrames with quotes
        """
        logger.info(
            f"Collecting quotes for {len(symbols)} symbols on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect quotes for each symbol in parallel
        futures = {}
        results = {}

        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        for symbol in symbols:
            future = self.executor.submit(
                self._collect_quotes_for_symbol, symbol, date_str
            )
            futures[future] = symbol

        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        # Store in TimescaleDB
                        self.storage.store_quotes(df)

            except Exception as e:
                logger.error(f"Error collecting quotes for {symbol}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Collected quotes for {len(results)} symbols")
        return results

    def _collect_quotes_for_symbol(self, symbol: str, date_str: str) -> pd.DataFrame:
        """
        Collect quotes for a single symbol.

        Args:
            symbol: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with quotes
        """
        try:
            logger.debug(f"Collecting quotes for {symbol} on {date_str}")
            df = self.polygon.get_quotes(symbol, date_str)

            if df.empty:
                logger.warning(f"No quotes found for {symbol} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} quotes for {symbol} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting quotes for {symbol} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_trades(
        self,
        symbols: List[str],
        date_to_collect: Union[str, datetime, date],
        store: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect trades for multiple symbols.

        Args:
            symbols: List of ticker symbols
            date_to_collect: Date to collect trades for
            store: Whether to store the data

        Returns:
            Dictionary mapping symbols to DataFrames with trades
        """
        logger.info(
            f"Collecting trades for {len(symbols)} symbols on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect trades for each symbol in parallel
        futures = {}
        results = {}

        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        for symbol in symbols:
            future = self.executor.submit(
                self._collect_trades_for_symbol, symbol, date_str
            )
            futures[future] = symbol

        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        # Store in TimescaleDB
                        self.storage.store_trades(df)

            except Exception as e:
                logger.error(f"Error collecting trades for {symbol}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Collected trades for {len(results)} symbols")
        return results

    def _collect_trades_for_symbol(self, symbol: str, date_str: str) -> pd.DataFrame:
        """
        Collect trades for a single symbol.

        Args:
            symbol: Ticker symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with trades
        """
        try:
            logger.debug(f"Collecting trades for {symbol} on {date_str}")
            df = self.polygon.get_trades(symbol, date_str)

            if df.empty:
                logger.warning(f"No trades found for {symbol} on {date_str}")
                return df

            logger.debug(f"Collected {len(df)} trades for {symbol} on {date_str}")
            return df

        except Exception as e:
            logger.error(f"Error collecting trades for {symbol} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def collect_ticker_details(
        self, symbols: List[str], store: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """ 
        Collect ticker details for multiple symbols.

        Args:
            symbols: List of ticker symbols
            store: Whether to store the data

        Returns:
            Dictionary mapping symbols to ticker details
        """
        logger.info(f"Collecting ticker details for {len(symbols)} symbols")

        # Collect ticker details for each symbol in parallel
        futures = {}
        results = {}

        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        for symbol in symbols:
            future = self.executor.submit(
                self._collect_ticker_details_for_symbol, symbol
            )
            futures[future] = symbol

        # Process results as they complete
        ticker_details_list = []

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                details = future.result()
                if details:
                    results[symbol] = details

                    # Convert to DataFrame row for storage
                    df_row = pd.DataFrame(
                        [
                            {
                                # Required fields
                                "ticker": symbol,
                                "name": details.get("name", ""),
                                "market": details.get("market", ""),
                                "locale": details.get("locale", ""),
                                "type": details.get("type", ""),
                                "currency": details.get("currency", "USD"), 
                                "last_updated": datetime.now(), 
                                # Additional fields with NULL values to ensure upsert works
                                "active": True,
                                "primary_exchange": None,
                                "description": None,
                                "sic_code": None,
                                "sic_description": None,
                                "ticker_root": None,
                                "homepage_url": None,
                                "total_employees": None,
                                "list_date": None,
                                "share_class_shares_outstanding": None,
                                "weighted_shares_outstanding": None,
                                "market_cap": None,
                                "phone_number": None,
                                "address": None,
                                "metadata": None
                            }
                        ]
                    )

                    ticker_details_list.append(df_row)
            except Exception as e:
                logger.error(f"Error collecting ticker details for {symbol}: {e}")
                logger.debug(traceback.format_exc())

        # Store ticker details
        if store and ticker_details_list:
            ticker_details_df = pd.concat(ticker_details_list, ignore_index=True)
            self.storage.store_ticker_details(ticker_details_df)


        logger.info(f"Collected ticker details for {len(results)} symbols")
        return results

    def _collect_ticker_details_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Collect ticker details for a single symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with ticker details
        """
        try:
            logger.debug(f"Collecting ticker details for {symbol}")
            details = self.polygon.get_ticker_details(symbol)

            if not details:
                logger.warning(f"No ticker details found for {symbol}")
                return {}

            logger.debug(f"Collected ticker details for {symbol}")
            return details

        except Exception as e:
            logger.error(f"Error collecting ticker details for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return {}

    def collect_market_status(self, store: bool = True) -> Dict[str, Any]:
        """
        Collect current market status.

        Args:
            store: Whether to store the data

        Returns:
            Dictionary with market status
        """
        try:
            logger.info("Collecting market status")
            status = self.polygon.get_market_status()

            if not status:
                logger.warning("No market status found")
                return {}

            # Convert to DataFrame for storage
            if store:
                now = datetime.now()
                markets = ["us_equity", "us_options", "forex", "crypto"]
                rows = []

                for market in markets:
                    market_status = status.get(market, {})
                    rows.append(
                        {
                            "timestamp": now,
                            "market": market,
                            "status": market_status.get("status", "unknown"),
                            "next_open": market_status.get("next_open"),
                            "next_close": market_status.get("next_close"),
                            "early_close": market_status.get("early_close", False),
                            "late_open": market_status.get("late_open", False),
                        }
                    )

                df = pd.DataFrame(rows)
                self.storage.store_market_status(df)


            logger.info("Collected market status")
            return status

        except Exception as e:
            logger.error(f"Error collecting market status: {e}")
            logger.debug(traceback.format_exc())
            return {}

    def collect_market_holidays(self, store: bool = True) -> List[Dict[str, Any]]:
        """
        Collect market holidays.

        Args:
            store: Whether to store the data

        Returns:
            List of market holidays
        """
        try:
            logger.info("Collecting market holidays")
            holidays = self.polygon.get_market_holidays()

            if not holidays:
                logger.warning("No market holidays found")
                return []

            # Convert to DataFrame for storage
            if store and holidays:
                rows = []

                for holiday in holidays:
                    date_str = holiday.get("date")
                    if date_str:
                        holiday_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        year = holiday_date.year

                        # Add entry for each market
                        for market in ["us_equity", "us_options", "forex", "crypto"]:
                            status = holiday.get(market, {}).get("status", "unknown")
                            if status != "unknown":
                                rows.append(
                                    {
                                        "date": holiday_date,
                                        "name": holiday.get("name", ""),
                                        "market": market,
                                        "status": status,
                                        "open_time": holiday.get(market, {}).get(
                                            "open"
                                        ),
                                        "close_time": holiday.get(market, {}).get(
                                            "close"
                                        ),
                                        "year": year,
                                    }
                                )

                if rows:
                    df = pd.DataFrame(rows)
                    self.storage.store_market_holidays(df)


            logger.info(f"Collected {len(holidays)} market holidays")
            return holidays

        except Exception as e:
            logger.error(f"Error collecting market holidays: {e}")
            logger.debug(traceback.format_exc())
            return []

    def collect_news(
        self, symbols: Optional[List[str]] = None, limit: int = 100, store: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Collect news articles.

        Args:
            symbols: Optional list of ticker symbols to filter by
            limit: Maximum number of articles to collect
            store: Whether to store the data

        Returns:
            List of news articles
        """
        try:
            if symbols:
                # Check if the polygon client has the get_news method
                if not hasattr(self.polygon, 'get_news'):
                    logger.warning("News collection not supported by this Polygon client version")
                    return []
                    
                logger.info(f"Collecting news for {len(symbols)} symbols")

                # Collect news for each symbol in parallel
                futures = {}
                all_articles = []

                # Normalize symbols
                symbols = [self.normalize_symbol(symbol) for symbol in symbols]

                for symbol in symbols:
                    future = self.executor.submit(self.polygon.get_news, symbol, limit)
                    futures[future] = symbol

                # Process results as they complete
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        articles = future.result()
                        all_articles.extend(articles)
                    except Exception as e:
                        logger.error(f"Error collecting news for {symbol}: {e}")
                        logger.debug(traceback.format_exc())

                # Remove duplicates
                article_ids = set()
                unique_articles = []

                for article in all_articles:
                    article_id = article.get("id")
                    if article_id and article_id not in article_ids:
                        article_ids.add(article_id)
                        unique_articles.append(article)

                articles = unique_articles

            else:
                logger.info("Collecting general news")
                articles = self.polygon.get_news(limit=limit)

            # Convert to DataFrame for storage
            if store and articles:
                rows = []

                for article in articles:
                    rows.append(
                        {
                            "article_id": article.get("id", ""),
                            "published_utc": article.get("published_utc"),
                            "title": article.get("title", ""),
                            "author": article.get("author", ""),
                            "article_url": article.get("article_url", ""),
                            "tickers": article.get("tickers", []),
                            "image_url": article.get("image_url", ""),
                            "description": article.get("description", ""),
                            "keywords": article.get("keywords", []),
                            "source": article.get("publisher", {}).get(
                                "name", "unknown"
                            ),
                        }
                    )

                if rows:
                    df = pd.DataFrame(rows)
                    self.storage.store_news_articles(df)


            logger.info(f"Collected {len(articles)} news articles")
            return articles

        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            logger.debug(traceback.format_exc())
            return []

    def collect_options_flow(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 1,
        limit: int = 1000,
        store: bool = True,
    ) -> List[Dict[str, Any]]:
        """ 
        Collect options flow data from Unusual Whales.

        Args:
            symbols: Optional list of ticker symbols to filter by
            days_back: Number of days to look back
            limit: Maximum number of results to return
            store: Whether to store the data

        Returns:
            List of options flow data
        """
        try:
            logger.info(f"Collecting options flow data for the past {days_back} days")

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)

            # Normalize symbols if provided
            normalized_symbols = None
            if symbols:
                normalized_symbols = [
                    self.normalize_symbol(symbol) for symbol in symbols
                ]
                logger.info(
                    f"Filtering options flow for {len(normalized_symbols)} symbols"
                )

            # Collect options flow data
            flow_data = []

            # Get historical flow data
            for symbol in normalized_symbols or [None]:
                try:
                    historical_flow = self.unusual_whales.get_historical_flow(
                        limit=limit,
                        from_date=start_date,
                        to_date=end_date,
                        ticker=symbol,
                    )

                    if historical_flow:
                        flow_data.extend(historical_flow)

                except Exception as e:
                    logger.error(
                        f"Error collecting historical options flow for {symbol}: {e}"
                    )
                    logger.debug(traceback.format_exc())

            # Get live flow data
            if normalized_symbols:
                for symbol in normalized_symbols:
                    try:
                        live_flow = self.unusual_whales.get_options_flow(
                            limit=limit, ticker=symbol
                        )

                        if live_flow:
                            flow_data.extend(live_flow)

                    except Exception as e:
                        logger.error(
                            f"Error collecting live options flow for {symbol}: {e}"
                        )
                        logger.debug(traceback.format_exc())
            else:
                try:
                    live_flow = self.unusual_whales.get_options_flow(limit=limit)
                    if live_flow:
                        flow_data.extend(live_flow)
                except Exception as e:
                    logger.error(f"Error collecting live options flow: {e}")
                    logger.debug(traceback.format_exc())

            # Remove duplicates
            seen_ids = set()
            unique_flow_data = []

            for item in flow_data:
                item_id = item.get("id")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_flow_data.append(item)

            flow_data = unique_flow_data

            # Store data if requested
            if store and flow_data:
                # Convert to DataFrame
                df = self.unusual_whales.flow_to_dataframe(flow_data)

                if not df.empty:
                    # Store in options_flow table
                    # Note: This assumes a table structure that matches the DataFrame
                    self.storage.execute_statement(
                        """
                        CREATE TABLE IF NOT EXISTS options_flow (
                            id TEXT PRIMARY KEY,
                            timestamp TIMESTAMPTZ NOT NULL,
                            symbol TEXT NOT NULL,
                            contract_type TEXT,
                            strike NUMERIC,
                            expiration_date TIMESTAMPTZ,
                            premium NUMERIC,
                            size NUMERIC,
                            open_interest NUMERIC,
                            implied_volatility NUMERIC,
                            delta NUMERIC,
                            gamma NUMERIC,
                            theta NUMERIC,
                            vega NUMERIC,
                            sentiment TEXT,
                            trade_type TEXT,
                            source TEXT
                        )
                    """
                    )

                    # Use to_sql with if_exists='append' to store the data
                    df.to_sql(
                        "options_flow",
                        self.storage.engine,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=10000,
                    )

                    logger.info(f"Stored {len(df)} options flow records")

            logger.info(f"Collected {len(flow_data)} options flow records")
            return flow_data

        except Exception as e:
            logger.error(f"Error collecting options flow: {e}")
            logger.debug(traceback.format_exc())
            return []

    def run_daily_collection(
        self,
        symbols: list[str], 
        timeframes: list[str] = ["1m", "5m", "15m", "1h", "1d"],
        include_quotes: bool = True,
        include_trades: bool = False,
        include_options_flow: bool = True,
        days_back: int = 1,
    ) -> dict[str, Any]:
        """
        Run a daily collection of market data.

        Args:
            symbols: List of ticker symbols
            timeframes: List of timeframes to collect
            include_quotes: Whether to collect quotes
            include_trades: Whether to collect trades
            include_options_flow: Whether to collect options flow data
            days_back: Number of days to look back

        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Starting daily collection for {len(symbols)} symbols")

        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        stats = {
            "symbols_processed": len(symbols),
            "timeframes_processed": len(timeframes),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "aggregates_collected": {},
            "quotes_collected": 0,
            "trades_collected": 0,
            "ticker_details_collected": 0,
            "news_articles_collected": 0,
            "errors": [],
        }

        # Normalize symbols
        normalized_symbols = [self.normalize_symbol(symbol) for symbol in symbols]

        try:
            # Collect ticker details
            ticker_details = self.collect_ticker_details(normalized_symbols)
            stats["ticker_details_collected"] = len(ticker_details)

            # Use multi-timeframe collector for more efficient data collection
            try:
                try:
                    multi_timeframe_results = self.collect_multi_timeframe_data(
                        normalized_symbols, timeframes, start_date, end_date
                    )

                    for timeframe in timeframes:
                        total_bars = sum(
                            len(data.get(timeframe, pd.DataFrame()))
                            for data in multi_timeframe_results.values()
                        )
                        stats["aggregates_collected"][timeframe] = total_bars

                        logger.info(
                            f"Collected {total_bars} {timeframe} bars for {len(multi_timeframe_results)} symbols"
                        )

                except Exception as e:
                    error_msg = f"Error collecting multi-timeframe data: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats["errors"].append(error_msg)
            except Exception as e:
                logger.error(f"Unexpected error in multi-timeframe collection: {e}")

            # Collect quotes
            if include_quotes:
                try:
                    for day_offset in range(days_back):
                        collection_date = end_date - timedelta(days=day_offset)
                        quotes_results = self.collect_quotes(
                            normalized_symbols, collection_date
                        )

                        total_quotes = sum(len(df) for df in quotes_results.values())
                        stats["quotes_collected"] += total_quotes

                        logger.info(
                            f"Collected {total_quotes} quotes for {len(quotes_results)} symbols on {collection_date}"
                        )

                except Exception as e:
                    error_msg = f"Error collecting quotes: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats["errors"].append(error_msg)

            # Collect trades
            if include_trades:
                try:
                    for day_offset in range(days_back):
                        collection_date = end_date - timedelta(days=day_offset)
                        trades_results = self.collect_trades(
                            normalized_symbols, collection_date
                        )

                        total_trades = sum(len(df) for df in trades_results.values())
                        stats["trades_collected"] += total_trades

                        logger.info(
                            f"Collected {total_trades} trades for {len(trades_results)} symbols on {collection_date}"
                        )

                except Exception as e:
                    error_msg = f"Error collecting trades: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats["errors"].append(error_msg)

            # Collect options flow data
            if include_options_flow:
                try:
                    # Use all symbols without filtering test tickers
                    real_symbols = normalized_symbols
                    
                    options_flow = self.collect_options_flow(
                        real_symbols, days_back
                    )
                    stats["options_flow_collected"] = len(options_flow)

                except Exception as e:
                    error_msg = f"Error collecting options flow: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats["errors"].append(error_msg)

            # Collect news
            try:
                news_articles = self.collect_news(normalized_symbols)
                if news_articles is not None:
                    stats["news_articles_collected"] = len(news_articles)
                else:
                    stats["news_articles_collected"] = 0
                    logger.warning(
                        "News collection not supported by this Polygon client version"
                    )

            except Exception as e:
                error_msg = f"Error collecting news: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                stats["errors"].append(error_msg)

            # Collect market status and holidays
            try:
                self.collect_market_status()
                self.collect_market_holidays()

            except Exception as e:
                error_msg = f"Error collecting market status/holidays: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                stats["errors"].append(error_msg)

            logger.info("Daily collection completed")
            return stats

        except Exception as e:
            error_msg = f"Error in daily collection: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            stats["errors"].append(error_msg)
            return stats

    def shutdown(self) -> None:
        """Shutdown the pipeline and release resources."""
        logger.info("Shutting down data pipeline")
        self.stop_requested = True


        self.executor.shutdown(wait=True)
        logger.info("Data pipeline shutdown complete")

    def calculate_order_book_imbalance(
        self,
        symbol: str,
        bid_sizes: List[float],
        ask_sizes: List[float],
        bid_prices: List[float],
        ask_prices: List[float],
    ) -> Dict[str, Any]:
        """
        Calculate order book imbalance.

        Args:
            symbol: Ticker symbol
            bid_sizes: List of bid sizes
            ask_sizes: List of ask sizes
            bid_prices: List of bid prices
            ask_prices: List of ask prices

        Returns:
            Order book imbalance dictionary
        """
        # Simple implementation
        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)
        
        if total_bid_size == 0 or total_ask_size == 0:
            return {"imbalance": 0.0, "is_valid": True}
            
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        return {
            "imbalance": imbalance,
            "is_valid": True,
            "timestamp": datetime.now(),
            "symbol": symbol
        }

    def calculate_trade_flow_imbalance(
        self, symbol: str, buy_volume: float, sell_volume: float
    ) -> Dict[str, Any]:
        """
        Calculate trade flow imbalance.

        Args:
            symbol: Ticker symbol
            buy_volume: Buy volume
            sell_volume: Sell volume

        Returns:
            Trade flow imbalance dictionary
        """
        # Simple implementation
        if buy_volume == 0 and sell_volume == 0:
            return {"imbalance": 0.0, "is_valid": True}
            
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.0
        
        return {
            "imbalance": imbalance,
            "is_valid": True,
            "timestamp": datetime.now(),
            "symbol": symbol
        }

    def calculate_vwpp(
        self,
        symbol: str,
        bid_sizes: List[float],
        ask_sizes: List[float],
        bid_prices: List[float],
        ask_prices: List[float],
    ) -> float:
        """
        Calculate volume-weighted price pressure.

        Args:
            symbol: Ticker symbol
            bid_sizes: List of bid sizes
            ask_sizes: List of ask sizes
            bid_prices: List of bid prices
            ask_prices: List of ask prices

        Returns:
            Volume-weighted price pressure value
        """
        # Simple implementation of volume-weighted price pressure
        total_bid_value = sum(size * price for size, price in zip(bid_sizes, bid_prices))
        total_ask_value = sum(size * price for size, price in zip(ask_sizes, ask_prices))
        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)
        
        if total_bid_size == 0 or total_ask_size == 0:
            return 0.0
            
        bid_vwap = total_bid_value / total_bid_size if total_bid_size > 0 else 0
        ask_vwap = total_ask_value / total_ask_size if total_ask_size > 0 else 0
        
        if bid_vwap == 0 or ask_vwap == 0:
            return 0.0
            
        return (bid_vwap - ask_vwap) / ((bid_vwap + ask_vwap) / 2)

    def calculate_relative_strength(
        self,
        symbol: str,
        current_price: float,
        reference_prices: list[float],
        reference_volumes: list[float] = None,
    ) -> float:
        """
        Calculate relative strength of a symbol compared to reference symbols.

        Args:
            symbol: Ticker symbol
            current_price: Current price of the symbol
            reference_prices: List of prices for reference symbols
            reference_volumes: Optional list of volumes for reference symbols

        Returns:
            Relative strength value
        """
        # Simple implementation of relative strength calculation
        if not reference_prices:
            return 0.0
            
        # If volumes are provided, calculate volume-weighted average
        if reference_volumes and len(reference_volumes) == len(reference_prices):
            total_volume = sum(reference_volumes)
            if total_volume == 0:
                avg_reference_price = sum(reference_prices) / len(reference_prices)
            else:
                avg_reference_price = sum(p * v for p, v in zip(reference_prices, reference_volumes)) / total_volume
        else:
            # Simple average if no volumes or mismatched lengths
            avg_reference_price = sum(reference_prices) / len(reference_prices)
            
        if avg_reference_price == 0:
            return 0.0
            
        # Calculate relative strength as percentage difference
        return (current_price - avg_reference_price) / avg_reference_price
