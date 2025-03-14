"""
Feature Store

This module provides functionality for storing and retrieving features from the database.
It integrates with the feature cache and registry to provide efficient access to features.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Try both import styles to handle different execution contexts
try:
    from autonomous_trading_system.src.feature_engineering.store.feature_cache import (
        feature_store_cache,
    )
    from autonomous_trading_system.src.feature_engineering.store.feature_registry import (
        feature_registry,
    )
    from autonomous_trading_system.src.utils.database.timescale_manager import (
        TimescaleManager,
    )
except ImportError:
    # For when running from within the autonomous_trading_system directory
    from src.feature_engineering.store.feature_cache import feature_store_cache
    from src.feature_engineering.store.feature_registry import feature_registry
    from src.utils.database.timescale_manager import TimescaleManager


logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Store for managing feature data.

    This class provides methods for:
    - Storing features in the database
    - Retrieving features from the database or cache
    - Managing feature metadata
    - Batch operations for efficient processing
    """

    def __init__(
        self,
        db_manager: Optional[TimescaleManager] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        batch_size: int = 10000,
        max_workers: int = 4,
    ):
        """
        Initialize the feature store.

        Args:
            db_manager: Database manager for storing/retrieving features
            use_cache: Whether to use the feature cache
            cache_ttl: Time-to-live for cached features (in seconds)
            batch_size: Batch size for database operations
            max_workers: Maximum number of worker threads for parallel operations
        """
        self.db_manager = db_manager
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Initialize database tables if needed
        if self.db_manager:
            self._initialize_tables()

        logger.info("Initialized FeatureStore")

    def store_feature(
        self,
        symbol: str,
        feature_name: str,
        feature_value: float,
        timestamp: datetime,
        timeframe: str,
        feature_group: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a single feature value in the database.

        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            feature_value: Value of the feature
            timestamp: Timestamp for the feature value
            timeframe: Timeframe of the data (e.g., '1m', '5m', '1h', '1d')
            feature_group: Group the feature belongs to (default: from registry)
            tags: Additional tags for the feature value

        Returns:
            True if successful, False otherwise
        """
        if self.db_manager is None:
            logger.error("Database manager is required for storing features")
            return False

        # Get feature metadata from registry
        metadata = feature_registry.get_feature_metadata(feature_name)
        if metadata is None and feature_group is None:
            logger.warning(
                f"Feature {feature_name} not found in registry and no feature_group provided"
            )
            return False

        # Use feature group from metadata if not provided
        if feature_group is None:
            feature_group = metadata["feature_group"]

        # Create DataFrame with feature data
        df = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                    "timeframe": timeframe,
                    "feature_group": feature_group,
                    "tags": json.dumps(tags or {}),
                }
            ]
        )

        try:
            # Store in database
            df.to_sql(
                "features",
                self.db_manager.engine,
                if_exists="append",
                index=False,
                method="multi",
            )

            logger.debug(f"Stored feature {feature_name} for {symbol} at {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Error storing feature: {e}")
            return False

    def store_features_batch(self, features_df: pd.DataFrame) -> int:
        """
        Store multiple features in the database.

        Args:
            features_df: DataFrame with feature data
                Required columns: timestamp, symbol, feature_name, feature_value, timeframe, feature_group
                Optional columns: tags

        Returns:
            Number of features stored
        """
        if self.db_manager is None:
            logger.error("Database manager is required for storing features")
            return 0

        if features_df.empty:
            logger.warning("No features to store")
            return 0

        # Ensure required columns are present
        required_columns = [
            "timestamp",
            "symbol",
            "feature_name",
            "feature_value",
            "timeframe",
            "feature_group",
        ]
        missing_columns = [
            col for col in required_columns if col not in features_df.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return 0

        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(features_df["timestamp"]):
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])

        # Ensure tags column exists
        if "tags" not in features_df.columns:
            features_df["tags"] = "{}"
        elif not features_df["tags"].apply(lambda x: isinstance(x, str)).all():
            # Convert tags to JSON strings if they're not already
            features_df["tags"] = features_df["tags"].apply(
                lambda x: json.dumps(x) if not isinstance(x, str) else x
            )

        try:
            # Store in database in batches
            total_stored = 0
            for i in range(0, len(features_df), self.batch_size):
                batch = features_df.iloc[i : i + self.batch_size]
                batch.to_sql(
                    "features",
                    self.db_manager.engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                )
                total_stored += len(batch)
                logger.debug(
                    f"Stored batch of {len(batch)} features ({i+1}-{i+len(batch)} of {len(features_df)})"
                )

            logger.info(f"Stored {total_stored} features")
            return total_stored
        except Exception as e:
            logger.error(f"Error storing features batch: {e}")
            return 0

    def get_feature(
        self, symbol: str, feature_name: str, timestamp: datetime, timeframe: str
    ) -> Optional[float]:
        """
        Get a single feature value from the database or cache.

        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            timestamp: Timestamp for the feature value
            timeframe: Timeframe of the data

        Returns:
            Feature value or None if not found
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = f"feature:{symbol}:{feature_name}:{timeframe}"
            cache_params = {
                "symbol": symbol,
                "feature_name": feature_name,
                "timestamp": timestamp.isoformat(),
                "timeframe": timeframe,
            }

            cached_value = feature_store_cache.get_feature(cache_key, cache_params)
            if cached_value is not None:
                logger.debug(
                    f"Cache hit for feature {feature_name} for {symbol} at {timestamp}"
                )
                return cached_value

        # Fall back to database
        if self.db_manager is None:
            logger.error("Database manager is required for retrieving features")
            return None

        try:
            query = """
                SELECT feature_value
                FROM features
                WHERE symbol = :symbol
                AND feature_name = :feature_name
                AND timeframe = :timeframe
                AND timestamp = :timestamp
            """

            params = {
                "symbol": symbol,
                "feature_name": feature_name,
                "timeframe": timeframe,
                "timestamp": timestamp,
            }

            result = self.db_manager.execute_query(query, params)

            if result.empty:
                logger.debug(
                    f"Feature {feature_name} for {symbol} at {timestamp} not found"
                )
                return None

            feature_value = float(result.iloc[0]["feature_value"])

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_feature(
                    cache_key, cache_params, feature_value, self.cache_ttl
                )

            return feature_value
        except Exception as e:
            logger.error(f"Error retrieving feature: {e}")
            return None

    def get_features_for_symbol(
        self,
        symbol: str,
        feature_names: list[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get multiple features for a symbol within a time range.

        Args:
            symbol: Ticker symbol
            feature_names: List of feature names
            start_time: Start time for the range
            end_time: End time for the range
            timeframe: Timeframe of the data

        Returns:
            DataFrame with feature values or None if error
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = f"features:{symbol}:{timeframe}:{start_time.isoformat()}:{end_time.isoformat()}"
            cache_params = {
                "symbol": symbol,
                "feature_names": feature_names,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeframe": timeframe,
            }

            cached_df = feature_store_cache.get_dataframe(cache_key, cache_params)
            if cached_df is not None:
                logger.debug(
                    f"Cache hit for features for {symbol} from {start_time} to {end_time}"
                )
                return cached_df

        # Fall back to database
        if self.db_manager is None:
            logger.error("Database manager is required for retrieving features")
            return None

        try:
            query = """
                SELECT timestamp, feature_name, feature_value
                FROM features
                WHERE symbol = :symbol
                AND feature_name IN :feature_names
                AND timeframe = :timeframe
                AND timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp, feature_name
            """

            params = {
                "symbol": symbol,
                "feature_names": tuple(feature_names),
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            }

            result = self.db_manager.execute_query(query, params)

            if result.empty:
                logger.debug(
                    f"No features found for {symbol} from {start_time} to {end_time}"
                )
                return pd.DataFrame()

            # Pivot the result to get features as columns
            df = result.pivot(
                index="timestamp", columns="feature_name", values="feature_value"
            )

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_dataframe(
                    cache_key, cache_params, df, self.cache_ttl
                )

            return df
        except Exception as e:
            logger.error(f"Error retrieving features for symbol: {e}")
            return None

    def get_latest_features(
        self, symbol: str, feature_names: list[str], timeframe: str, limit: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Get the latest feature values for a symbol.

        Args:
            symbol: Ticker symbol
            feature_names: List of feature names
            timeframe: Timeframe of the data
            limit: Maximum number of rows to return

        Returns:
            DataFrame with feature values or None if error
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = f"latest_features:{symbol}:{timeframe}:{limit}"
            cache_params = {
                "symbol": symbol,
                "feature_names": feature_names,
                "timeframe": timeframe,
                "limit": limit,
            }

            cached_df = feature_store_cache.get_dataframe(cache_key, cache_params)
            if cached_df is not None:
                logger.debug(f"Cache hit for latest features for {symbol}")
                return cached_df

        # Fall back to database
        if self.db_manager is None:
            logger.error("Database manager is required for retrieving features")
            return None

        try:
            query = """
                WITH latest_timestamps AS (
                    SELECT timestamp
                    FROM features
                    WHERE symbol = :symbol
                    AND timeframe = :timeframe
                    GROUP BY timestamp
                    ORDER BY timestamp DESC
                    LIMIT :limit
                )
                SELECT f.timestamp, f.feature_name, f.feature_value
                FROM features f
                JOIN latest_timestamps lt ON f.timestamp = lt.timestamp
                WHERE f.symbol = :symbol
                AND f.feature_name IN :feature_names
                AND f.timeframe = :timeframe
                ORDER BY f.timestamp DESC, f.feature_name
            """

            params = {
                "symbol": symbol,
                "feature_names": tuple(feature_names),
                "timeframe": timeframe,
                "limit": limit,
            }

            result = self.db_manager.execute_query(query, params)

            if result.empty:
                logger.debug(f"No latest features found for {symbol}")
                return pd.DataFrame()

            # Pivot the result to get features as columns
            df = result.pivot(
                index="timestamp", columns="feature_name", values="feature_value"
            )

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_dataframe(
                    cache_key, cache_params, df, self.cache_ttl
                )

            return df
        except Exception as e:
            logger.error(f"Error retrieving latest features: {e}")
            return None

    def get_features_for_symbols(
        self,
        symbols: list[str],
        feature_names: list[str],
        timestamp: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get feature values for multiple symbols at a specific timestamp.

        Args:
            symbols: List of ticker symbols
            feature_names: List of feature names
            timestamp: Timestamp for the feature values
            timeframe: Timeframe of the data

        Returns:
            DataFrame with feature values or None if error
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = f"features_for_symbols:{timestamp.isoformat()}:{timeframe}"
            cache_params = {
                "symbols": symbols,
                "feature_names": feature_names,
                "timestamp": timestamp.isoformat(),
                "timeframe": timeframe,
            }

            cached_df = feature_store_cache.get_dataframe(cache_key, cache_params)
            if cached_df is not None:
                logger.debug(
                    f"Cache hit for features for multiple symbols at {timestamp}"
                )
                return cached_df

        # Fall back to database
        if self.db_manager is None:
            logger.error("Database manager is required for retrieving features")
            return None

        try:
            query = """
                SELECT symbol, feature_name, feature_value
                FROM features
                WHERE symbol IN :symbols
                AND feature_name IN :feature_names
                AND timeframe = :timeframe
                AND timestamp = :timestamp
            """

            params = {
                "symbols": tuple(symbols),
                "feature_names": tuple(feature_names),
                "timeframe": timeframe,
                "timestamp": timestamp,
            }

            result = self.db_manager.execute_query(query, params)

            if result.empty:
                logger.debug(f"No features found for symbols at {timestamp}")
                return pd.DataFrame()

            # Pivot the result to get features as columns
            df = result.pivot(
                index="symbol", columns="feature_name", values="feature_value"
            )

            # Reset index to make symbol a column
            df = df.reset_index()

            # Add timestamp column
            df["timestamp"] = timestamp

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_dataframe(
                    cache_key, cache_params, df, self.cache_ttl
                )

            return df
        except Exception as e:
            logger.error(f"Error retrieving features for symbols: {e}")
            return None

    def get_feature_matrix(
        self,
        symbol: str,
        feature_names: list[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a feature matrix for machine learning.

        Args:
            symbol: Ticker symbol
            feature_names: List of feature names
            start_time: Start time for the range
            end_time: End time for the range
            timeframe: Timeframe of the data

        Returns:
            Tuple of (features, timestamps) as numpy arrays or (None, None) if error
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = f"feature_matrix:{symbol}:{timeframe}:{start_time.isoformat()}:{end_time.isoformat()}"
            cache_params = {
                "symbol": symbol,
                "feature_names": feature_names,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "timeframe": timeframe,
            }

            cached_result = feature_store_cache.get_feature(cache_key, cache_params)
            if cached_result is not None:
                logger.debug(
                    f"Cache hit for feature matrix for {symbol} from {start_time} to {end_time}"
                )
                return cached_result

        # Get features as DataFrame
        df = self.get_features_for_symbol(
            symbol, feature_names, start_time, end_time, timeframe
        )

        if df is None or df.empty:
            logger.warning(
                f"No features found for {symbol} from {start_time} to {end_time}"
            )
            return None, None

        try:
            # Extract timestamps
            timestamps = df["timestamp"].values

            # Extract features
            features = df[feature_names].values

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_feature(
                    cache_key, cache_params, (features, timestamps), self.cache_ttl
                )

            return features, timestamps
        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return None, None

    def get_feature_matrix_for_symbols(
        self,
        symbols: list[str],
        feature_names: list[str],
        timestamp: datetime,
        timeframe: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a feature matrix for multiple symbols at a specific timestamp.

        Args:
            symbols: List of ticker symbols
            feature_names: List of feature names
            timestamp: Timestamp for the feature values
            timeframe: Timeframe of the data

        Returns:
            Tuple of (features, symbols) as numpy arrays or (None, None) if error
        """
        # Try cache first if enabled
        if self.use_cache:
            cache_key = (
                f"feature_matrix_for_symbols:{timestamp.isoformat()}:{timeframe}"
            )
            cache_params = {
                "symbols": symbols,
                "feature_names": feature_names,
                "timestamp": timestamp.isoformat(),
                "timeframe": timeframe,
            }

            cached_result = feature_store_cache.get_feature(cache_key, cache_params)
            if cached_result is not None:
                logger.debug(
                    f"Cache hit for feature matrix for multiple symbols at {timestamp}"
                )
                return cached_result

        # Get features as DataFrame
        df = self.get_features_for_symbols(symbols, feature_names, timestamp, timeframe)

        if df is None or df.empty:
            logger.warning(f"No features found for symbols at {timestamp}")
            return None, None

        try:
            # Extract symbols
            symbol_array = df["symbol"].values

            # Extract features
            features = df[feature_names].values

            # Cache the result if enabled
            if self.use_cache:
                feature_store_cache.set_feature(
                    cache_key, cache_params, (features, symbol_array), self.cache_ttl
                )

            return features, symbol_array
        except Exception as e:
            logger.error(f"Error creating feature matrix for symbols: {e}")
            return None, None

    def delete_features(
        self,
        symbol: Optional[str] = None,
        feature_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """
        Delete features from the database.

        Args:
            symbol: Ticker symbol (if None, delete for all symbols)
            feature_name: Name of the feature (if None, delete all features)
            start_time: Start time for the range (if None, no lower bound)
            end_time: End time for the range (if None, no upper bound)
            timeframe: Timeframe of the data (if None, delete for all timeframes)

        Returns:
            Number of features deleted
        """
        if self.db_manager is None:
            logger.error("Database manager is required for deleting features")
            return 0

        try:
            # Build query
            query = "DELETE FROM features WHERE 1=1"
            params = {}

            if symbol:
                query += " AND symbol = :symbol"
                params["symbol"] = symbol

            if feature_name:
                query += " AND feature_name = :feature_name"
                params["feature_name"] = feature_name

            if start_time:
                query += " AND timestamp >= :start_time"
                params["start_time"] = start_time

            if end_time:
                query += " AND timestamp <= :end_time"
                params["end_time"] = end_time

            if timeframe:
                query += " AND timeframe = :timeframe"
                params["timeframe"] = timeframe

            # Execute query
            result = self.db_manager.execute_statement(query, params)

            # Invalidate cache if enabled
            if self.use_cache:
                pattern = "feature:*"
                if symbol:
                    pattern = f"feature:{symbol}:*"
                feature_store_cache.invalidate_by_pattern(pattern)

            logger.info("Deleted features matching criteria")
            return result.rowcount if hasattr(result, "rowcount") else 0
        except Exception as e:
            logger.error(f"Error deleting features: {e}")
            return 0

    def get_available_features(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> list[str]:
        """
        Get a list of available features in the database.

        Args:
            symbol: Ticker symbol (if None, get for all symbols)
            timeframe: Timeframe of the data (if None, get for all timeframes)

        Returns:
            List of feature names
        """
        if self.db_manager is None:
            logger.error(
                "Database manager is required for retrieving available features"
            )
            return []

        try:
            # Build query
            query = "SELECT DISTINCT feature_name FROM features WHERE 1=1"
            params = {}

            if symbol:
                query += " AND symbol = :symbol"
                params["symbol"] = symbol

            if timeframe:
                query += " AND timeframe = :timeframe"
                params["timeframe"] = timeframe

            # Execute query
            result = self.db_manager.execute_query(query, params)

            if result.empty:
                return []

            return result["feature_name"].tolist()
        except Exception as e:
            logger.error(f"Error retrieving available features: {e}")
            return []

    def get_available_symbols(
        self, feature_name: Optional[str] = None, timeframe: Optional[str] = None
    ) -> list[str]:
        """
        Get a list of available symbols in the database.

        Args:
            feature_name: Name of the feature (if None, get for all features)
            timeframe: Timeframe of the data (if None, get for all timeframes)

        Returns:
            List of symbols
        """
        if self.db_manager is None:
            logger.error(
                "Database manager is required for retrieving available symbols"
            )
            return []

        try:
            # Build query
            query = "SELECT DISTINCT symbol FROM features WHERE 1=1"
            params = {}

            if feature_name:
                query += " AND feature_name = :feature_name"
                params["feature_name"] = feature_name

            if timeframe:
                query += " AND timeframe = :timeframe"
                params["timeframe"] = timeframe

            # Execute query
            result = self.db_manager.execute_query(query, params)

            if result.empty:
                return []

            return result["symbol"].tolist()
        except Exception as e:
            logger.error(f"Error retrieving available symbols: {e}")
            return []

    def get_available_timeframes(
        self, symbol: Optional[str] = None, feature_name: Optional[str] = None
    ) -> list[str]:
        """
        Get a list of available timeframes in the database.

        Args:
            symbol: Ticker symbol (if None, get for all symbols)
            feature_name: Name of the feature (if None, get for all features)

        Returns:
            List of timeframes
        """
        if self.db_manager is None:
            logger.error(
                "Database manager is required for retrieving available timeframes"
            )
            return []

        try:
            # Build query
            query = "SELECT DISTINCT timeframe FROM features WHERE 1=1"
            params = {}

            if symbol:
                query += " AND symbol = :symbol"
                params["symbol"] = symbol

            if feature_name:
                query += " AND feature_name = :feature_name"
                params["feature_name"] = feature_name

            # Execute query
            result = self.db_manager.execute_query(query, params)

            if result.empty:
                return []

            return result["timeframe"].tolist()
        except Exception as e:
            logger.error(f"Error retrieving available timeframes: {e}")
            return []

    def get_feature_statistics(
        self,
        symbol: str,
        feature_name: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, float]:
        """
        Get statistics for a feature.

        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            timeframe: Timeframe of the data
            start_time: Start time for the range (if None, no lower bound)
            end_time: End time for the range (if None, no upper bound)

        Returns:
            Dictionary with statistics (min, max, mean, std, count)
        """
        if self.db_manager is None:
            logger.error(
                "Database manager is required for retrieving feature statistics"
            )
            return {}

        try:
            # Build query
            query = """
                SELECT
                    MIN(feature_value) as min_value,
                    MAX(feature_value) as max_value,
                    AVG(feature_value) as mean_value,
                    STDDEV(feature_value) as std_value,
                    COUNT(*) as count
                FROM features
                WHERE symbol = :symbol
                AND feature_name = :feature_name
                AND timeframe = :timeframe
            """
            params = {
                "symbol": symbol,
                "feature_name": feature_name,
                "timeframe": timeframe,
            }

            if start_time:
                query += " AND timestamp >= :start_time"
                params["start_time"] = start_time

            if end_time:
                query += " AND timestamp <= :end_time"
                params["end_time"] = end_time

            # Execute query
            result = self.db_manager.execute_query(query, params)

            if result.empty:
                return {}

            # Convert to dictionary
            stats = {
                "min": float(result.iloc[0]["min_value"])
                if not pd.isna(result.iloc[0]["min_value"])
                else None,
                "max": float(result.iloc[0]["max_value"])
                if not pd.isna(result.iloc[0]["max_value"])
                else None,
                "mean": float(result.iloc[0]["mean_value"])
                if not pd.isna(result.iloc[0]["mean_value"])
                else None,
                "std": float(result.iloc[0]["std_value"])
                if not pd.isna(result.iloc[0]["std_value"])
                else None,
                "count": int(result.iloc[0]["count"]),
            }

            return stats
        except Exception as e:
            logger.error(f"Error retrieving feature statistics: {e}")
            return {}

    def _initialize_tables(self) -> None:
        """
        Initialize database tables if they don't exist.
        """
        try:
            # Create features table
            self.db_manager.execute_statement(
                """
                CREATE TABLE IF NOT EXISTS features (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value FLOAT NOT NULL,
                    timeframe TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    tags TEXT,
                    PRIMARY KEY (timestamp, symbol, feature_name, timeframe)
                )
            """
            )

            # Create hypertable if TimescaleDB is available
            try:
                self.db_manager.execute_statement(
                    """
                    SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE)
                """
                )
                logger.info("Created hypertable for features")
            except Exception as e:
                logger.warning(f"Could not create hypertable for features: {e}")

            # Create indexes
            self.db_manager.execute_statement(
                """
                CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol)
            """
            )

            self.db_manager.execute_statement(
                """
                CREATE INDEX IF NOT EXISTS idx_features_feature_name ON features (feature_name)
            """
            )

            self.db_manager.execute_statement(
                """
                CREATE INDEX IF NOT EXISTS idx_features_timeframe ON features (timeframe)
            """
            )

            self.db_manager.execute_statement(
                """
                CREATE INDEX IF NOT EXISTS idx_features_feature_group ON features (feature_group)
            """
            )

            logger.info("Initialized feature store tables")
        except Exception as e:
            logger.error(f"Error initializing feature store tables: {e}")


# Create a singleton instance
feature_store = FeatureStore()
