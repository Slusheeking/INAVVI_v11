"""
Feature Engineering Pipeline

This module provides the core functionality for generating features from raw market data.
It implements the FeaturePipeline class that orchestrates the feature generation process.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import standard Python modules
import logging
import concurrent.futures

# Import project modules
from src.feature_engineering.store.feature_store import feature_store

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Main pipeline for feature engineering.
    
    This class provides a simplified interface for generating features from raw market data.
    It wraps the FeatureEngineer class and provides methods for generating different types of features.
    """
    
    def __init__(self, db_manager=None, use_redis_cache=True, redis_cache_ttl=3600, max_workers=4):
        """
        Initialize the feature pipeline.
        
        Args:
            db_manager: Database manager for loading data and storing features
            use_redis_cache: Whether to use Redis caching
            redis_cache_ttl: Time-to-live for cached features (in seconds)
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.engineer = FeatureEngineer(
            db_manager=db_manager,
            use_redis_cache=use_redis_cache,
            redis_cache_ttl=redis_cache_ttl,
            max_workers=max_workers
        )
        logger.info("Initialized FeaturePipeline")
    
    def generate_technical_indicators(self, df):
        """
        Generate technical indicators for the data.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with technical indicators
        """
        return self.engineer._calculate_technical_indicators(df)
    
    def generate_price_features(self, df):
        """
        Generate price-based features.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with price features
        """
        # Extract price-related features from market microstructure
        result_df = df.copy()
        
        # Calculate price volatility
        result_df["price_volatility"] = (
            result_df["close"].pct_change().rolling(window=20).std()
        )
        
        # Calculate price acceleration
        result_df["price_acceleration"] = result_df["close"].pct_change().diff()
        
        # Calculate price gap
        result_df["price_gap"] = result_df["open"] / result_df["close"].shift(1) - 1
        
        # Calculate price trend strength
        result_df["trend_strength"] = (
            result_df["close"].rolling(window=20).mean()
            / result_df["close"].rolling(window=50).mean()
            - 1
        )
        
        return result_df
    
    def generate_volume_features(self, df):
        """
        Generate volume-based features.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with volume features
        """
        # Extract volume-related features from market microstructure
        result_df = df.copy()
        
        # Calculate volume volatility
        result_df["volume_volatility"] = (
            result_df["volume"].pct_change().rolling(window=20).std()
        )
        
        # Calculate price-volume correlation
        result_df["price_volume_corr"] = (
            result_df["close"]
            .pct_change()
            .rolling(window=20)
            .corr(result_df["volume"].pct_change())
        )
        
        # Calculate volume acceleration
        result_df["volume_acceleration"] = result_df["volume"].pct_change().diff()
        
        # Calculate volume surprise
        result_df["volume_surprise"] = (
            result_df["volume"] / result_df["volume"].rolling(window=20).mean() - 1
        )
        
        # Calculate volume gap
        result_df["volume_gap"] = result_df["volume"] / result_df["volume"].shift(1) - 1
        
        # Calculate OBV
        result_df = self.engineer._calculate_obv(result_df, {})
        
        return result_df
    
    def generate_volatility_features(self, df):
        """
        Generate volatility-based features.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with volatility features
        """
        # Extract volatility-related features
        result_df = df.copy()
        
        # Calculate Bollinger Bands
        result_df = self.engineer._calculate_bollinger_bands(result_df, {"parameters": {"window": 20, "num_std_dev": 2}})
        
        # Calculate ATR
        result_df = self.engineer._calculate_atr(result_df, {"parameters": {"window": 14}})
        
        # Calculate range volatility ratio
        result_df["range_volatility_ratio"] = (result_df["high"] - result_df["low"]) / (
            result_df["close"].rolling(window=20).std() * result_df["close"]
        )
        
        return result_df
    
    def generate_all_features(self, df):
        """
        Generate all features for the data.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with all features
        """
        # Generate all types of features
        df = self.generate_technical_indicators(df)
        df = self.generate_price_features(df)
        df = self.generate_volume_features(df)
        df = self.generate_volatility_features(df)
        
        # Apply transformations
        df = self.engineer._apply_feature_transformations(df)
        
        return df
    
    def create_target_variable(self, df):
        """
        Create target variable for model training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target variable
        """
        return self.engineer._create_target_variable(df)
    
    def get_health(self):
        """
        Get the health status of the feature pipeline.
        
        Returns:
            Dictionary with health status
        """
        return {"healthy": True, "reason": "Feature pipeline is operational"}
    
    def initialize(self):
        """
        Initialize the feature pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing feature pipeline")
        return True
    
    def get_status(self):
        """
        Get the current status of the feature pipeline.
        
        Returns:
            Status string
        """
        return "active"
    
    def shutdown(self):
        """
        Shutdown the feature pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Shutting down feature pipeline")
        return True


class FeatureEngineer:
    """
    Core class for generating features from raw market data.

    This class provides methods for:
    - Loading raw market data from TimescaleDB
    - Calculating technical indicators
    - Generating market microstructure features
    - Creating target variables
    - Normalizing and transforming features
    """

    def __init__(
        self,
        db_manager: Optional[Any] = None,
        use_redis_cache: bool = False,  # Changed to False by default since we're not using Redis
        redis_cache_ttl: int = 3600,  # 1 hour
        indicators_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the feature engineer.

        Args:
            db_manager: Database manager for loading data and storing features
            use_redis_cache: Whether to use Redis caching (not used in this simplified version)
            redis_cache_ttl: Time-to-live for cached features (in seconds)
            indicators_config: Configuration for technical indicators
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.db_manager = db_manager
        self.use_redis_cache = False  # Disable Redis caching in this simplified version
        self.redis_cache_ttl = redis_cache_ttl
        self.indicators_config = indicators_config or {}
        self.max_workers = max_workers

        # No multi-timeframe processor in this simplified version

        # Initialize feature calculators
        self._initialize_feature_calculators()

        logger.info("Initialized FeatureEngineer")

    def generate_multi_timeframe_features(
        self,
        symbol: str,
        timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"],
        lookback_days: int = 365,
        parallel: bool = True,
        store_features: bool = True,
        include_target: bool = True,
    ) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Generate features for multiple timeframes.

        Args:
            symbol: Ticker symbol
            timeframes: List of timeframes to generate features for
            lookback_days: Number of days to look back for data
            parallel: Whether to process timeframes in parallel
            store_features: Whether to store features in the database
            include_target: Whether to include target variable

        Returns:
            Dictionary mapping timeframes to tuples of (features, targets)
        """
        logger.info(
            f"Generating multi-timeframe features for {symbol} across {timeframes}"
        )

        results = {}

        # No caching in this simplified version
        timeframes_to_process = timeframes

        # Generate features for each timeframe
        if parallel and len(timeframes_to_process) > 1:
            # Use parallel processing with concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(timeframes_to_process))) as executor:
                # Submit tasks
                future_to_timeframe = {
                    executor.submit(
                        self.generate_features,
                        symbol,
                        tf,
                        lookback_days,
                        store_features,
                        include_target,
                    ): tf for tf in timeframes_to_process
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        features, targets = future.result()
                        results[timeframe] = (features, targets)
                    except Exception as e:
                        logger.error(f"Error generating features for {symbol} {timeframe}: {e}")
        else:
            # Sequential processing
            for timeframe in timeframes_to_process:
                try:
                    features, targets = self.generate_features(
                        symbol, timeframe, lookback_days, store_features, include_target
                    )
                    results[timeframe] = (features, targets)
                except Exception as e:
                    logger.error(
                        f"Error generating features for {symbol} {timeframe}: {e}"
                    )

        return results

    def generate_features(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 365,
        store_features: bool = True,
        include_target: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate features for a single timeframe.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe to generate features for
            lookback_days: Number of days to look back for data
            store_features: Whether to store features in the database
            include_target: Whether to include target variable

        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        logger.info(f"Generating features for {symbol} {timeframe}")

        # Load raw data
        df = self._load_raw_data(symbol, timeframe, lookback_days)

        if df is None or df.empty:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return None, None

        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)

        # Generate market microstructure features
        df = self._generate_market_microstructure_features(df, symbol)

        # Apply feature transformations
        df = self._apply_feature_transformations(df)

        # Create target variable if requested
        if include_target:
            df = self._create_target_variable(df)

        # Store features if requested
        if store_features and self.db_manager is not None:
            self._store_features(df, symbol, timeframe)

        # Extract features and target
        feature_cols = [
            col
            for col in df.columns
            if col not in ["timestamp", "symbol", "target", "future_return"]
        ]

        if not feature_cols:
            logger.warning(f"No features generated for {symbol} {timeframe}")
            return None, None

        # Convert to numpy arrays
        features = df[feature_cols].values

        # Normalize features
        features = self._normalize_features(features)

        # Extract target if available
        targets = (
            df["target"].values if include_target and "target" in df.columns else None
        )

        return features, targets

    def _load_raw_data(
        self, symbol: str, timeframe: str, lookback_days: int
    ) -> Optional[pd.DataFrame]:
        """
        Load raw market data from TimescaleDB.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe to load
            lookback_days: Number of days to look back

        Returns:
            DataFrame with raw data or None if error
        """
        if self.db_manager is None:
            logger.error("Database manager is required for loading data")
            return None

        # Calculate start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        try:
            # Direct SQL query without caching
            query = """
            SELECT * FROM stock_aggs
            WHERE symbol = :symbol
            AND timeframe = :timeframe
            AND timestamp BETWEEN :start_time AND :end_time
            ORDER BY timestamp
            """

            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            }

            # Execute query
            df = self.db_manager.execute_query(query, params)

            if df.empty:
                logger.warning(
                    f"No data found for {symbol} {timeframe} from {start_time} to {end_time}"
                )
                return None

            return df
        except Exception as e:
            logger.error(f"Error loading raw data for {symbol} {timeframe}: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data.

        Args:
            df: DataFrame with raw data

        Returns:
            DataFrame with technical indicators
        """
        if df is None or df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get indicators to calculate
        indicators = self.indicators_config.get("indicators", [])

        # Calculate each indicator
        for indicator in indicators:
            indicator_type = indicator.get("type")

            if indicator_type == "rsi":
                result_df = self._calculate_rsi(result_df, indicator)
            elif indicator_type == "macd":
                result_df = self._calculate_macd(result_df, indicator)
            elif indicator_type == "bollinger_bands":
                result_df = self._calculate_bollinger_bands(result_df, indicator)
            elif indicator_type == "atr":
                result_df = self._calculate_atr(result_df, indicator)
            elif indicator_type == "obv":
                result_df = self._calculate_obv(result_df, indicator)
            elif indicator_type == "sma":
                result_df = self._calculate_sma(result_df, indicator)
            elif indicator_type == "ema":
                result_df = self._calculate_ema(result_df, indicator)
            elif indicator_type == "stochastic":
                result_df = self._calculate_stochastic(result_df, indicator)
            elif indicator_type == "adx":
                result_df = self._calculate_adx(result_df, indicator)
            elif indicator_type == "ichimoku":
                result_df = self._calculate_ichimoku(result_df, indicator)
            else:
                logger.warning(f"Unknown indicator type: {indicator_type}")

        return result_df

    def _generate_market_microstructure_features(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        Generate market microstructure features.

        Args:
            df: DataFrame with raw data
            symbol: Ticker symbol

        Returns:
            DataFrame with market microstructure features
        """
        if df is None or df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Calculate price volatility
        result_df["price_volatility"] = (
            result_df["close"].pct_change().rolling(window=20).std()
        )

        # Calculate volume volatility
        result_df["volume_volatility"] = (
            result_df["volume"].pct_change().rolling(window=20).std()
        )

        # Calculate price-volume correlation
        result_df["price_volume_corr"] = (
            result_df["close"]
            .pct_change()
            .rolling(window=20)
            .corr(result_df["volume"].pct_change())
        )

        # Calculate price acceleration (second derivative of price)
        result_df["price_acceleration"] = result_df["close"].pct_change().diff()

        # Calculate volume acceleration
        result_df["volume_acceleration"] = result_df["volume"].pct_change().diff()

        # Calculate price range relative to volatility
        result_df["range_volatility_ratio"] = (result_df["high"] - result_df["low"]) / (
            result_df["close"].rolling(window=20).std() * result_df["close"]
        )

        # Calculate volume surprise (volume relative to recent average)
        result_df["volume_surprise"] = (
            result_df["volume"] / result_df["volume"].rolling(window=20).mean() - 1
        )

        # Calculate price gap (overnight gap)
        result_df["price_gap"] = result_df["open"] / result_df["close"].shift(1) - 1

        # Calculate volume gap
        result_df["volume_gap"] = result_df["volume"] / result_df["volume"].shift(1) - 1

        # Calculate price trend strength
        result_df["trend_strength"] = (
            result_df["close"].rolling(window=20).mean()
            / result_df["close"].rolling(window=50).mean()
            - 1
        )

        return result_df

    def _apply_feature_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature transformations.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with transformed features
        """
        if df is None or df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get transformations to apply
        transformations = self.indicators_config.get("transformations", [])

        # Apply each transformation
        for transformation in transformations:
            transformation_type = transformation.get("type")

            if transformation_type == "lag":
                result_df = self._apply_lag_transformation(result_df, transformation)
            elif transformation_type == "diff":
                result_df = self._apply_diff_transformation(result_df, transformation)
            elif transformation_type == "pct_change":
                result_df = self._apply_pct_change_transformation(
                    result_df, transformation
                )
            elif transformation_type == "rolling":
                result_df = self._apply_rolling_transformation(
                    result_df, transformation
                )
            elif transformation_type == "crossover":
                result_df = self._apply_crossover_transformation(
                    result_df, transformation
                )
            else:
                logger.warning(f"Unknown transformation type: {transformation_type}")

        return result_df

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for model training.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with target variable
        """
        if df is None or df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Calculate future returns (next day's return)
        result_df["future_return"] = result_df["close"].pct_change(1).shift(-1)

        # Create classification target (1 for positive return, 0 for negative return)
        result_df["target"] = np.where(result_df["future_return"] > 0, 1, 0)

        return result_df

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to have similar scales.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Normalized feature matrix
        """
        if features is None or features.size == 0:
            return features

        # Simple min-max normalization
        min_vals = np.nanmin(features, axis=0)
        max_vals = np.nanmax(features, axis=0)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        normalized_features = (features - min_vals) / range_vals

        # Replace NaN values with 0
        normalized_features = np.nan_to_num(normalized_features)

        return normalized_features

    def _store_features(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Store features in the database.

        Args:
            df: DataFrame with features
            symbol: Ticker symbol
            timeframe: Timeframe of the data

        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty or self.db_manager is None:
            return False

        try:
            # Prepare features for storage
            features_data = []

            # Get timestamp column
            timestamp_col = "timestamp"

            # Get feature columns (exclude timestamp, symbol, target, future_return)
            feature_cols = [
                col
                for col in df.columns
                if col not in [timestamp_col, "symbol", "target", "future_return"]
            ]

            # Create feature records
            for _, row in df.iterrows():
                timestamp = row[timestamp_col]

                for feature_name in feature_cols:
                    feature_value = row[feature_name]

                    # Skip NaN values
                    if pd.isna(feature_value):
                        continue

                    # Determine feature group
                    if feature_name in ["open", "high", "low", "close", "volume"]:
                        feature_group = "price"
                    elif feature_name in [
                        "rsi",
                        "macd",
                        "macd_signal",
                        "macd_histogram",
                    ]:
                        feature_group = "momentum"
                    elif feature_name in [
                        "bb_upper",
                        "bb_middle",
                        "bb_lower",
                        "bb_bandwidth",
                        "bb_percent_b",
                    ]:
                        feature_group = "volatility"
                    elif feature_name in [
                        "price_volatility",
                        "volume_volatility",
                        "atr",
                    ]:
                        feature_group = "volatility"
                    elif feature_name in [
                        "price_volume_corr",
                        "volume_surprise",
                        "obv",
                    ]:
                        feature_group = "volume"
                    elif feature_name in [
                        "price_gap",
                        "volume_gap",
                        "price_acceleration",
                        "volume_acceleration",
                    ]:
                        feature_group = "microstructure"
                    elif feature_name in ["range_volatility_ratio", "trend_strength"]:
                        feature_group = "trend"
                    else:
                        feature_group = "other"

                    # Add feature record
                    features_data.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "feature_name": feature_name,
                            "feature_value": float(feature_value),
                            "timeframe": timeframe,
                            "feature_group": feature_group,
                            "tags": "{}",
                        }
                    )

            # Create DataFrame
            features_df = pd.DataFrame(features_data)

            # Store in database
            if not features_df.empty:
                feature_store.store_features_batch(features_df)
                logger.info(
                    f"Stored {len(features_df)} features for {symbol} {timeframe}"
                )
                return True

            return False
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False

    def _initialize_feature_calculators(self) -> None:
        """
        Initialize feature calculators with default configurations.
        """
        # Set default indicators config if not provided
        if not self.indicators_config:
            self.indicators_config = {
                "indicators": [
                    {"type": "rsi", "parameters": {"window": 14}},
                    {
                        "type": "macd",
                        "parameters": {
                            "fast_period": 12,
                            "slow_period": 26,
                            "signal_period": 9,
                        },
                    },
                    {
                        "type": "bollinger_bands",
                        "parameters": {"window": 20, "num_std_dev": 2},
                    },
                    {"type": "atr", "parameters": {"window": 14}},
                    {"type": "obv"},
                    {"type": "sma", "parameters": {"window": 20, "column": "close"}},
                    {"type": "sma", "parameters": {"window": 50, "column": "close"}},
                    {"type": "sma", "parameters": {"window": 200, "column": "close"}},
                    {"type": "ema", "parameters": {"window": 20, "column": "close"}},
                    {"type": "ema", "parameters": {"window": 50, "column": "close"}},
                    {
                        "type": "stochastic",
                        "parameters": {"k_period": 14, "d_period": 3},
                    },
                    {"type": "adx", "parameters": {"window": 14}},
                ],
                "transformations": [
                    {"type": "lag", "parameters": {"periods": [1, 2, 3, 5, 10]}},
                    {"type": "diff", "parameters": {"periods": [1, 2, 3]}},
                    {"type": "pct_change", "parameters": {"periods": [1, 2, 3, 5, 10]}},
                    {
                        "type": "rolling",
                        "parameters": {
                            "windows": [5, 10, 20],
                            "functions": ["mean", "std", "min", "max"],
                        },
                    },
                    {
                        "type": "crossover",
                        "parameters": {
                            "pairs": [["sma_20", "sma_50"], ["ema_20", "ema_50"]]
                        },
                    },
                ],
            }

    # Technical Indicator Calculation Methods

    def _calculate_rsi(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with price data
            config: Configuration for RSI calculation

        Returns:
            DataFrame with RSI
        """
        window = config.get("parameters", {}).get("window", 14)

        # Calculate price changes
        delta = df["close"].diff()

        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Add to DataFrame
        df["rsi"] = rsi

        return df

    def _calculate_macd(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            df: DataFrame with price data
            config: Configuration for MACD calculation

        Returns:
            DataFrame with MACD
        """
        fast_period = config.get("parameters", {}).get("fast_period", 12)
        slow_period = config.get("parameters", {}).get("slow_period", 26)
        signal_period = config.get("parameters", {}).get("signal_period", 9)

        # Calculate EMAs
        ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD and signal line
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        # Add to DataFrame
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_histogram"] = histogram

        return df

    def _calculate_bollinger_bands(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data
            config: Configuration for Bollinger Bands calculation

        Returns:
            DataFrame with Bollinger Bands
        """
        window = config.get("parameters", {}).get("window", 20)
        num_std_dev = config.get("parameters", {}).get("num_std_dev", 2)

        # Calculate middle band (SMA)
        middle_band = df["close"].rolling(window=window).mean()

        # Calculate standard deviation
        std_dev = df["close"].rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)

        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (df["close"] - lower_band) / (upper_band - lower_band)

        # Add to DataFrame
        df["bb_upper"] = upper_band
        df["bb_middle"] = middle_band
        df["bb_lower"] = lower_band
        df["bb_bandwidth"] = bandwidth
        df["bb_percent_b"] = percent_b

        return df

    def _calculate_atr(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        Args:
            df: DataFrame with price data
            config: Configuration for ATR calculation

        Returns:
            DataFrame with ATR
        """
        window = config.get("parameters", {}).get("window", 14)

        # Calculate true range
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR
        atr = true_range.rolling(window=window).mean()

        # Add to DataFrame
        df["atr"] = atr

        return df

    def _calculate_obv(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            df: DataFrame with price and volume data
            config: Configuration for OBV calculation

        Returns:
            DataFrame with OBV
        """
        # Initialize OBV with zeros
        obv = np.zeros(len(df))

        # Calculate price changes
        price_changes = df["close"].diff().values

        # Calculate OBV
        for i in range(1, len(df)):
            if price_changes[i] > 0:  # Price went up
                obv[i] = obv[i - 1] + df["volume"].iloc[i]
            elif price_changes[i] < 0:  # Price went down
                obv[i] = obv[i - 1] - df["volume"].iloc[i]
            else:  # Price unchanged
                obv[i] = obv[i - 1]

        # Add to DataFrame
        df["obv"] = obv

        return df

    def _calculate_sma(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            df: DataFrame with price data
            config: Configuration for SMA calculation

        Returns:
            DataFrame with SMA
        """
        window = config.get("parameters", {}).get("window", 20)
        column = config.get("parameters", {}).get("column", "close")

        # Calculate SMA
        sma = df[column].rolling(window=window).mean()

        # Add to DataFrame
        df[f"sma_{window}"] = sma

        return df

    def _calculate_ema(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            df: DataFrame with price data
            config: Configuration for EMA calculation

        Returns:
            DataFrame with EMA
        """
        window = config.get("parameters", {}).get("window", 20)
        column = config.get("parameters", {}).get("column", "close")

        # Calculate EMA
        ema = df[column].ewm(span=window, adjust=False).mean()

        # Add to DataFrame
        df[f"ema_{window}"] = ema

        return df

    def _calculate_stochastic(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.

        Args:
            df: DataFrame with price data
            config: Configuration for Stochastic Oscillator calculation

        Returns:
            DataFrame with Stochastic Oscillator
        """
        k_period = config.get("parameters", {}).get("k_period", 14)
        d_period = config.get("parameters", {}).get("d_period", 3)

        # Calculate %K
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        k = 100 * ((df["close"] - low_min) / (high_max - low_min))

        # Calculate %D
        d = k.rolling(window=d_period).mean()

        # Add to DataFrame
        df["stoch_k"] = k
        df["stoch_d"] = d

        return df

    def _calculate_adx(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).

        Args:
            df: DataFrame with price data
            config: Configuration for ADX calculation

        Returns:
            DataFrame with ADX
        """
        window = config.get("parameters", {}).get("window", 14)

        # Calculate True Range
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = df["high"].diff()
        down_move = df["low"].diff().mul(-1)

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate Smoothed True Range and Directional Movement
        tr_smooth = true_range.rolling(window=window).sum()
        pos_dm_smooth = pd.Series(pos_dm).rolling(window=window).sum()
        neg_dm_smooth = pd.Series(neg_dm).rolling(window=window).sum()

        # Calculate Directional Indicators
        pos_di = 100 * (pos_dm_smooth / tr_smooth)
        neg_di = 100 * (neg_dm_smooth / tr_smooth)

        # Calculate Directional Index
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)

        # Calculate ADX
        adx = dx.rolling(window=window).mean()

        # Add to DataFrame
        df["adx"] = adx
        df["pos_di"] = pos_di
        df["neg_di"] = neg_di

        return df

    def _calculate_ichimoku(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud.

        Args:
            df: DataFrame with price data
            config: Configuration for Ichimoku Cloud calculation

        Returns:
            DataFrame with Ichimoku Cloud
        """
        tenkan_period = config.get("parameters", {}).get("tenkan_period", 9)
        kijun_period = config.get("parameters", {}).get("kijun_period", 26)
        senkou_span_b_period = config.get("parameters", {}).get(
            "senkou_span_b_period", 52
        )

        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = (
            df["high"].rolling(window=tenkan_period).max()
            + df["low"].rolling(window=tenkan_period).min()
        ) / 2

        # Calculate Kijun-sen (Base Line)
        kijun_sen = (
            df["high"].rolling(window=kijun_period).max()
            + df["low"].rolling(window=kijun_period).min()
        ) / 2

        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = (
            (
                df["high"].rolling(window=senkou_span_b_period).max()
                + df["low"].rolling(window=senkou_span_b_period).min()
            )
            / 2
        ).shift(kijun_period)

        # Calculate Chikou Span (Lagging Span)
        chikou_span = df["close"].shift(-kijun_period)

        # Add to DataFrame
        df["ichimoku_tenkan_sen"] = tenkan_sen
        df["ichimoku_kijun_sen"] = kijun_sen
        df["ichimoku_senkou_span_a"] = senkou_span_a
        df["ichimoku_senkou_span_b"] = senkou_span_b
        df["ichimoku_chikou_span"] = chikou_span

        return df

    # Feature Transformation Methods

    def _apply_lag_transformation(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply lag transformation to features.

        Args:
            df: DataFrame with features
            config: Configuration for lag transformation

        Returns:
            DataFrame with lagged features
        """
        periods = config.get("parameters", {}).get("periods", [1, 2, 3, 5, 10])

        # Get numeric columns (excluding target if it exists)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        # Create lagged features
        for col in numeric_cols:
            for period in periods:
                df[f"{col}_lag_{period}"] = df[col].shift(period)

        return df

    def _apply_diff_transformation(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply difference transformation to features.

        Args:
            df: DataFrame with features
            config: Configuration for difference transformation

        Returns:
            DataFrame with differenced features
        """
        periods = config.get("parameters", {}).get("periods", [1, 2, 3])

        # Get numeric columns (excluding target if it exists)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        # Create differenced features
        for col in numeric_cols:
            for period in periods:
                df[f"{col}_diff_{period}"] = df[col].diff(period)

        return df

    def _apply_pct_change_transformation(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply percentage change transformation to features.

        Args:
            df: DataFrame with features
            config: Configuration for percentage change transformation

        Returns:
            DataFrame with percentage change features
        """
        periods = config.get("parameters", {}).get("periods", [1, 2, 3, 5, 10])

        # Get numeric columns (excluding target if it exists)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        # Create percentage change features
        for col in numeric_cols:
            for period in periods:
                df[f"{col}_pct_{period}"] = df[col].pct_change(period)

        return df

    def _apply_rolling_transformation(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply rolling window transformation to features.

        Args:
            df: DataFrame with features
            config: Configuration for rolling window transformation

        Returns:
            DataFrame with rolling window features
        """
        windows = config.get("parameters", {}).get("windows", [5, 10, 20])
        functions = config.get("parameters", {}).get(
            "functions", ["mean", "std", "min", "max"]
        )

        # Get numeric columns (excluding target if it exists)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        # Create rolling window features
        for col in numeric_cols:
            for window in windows:
                for func in functions:
                    if func == "mean":
                        df[f"{col}_roll_{window}_mean"] = (
                            df[col].rolling(window=window).mean()
                        )
                    elif func == "std":
                        df[f"{col}_roll_{window}_std"] = (
                            df[col].rolling(window=window).std()
                        )
                    elif func == "min":
                        df[f"{col}_roll_{window}_min"] = (
                            df[col].rolling(window=window).min()
                        )
                    elif func == "max":
                        df[f"{col}_roll_{window}_max"] = (
                            df[col].rolling(window=window).max()
                        )

        return df

    def _apply_crossover_transformation(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply crossover transformation to features.

        Args:
            df: DataFrame with features
            config: Configuration for crossover transformation

        Returns:
            DataFrame with crossover features
        """
        pairs = config.get("parameters", {}).get("pairs", [])

        # Create crossover features
        for pair in pairs:
            if len(pair) == 2 and all(col in df.columns for col in pair):
                col1, col2 = pair
                # Crossover: 1 if col1 crosses above col2, -1 if col1 crosses below col2, 0 otherwise
                df[f"{col1}_{col2}_crossover"] = np.where(
                    (df[col1] > df[col2]) & (df[col1].shift(1) <= df[col2].shift(1)),
                    1,
                    np.where(
                        (df[col1] < df[col2])
                        & (df[col1].shift(1) >= df[col2].shift(1)),
                        -1,
                        0,
                    ),
                )

        return df
