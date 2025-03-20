#!/usr/bin/env python3
"""
ML Data Processor

This module provides data processing functionality for the ML model trainer.
It handles data loading, feature engineering, and target generation.
"""

import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_data_processor')


class MLDataProcessor:
    """
    Data processing for ML model training
    Handles data loading, feature engineering, and target generation
    """

    def __init__(self, data_loader, redis_client=None, config=None):
        self.data_loader = data_loader
        self.redis = redis_client

        # Default configuration
        self.config = config or {
            'data_dir': os.environ.get('DATA_DIR', './data'),
            'monitoring_dir': os.environ.get('MONITORING_DIR', './monitoring'),
            'min_samples': 1000,
            'lookback_days': 30,
            'monitoring': {'enabled': True, 'drift_threshold': 0.05},
        }

        # Ensure directories exist
        self._ensure_directories()

        logger.info("ML Data Processor initialized")

    def _ensure_directories(self):
        """Ensure all required directories exist with proper permissions"""
        for directory in [self.config['data_dir'], self.config['monitoring_dir']]:
            # Convert relative paths to absolute if needed
            if directory.startswith('./'):
                directory = directory.replace('./', '/app/', 1)
            elif not directory.startswith('/'):
                directory = f'/app/{directory}'

            try:
                os.makedirs(directory, exist_ok=True)
                # Set permissive permissions
                os.chmod(directory, 0o777)
                logger.info(
                    f"Created directory with full permissions: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")

    def load_historical_data(self):
        """Load historical data for model training"""
        try:
            # Get active tickers
            active_tickers = self.get_active_tickers()

            if not active_tickers:
                logger.warning("No active tickers found")
                return None

            logger.info(f"Loading data for {len(active_tickers)} tickers")

            # Calculate date ranges
            end_date = datetime.datetime.now()
            # Focus on the previous day's data
            start_date = end_date - datetime.timedelta(days=1)
            logger.info(f"Training on data from {start_date} to {end_date}")

            # Load price data
            price_data = self.data_loader.load_price_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe='1m'
            )

            # Load options data
            options_data = self.data_loader.load_options_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date
            )

            # Load market data
            market_data = self.data_loader.load_market_data(
                start_date=start_date,
                end_date=end_date,
                symbols=['SPY', 'QQQ', 'IWM', 'VIX']
            )

            # Combine data
            combined_data = self.prepare_training_data(
                price_data=price_data,
                options_data=options_data,
                market_data=market_data
            )

            if len(combined_data) < self.config['min_samples']:
                logger.warning(
                    f"Insufficient data: {len(combined_data)} samples (min: {self.config['min_samples']})")
                return None

            logger.info(f"Loaded {len(combined_data)} samples for training")

            # Save data for future reference
            # Skip saving data to disk to avoid directory permission issues
            logger.info(
                "Skipping data save - using fresh data for each training run")

            return combined_data

        except Exception as e:
            logger.error(
                f"Error loading historical data: {str(e)}", exc_info=True)
            return None

    def prepare_training_data(self, price_data, options_data, market_data):
        """Prepare data for model training"""
        try:
            # Check if we have valid price data
            if not price_data or (isinstance(price_data, dict) and len(price_data) == 0):
                logger.error("No price data available for training")
                return pd.DataFrame()

            # Create master dataframe from price data
            dfs = []
            if isinstance(price_data, dict):
                for ticker, df in price_data.items():
                    if df is not None and not df.empty:
                        df = df.copy()
                        df['ticker'] = ticker
                        dfs.append(df)
            elif isinstance(price_data, pd.DataFrame) and not price_data.empty:
                dfs.append(price_data)

            # Check if we have any valid dataframes to concatenate
            if not dfs:
                logger.error("No valid price dataframes to concatenate")
                return pd.DataFrame()

            # Concatenate the dataframes
            combined_price = pd.concat(dfs, ignore_index=True)

            # Ensure timestamp index
            if 'timestamp' in combined_price.columns:
                combined_price.set_index('timestamp', inplace=True)

            # Calculate technical indicators
            combined_price = self.calculate_technical_indicators(
                combined_price)

            # Merge options data if available
            if options_data is not None:
                if isinstance(options_data, dict):
                    # Convert dict of dataframes to a single dataframe
                    options_dfs = []
                    for ticker, data in options_data.items():
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            ticker_df = data.copy()
                            if 'ticker' not in ticker_df.columns:
                                ticker_df['ticker'] = ticker
                            options_dfs.append(ticker_df)

                    if options_dfs:
                        options_df = pd.concat(options_dfs, ignore_index=True)
                    else:
                        logger.warning(
                            "No valid options dataframes to concatenate")
                        options_df = pd.DataFrame()
                else:
                    options_df = options_data

                if not options_df.empty:
                    # Ensure datetime index for merging
                    if 'timestamp' in options_df.columns:
                        options_df.set_index('timestamp', inplace=True)

                # Ensure timestamp dtypes match
                if 'timestamp' in combined_price.reset_index().columns and 'timestamp' in options_df.reset_index().columns:
                    # Convert both to nanosecond precision
                    combined_price_reset = combined_price.reset_index().copy()
                    combined_price_reset['timestamp'] = pd.to_datetime(
                        combined_price_reset['timestamp']).astype('datetime64[ns]')
                    options_df_reset = options_df.reset_index().copy()
                    combined_price_reset = combined_price_reset.sort_values(
                        'timestamp')
                    options_df_reset['timestamp'] = pd.to_datetime(
                        options_df_reset['timestamp']).astype('datetime64[ns]')

                    # Merge on timestamp and ticker
                    combined_data = pd.merge_asof(
                        combined_price_reset,
                        options_df_reset,
                        on='timestamp',
                        by='ticker',
                        direction='backward',
                        tolerance=pd.Timedelta('1h'),
                        suffixes=('', '_options')
                    )
                else:
                    combined_data = combined_price.reset_index()
            else:
                combined_data = combined_price.reset_index()

            # Merge market data
            if market_data is not None and not market_data.empty:
                # Ensure datetime index
                if 'timestamp' in market_data.columns:
                    market_data.set_index('timestamp', inplace=True)

                # Ensure timestamp dtypes match
                combined_data['timestamp'] = pd.to_datetime(
                    combined_data['timestamp']).astype('datetime64[ns]')
                market_data_reset = market_data.reset_index().copy()
                combined_data = combined_data.sort_values('timestamp')
                market_data_reset['timestamp'] = pd.to_datetime(
                    market_data_reset['timestamp']).astype('datetime64[ns]')

                # Merge on timestamp
                combined_data = pd.merge_asof(
                    combined_data,
                    market_data_reset,
                    on='timestamp',
                    direction='backward',
                    suffixes=('', '_market')
                )

            # Generate target variables
            combined_data = self.generate_targets(combined_data)

            # Drop rows with missing values in critical columns
            critical_columns = ['close', 'high', 'low', 'volume', 'timestamp']
            combined_data = combined_data.dropna(subset=critical_columns)

            # Clean up extreme values
            # Safe approach to avoid SettingWithCopyWarning
            numeric_cols = combined_data.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                if col in combined_data.columns:
                    q1, q3 = combined_data[col].quantile([0.01, 0.99])
                    iqr = q3 - q1
                    # Use loc[] to avoid the SettingWithCopyWarning
                    combined_data.loc[:, col] = combined_data[col].clip(
                        q1 - 3 * iqr, q3 + 3 * iqr)

            return combined_data

        except Exception as e:
            logger.error(
                f"Error preparing training data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for price data"""
        try:
            # Group by ticker if multiple tickers
            if 'ticker' in df.columns:
                grouped = df.groupby('ticker')
                result_dfs = []

                for ticker, group_df in grouped:
                    result_df = self._calculate_indicators_for_group(group_df)
                    result_df['ticker'] = ticker
                    result_dfs.append(result_df)

                result = pd.concat(result_dfs)
                return result
            else:
                return self._calculate_indicators_for_group(df)

        except Exception as e:
            logger.error(
                f"Error calculating technical indicators: {str(e)}", exc_info=True)
            return df

    def _calculate_indicators_for_group(self, df):
        """Calculate indicators for a single ticker dataframe"""
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Simple Moving Averages
        result['sma5'] = result['close'].rolling(window=5).mean()
        result['sma10'] = result['close'].rolling(window=10).mean()
        result['sma20'] = result['close'].rolling(window=20).mean()

        # Exponential Moving Averages
        result['ema5'] = result['close'].ewm(span=5, adjust=False).mean()
        result['ema10'] = result['close'].ewm(span=10, adjust=False).mean()
        result['ema20'] = result['close'].ewm(span=20, adjust=False).mean()

        # MACD
        result['ema12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema26'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema12'] - result['ema26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']

        # Relative strength to price
        result['price_rel_sma5'] = result['close'] / result['sma5'] - 1
        result['price_rel_sma10'] = result['close'] / result['sma10'] - 1
        result['price_rel_sma20'] = result['close'] / result['sma20'] - 1

        # Momentum
        result['mom1'] = result['close'].pct_change(1)
        result['mom5'] = result['close'].pct_change(5)
        result['mom10'] = result['close'].pct_change(10)

        # Volatility
        result['volatility'] = result['close'].rolling(
            window=10).std() / result['close'].rolling(window=10).mean()

        # Volume-based indicators
        if 'volume' in result.columns:
            result['volume_sma5'] = result['volume'].rolling(window=5).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma5']

            # Money Flow Index (simplified)
            result['money_flow'] = result['close'] * result['volume']
            result['money_flow_sma'] = result['money_flow'].rolling(
                window=14).mean()

        # RSI (14)
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Handle division by zero
        rs = gain / loss.replace(0, 1e-9)
        result['rsi'] = 100 - (100 / (1 + rs))

        # Handle any remaining infinity values
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            result[col] = result[col].ffill().bfill().fillna(0)

        # Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        result['bb_width'] = (result['bb_upper'] -
                              result['bb_lower']) / result['bb_middle']

        return result

    def generate_targets(self, df):
        """Generate target variables for supervised learning"""
        try:
            # Make a copy
            result = df.copy()

            # Signal detection target (1 if price increases by 1% within next 10 bars)
            # Use a lower threshold to get more positive samples (was 0.01 or 1%)
            future_returns = result.groupby(
                'ticker')['close'].pct_change(10).shift(-10)
            # Ensure we have both positive and negative samples by using a lower threshold
            result['signal_target'] = (future_returns > 0.001).astype(int)

            # Check if we have both classes represented
            if result['signal_target'].nunique() < 2:
                # If not, force some samples to be positive to ensure balanced classes
                logger.warning(
                    "Only one class detected in targets, creating synthetic samples")
                # Set the top 20% of future returns as positive signals regardless of threshold
                # At least 100 positive samples
                positive_count = max(int(len(result) * 0.2), 100)
                top_indices = future_returns.nlargest(positive_count).index
                result.loc[top_indices, 'signal_target'] = 1
                logger.info(
                    f"Created {positive_count} synthetic positive samples")

            # Price prediction targets
            result['future_return_5min'] = result.groupby(
                'ticker')['close'].pct_change(5).shift(-5)
            result['future_return_10min'] = result.groupby(
                'ticker')['close'].pct_change(10).shift(-10)
            result['future_return_30min'] = result.groupby(
                'ticker')['close'].pct_change(30).shift(-30)

            # Direction target (1 for up, 0 for down)
            result['future_direction'] = (
                result['future_return_10min'] > 0).astype(int)

            # Risk assessment target (ATR as % of price)
            high_low = result['high'] - result['low']
            high_close = abs(result['high'] - result['close'].shift())
            low_close = abs(result['low'] - result['close'].shift())
            tr = pd.concat([high_low, high_close, low_close],
                           axis=1).max(axis=1)
            result['atr14'] = tr.rolling(14).mean()
            result['atr_pct'] = result['atr14'] / result['close']

            # Handle any NaN or Inf values in targets
            for col in ['signal_target', 'future_return_5min', 'future_return_10min',
                        'future_return_30min', 'future_direction', 'atr_pct']:
                if col in result.columns:
                    result[col] = result[col].replace(
                        [np.inf, -np.inf], np.nan)
                    result[col] = result[col].ffill().bfill().fillna(0)

            # Exit strategy target (optimal exit time within next 30 bars)
            # This is simplified; in practice would be more sophisticated
            for ticker, group in result.groupby('ticker'):
                future_prices = [
                    group['close'].shift(-i) for i in range(1, 31)]
                future_prices_df = pd.concat(future_prices, axis=1)
                max_price = future_prices_df.max(axis=1)
                optimal_exit = (max_price / group['close'] - 1)
                result.loc[group.index, 'optimal_exit'] = optimal_exit

            # Handle NaN in optimal_exit
            if 'optimal_exit' in result.columns:
                result['optimal_exit'] = result['optimal_exit'].replace(
                    [np.inf, -np.inf], np.nan)
                result['optimal_exit'] = result['optimal_exit'].ffill(
                ).bfill().fillna(0)

            return result

        except Exception as e:
            logger.error(f"Error generating targets: {str(e)}", exc_info=True)
            return df

    def store_reference_data(self, data):
        """Store reference data for drift detection"""
        if not self.config['monitoring']['enabled']:
            logger.info("Monitoring disabled, skipping reference data storage")
            return

        try:
            # Skip saving reference data to disk
            logger.info(
                "Skipping reference data storage - using fresh data for each run")

            # If needed, we could store reference data in memory for the current session
            # self.reference_data = data.sample(min(10000, len(data)), random_state=42) if len(data) > 0 else data.copy()
            # logger.info(f"Stored reference data in memory: {len(self.reference_data)} samples")

        except Exception as e:
            logger.error(f"Error storing reference data: {str(e)}")

    def get_active_tickers(self):
        """Get list of active tickers for training"""
        try:
            if self.redis is None:
                # Return default list if no Redis connection
                return ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLP", "XLI", "XLU"]

            # Get from Redis
            watchlist_tickers = self.redis.zrange("watchlist:active", 0, -1)

            # Convert from bytes if needed
            watchlist_tickers = [
                t.decode('utf-8') if isinstance(t, bytes) else t for t in watchlist_tickers]

            # If no watchlist, use a default list
            if not watchlist_tickers:
                watchlist_tickers = ["SPY", "QQQ", "IWM", "DIA", "XLK",
                                     "XLF", "XLV", "XLP", "XLI", "XLU"]

            logger.info(
                f"Using {len(watchlist_tickers)} active tickers for training")

            return watchlist_tickers

        except Exception as e:
            logger.error(
                f"Error getting active tickers: {str(e)}", exc_info=True)
            # Return default list on error
            return ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV"]

    def prepare_signal_detection_data(self, data):
        """Prepare data for signal detection model"""
        try:
            # Select features
            feature_columns = [
                # Price-based features
                'close', 'open', 'high', 'low', 'volume',

                # Technical indicators
                'sma5', 'sma10', 'sma20',
                'ema5', 'ema10', 'ema20',
                'macd', 'macd_signal', 'macd_hist',
                'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20',
                'mom1', 'mom5', 'mom10',
                'volatility', 'volume_ratio', 'rsi',
                'bb_width',

                # Market features (if available)
                'spy_close', 'vix_close', 'spy_change', 'vix_change',

                # Options features (if available)
                'put_call_ratio', 'implied_volatility', 'option_volume'
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 5:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()

            # Select data
            X = data[available_columns].copy()
            y = data['signal_target'].copy()

            # Drop rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            # Handle any remaining infinity values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.mean(), inplace=True)

            logger.info(
                f"Prepared signal detection data with {len(X)} samples and {len(available_columns)} features")

            return X, y

        except Exception as e:
            logger.error(
                f"Error preparing signal detection data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def prepare_price_prediction_data(self, data):
        """Prepare data for price prediction model"""
        try:
            # Select features
            feature_columns = [
                # Price-based features
                'close', 'high', 'low', 'volume',

                # Technical indicators
                'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20',
                'macd', 'rsi', 'volatility',

                # Market features (if available)
                'spy_close', 'vix_close'
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 4:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return np.array([]), np.array([])

            # Target columns
            target_columns = ['future_return_5min',
                              'future_return_10min', 'future_return_30min']
            available_targets = [
                col for col in target_columns if col in data.columns]

            if len(available_targets) == 0:
                logger.warning("No target variables available")
                return np.array([]), np.array([])

            # Group by ticker to create sequences
            sequences = []
            targets = []

            for ticker, group in data.groupby('ticker'):
                # Sort by timestamp
                group = group.sort_index()

                # Select features and targets
                X = group[available_columns].values
                y = group[available_targets].values

                # Create sequences (lookback of 20 intervals)
                for i in range(20, len(X)):
                    sequences.append(X[i-20:i])
                    targets.append(y[i])

            # Convert to numpy arrays
            X_array = np.array(sequences)
            y_array = np.array(targets)

            # More robust handling of NaN or infinite values
            if np.isnan(X_array).any() or np.isinf(X_array).any() or np.isnan(y_array).any() or np.isinf(y_array).any():
                logger.warning(
                    "NaN or infinite values detected in input data. Performing robust cleaning...")

                # First, identify rows with NaN or inf in either X or y
                X_has_invalid = np.any(
                    np.isnan(X_array) | np.isinf(X_array), axis=(1, 2))
                y_has_invalid = np.any(
                    np.isnan(y_array) | np.isinf(y_array), axis=1)
                valid_indices = ~(X_has_invalid | y_has_invalid)

                # If we have enough valid data, filter out invalid rows
                if np.sum(valid_indices) > 100:
                    logger.info(
                        f"Filtering out {np.sum(~valid_indices)} invalid rows, keeping {np.sum(valid_indices)} valid rows")
                    X_array = X_array[valid_indices]
                    y_array = y_array[valid_indices]
                else:
                    # If not enough valid data, replace NaN and inf with zeros/means
                    logger.warning(
                        "Not enough valid rows, replacing NaN values instead of filtering")
                    X_array = np.nan_to_num(
                        X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    y_array = np.nan_to_num(
                        y_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Handle outliers by clipping values to reasonable ranges
            # Clip to 5 standard deviations from mean
            X_flat = X_array.reshape(-1, X_array.shape[-1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mean, std = np.mean(col[~np.isnan(col)]), np.std(
                    col[~np.isnan(col)])
                if not np.isnan(mean) and not np.isnan(std) and std > 0:
                    X_flat[:, i] = np.clip(col, mean - 5*std, mean + 5*std)
            X_array = X_flat.reshape(X_array.shape)

            # Scale features
            scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_array.shape
            X_reshaped = X_array.reshape(n_samples * n_timesteps, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Store scaler in memory instead of saving to disk
            self.price_prediction_scaler = scaler
            logger.info("Stored price prediction scaler in memory")

            logger.info(
                f"Prepared price prediction data with {len(sequences)} sequences")

            return X_array, y_array

        except Exception as e:
            logger.error(
                f"Error preparing price prediction data: {str(e)}", exc_info=True)
            return np.array([]), np.array([])

    def select_features(self, features, target, task_type='classification'):
        """
        Select important features based on the feature selection method specified in config

        Parameters:
        -----------
        features : DataFrame
            Input features
        target : Series
            Target variable
        task_type : str
            Type of task: 'classification' or 'regression'

        Returns:
        --------
        DataFrame
            Selected features
        """
        try:
            logger.info(f"Performing feature selection for {task_type} task")

            if not self.config['feature_selection']['enabled']:
                return features

            if len(features) == 0 or len(target) == 0:
                logger.warning(
                    "Empty features or target, skipping feature selection")
                return features

            method = self.config['feature_selection']['method']

            # Create appropriate estimator based on task type
            if task_type == 'classification':
                if method == 'importance':
                    estimator = XGBClassifier(
                        n_estimators=100, learning_rate=0.05)
                elif method == 'rfe':
                    estimator = RandomForestClassifier(n_estimators=50)
                else:  # mutual_info
                    return self._select_using_mutual_info(features, target, task_type)
            else:  # regression
                if method == 'importance':
                    estimator = XGBRegressor(
                        n_estimators=100, learning_rate=0.05)
                elif method == 'rfe':
                    estimator = RandomForestRegressor(n_estimators=50)
                else:  # mutual_info
                    return self._select_using_mutual_info(features, target, task_type)

            # Apply selection method
            if method == 'importance':
                threshold = self.config['feature_selection']['threshold']
                selector = SelectFromModel(estimator, threshold=threshold)
                selector.fit(features, target)
                selected_features_mask = selector.get_support()
            elif method == 'rfe':
                n_features = min(
                    self.config['feature_selection']['n_features'], features.shape[1])
                selector = RFE(estimator, n_features_to_select=n_features)
                selector.fit(features, target)
                selected_features_mask = selector.get_support()

            # Get selected feature names
            selected_features = features.columns[selected_features_mask].tolist(
            )

            if not selected_features:
                logger.warning("No features selected, using all features")
                return features

            logger.info(
                f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
            return features[selected_features]

        except Exception as e:
            logger.error(
                f"Error in feature selection: {str(e)}", exc_info=True)
            return features

    def _select_using_mutual_info(self, features, target, task_type):
        """Helper method for mutual information based feature selection"""
        try:
            # Calculate mutual information
            if task_type == 'classification':
                mi_scores = mutual_info_classif(features, target)
            else:
                mi_scores = mutual_info_regression(features, target)

            # Create a ranking of features
            mi_ranking = pd.Series(mi_scores, index=features.columns)
            mi_ranking = mi_ranking.sort_values(ascending=False)

            # Select top features
            n_features = min(
                self.config['feature_selection']['n_features'], features.shape[1])
            selected_features = mi_ranking.index[:n_features].tolist()

            logger.info(
                f"Selected {len(selected_features)} features using mutual info: {', '.join(selected_features[:5])}...")
            return features[selected_features]

        except Exception as e:
            logger.error(
                f"Error in mutual info feature selection: {str(e)}", exc_info=True)
            return features

    def create_time_series_splits(self, features, target):
        """
        Create time series cross-validation splits

        Parameters:
        -----------
        features : DataFrame
            Input features
        target : Series
            Target variable

        Returns:
        --------
        list
            List of (train_idx, test_idx) tuples
        """
        try:
            n_splits = self.config['time_series_cv']['n_splits']
            embargo_size = self.config['time_series_cv']['embargo_size']

            # Total data size
            n_samples = len(features)

            # Calculate split sizes
            test_size = int(n_samples / (n_splits + 1))

            splits = []
            for i in range(n_splits):
                # Calculate indices
                test_start = (i + 1) * test_size
                test_end = test_start + test_size

                # Apply embargo - gap between train and test
                if embargo_size > 0:
                    train_end = max(0, test_start - embargo_size)
                else:
                    train_end = test_start

                # Create index arrays
                train_idx = list(range(0, train_end))
                test_idx = list(range(test_start, min(test_end, n_samples)))

                splits.append((train_idx, test_idx))

            return splits

        except Exception as e:
            logger.error(
                f"Error creating time series splits: {str(e)}", exc_info=True)
            # Return a simple 80/20 split as fallback
            n_samples = len(features)
            split_idx = int(n_samples * 0.8)
            return [(list(range(0, split_idx)), list(range(split_idx, n_samples)))]
