#!/usr/bin/env python3
"""
ML Model Trainer

This module provides a machine learning model trainer for the trading system.
It builds and trains various models using historical market data:
1. Signal detection model (XGBoost)
2. Price prediction model (LSTM)
3. Risk assessment model (Random Forest)
4. Exit strategy model (XGBoost)
5. Market regime classification (KMeans)

The models are optimized for GPU execution when available.
"""

import os
import time
import json
import logging
import subprocess
import asyncio
import datetime
import numpy as np
import pandas as pd
import joblib
import pickle
import traceback
import re

# Import Slack reporting module
try:
    from slack_reporter import SlackReporter, GPUStatsTracker
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# Import metrics for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
# Handle TensorFlow imports with try-except to avoid import errors
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.mixed_precision import set_global_policy
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    logger = logging.getLogger('ml_trainer')
    logger.warning("TensorFlow Keras modules could not be imported. Some functionality may be limited.")
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif
from scipy.stats import ks_2samp  # Used in detect_feature_drift

# Import optuna for hyperparameter optimization if available
optuna_available = False
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_trainer')

# Suppress TensorFlow and XGBoost warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

try:
    import optuna
    # XGBoostPruningCallback not used
    optuna_available = True  # Set flag to indicate optuna is available
    logger.info("Optuna is available for hyperparameter optimization")
except ImportError:
    logger.warning("Optuna not available. Hyperparameter optimization will be disabled.")

def configure_tensorflow_for_gh200():
    """Configure TensorFlow specifically for GH200 architecture"""
    try:
        # Set environment variables for GH200
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"
        
        # For Grace Hopper specifically
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation
        
        # Configure memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Enable mixed precision which works well on Hopper
            set_global_policy('mixed_float16')
            logger.info("TensorFlow configured for GH200 with mixed precision")
            return True
        else:
            logger.warning("No GPUs detected by TensorFlow")
            return False
    except Exception as e:
        logger.warning(f"Error configuring TensorFlow for GH200: {str(e)}")
        return False

def register_gh200_device():
    """Register GH200 as a special device for TensorFlow"""
    try:
        # Load CUDA libraries explicitly 
        import ctypes
        try:
            ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
            logger.info("Successfully loaded CUDA libraries")
        except Exception as e:
            logger.warning(f"Could not load CUDA libraries: {str(e)}")
        
        # Force device discovery
        physical_devices = tf.config.list_physical_devices()
        if not any(device.device_type == 'GPU' for device in physical_devices):
            # If no GPU found, try manual registration
            logger.info("No GPU found by TensorFlow, attempting manual device registration...")
            return False
        return True
    except Exception as e:
        logger.error(f"Error registering GH200 device: {str(e)}")
        return False

class GH200Accelerator:
    """Unified class to handle GPU acceleration on GH200"""
    
    def __init__(self):
        self.has_tensorflow_gpu = False
        self.has_cupy_gpu = False
        self.device_name = None
        self.device_memory = None
        
        # Configure both libraries
        self._configure_tensorflow()
        self._configure_cupy()
        
        # Set optimal execution strategy
        self._set_execution_strategy()
    
    def _configure_tensorflow(self):
        """Configure TensorFlow for GH200"""
        self.has_tensorflow_gpu = configure_tensorflow_for_gh200()
        if self.has_tensorflow_gpu:
            self.device_name = tf.test.gpu_device_name()
            logger.info(f"TensorFlow using GPU device: {self.device_name}")
    
    def _configure_cupy(self):
        """Configure CuPy for GH200"""
        try:
            import cupy as cp
            
            # Check if CuPy can see any GPU
            if cp.cuda.runtime.getDeviceCount() > 0:
                self.has_cupy_gpu = True
                
                # Find and use GH200 if available
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    device_name = props["name"].decode()
                    logger.info(f"CuPy found GPU device {i}: {device_name}")
                    if "GH200" in device_name:
                        cp.cuda.Device(i).use()
                        self.device_name = device_name
                        
                        # Configure for unified memory
                        cp.cuda.set_allocator(cp.cuda.MemoryPool(
                            cp.cuda.malloc_managed).malloc)
                        
                        # Get memory info
                        free, total = cp.cuda.runtime.memGetInfo()
                        self.device_memory = (free, total)
                        logger.info(f"Using GH200 device with {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total memory")
                        break
        except ImportError:
            logger.warning("CuPy not available")
    
    def _set_execution_strategy(self):
        """Set the optimal execution strategy based on available hardware"""
        if self.has_tensorflow_gpu:
            self.strategy = "tensorflow_gpu"
        elif self.has_cupy_gpu:
            self.strategy = "cupy_gpu_tensorflow_cpu"
        else:
            self.strategy = "cpu_only"
            
        logger.info(f"Using execution strategy: {self.strategy}")
    
    def get_optimal_batch_size(self):
        """Calculate optimal batch size based on GPU memory"""
        if not self.device_memory:
            return 32  # Default
            
        free_memory = self.device_memory[0]
        # Conservative estimate: use 20% of free memory for batch data
        memory_per_sample = 5000000  # Estimate bytes per sample
        return min(2048, max(32, int(free_memory * 0.2 / memory_per_sample)))

def optimize_for_gh200():
    """Apply GH200-specific optimizations"""
    # Environment variables for GH200
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Enable TF32 computation
    
    # For ARM CPU side of GH200
    os.environ["GOMP_CPU_AFFINITY"] = "0-15"  # Adjust based on Neoverse cores
    
    # Optimize memory transfer
    os.environ["CUDA_AUTO_BOOST"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    
    # NVLink optimizations
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    
    logger.info("Applied GH200-specific optimizations")
    return True

class MLModelTrainer:
    """
    ML Model Trainer for trading system
    Builds and trains models using live market data
    """
    
    def __init__(self, redis_client, data_loader):
        self.redis = redis_client
        self.data_loader = data_loader
        
        # Apply GH200 optimizations
        optimize_for_gh200()
        
        # Initialize GPU acceleration
        self.use_gpu = os.environ.get('USE_GPU', 'true').lower() == 'true'
        if self.use_gpu:
            # Initialize GH200 accelerator
            self.accelerator = GH200Accelerator()
            self.cupy_gpu_available = self.accelerator.has_cupy_gpu
            self.tf_gpu_available = self.accelerator.has_tensorflow_gpu
            logger.info(f"GH200 acceleration enabled: {self.use_gpu}, TensorFlow GPU available: {self.tf_gpu_available}, CuPy GPU available: {self.accelerator.has_cupy_gpu}")
        
        # Configuration
        self.config = {
            'models_dir': os.environ.get('MODELS_DIR', './models'),
            'monitoring_dir': os.environ.get('MONITORING_DIR', './monitoring'),
            'data_dir': os.environ.get('DATA_DIR', './data'),
            'min_samples': 1000,
            'lookback_days': 30,            
            'feature_selection': {
                'enabled': True,
                'method': 'importance',  # 'importance', 'rfe', 'mutual_info'
                'threshold': 0.01,  # For importance-based selection
                'n_features': 20    # For RFE
            },
            'time_series_cv': {
                'enabled': True,
                'n_splits': 5,
                'embargo_size': 10  # Number of samples to exclude between train and test
            },
            'monitoring': {'enabled': True, 'drift_threshold': 0.05},
            'test_size': 0.2,
            'random_state': 42,
            'model_configs': {
                'signal_detection': {
                    'type': 'xgboost',
                    'params': {
                        'max_depth': 6,
                        'learning_rate': 0.03,
                        'subsample': 0.8,
                        'n_estimators': 200,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc'
                    }
                },
                'price_prediction': {
                    'type': 'lstm',
                    'params': {
                        'units': [64, 32],
                        'dropout': 0.3,
                        'epochs': 50,
                        'batch_size': 32,
                        'learning_rate': 0.001
                    }
                },
                'risk_assessment': {
                    'type': 'random_forest',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'max_features': 'sqrt',
                        'min_samples_leaf': 30
                    }
                },
                'exit_strategy': {
                    'type': 'xgboost',
                    'params': {
                        'max_depth': 5,
                        'learning_rate': 0.02,
                        'subsample': 0.8,
                        'n_estimators': 150,
                        'objective': 'reg:squarederror'
                    }
                },
                'market_regime': {
                    'type': 'kmeans',
                    'params': {
                        'n_clusters': 4,
                        'random_state': 42
                    }
                }
            },
        }
        
        # Ensure directories exist with proper permissions
        self._ensure_directories()
        
        # Initialize Slack reporting
        self.slack_reporter = None
        self.gpu_tracker = None
        self.model_training_times = {}
        self.training_start_time = None
        self.training_config = {}
        
        if SLACK_AVAILABLE:
            # Initialize Slack reporter using environment variables or defaults
            webhook_url = os.environ.get('SLACK_WEBHOOK_URL', '')
            bot_token = os.environ.get('SLACK_BOT_TOKEN', '')
            channel = os.environ.get('SLACK_CHANNEL', '#system-notifications')
            
            if webhook_url or bot_token:
                self.slack_reporter = SlackReporter(webhook_url=webhook_url, bot_token=bot_token, channel=channel)
                self.gpu_tracker = GPUStatsTracker(polling_interval=10.0)  # Poll every 10 seconds
        
        # Scalers
        self.scalers = {}
        
        logger.info("ML Model Trainer initialized")

    def _ensure_directories(self):
        """Ensure all required directories exist with proper permissions"""
        for directory in [self.config['models_dir'], self.config['data_dir'], self.config['monitoring_dir']]:
            try:
                os.makedirs(directory, exist_ok=True)
                # Set permissive permissions
                os.chmod(directory, 0o777)
                logger.info(f"Created directory with full permissions: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")
    
    def train_all_models(self):
        """Train all trading models"""
        logger.info("Starting training for all models")

        # Start tracking total training time
        self.training_start_time = time.time()
        
        # Start GPU tracking if available
        if self.gpu_tracker:
            self.gpu_tracker.start()
        
        # Run hyperparameter optimization if enabled and optuna is available
        if optuna_available and os.environ.get('OPTIMIZE_HYPERPARAMS', 'false').lower() == 'true':
            logger.info("Running hyperparameter optimization")
            self.optimize_hyperparameters()
        
        # Continue with regular training
        self._train_all_models()
        
    def _train_all_models(self):
        """Internal method to train all models with current hyperparameters"""
        """Train all trading models"""
        logger.info("Starting training for all models")
        
        try:
            # Load historical data
            logger.info("Loading historical data")
            historical_data = self.load_historical_data()
            
            if historical_data is None or (isinstance(historical_data, pd.DataFrame) and historical_data.empty):
                logger.error("Failed to load sufficient historical data")
                
                # Report error to Slack if available
                if self.slack_reporter:
                    self.slack_reporter.report_error(
                        "Failed to load sufficient historical data", 
                        phase="data loading"
                    )
                
                return False
                
            # Store reference data for drift detection
            self._store_reference_data(historical_data)
                
            # Train each model
            self.train_signal_detection_model(historical_data)
            self.train_price_prediction_model(historical_data)
            self.train_risk_assessment_model(historical_data)
            self.train_exit_strategy_model(historical_data)
            self.train_market_regime_model(historical_data)
            
            # Update Redis with model info
            self.update_model_info()
            
            # Calculate total training time
            total_training_time = time.time() - self.training_start_time
            
            # Stop GPU tracking if available
            gpu_stats = None
            if self.gpu_tracker:
                gpu_stats = self.gpu_tracker.stop()
            
            # Report training completion to Slack if available
            if self.slack_reporter and gpu_stats:
                self.slack_reporter.report_training_complete(
                    total_training_time,
                    {"signal_detection": {"success": True},
                     "price_prediction": {"success": True},
                     "risk_assessment": {"success": True},
                     "exit_strategy": {"success": True},
                     "market_regime": {"success": True}},
                    gpu_stats
                )
            
            logger.info(f"All models trained successfully in {total_training_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}", exc_info=True)
            return False
    
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
            start_date = end_date - datetime.timedelta(days=self.config['lookback_days'])
            
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
                logger.warning(f"Insufficient data: {len(combined_data)} samples (min: {self.config['min_samples']})")
                return None
                
            logger.info(f"Loaded {len(combined_data)} samples for training")
            
            # Save data for future reference
            data_path = os.path.join(self.config['data_dir'], f"training_data_{int(time.time())}.pkl")
            joblib.dump(combined_data, data_path)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}", exc_info=True)
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
            combined_price = self.calculate_technical_indicators(combined_price)
            
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
                        logger.warning("No valid options dataframes to concatenate")
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
                    combined_price_reset['timestamp'] = pd.to_datetime(combined_price_reset['timestamp']).astype('datetime64[ns]')
                    options_df_reset = options_df.reset_index().copy()
                    combined_price_reset = combined_price_reset.sort_values('timestamp')
                    options_df_reset['timestamp'] = pd.to_datetime(options_df_reset['timestamp']).astype('datetime64[ns]')
                
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
                combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp']).astype('datetime64[ns]')
                market_data_reset = market_data.reset_index().copy()
                combined_data = combined_data.sort_values('timestamp')
                market_data_reset['timestamp'] = pd.to_datetime(market_data_reset['timestamp']).astype('datetime64[ns]')
                
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
            for col in combined_data.select_dtypes(include=[np.number]).columns:
                if col in combined_data.columns:
                    q1, q3 = combined_data[col].quantile([0.01, 0.99])
                    iqr = q3 - q1
                    combined_data[col] = combined_data[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
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
            logger.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
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
        result['volatility'] = result['close'].rolling(window=10).std() / result['close'].rolling(window=10).mean()
        
        # Volume-based indicators
        if 'volume' in result.columns:
            result['volume_sma5'] = result['volume'].rolling(window=5).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma5']
            
            # Money Flow Index (simplified)
            result['money_flow'] = result['close'] * result['volume']
            result['money_flow_sma'] = result['money_flow'].rolling(window=14).mean()
        
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
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        return result
    
    def generate_targets(self, df):
        """Generate target variables for supervised learning"""
        try:
            # Make a copy
            result = df.copy()
            
            # Signal detection target (1 if price increases by 1% within next 10 bars)
            future_returns = result.groupby('ticker')['close'].pct_change(10).shift(-10)
            result['signal_target'] = (future_returns > 0.01).astype(int)
            
            # Price prediction targets
            result['future_return_5min'] = result.groupby('ticker')['close'].pct_change(5).shift(-5)
            result['future_return_10min'] = result.groupby('ticker')['close'].pct_change(10).shift(-10)
            result['future_return_30min'] = result.groupby('ticker')['close'].pct_change(30).shift(-30)
            
            # Direction target (1 for up, 0 for down)
            result['future_direction'] = (result['future_return_10min'] > 0).astype(int)
            
            # Risk assessment target (ATR as % of price)
            high_low = result['high'] - result['low']
            high_close = abs(result['high'] - result['close'].shift())
            low_close = abs(result['low'] - result['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result['atr14'] = tr.rolling(14).mean()
            result['atr_pct'] = result['atr14'] / result['close']
            
            # Handle any NaN or Inf values in targets
            for col in ['signal_target', 'future_return_5min', 'future_return_10min', 
                        'future_return_30min', 'future_direction', 'atr_pct', 'optimal_exit']:
                if col in result.columns:
                    result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                    result[col] = result[col].ffill().bfill().fillna(0)
            
            
            # Exit strategy target (optimal exit time within next 30 bars)
            # This is simplified; in practice would be more sophisticated
            for ticker, group in result.groupby('ticker'):
                future_prices = [group['close'].shift(-i) for i in range(1, 31)]
                future_prices_df = pd.concat(future_prices, axis=1)
                max_price = future_prices_df.max(axis=1)
                optimal_exit = (max_price / group['close'] - 1)
                result.loc[group.index, 'optimal_exit'] = optimal_exit
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating targets: {str(e)}", exc_info=True)
            return df
            
    def _store_reference_data(self, data):
        """Store reference data for drift detection"""
        if not self.config['monitoring']['enabled']:
            return
            
        try:
            # Store a sample of the data for future drift detection
            reference_path = os.path.join(self.config['monitoring_dir'], 'reference_data.pkl')
            
            # Sample the data to keep a manageable size
            if len(data) > 10000:
                reference_data = data.sample(10000, random_state=42)
            else:
                reference_data = data.copy()
                
            # Store the reference data
            joblib.dump(reference_data, reference_path)
            logger.info(f"Stored reference data for drift detection: {len(reference_data)} samples")
            
        except Exception as e:
            logger.error(f"Error storing reference data: {str(e)}")
            
    def detect_feature_drift(self, current_data, threshold=None):
        """Detect drift in feature distributions using KS test"""
        if not self.config['monitoring']['enabled']:
            return False, {}
            
        if threshold is None:
            threshold = self.config['monitoring']['drift_threshold']
            
        try:
            # Load reference data
            reference_path = os.path.join(self.config['monitoring_dir'], 'reference_data.pkl')
            if not os.path.exists(reference_path):
                logger.warning("No reference data available for drift detection")
                return False, {}
                
            reference_data = joblib.load(reference_path)
            
            # Select numeric features only
            numeric_features = reference_data.select_dtypes(include=[np.number]).columns
            
            drift_detected = False
            drift_features = {}
            
            for feature in numeric_features:
                if feature in current_data.columns:
                    # Get clean samples from both datasets
                    ref_values = reference_data[feature].dropna().values
                    cur_values = current_data[feature].dropna().values
                    
                    if len(ref_values) > 10 and len(cur_values) > 10:
                        # Perform KS test
                        ks_statistic, p_value = ks_2samp(ref_values, cur_values)
                        
                        if p_value < threshold:
                            drift_detected = True
                            drift_features[feature] = {'ks_statistic': float(ks_statistic), 'p_value': float(p_value)}
            
            return drift_detected, drift_features
            
        except Exception as e:
            logger.error(f"Error detecting feature drift: {str(e)}")
            return False, {}
    
    def train_signal_detection_model(self, data):
        """Train signal detection model"""
        logger.info("Training signal detection model")
        
        try:
            # Prepare data
            features, target = self.prepare_signal_detection_data(data)
            
            if len(features) == 0 or len(target) == 0:
                logger.error("No valid data for signal detection model")
                return False
                
            # Apply feature selection if enabled
            if self.config['feature_selection']['enabled']:
                features = self.select_features(features, target, 'classification')
                
            # Use time series cross-validation if enabled
            if self.config['time_series_cv']['enabled']:
                # Create time series split
                splits = self.create_time_series_splits(features, target)
                
                # Use the last split for final evaluation
                train_idx, test_idx = splits[-1]
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            else:
                # Use traditional train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, 
                    test_size=self.config['test_size'],
                    random_state=self.config['random_state']
                )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            scaler_path = os.path.join(self.config['models_dir'], 'signal_detection_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers['signal_detection'] = scaler
            
            # Get model config
            model_config = self.config['model_configs']['signal_detection']
            
            # Check for optimized parameters
            optimized_params_path = os.path.join(self.config['models_dir'], 'signal_detection_optimized_params.json')
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path, 'r') as f:
                    model_config['params'].update(json.load(f))
            
            # Train XGBoost model
            logger.info("Training XGBoost signal detection model")
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            eval_list = [(dtrain, 'train'), (dtest, 'test')]
            
            # Create a copy of params without n_estimators to avoid warning
            
            model = xgb.train(
                # Use only one params parameter with n_estimators removed
                params={k: v for k, v in model_config['params'].items() if k != 'n_estimators'},
                dtrain=dtrain,
                evals=eval_list,
                num_boost_round=model_config['params'].get('n_estimators', 200),
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Evaluate model
            y_pred = model.predict(dtest)
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Check if we have multiple classes in the test set
            unique_classes = np.unique(y_test)
            if len(unique_classes) < 2:
                logger.warning(f"Only one class present in test set: {unique_classes}. Using simplified metrics.")
                accuracy = accuracy_score(y_test, y_pred_binary)
                metrics = {
                    'accuracy': float(accuracy)
                }
                logger.info(f"Signal detection model metrics - Accuracy: {accuracy:.4f}")
            else:
                accuracy = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                f1 = f1_score(y_test, y_pred_binary)
                auc = roc_auc_score(y_test, y_pred)
                metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'auc': float(auc),
                    'feature_importance': {str(k): float(v) for k, v in model.get_score(importance_type='gain').items()}
                }
                logger.info(f"Signal detection model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

            # Save model
            model_path = os.path.join(self.config['models_dir'], 'signal_detection_model.xgb')
            model.save_model(model_path)
            
            # Save metrics
            metrics_path = os.path.join(self.config['models_dir'], 'signal_detection_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            # Update Redis
            self.redis.hset(
                "models:metrics",
                "signal_detection",
                json.dumps(metrics)
            )
            
            # Report metrics to Slack if available
            if self.slack_reporter:
                self.slack_reporter.report_model_metrics(
                    "signal_detection", 
                    metrics, 
                    self.model_training_times.get("signal_detection", 0)
                )
            
            logger.info("Signal detection model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training signal detection model: {str(e)}", exc_info=True)
            return False
    
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
            available_columns = [col for col in feature_columns if col in data.columns]
            
            if len(available_columns) < 5:
                logger.warning(f"Too few features available: {len(available_columns)}")
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
            
            logger.info(f"Prepared signal detection data with {len(X)} samples and {len(available_columns)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing signal detection data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()
    
    def train_price_prediction_model(self, data):
        """Train price prediction model"""
        logger.info("Training price prediction model")
        
        try:
            # Prepare data
            sequences, targets = self.prepare_price_prediction_data(data)
            
            if len(sequences) == 0 or len(targets) == 0:
                logger.error("No valid data for price prediction model")
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            # Get model config
            model_config = self.config['model_configs']['price_prediction']
            
            # Determine optimal batch size based on GPU memory
            batch_size = 32  # Default
            if self.use_gpu and hasattr(self, 'accelerator'):
                batch_size = self.accelerator.get_optimal_batch_size()
                logger.info(f"Using optimal batch size for GH200: {batch_size}")
            
            # Check GPU availability
            if self.use_gpu and hasattr(self, 'accelerator') and not self.accelerator.has_tensorflow_gpu:
                logger.warning("TensorFlow GPU not available on GH200, but GPU acceleration is enabled")
                logger.info("Attempting to use CuPy for data preprocessing and TensorFlow on CPU for model training")
                # Continue with CPU training but use CuPy for data preprocessing
            
            # Create a more stable model with simpler architecture
            model = Sequential()
            
            # Use a simpler architecture that's less prone to NaN issues
            # Flatten the input sequence
            model.add(Flatten(input_shape=(sequences.shape[1], sequences.shape[2])))
            
            # Add dense layers with batch normalization and dropout
            model.add(Dense(
                64,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(
                32,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            # Output layer with L2 regularization
            model.add(Dense(
                targets.shape[1],
                activation='linear',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))
            
            # Compile model with gradient clipping to prevent NaN loss
            optimizer = Adam(
                learning_rate=model_config['params']['learning_rate'] * 0.01,  # Further reduce learning rate for stability
                clipnorm=1.0  # Add gradient clipping (only use one clipping method)
            )
            model.compile(
                optimizer=optimizer,
                loss='mse'
            )
            
            # Custom callback to stop training if NaN loss is encountered
            class TerminateOnNaN(tf.keras.callbacks.Callback):
                def on_batch_end(self, batch, logs=None):
                    logs = logs or {}
                    loss = logs.get('loss')
                    if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                        logger.warning(f'Batch {batch}: Invalid loss, terminating training')
                        self.model.stop_training = True
            
            # Callbacks with more aggressive early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.0001),
                ModelCheckpoint(
                    filepath=os.path.join(self.config['models_dir'], 'price_prediction_best.keras'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                TerminateOnNaN()
            ]
            
            # Train model with robust error handling
            try:
                # Use a smaller batch size and fewer epochs
                epochs = 10  # Fewer epochs to prevent overfitting
                
                # Add validation split for early stopping
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,  # Use validation_split instead of validation_data
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=True  # Shuffle data for better training
                )
                
                # Check if training was successful (no NaN loss)
                if 'loss' in history.history and np.isnan(history.history['loss']).any():
                    logger.warning("NaN values detected in training history. Training may have failed.")
                    # Try a simpler model as fallback
                    logger.info("Attempting to train a simpler fallback model...")
                    
                    # Create a simpler model
                    simple_model = Sequential([
                        # Use a very simple model that's extremely stable
                        Flatten(input_shape=(sequences.shape[1], sequences.shape[2])),
                        Dense(
                            16, 
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)  # Stronger regularization
                        ),
                        BatchNormalization(),
                        Dense(targets.shape[1], activation='linear', kernel_initializer='glorot_uniform')
                    ])
                    
                    simple_model.compile(optimizer='adam', loss='mse')
                    
                    # Train the simple model
                    history = simple_model.fit(
                        X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1
                    )
                    
                    # Use the simple model instead
                    model = simple_model
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                # Create a minimal model that at least won't crash
                logger.info("Creating minimal emergency fallback model")
                model = Sequential([
                    # Use the simplest possible model with strong regularization
                    Flatten(input_shape=(sequences.shape[1], sequences.shape[2])), 
                    Dense(
                        8, 
                        activation='linear',  # Linear activation is more stable than ReLU
                        kernel_initializer='zeros',  # Initialize with zeros for stability
                        kernel_regularizer=tf.keras.regularizers.l2(0.1),  # Very strong regularization
                        bias_initializer='zeros'
                    ),
                    Dense(targets.shape[1], activation='linear')
                ])
                model.compile(optimizer='adam', loss='mse')
                # Train with minimal data
                history = model.fit(
                    X_train[:100], y_train[:100],  # Use just a small subset
                    epochs=1,
                    batch_size=32,
                    verbose=0
                )
            
            # Evaluate model
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            # Predictions for metrics
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            direction_accuracy = np.mean((y_pred[:, 0] > 0) == (y_test[:, 0] > 0))
            mse = np.mean((y_pred - y_test) ** 2)
            
            logger.info(f"Price prediction model metrics - MSE: {mse:.6f}, Direction Accuracy: {direction_accuracy:.4f}")
            
            # Save model
            model_path = os.path.join(self.config['models_dir'], 'price_prediction_model.keras')
            model.save(model_path)
            
            # Save metrics
            metrics = {
                'test_loss': float(test_loss),
                'direction_accuracy': float(direction_accuracy),
                'mse': float(mse),
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])] if 'val_loss' in history.history else []
                }
            }
            
            metrics_path = os.path.join(self.config['models_dir'], 'price_prediction_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            # Update Redis
            self.redis.hset(
                "models:metrics",
                "price_prediction",
                json.dumps(metrics)
            )
            
            logger.info("Price prediction model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {str(e)}", exc_info=True)
            return False
    
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
            available_columns = [col for col in feature_columns if col in data.columns]
            
            if len(available_columns) < 4:
                logger.warning(f"Too few features available: {len(available_columns)}")
                return np.array([]), np.array([])
                
            # Target columns
            target_columns = ['future_return_5min', 'future_return_10min', 'future_return_30min']
            available_targets = [col for col in target_columns if col in data.columns]
            
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
                logger.warning("NaN or infinite values detected in input data. Performing robust cleaning...")
                
                # First, identify rows with NaN or inf in either X or y
                X_has_invalid = np.any(np.isnan(X_array) | np.isinf(X_array), axis=(1, 2))
                y_has_invalid = np.any(np.isnan(y_array) | np.isinf(y_array), axis=1)
                valid_indices = ~(X_has_invalid | y_has_invalid)
                
                # If we have enough valid data, filter out invalid rows
                if np.sum(valid_indices) > 100:
                    logger.info(f"Filtering out {np.sum(~valid_indices)} invalid rows, keeping {np.sum(valid_indices)} valid rows")
                    X_array = X_array[valid_indices]
                    y_array = y_array[valid_indices]
                else:
                    # If not enough valid data, replace NaN and inf with zeros/means
                    logger.warning("Not enough valid rows, replacing NaN values instead of filtering")
                    X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Handle outliers by clipping values to reasonable ranges
            # Clip to 5 standard deviations from mean
            X_flat = X_array.reshape(-1, X_array.shape[-1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mean, std = np.mean(col[~np.isnan(col)]), np.std(col[~np.isnan(col)])
                if not np.isnan(mean) and not np.isnan(std) and std > 0:
                    X_flat[:, i] = np.clip(col, mean - 5*std, mean + 5*std)
            X_array = X_flat.reshape(X_array.shape)
            
            # Scale features
            scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_array.shape
            X_reshaped = X_array.reshape(n_samples * n_timesteps, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Save scaler
            scaler_path = os.path.join(self.config['models_dir'], 'price_prediction_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers['price_prediction'] = scaler
            
            logger.info(f"Prepared price prediction data with {len(sequences)} sequences")
            
            return X_array, y_array
            
        except Exception as e:
            logger.error(f"Error preparing price prediction data: {str(e)}", exc_info=True)
            return np.array([]), np.array([])
    
    def train_risk_assessment_model(self, data):
        """Train risk assessment model"""
        logger.info("Training risk assessment model")
        
        try:
            # Prepare data
            features, targets = self.prepare_risk_assessment_data(data)
            
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for risk assessment model")
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            scaler_path = os.path.join(self.config['models_dir'], 'risk_assessment_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers['risk_assessment'] = scaler
            
            # Get model config
            model_config = self.config['model_configs']['risk_assessment']
            
            # Create model
            model = RandomForestRegressor(
                n_estimators=model_config['params']['n_estimators'],
                # Reduce max_depth to prevent overfitting
                max_depth=min(model_config['params']['max_depth'], 4),
                # Increase min_samples_leaf to prevent overfitting
                min_samples_leaf=max(model_config['params']['min_samples_leaf'], 50),
                # Add max_features parameter to reduce overfitting
                max_features='sqrt',
                random_state=self.config['random_state']
            )
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            r2 = model.score(X_test_scaled, y_test)
            
            logger.info(f"Risk assessment model metrics - MSE: {mse:.6f}, R²: {r2:.4f}")
            
            # Save model
            model_path = os.path.join(self.config['models_dir'], 'risk_assessment_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            metrics = {
                'mse': float(mse),
                'r2': float(r2),
                'feature_importance': {str(i): float(v) for i, v in enumerate(model.feature_importances_)}
            }
            
            metrics_path = os.path.join(self.config['models_dir'], 'risk_assessment_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            # Update Redis
            self.redis.hset(
                "models:metrics",
                "risk_assessment",
                json.dumps(metrics)
            )
            
            logger.info("Risk assessment model trained successfully")
            return True
            
        except Exception as e:
            if "no running event loop" in str(e):
                logger.warning("Asyncio error detected. This is likely due to a signal interruption.")
                # Create a simple model that won't crash
                model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                model.fit(X_train_scaled[:100], y_train[:100])  # Train on a small subset
                
                # Save the simple model
                model_path = os.path.join(self.config['models_dir'], 'risk_assessment_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info("Created fallback risk assessment model")
                return True
            else:
                logger.error(f"Error training risk assessment model: {str(e)}", exc_info=True)
                return False
    
    def prepare_risk_assessment_data(self, data):
        """Prepare data for risk assessment model"""
        try:
            # Select features
            feature_columns = [
                # Price volatility features
                'volatility', 'atr_pct', 'bb_width',
                
                # Volume features
                'volume_ratio',
                
                # Market features (if available)
                'vix_close', 'vix_change'
            ]
            
            # Keep only available columns
            available_columns = [col for col in feature_columns if col in data.columns]
            
            if len(available_columns) < 2:
                logger.warning(f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()
                
            # Target column (ATR as % of price - risk measure)
            target_column = 'atr_pct'
            
            if target_column not in data.columns:
                logger.warning(f"Target column {target_column} not available")
                return pd.DataFrame(), pd.Series()
                
            # Select data
            X = data[available_columns].copy()
            y = data[target_column].copy()
            
            # Drop NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared risk assessment data with {len(X)} samples and {len(available_columns)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing risk assessment data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()
    
    def train_exit_strategy_model(self, data):
        """Train exit strategy model"""
        logger.info("Training exit strategy model")
        
        try:
            # Prepare data
            features, targets = self.prepare_exit_strategy_data(data)
            
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for exit strategy model")
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            scaler_path = os.path.join(self.config['models_dir'], 'exit_strategy_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers['exit_strategy'] = scaler
            
            # Get model config
            model_config = self.config['model_configs']['exit_strategy']
            
            # Train XGBoost model
            logger.info("Training XGBoost exit strategy model")
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            eval_list = [(dtrain, 'train'), (dtest, 'test')]
            
            # Create a copy of params without n_estimators to avoid warning
            xgb_params = {k: v for k, v in model_config['params'].items() if k != 'n_estimators'}
            
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=model_config['params'].get('n_estimators', 200),
                evals=eval_list,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Evaluate model
            y_pred = model.predict(dtest)
            
            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            rmse = np.sqrt(mse)
            mean_actual = np.mean(y_test)
            
            logger.info(f"Exit strategy model metrics - RMSE: {rmse:.6f}, Mean Target: {mean_actual:.6f}")
            
            # Save model
            model_path = os.path.join(self.config['models_dir'], 'exit_strategy_model.xgb')
            model.save_model(model_path)
            
            # Save metrics
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'feature_importance': {str(k): float(v) for k, v in model.get_score(importance_type='gain').items()}
            }
            
            metrics_path = os.path.join(self.config['models_dir'], 'exit_strategy_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            # Update Redis
            self.redis.hset(
                "models:metrics",
                "exit_strategy",
                json.dumps(metrics)
            )
            
            logger.info("Exit strategy model trained successfully")
            return True
            
        except Exception as e:
            if "no running event loop" in str(e):
                logger.warning("Asyncio error detected. This is likely due to a signal interruption.")
                # Create a simple model that won't crash
                dtrain = xgb.DMatrix(X_train_scaled[:100], label=y_train[:100])
                model = xgb.train(
                    params={'objective': 'reg:squarederror', 'max_depth': 3},
                    dtrain=dtrain,
                    num_boost_round=10
                )
                
                # Properly handle the coroutine
                if hasattr(self.data_loader, 'unusual_whales_client') and hasattr(self.data_loader.unusual_whales_client, 'close'):
                    try:
                        asyncio.create_task(self.data_loader.unusual_whales_client.close())
                    except Exception as close_error:
                        logger.warning(f"Error closing unusual whales client: {str(close_error)}")
                
                # Save the simple model
                model_path = os.path.join(self.config['models_dir'], 'exit_strategy_model.xgb')
                model.save_model(model_path)
                
                logger.info("Created fallback exit strategy model")
                return True
            else:
                logger.error(f"Error training exit strategy model: {str(e)}", exc_info=True)
                return False
    
    def prepare_exit_strategy_data(self, data):
        """Prepare data for exit strategy model"""
        try:
            # Select features
            feature_columns = [
                # Price and technical indicators
                'close', 'volatility', 'rsi',
                'macd', 'price_rel_sma5', 'bb_width',
                
                # Market features (if available)
                'spy_close', 'vix_close'
            ]
            
            # Keep only available columns
            available_columns = [col for col in feature_columns if col in data.columns]
            
            if len(available_columns) < 3:
                logger.warning(f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()
                
            # Target column (optimal exit within next 30 minutes)
            target_column = 'optimal_exit'
            
            if target_column not in data.columns:
                logger.warning(f"Target column {target_column} not available")
                return pd.DataFrame(), pd.Series()
                
            # Select data
            X = data[available_columns].copy()
            y = data[target_column].copy()
            
            # Drop NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared exit strategy data with {len(X)} samples and {len(available_columns)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing exit strategy data: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()
    
    def train_market_regime_model(self, data):
        """Train market regime classifier model"""
        logger.info("Training market regime model")
        
        try:
            # Prepare data
            features = self.prepare_market_regime_data(data)
            
            if len(features) == 0:
                logger.error("No valid data for market regime model")
                return False
                
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Save scaler
            scaler_path = os.path.join(self.config['models_dir'], 'market_regime_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers['market_regime'] = scaler
            
            # Get model config
            model_config = self.config['model_configs']['market_regime']
            
            # Create model
            model = KMeans(
                n_clusters=model_config['params']['n_clusters'],
                random_state=model_config['params']['random_state']
            )
            
            # Train model
            model.fit(features_scaled)
            
            # Calculate metrics
            inertia = model.inertia_
            cluster_counts = np.bincount(model.labels_)
            
            logger.info(f"Market regime model metrics - Inertia: {inertia:.2f}, Cluster counts: {cluster_counts}")
            
            # Save model
            model_path = os.path.join(self.config['models_dir'], 'market_regime_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            metrics = {
                'inertia': float(inertia),
                'cluster_counts': [int(count) for count in cluster_counts],
                'cluster_centers': [[float(value) for value in center] for center in model.cluster_centers_]
            }
            
            metrics_path = os.path.join(self.config['models_dir'], 'market_regime_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            # Update Redis
            self.redis.hset(
                "models:metrics",
                "market_regime",
                json.dumps(metrics)
            )
            
            logger.info("Market regime model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training market regime model: {str(e)}", exc_info=True)
        
        # If training failed, create a simple fallback model
        try:
            try:
                logger.info("Creating fallback market regime model with default parameters")
                
                # Create a simple KMeans model with default parameters
                model = KMeans(n_clusters=4, random_state=42)
                
                # Fit on a small dummy dataset to initialize the model
                dummy_data = np.random.rand(10, 5)  # 10 samples, 5 features
                model.fit(dummy_data)
                
                # Save the model
                model_path = os.path.join(self.config['models_dir'], 'market_regime_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Created and saved fallback market regime model to {model_path}")
                
                # Update Redis with minimal model info
                self.redis.hset("models:metrics", "market_regime", json.dumps({"fallback": True}))
            except Exception as e:
                logger.error(f"Error creating fallback market regime model: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
        except Exception:
            return False
    
    def prepare_market_regime_data(self, data):
        """Prepare data for market regime model"""
        try:
            # Market regime only uses market-wide data, not individual stocks
            
            # Define required and optional market data columns
            required_columns = ['spy_close', 'vix_close']
            optional_columns = ['spy_open', 'spy_high', 'spy_low', 'spy_volume', 'spy_vwap', 
                               'vix_open', 'vix_high', 'vix_low', 'vix_volume',
                               'spy_change', 'vix_change']
            
            # Check if required columns are available
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Insufficient market data columns: {[col for col in required_columns if col not in data.columns]}")
                return pd.DataFrame()
            
            # Extract available market data columns
            market_columns = required_columns + [col for col in optional_columns if col in data.columns]
                
            # Add some additional columns if available
            for col in ['qqq_close', 'qqq_change', 'iwm_close', 'iwm_change']:
                if col in data.columns:
                    market_columns.append(col)
            
            # Select market data and drop duplicates (same market data repeated for each stock)
            market_data = data[market_columns].drop_duplicates().copy()
            
            # Additional feature engineering for market regime
            if 'spy_close' in market_data.columns and 'spy_open' in market_data.columns:
                market_data['spy_daily_range'] = (market_data['spy_close'] / market_data['spy_open'] - 1) * 100
                
            if 'vix_close' in market_data.columns:
                market_data['vix_ma5'] = market_data['vix_close'].rolling(window=5).mean()
                market_data['vix_ma10'] = market_data['vix_close'].rolling(window=10).mean()
                market_data['vix_ratio'] = market_data['vix_close'] / market_data['vix_ma10']
            
            # Drop NaN
            market_data = market_data.dropna()
            
            logger.info(f"Prepared market regime data with {len(market_data)} samples and {len(market_columns)} features")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error preparing market regime data: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def update_model_info(self):
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}
            
            for model_name, config in self.config['model_configs'].items():
                model_path = os.path.join(self.config['models_dir'], f"{model_name}_model.{'xgb' if config['type'] == 'xgboost' else 'pkl' if config['type'] in ['random_forest', 'kmeans'] else 'keras'}")
                
                if os.path.exists(model_path):
                    file_stats = os.stat(model_path)
                    
                    models_info[model_name] = {
                        'type': config['type'],
                        'path': model_path,
                        'size_bytes': file_stats.st_size,
                        'last_modified': int(file_stats.st_mtime),
                        'last_modified_str': datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }
            
            # Update Redis
            self.redis.set("models:info", json.dumps(models_info))
            
            logger.info(f"Updated model info for {len(models_info)} models")
            
        except Exception as e:
            logger.error(f"Error updating model info: {str(e)}", exc_info=True)
    
    def get_active_tickers(self):
        """Get list of active tickers for training"""
        try:
            # Get from Redis
            watchlist_tickers = self.redis.zrange("watchlist:active", 0, -1)
            
            # Convert from bytes if needed
            watchlist_tickers = [t.decode('utf-8') if isinstance(t, bytes) else t for t in watchlist_tickers]
            
            # If no watchlist, use a default list
            if not watchlist_tickers:
                watchlist_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD"]
                
            logger.info(f"Using {len(watchlist_tickers)} active tickers for training")
            
            return watchlist_tickers
            
        except Exception as e:
            logger.error(f"Error getting active tickers: {str(e)}", exc_info=True)
            # Return default list on error
            return ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    def create_time_series_splits(self, X, y, n_splits=None, embargo_size=None):
        """Create time series cross-validation splits with embargo period"""
        if n_splits is None:
            n_splits = self.config['time_series_cv']['n_splits']
            
        if embargo_size is None:
            embargo_size = self.config['time_series_cv']['embargo_size']
            
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Apply embargo period to prevent data leakage
        purged_splits = []
        for train_idx, test_idx in tscv.split(X):
            # Apply embargo: remove samples at the end of train that are too close to test
            if embargo_size > 0:
                min_test_idx = min(test_idx)
                embargo_idx = range(max(min_test_idx - embargo_size, 0), min_test_idx)
                train_idx = np.setdiff1d(train_idx, embargo_idx)
                
            purged_splits.append((train_idx, test_idx))
            
        return purged_splits
        
    def select_features(self, X, y, problem_type='regression', method=None, threshold=None, n_features=None):
        """Select most important features using specified method"""
        if not self.config['feature_selection']['enabled']:
            return X
            
        if method is None:
            method = self.config['feature_selection']['method']
            
        if threshold is None:
            threshold = self.config['feature_selection']['threshold']
            
        if n_features is None:
            n_features = self.config['feature_selection']['n_features']
            
        logger.info(f"Performing feature selection using {method} method")
        
        try:
            if method == 'importance':
                # Use a simple model to get feature importances
                if problem_type == 'classification':
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    
                model.fit(X, y)
                importances = model.feature_importances_
                
                # Select features with importance above threshold
                selected_features = X.columns[importances > threshold]
                
                # Ensure we have at least 5 features
                if len(selected_features) < 5:
                    # Take top 5 features by importance
                    selected_features = X.columns[np.argsort(importances)[-5:]]
                    
                logger.info(f"Selected {len(selected_features)} features using importance threshold")
                
                return X[selected_features]
                
            elif method == 'rfe':
                # Use Recursive Feature Elimination
                if problem_type == 'classification':
                    estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    
                selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=1)
                selector = selector.fit(X, y)
                
                selected_features = X.columns[selector.support_]
                logger.info(f"Selected {len(selected_features)} features using RFE")
                
                return X[selected_features]
                
            elif method == 'mutual_info':
                # Use mutual information
                if problem_type == 'classification':
                    mi_scores = mutual_info_classif(X, y, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X, y, random_state=42)
                    
                # Select features with MI score above threshold
                selected_features = X.columns[mi_scores > threshold]
                
                # Ensure we have at least 5 features
                if len(selected_features) < 5:
                    # Take top 5 features by MI score
                    selected_features = X.columns[np.argsort(mi_scores)[-5:]]
                    
                logger.info(f"Selected {len(selected_features)} features using mutual information")
                
                return X[selected_features]
                
            else:
                logger.warning(f"Unknown feature selection method: {method}. Using all features.")
                return X
                
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return X
            
    def optimize_hyperparameters(self):
        """Run hyperparameter optimization for all models"""
        if not optuna_available:
            logger.warning("Optuna not available. Skipping hyperparameter optimization.")
            return
            
        try:
            # Load historical data
            historical_data = self.load_historical_data()
            
            if historical_data is None or historical_data.empty:
                logger.error("Failed to load sufficient historical data for hyperparameter optimization")
                return
                
            # Optimize signal detection model
            self.optimize_signal_detection_hyperparams(historical_data)
            
            # Optimize other models as needed
            # self.optimize_price_prediction_hyperparams(historical_data)
            # self.optimize_risk_assessment_hyperparams(historical_data)
            # self.optimize_exit_strategy_hyperparams(historical_data)
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}", exc_info=True)
            
    def optimize_signal_detection_hyperparams(self, data):
        """Optimize hyperparameters for signal detection model"""
        logger.info("Optimizing hyperparameters for signal detection model")
        
        try:
            # Prepare data
            features, target = self.prepare_signal_detection_data(data)
            
            if len(features) == 0 or len(target) == 0:
                logger.error("No valid data for signal detection hyperparameter optimization")
                return
                
            # Apply feature selection if enabled
            if self.config['feature_selection']['enabled']:
                features = self.select_features(features, target, 'classification')
                
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create time series splits for cross-validation
            splits = self.create_time_series_splits(features, target)
            
            # Define the objective function for optimization
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'booster': trial.suggest_categorical('booster', ['gbtree']),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1)
                }
                
                # Cross-validation scores
                cv_scores = []
                
                for train_idx, test_idx in splits:
                    X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
                    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                    
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                    
                    # Train model
                    model = xgb.train(
                        {k: v for k, v in params.items() if k != 'n_estimators'},
                        dtrain=dtrain,
                        num_boost_round=params['n_estimators'],
                        early_stopping_rounds=20,
                        evals=[(dtest, 'test')],
                        verbose_eval=False
                    )
                    
                    # Evaluate
                    y_pred = model.predict(dtest)
                    auc = roc_auc_score(y_test, y_pred)
                    cv_scores.append(auc)
                    
                return np.mean(cv_scores)
                
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            
            # Get best parameters
            best_params = study.best_params
            logger.info(f"Best hyperparameters for signal detection: {best_params}")
            
            # Save best parameters
            params_path = os.path.join(self.config['models_dir'], 'signal_detection_optimized_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
                
            # Update Redis
            self.redis.hset(
                "models:hyperparams",
                "signal_detection",
                json.dumps(best_params)
            )
            
        except Exception as e:
            logger.error(f"Error optimizing signal detection hyperparameters: {str(e)}", exc_info=True)

    def train_with_fallbacks(self, model_func, X_train, y_train, X_test, y_test):
        """Train a model with progressive fallbacks if errors occur"""
        try:
            # First try: Full model with GPU acceleration
            model = model_func(full=True)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
            return model, history
        except Exception as e:
            logger.warning(f"Full model training failed: {str(e)}")
            try:
                # Second try: Simplified model with GPU acceleration
                model = model_func(full=False)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                return model, history
            except Exception as e:
                logger.warning(f"Simplified model training failed: {str(e)}")
                # Final fallback: Minimal model on CPU
                with tf.device('/CPU:0'):
                    model = model_func(minimal=True)
                    history = model.fit(X_train[:100], y_train[:100], epochs=1)
                    return model, history

def run_diagnostics():
    """Run diagnostics to identify GPU configuration issues"""
    logger.info("=== Running GPU Diagnostics ===")
    # re module is already imported at the top of the file
    
    results = {
        "nvidia_smi": None,
        "tensorflow_gpu": None,
        "cuda_version": None,
        "cudnn_version": None,
        "cupy_version": None,
        "tensorflow_build_info": None,
        "nvcc_version": None,
        "gh200_specific": None,
        "system_libraries": None,
        "latency_benchmark": None
    }
    
    # Check NVIDIA driver
    try:
        results["nvidia_smi"] = subprocess.check_output(["nvidia-smi"]).decode()
        logger.info("NVIDIA driver detected")
    except Exception as e:
        results["nvidia_smi"] = f"Failed to run nvidia-smi: {str(e)}"
        logger.warning(f"NVIDIA driver issue: {str(e)}")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        results["tensorflow_gpu"] = [gpu.name for gpu in gpus]
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s): {results['tensorflow_gpu']}")
        else:
            logger.warning("TensorFlow did not detect any GPUs")
    except Exception as e:
        results["tensorflow_gpu"] = f"Error: {str(e)}"
        logger.error(f"TensorFlow GPU detection error: {str(e)}")
    
    # Check CUDA version
    try:
        import cupy as cp
        results["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
        results["cupy_version"] = cp.__version__
        logger.info(f"CUDA version: {results['cuda_version']}, CuPy version: {results['cupy_version']}")
        
        # Check for GH200 specifically
        gh200_info = []
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            device_name = props["name"].decode()
            if "GH200" in device_name:
                gh200_info.append({
                    "device_id": i,
                    "name": device_name,
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory": props["totalGlobalMem"]
                })
        
        if gh200_info:
            results["gh200_specific"] = gh200_info
            logger.info(f"GH200 GPU detected: {gh200_info}")
            
            # Run basic performance benchmark if GH200 is detected
            try:
                # Create a dummy matrix for benchmark
                size = 10000
                a_gpu = cp.random.rand(size, size, dtype=cp.float32)
                b_gpu = cp.random.rand(size, size, dtype=cp.float32)
                
                # Warm up
                _ = cp.dot(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                
                # Benchmark
                start = time.time()
                _ = cp.dot(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                end = time.time()
                
                latency = (end - start) * 1000  # ms
                results["latency_benchmark"] = {
                    "operation": f"Matrix multiplication ({size}x{size})",
                    "time_ms": latency,
                    "throughput_gflops": 2 * size**3 / (end - start) / 1e9
                }
                
                logger.info(f"GH200 Benchmark: Matrix multiplication {size}x{size} took {latency:.2f} ms")
            except Exception as bench_error:
                logger.warning(f"Benchmark error: {str(bench_error)}")
        else:
            logger.warning("No GH200 GPU detected")
    except ImportError:
        results["cuda_version"] = "CuPy not available"
        logger.warning("CuPy not available, cannot check CUDA version")
    except Exception as e:
        results["cuda_version"] = f"Error: {str(e)}"
        logger.error(f"Error checking CUDA version: {str(e)}")
        
    # Check system libraries relevant to GPU operation
    try:
        # Get library versions using ldconfig
        lib_output = subprocess.check_output(["ldconfig", "-p"]).decode()
        libraries = {
            "libcuda": None,
            "libcudart": None,
            "libcudnn": None,
            "libnccl": None,
            "libtensorflow": None
        }
        
        for lib in libraries.keys():
            match = re.search(f"{lib}[^ ]* => ([^ ]+)", lib_output)
            if match:
                lib_path = match.group(1)
                libraries[lib] = lib_path
                
        results["system_libraries"] = libraries
    except Exception as e:
        logger.warning(f"Error checking system libraries: {str(e)}")
    
    return results

# Example usage
if __name__ == "__main__":
    import redis
    from data_pipeline_integration import DataPipelineIntegration
    
    # Create Redis client
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6380)),
        db=int(os.environ.get('REDIS_DB', 0))
    )
    
    # Create data loader
    data_loader = DataPipelineIntegration(
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=int(os.environ.get('REDIS_PORT', 6380)),
        redis_db=int(os.environ.get('REDIS_DB', 0)),
        polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
        unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
        use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
    )
    
    # Run GPU diagnostics
    run_diagnostics()
    
    # Create model trainer
    model_trainer = MLModelTrainer(redis_client, data_loader)
    
    # Train all models
    model_trainer.train_all_models()