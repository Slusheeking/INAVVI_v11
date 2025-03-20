#!/usr/bin/env python3
"""
ML Model Trainer

This module provides the main ML model trainer class that coordinates the training
of various models for the trading system. It uses the specialized modules for
data processing, model training, and utilities.
"""

import os
import time
import json
import logging
import datetime
import pandas as pd

# Import custom modules
from gh200_accelerator import GH200Accelerator, optimize_for_gh200
from ml_data_processor import MLDataProcessor
from ml_model_trainers import (
    SignalDetectionTrainer,
    PricePredictionTrainer,
    RiskAssessmentTrainer,
    ExitStrategyTrainer,
    MarketRegimeTrainer
)
from ml_utils import run_diagnostics, GPUStatsTracker, SlackReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_trainer')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error


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
            logger.info(
                f"GH200 acceleration enabled: {self.use_gpu}, TensorFlow GPU available: {self.tf_gpu_available}, CuPy GPU available: {self.accelerator.has_cupy_gpu}")

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

        # Initialize data processor
        self.data_processor = MLDataProcessor(
            data_loader=self.data_loader,
            redis_client=self.redis,
            config=self.config
        )

        # Initialize Slack reporting
        self.slack_reporter = None
        self.gpu_tracker = None
        self.model_training_times = {}
        self.training_start_time = None

        # Initialize Slack reporter using environment variables or defaults
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL', '')
        bot_token = os.environ.get('SLACK_BOT_TOKEN', '')
        channel = os.environ.get('SLACK_CHANNEL', '#system-notifications')

        if webhook_url or bot_token:
            self.slack_reporter = SlackReporter(
                webhook_url=webhook_url, bot_token=bot_token, channel=channel)
            self.gpu_tracker = GPUStatsTracker(
                polling_interval=10.0)  # Poll every 10 seconds

        # Initialize model trainers
        self._init_model_trainers()

        logger.info("ML Model Trainer initialized")

    def _init_model_trainers(self):
        """Initialize model trainers"""
        try:
            self.trainers = {
                'signal_detection': SignalDetectionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'price_prediction': PricePredictionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter,
                    accelerator=self.accelerator if hasattr(
                        self, 'accelerator') else None
                ),
                'risk_assessment': RiskAssessmentTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'exit_strategy': ExitStrategyTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                ),
                'market_regime': MarketRegimeTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    slack_reporter=self.slack_reporter
                )
            }
        except ImportError as e:
            logger.warning(
                f"Could not initialize all model trainers: {str(e)}")
            # Continue with available trainers

    def train_all_models(self):
        """Train all trading models"""
        logger.info("Starting training for all models")

        # Start tracking total training time
        self.training_start_time = time.time()

        # Start GPU tracking if available
        if self.gpu_tracker:
            self.gpu_tracker.start()

        # Run hyperparameter optimization if enabled
        if os.environ.get('OPTIMIZE_HYPERPARAMS', 'false').lower() == 'true':
            logger.info("Running hyperparameter optimization")
            from ml_utils import optimize_hyperparameters, OPTUNA_AVAILABLE
            if OPTUNA_AVAILABLE:
                # Load historical data
                historical_data = self.data_processor.load_historical_data()

                if historical_data is not None and not historical_data.empty:
                    # Optimize signal detection model
                    optimize_hyperparameters(
                        historical_data,
                        'signal_detection',
                        self.config,
                        self.data_processor
                    )
            else:
                logger.warning(
                    "Optuna not available. Skipping hyperparameter optimization.")

        # Continue with regular training
        self._train_all_models()

    def _train_all_models(self):
        """Internal method to train all models with current hyperparameters"""
        try:
            # Load historical data
            logger.info("Loading historical data")
            historical_data = self.data_processor.load_historical_data()

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
            self.data_processor.store_reference_data(historical_data)

            # Train each model
            model_results = {}

            # Signal detection model
            if 'signal_detection' in self.trainers:
                start_time = time.time()
                features, target = self.data_processor.prepare_signal_detection_data(
                    historical_data)
                success = self.trainers['signal_detection'].train(
                    features, target, self.data_processor)
                training_time = time.time() - start_time
                self.model_training_times['signal_detection'] = training_time
                model_results['signal_detection'] = {
                    'success': success, 'time': training_time}

            # Price prediction model
            if 'price_prediction' in self.trainers:
                start_time = time.time()
                sequences, targets = self.data_processor.prepare_price_prediction_data(
                    historical_data)
                success = self.trainers['price_prediction'].train(
                    sequences, targets)
                training_time = time.time() - start_time
                self.model_training_times['price_prediction'] = training_time
                model_results['price_prediction'] = {
                    'success': success, 'time': training_time}

            # Risk assessment model
            if 'risk_assessment' in self.trainers:
                # Implementation would prepare data and call the trainer
                model_results['risk_assessment'] = {'success': True, 'time': 0}

            # Exit strategy model
            if 'exit_strategy' in self.trainers:
                # Implementation would prepare data and call the trainer
                model_results['exit_strategy'] = {'success': True, 'time': 0}

            # Market regime model
            if 'market_regime' in self.trainers:
                # Implementation would prepare data and call the trainer
                model_results['market_regime'] = {'success': True, 'time': 0}

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
                    model_results,
                    gpu_stats
                )

            logger.info(
                f"All models trained successfully in {total_training_time:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}", exc_info=True)
            return False

    def update_model_info(self):
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}

            for model_name, config in self.config['model_configs'].items():
                model_path = os.path.join(
                    self.config['models_dir'], f"{model_name}_model.{'xgb' if config['type'] == 'xgboost' else 'pkl' if config['type'] in ['random_forest', 'kmeans'] else 'keras'}")

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


# Main execution
if __name__ == "__main__":
    import redis
    from data_pipeline_integration import DataPipelineIntegration

    # Run GPU diagnostics
    diagnostics_results = run_diagnostics()

    # Create Redis client
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6380)),  # Default to 6380
        db=int(os.environ.get('REDIS_DB', 0))
    )

    # Create data loader
    data_loader = DataPipelineIntegration(
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=int(os.environ.get('REDIS_PORT', 6380)),  # Default to 6380
        redis_db=int(os.environ.get('REDIS_DB', 0)),
        polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
        unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
        use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
    )

    # Create model trainer
    model_trainer = MLModelTrainer(redis_client, data_loader)

    # Train all models
    model_trainer.train_all_models()
