#!/usr/bin/env python3
"""
Learning Engine Module

This module provides a unified learning engine with continual model training, evaluation,
and deployment capabilities. It integrates with the ML Engine and Trading System to ensure
models stay relevant as market conditions evolve. Key features include:

1. Automated model training and updating with new market data
2. Performance monitoring and validation of models
3. Model versioning and deployment management
4. Drift detection and adaptive retraining
5. A/B testing of model variants
6. GPU acceleration optimized for NVIDIA GH200 Grace Hopper Superchips

The Learning Engine serves as the "brain" of the trading system, constantly improving
the quality of predictions and trading signals through automated learning processes.
"""

import os
import json
import time
import uuid
import logging
import datetime
import threading
import numpy as np
import pandas as pd
import schedule
import pickle
import requests

# Import Prometheus client for metrics
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.getLogger('learning_engine').warning(
        "Prometheus client not available. Metrics will not be exposed.")
from typing import Dict, List, Optional, Any, Tuple
from config import config

# ML frameworks
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import xgboost as xgb

# Import TensorRT with error handling
try:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.getLogger('learning_engine').warning(
        "TensorRT not available. Model optimization will be limited.")

# Import CuPy with error handling
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.getLogger('learning_engine').warning(
        "CuPy not available. Some GPU operations will be slower.")
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from scipy.stats import ks_2samp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.environ.get(
            'LOGS_DIR', './logs'), 'learning_engine.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('learning_engine')

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # Model training metrics
    MODEL_TRAINING_TIME = prom.Histogram(
        'learning_engine_model_training_time_seconds',
        'Time spent training models',
        ['model_name', 'model_type', 'training_type']
    )

    MODEL_EVALUATION_METRICS = prom.Gauge(
        'learning_engine_model_metrics',
        'Model evaluation metrics',
        ['model_name', 'model_type', 'metric']
    )

    MODEL_VERSIONS = prom.Gauge(
        'learning_engine_model_versions',
        'Number of model versions',
        ['model_name']
    )

    DRIFT_DETECTION = prom.Counter(
        'learning_engine_drift_detection_total',
        'Number of drift detections',
        ['model_name', 'drift_type', 'result']
    )

    GPU_MEMORY_USAGE = prom.Gauge(
        'learning_engine_gpu_memory_usage_bytes',
        'GPU memory usage in bytes',
        ['device']
    )

    MODEL_DEPLOYMENT_COUNT = prom.Counter(
        'learning_engine_model_deployments_total',
        'Number of model deployments',
        ['model_name', 'environment']
    )

    FEATURE_IMPORTANCE = prom.Gauge(
        'learning_engine_feature_importance',
        'Feature importance values',
        ['model_name', 'feature']
    )

    logger.info("Prometheus metrics initialized for Learning Engine")


class SlackReporter:
    """
    Slack notification system for learning engine events
    """

    def __init__(self, webhook_url=None, bot_token=None, channel='#learning-system'):
        """
        Initialize the Slack reporter

        Args:
            webhook_url: Slack webhook URL for notifications
            bot_token: Slack bot token for API access
            channel: Default Slack channel for notifications
        """
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.bot_token = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        self.channel = channel

        # Check if Slack integration is available
        if not self.webhook_url and not self.bot_token:
            logging.getLogger('learning_engine').warning(
                "No Slack webhook URL or bot token provided. Notifications will be logged only.")
            self.slack_available = False
        else:
            self.slack_available = True
            logging.getLogger('learning_engine').info(
                "Slack reporter initialized")

    def report_error(self, error_message, details=None):
        """Report an error to Slack"""
        if not self.slack_available:
            return

        try:
            # Format message
            message = f"*ERROR:* {error_message}"
            if details:
                message += f"\n```{json.dumps(details, indent=2)}```"

            # Send to Slack
            if self.webhook_url:
                payload = {
                    "text": message,
                    "channel": self.channel,
                    "icon_emoji": ":x:",
                    "username": "Learning Engine"
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": message
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)
        except Exception as e:
            logging.getLogger('learning_engine').error(
                f"Error sending Slack notification: {str(e)}")

    def report_model_metrics(self, model_name, metrics):
        """Report model metrics to Slack"""
        if not self.slack_available:
            return

        try:
            # Format message
            message = f"*Model Metrics:* {model_name}\n"

            # Add metrics
            for key, value in metrics.items():
                if isinstance(value, float):
                    message += f"• {key}: {value:.4f}\n"
                else:
                    message += f"• {key}: {value}\n"

            # Send to Slack
            if self.webhook_url:
                payload = {
                    "text": message,
                    "channel": self.channel,
                    "icon_emoji": ":chart_with_upwards_trend:",
                    "username": "Learning Engine"
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": message
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)
        except Exception as e:
            logging.getLogger('learning_engine').error(
                f"Error sending Slack notification: {str(e)}")

    def report_training_complete(self, training_time, model_name, version, metrics=None):
        """Report training completion to Slack"""
        if not self.slack_available:
            return

        try:
            # Format message
            message = f"*Training Complete:* {model_name} v{version}\n"
            message += f"• Training Time: {training_time:.2f} seconds\n"

            # Add metrics if provided
            if metrics:
                message += "*Metrics:*\n"
                for key, value in metrics.items():
                    if isinstance(value, float):
                        message += f"• {key}: {value:.4f}\n"
                    else:
                        message += f"• {key}: {value}\n"

            # Send to Slack
            if self.webhook_url:
                payload = {
                    "text": message,
                    "channel": self.channel,
                    "icon_emoji": ":white_check_mark:",
                    "username": "Learning Engine"
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": message
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)
        except Exception as e:
            logging.getLogger('learning_engine').error(
                f"Error sending Slack notification: {str(e)}")


class ModelPerformanceTracker:
    """
    Tracks and compares model performance metrics over time
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.metrics_history = {}
        self.baseline_metrics = {}

    def log_metrics(self, model_name: str, version: int, metrics: Dict[str, float]):
        """Log model metrics for a specific version"""
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = {}

        # Add timestamp to metrics
        metrics_with_time = metrics.copy()
        metrics_with_time['timestamp'] = time.time()
        metrics_with_time['version'] = version

        # Store metrics in history
        self.metrics_history[model_name][version] = metrics_with_time

        # Store in Redis if available
        if self.redis:
            try:
                # Store latest metrics
                self.redis.hset(
                    f"model:metrics:{model_name}:{version}",
                    mapping=self._prepare_for_redis(metrics_with_time)
                )

                # Store version history
                self.redis.zadd(
                    f"model:versions:{model_name}",
                    {str(version): metrics_with_time['timestamp']}
                )

                # Log successful metrics update
                logger.info(f"Logged metrics for {model_name} v{version}")
            except Exception as e:
                logger.error(f"Error storing metrics in Redis: {str(e)}")

    def set_baseline(self, model_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for model performance comparison"""
        self.baseline_metrics[model_name] = metrics

        # Store in Redis if available
        if self.redis:
            try:
                self.redis.hset(
                    f"model:baseline:{model_name}",
                    mapping=self._prepare_for_redis(metrics)
                )
            except Exception as e:
                logger.error(f"Error storing baseline in Redis: {str(e)}")

    def compare_to_baseline(self, model_name: str, version: int) -> Dict[str, float]:
        """Compare metrics to baseline and return differences"""
        if model_name not in self.metrics_history or version not in self.metrics_history[model_name]:
            logger.warning(f"No metrics found for {model_name} v{version}")
            return {}

        if model_name not in self.baseline_metrics:
            logger.warning(f"No baseline metrics found for {model_name}")
            return {}

        current = self.metrics_history[model_name][version]
        baseline = self.baseline_metrics[model_name]

        # Calculate differences for each metric
        diffs = {}
        for key in baseline.keys():
            if key in current and isinstance(baseline[key], (int, float)) and key != 'timestamp' and key != 'version':
                diffs[key] = current[key] - baseline[key]
                diffs[f"{key}_pct"] = (
                    (current[key] / baseline[key]) - 1) * 100 if baseline[key] != 0 else float('inf')

        return diffs

    def get_best_version(self, model_name: str, metric: str, higher_is_better: bool = True) -> Optional[int]:
        """Get the best performing model version based on a specific metric"""
        if model_name not in self.metrics_history:
            return None

        versions = self.metrics_history[model_name]
        if not versions:
            return None

        # Filter versions that have the specified metric
        valid_versions = {v: metrics for v,
                          metrics in versions.items() if metric in metrics}
        if not valid_versions:
            return None

        # Find best version
        if higher_is_better:
            best_version = max(valid_versions.items(),
                               key=lambda x: x[1][metric])[0]
        else:
            best_version = min(valid_versions.items(),
                               key=lambda x: x[1][metric])[0]

        return best_version

    def _prepare_for_redis(self, metrics: Dict) -> Dict:
        """Prepare metrics dictionary for Redis storage"""
        result = {}
        for k, v in metrics.items():
            # Convert numpy values to native Python types
            if isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                result[k] = int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                result[k] = float(v)
            else:
                result[k] = v
        return result


class ModelRegistry:
    """
    Registry for ML models with versioning and metadata
    """

    def __init__(self, models_dir: str, redis_client=None):
        self.models_dir = models_dir
        self.redis = redis_client
        self.models_metadata = {}

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Load existing model metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata for existing models"""
        # Check if metadata file exists
        metadata_path = os.path.join(self.models_dir, 'models_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.models_metadata = json.load(f)
                logger.info(
                    f"Loaded metadata for {len(self.models_metadata)} models")
            except Exception as e:
                logger.error(f"Error loading model metadata: {str(e)}")
                self.models_metadata = {}

    def _save_metadata(self):
        """Save metadata for all models"""
        metadata_path = os.path.join(self.models_dir, 'models_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.models_metadata, f, indent=2)
            logger.info(
                f"Saved metadata for {len(self.models_metadata)} models")
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")

    def register_model(self, model_name: str, model, model_type: str, metrics: Dict[str, float],
                       feature_names: List[str] = None, hyperparams: Dict = None) -> int:
        """
        Register a new model in the registry

        Args:
            model_name: Name of the model
            model: The model object
            model_type: Type of model ('xgboost', 'keras', 'sklearn')
            metrics: Dict of performance metrics
            feature_names: List of feature names
            hyperparams: Dict of hyperparameters

        Returns:
            Version number of the new model
        """
        # Initialize model entry if it doesn't exist
        if model_name not in self.models_metadata:
            self.models_metadata[model_name] = {
                'versions': {},
                'latest_version': 0,
                'model_type': model_type,
                'creation_date': datetime.datetime.now().isoformat()
            }

        # Get new version number
        new_version = self.models_metadata[model_name]['latest_version'] + 1

        # Save the model
        model_filename = self._get_model_filename(
            model_name, new_version, model_type)
        model_path = os.path.join(self.models_dir, model_filename)

        try:
            # Save model based on its type
            if model_type == 'xgboost':
                model.save_model(model_path)
            elif model_type == 'keras':
                model.save(model_path)
            elif model_type == 'sklearn':
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Update metadata
            version_metadata = {
                'version': new_version,
                'timestamp': time.time(),
                'date': datetime.datetime.now().isoformat(),
                'metrics': metrics,
                'file': model_filename,
                'feature_names': feature_names,
                'hyperparams': hyperparams
            }

            self.models_metadata[model_name]['versions'][str(
                new_version)] = version_metadata
            self.models_metadata[model_name]['latest_version'] = new_version

            # Save metadata
            self._save_metadata()

            # Update Redis if available
            if self.redis:
                self._update_redis_metadata(
                    model_name, new_version, version_metadata)

            # Record metrics in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    # Record model version count
                    MODEL_VERSIONS.labels(
                        model_name=model_name).set(new_version)

                    # Record model metrics
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            MODEL_EVALUATION_METRICS.labels(
                                model_name=model_name,
                                model_type=model_type,
                                metric=metric_name
                            ).set(metric_value)

                    # Record feature importance if available
                    if model_type == 'xgboost' and hasattr(model, 'get_score'):
                        importance_map = model.get_score(
                            importance_type='gain')
                        for feature, importance in importance_map.items():
                            FEATURE_IMPORTANCE.labels(
                                model_name=model_name,
                                feature=feature
                            ).set(importance)
                    elif model_type == 'sklearn' and hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if feature_names and len(importances) == len(feature_names):
                            for i, feature in enumerate(feature_names):
                                FEATURE_IMPORTANCE.labels(
                                    model_name=model_name,
                                    feature=feature
                                ).set(importances[i])
                except Exception as prom_e:
                    logger.warning(
                        f"Error recording model metrics in Prometheus: {prom_e}")

            logger.info(
                f"Registered {model_name} v{new_version} in model registry")
            return new_version

        except Exception as e:
            logger.error(f"Error registering model {model_name}: {str(e)}")
            raise

    def load_model(self, model_name: str, version: int = None) -> Tuple[Any, Dict]:
        """
        Load a model from the registry

        Args:
            model_name: Name of the model
            version: Version number, or None for latest

        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.models_metadata:
            raise ValueError(f"Model {model_name} not found in registry")

        # Get version (latest if not specified)
        if version is None:
            version = self.models_metadata[model_name]['latest_version']

        version_str = str(version)
        if version_str not in self.models_metadata[model_name]['versions']:
            raise ValueError(
                f"Version {version} of model {model_name} not found")

        # Get metadata and model path
        metadata = self.models_metadata[model_name]['versions'][version_str]
        model_path = os.path.join(self.models_dir, metadata['file'])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model based on its type
        model_type = self.models_metadata[model_name]['model_type']
        try:
            if model_type == 'xgboost':
                model = xgb.Booster()
                model.load_model(model_path)
            elif model_type == 'keras':
                model = tf_load_model(model_path)
            elif model_type == 'sklearn':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            logger.info(f"Loaded {model_name} v{version} from registry")
            return model, metadata

        except Exception as e:
            logger.error(
                f"Error loading model {model_name} v{version}: {str(e)}")
            raise

    def get_all_models(self) -> Dict[str, Dict]:
        """Get metadata for all models in the registry"""
        return self.models_metadata

    def get_model_versions(self, model_name: str) -> List[int]:
        """Get all versions of a specific model"""
        if model_name not in self.models_metadata:
            return []

        return [int(v) for v in self.models_metadata[model_name]['versions'].keys()]

    def delete_model_version(self, model_name: str, version: int) -> bool:
        """Delete a specific version of a model"""
        if model_name not in self.models_metadata:
            return False

        version_str = str(version)
        if version_str not in self.models_metadata[model_name]['versions']:
            return False

        # Get file path
        metadata = self.models_metadata[model_name]['versions'][version_str]
        model_path = os.path.join(self.models_dir, metadata['file'])

        # Delete file if it exists
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception as e:
                logger.error(
                    f"Error deleting model file {model_path}: {str(e)}")
                return False

        # Update metadata
        del self.models_metadata[model_name]['versions'][version_str]

        # Update latest version if needed
        if self.models_metadata[model_name]['latest_version'] == version:
            if self.models_metadata[model_name]['versions']:
                self.models_metadata[model_name]['latest_version'] = max(
                    int(v) for v in self.models_metadata[model_name]['versions'].keys()
                )
            else:
                self.models_metadata[model_name]['latest_version'] = 0

        # Save metadata
        self._save_metadata()

        # Update Redis if available
        if self.redis:
            try:
                self.redis.hdel(f"model:registry:{model_name}", version_str)
                self.redis.zrem(f"model:versions:{model_name}", version_str)
            except Exception as e:
                logger.error(
                    f"Error updating Redis after model deletion: {str(e)}")

        logger.info(f"Deleted {model_name} v{version} from registry")
        return True

    def _get_model_filename(self, model_name: str, version: int, model_type: str) -> str:
        """Generate filename for a model"""
        if model_type == 'xgboost':
            ext = 'xgb'
        elif model_type == 'keras':
            ext = 'h5'
        elif model_type == 'sklearn':
            ext = 'pkl'
        else:
            ext = 'model'

        return f"{model_name}_v{version}.{ext}"

    def _update_redis_metadata(self, model_name: str, version: int, metadata: Dict):
        """Update model metadata in Redis"""
        if not self.redis:
            return

        try:
            # Store versions set
            self.redis.zadd(
                f"model:versions:{model_name}",
                {str(version): metadata['timestamp']}
            )

            # Store metadata
            self.redis.hset(
                f"model:registry:{model_name}",
                str(version),
                json.dumps(metadata)
            )

            # Update current version pointer
            self.redis.set(f"model:current:{model_name}", version)

        except Exception as e:
            logger.error(f"Error updating Redis model metadata: {str(e)}")


class ModelDeployer:
    """
    Manages model deployment, switching, and rollback
    """

    def __init__(self, model_registry: ModelRegistry, redis_client=None):
        self.registry = model_registry
        self.redis = redis_client
        self.deployed_models = {}
        self.deployment_history = {}

        # Load currently deployed models
        self._load_deployed_models()

    def _load_deployed_models(self):
        """Load information about currently deployed models"""
        if self.redis:
            try:
                # Get all model deployment keys
                for key in self.redis.keys("model:deployed:*"):
                    model_name = key.split(":")[-1]
                    version = int(self.redis.get(key) or 0)
                    if version > 0:
                        self.deployed_models[model_name] = version

                logger.info(
                    f"Loaded {len(self.deployed_models)} deployed models")
            except Exception as e:
                logger.error(
                    f"Error loading deployed models from Redis: {str(e)}")

    def deploy_model(self, model_name: str, version: int, environment: str = 'production') -> bool:
        """
        Deploy a specific model version

        Args:
            model_name: Name of the model
            version: Version to deploy
            environment: Deployment environment (production, staging, etc.)

        Returns:
            Success status
        """
        try:
            # Start timing for Prometheus metrics
            start_time = time.time()

            # Verify model exists
            if model_name not in self.registry.models_metadata:
                logger.error(f"Model {model_name} not found in registry")
                return False

            version_str = str(version)
            if version_str not in self.registry.models_metadata[model_name]['versions']:
                logger.error(
                    f"Version {version} of model {model_name} not found in registry")
                return False

            # Update deployed models
            prev_version = self.deployed_models.get(model_name)
            self.deployed_models[model_name] = version

            # Record in deployment history
            if model_name not in self.deployment_history:
                self.deployment_history[model_name] = []

            deployment_record = {
                'version': version,
                'environment': environment,
                'timestamp': time.time(),
                'date': datetime.datetime.now().isoformat(),
                'previous_version': prev_version
            }

            self.deployment_history[model_name].append(deployment_record)

            # Update Redis if available
            if self.redis:
                try:
                    # Set current deployed version
                    self.redis.set(f"model:deployed:{model_name}", version)

                    # Add to deployment history
                    self.redis.lpush(
                        f"model:deployment_history:{model_name}",
                        json.dumps(deployment_record)
                    )

                    # Trim history to last 20 deployments
                    self.redis.ltrim(
                        f"model:deployment_history:{model_name}", 0, 19)

                except Exception as e:
                    logger.error(
                        f"Error updating Redis for model deployment: {str(e)}")

            # Record in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                try:
                    # Increment deployment counter
                    MODEL_DEPLOYMENT_COUNT.labels(
                        model_name=model_name,
                        environment=environment
                    ).inc()

                    # Record deployment time
                    deployment_time = time.time() - start_time
                    MODEL_TRAINING_TIME.labels(
                        model_name=model_name,
                        model_type=self.registry.models_metadata[model_name]['model_type'],
                        training_type='deployment'
                    ).observe(deployment_time)
                except Exception as prom_e:
                    logger.warning(
                        f"Error recording deployment in Prometheus: {prom_e}")

            logger.info(f"Deployed {model_name} v{version} to {environment}")
            return True

        except Exception as e:
            logger.error(
                f"Error deploying model {model_name} v{version}: {str(e)}")
            return False

    def rollback_model(self, model_name: str, to_version: int = None) -> bool:
        """
        Rollback a model to a previous version

        Args:
            model_name: Name of the model
            to_version: Version to rollback to, or None for previous version

        Returns:
            Success status
        """
        try:
            # Check if model is deployed
            if model_name not in self.deployed_models:
                logger.error(f"Model {model_name} is not currently deployed")
                return False

            # Get deployment history
            if not self.deployment_history.get(model_name, []):
                # Try to get from Redis
                if self.redis:
                    try:
                        history = self.redis.lrange(
                            f"model:deployment_history:{model_name}", 0, -1)
                        if history:
                            self.deployment_history[model_name] = [
                                json.loads(h) for h in history]
                    except Exception:
                        pass

            history = self.deployment_history.get(model_name, [])

            # Determine version to rollback to
            if to_version is None:
                # Use previous version from history if available
                if len(history) > 1:
                    # Current version is at index 0, previous at index 1
                    to_version = history[1]['version']
                else:
                    # Get previous version from current deployment
                    current = history[0] if history else {}
                    to_version = current.get('previous_version')

                    if to_version is None:
                        logger.error(
                            f"No previous version found for {model_name}")
                        return False

            # Verify rollback version exists
            version_str = str(to_version)
            if version_str not in self.registry.models_metadata[model_name]['versions']:
                logger.error(
                    f"Rollback version {to_version} not found for {model_name}")
                return False

            # Deploy the rollback version
            return self.deploy_model(model_name, to_version, environment='rollback')

        except Exception as e:
            logger.error(f"Error rolling back model {model_name}: {str(e)}")
            return False

    def get_deployed_model(self, model_name: str) -> Tuple[Any, Dict]:
        """
        Get the currently deployed model

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (model, metadata) or (None, None) if not deployed
        """
        if model_name not in self.deployed_models:
            return None, None

        version = self.deployed_models[model_name]
        try:
            return self.registry.load_model(model_name, version)
        except Exception as e:
            logger.error(
                f"Error loading deployed model {model_name} v{version}: {str(e)}")
            return None, None

    def get_deployment_history(self, model_name: str) -> List[Dict]:
        """Get deployment history for a specific model"""
        return self.deployment_history.get(model_name, [])

    def get_all_deployed_models(self) -> Dict[str, int]:
        """Get all currently deployed models with their versions"""
        return self.deployed_models.copy()


class DriftDetector:
    """
    Detects data drift and model performance degradation
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.reference_distributions = {}
        self.drift_thresholds = {
            'ks_threshold': 0.1,        # Kolmogorov-Smirnov test threshold
            'js_threshold': 0.05,       # Jensen-Shannon divergence threshold
            'performance_threshold': {
                'classification': 0.05,  # 5% drop in accuracy/f1
                'regression': 0.1        # 10% increase in error metrics
            }
        }

    def store_reference_distribution(self, name: str, data, metadata: Dict = None):
        """
        Store a reference distribution for future drift comparison

        Args:
            name: Identifier for the distribution
            data: Reference data (numpy array or pandas DataFrame)
            metadata: Additional information about the distribution
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Store descriptive statistics for each column
                stats = {}
                for column in data.columns:
                    column_data = data[column].dropna()
                    if len(column_data) > 0:
                        if np.issubdtype(column_data.dtype, np.number):
                            stats[column] = {
                                'mean': column_data.mean(),
                                'std': column_data.std(),
                                'min': column_data.min(),
                                'max': column_data.max(),
                                'median': column_data.median(),
                                'q1': column_data.quantile(0.25),
                                'q3': column_data.quantile(0.75),
                                'histogram': np.histogram(column_data, bins=20)[0].tolist()
                            }

                # Store in-memory
                self.reference_distributions[name] = {
                    'stats': stats,
                    'timestamp': time.time(),
                    'metadata': metadata or {}
                }

                # Store in Redis if available
                if self.redis:
                    try:
                        redis_data = {
                            'stats': json.dumps(stats),
                            'timestamp': time.time(),
                            'metadata': json.dumps(metadata or {})
                        }
                        self.redis.hset(
                            f"drift:reference:{name}", mapping=redis_data)
                        logger.info(
                            f"Stored reference distribution for {name} in Redis")
                    except Exception as e:
                        logger.error(
                            f"Error storing reference distribution in Redis: {str(e)}")

                logger.info(
                    f"Stored reference distribution for {name} with {len(stats)} features")

            else:
                logger.error(
                    f"Unsupported data type for reference distribution: {type(data)}")

        except Exception as e:
            logger.error(f"Error storing reference distribution: {str(e)}")

    def detect_drift(self, name: str, current_data) -> Dict[str, Any]:
        """
        Detect drift between current data and reference distribution

        Args:
            name: Identifier for the distribution
            current_data: Current data to compare

        Returns:
            Dictionary with drift metrics
        """
        if name not in self.reference_distributions:
            # Try to load from Redis
            if self.redis:
                try:
                    redis_data = self.redis.hgetall(f"drift:reference:{name}")
                    if redis_data and 'stats' in redis_data:
                        self.reference_distributions[name] = {
                            'stats': json.loads(redis_data['stats']),
                            'timestamp': float(redis_data['timestamp']),
                            'metadata': json.loads(redis_data.get('metadata', '{}'))
                        }
                        logger.info(
                            f"Loaded reference distribution for {name} from Redis")
                except Exception as e:
                    logger.error(
                        f"Error loading reference distribution from Redis: {str(e)}")

        if name not in self.reference_distributions:
            logger.warning(f"No reference distribution found for {name}")
            return {'drift_detected': False, 'error': 'No reference distribution'}

        try:
            reference = self.reference_distributions[name]

            if not isinstance(current_data, pd.DataFrame):
                return {'drift_detected': False, 'error': 'Current data must be a DataFrame'}

            # Calculate drift metrics for each feature
            drift_metrics = {}
            drift_detected = False

            for column, ref_stats in reference['stats'].items():
                if column in current_data.columns:
                    column_data = current_data[column].dropna()

                    if len(column_data) > 0 and np.issubdtype(column_data.dtype, np.number):
                        # Calculate current statistics
                        curr_stats = {
                            'mean': column_data.mean(),
                            'std': column_data.std(),
                            'min': column_data.min(),
                            'max': column_data.max(),
                            'median': column_data.median(),
                            'q1': column_data.quantile(0.25),
                            'q3': column_data.quantile(0.75)
                        }

                        # Calculate KS test for distribution comparison
                        try:
                            # Create samples of equal size for KS test
                            # This is a simplified approach - in production you'd use actual reference data
                            ref_mean = ref_stats['mean']
                            ref_std = ref_stats['std']
                            ref_samples = np.random.normal(
                                ref_mean, ref_std, size=len(column_data))

                            ks_stat, ks_pvalue = ks_2samp(
                                column_data, ref_samples)

                            # Calculate percent changes
                            mean_pct_change = abs(
                                (curr_stats['mean'] - ref_stats['mean']) / ref_stats['mean']) if ref_stats['mean'] != 0 else float('inf')
                            std_pct_change = abs(
                                (curr_stats['std'] - ref_stats['std']) / ref_stats['std']) if ref_stats['std'] != 0 else float('inf')

                            # Check for drift
                            feature_drift = (ks_stat > self.drift_thresholds['ks_threshold'] and
                                             (mean_pct_change > 0.2 or std_pct_change > 0.3))

                            # Store metrics
                            drift_metrics[column] = {
                                'ks_stat': ks_stat,
                                'ks_pvalue': ks_pvalue,
                                'mean_change': mean_pct_change,
                                'std_change': std_pct_change,
                                'drift_detected': feature_drift
                            }

                            if feature_drift:
                                drift_detected = True

                        except Exception as e:
                            logger.error(
                                f"Error calculating drift for {column}: {str(e)}")
                            drift_metrics[column] = {'error': str(e)}

            # Store drift detection results in Redis
            if self.redis:
                try:
                    result_summary = {
                        'drift_detected': drift_detected,
                        'timestamp': time.time(),
                        'num_features': len(drift_metrics),
                        'drifted_features': sum(1 for v in drift_metrics.values()
                                                if isinstance(v, dict) and v.get('drift_detected', False))
                    }
                    self.redis.hset(
                        f"drift:detection:{name}", mapping=result_summary)

                    # Store detailed results
                    self.redis.set(
                        f"drift:details:{name}:{int(time.time())}",
                        json.dumps(drift_metrics)
                    )

                    # Set expiry for detailed results (7 days)
                    self.redis.expire(
                        f"drift:details:{name}:{int(time.time())}", 60*60*24*7)
                except Exception as e:
                    logger.error(
                        f"Error storing drift results in Redis: {str(e)}")

            return {
                'drift_detected': drift_detected,
                'features': drift_metrics,
                'reference_timestamp': reference['timestamp'],
                'detection_timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return {'drift_detected': False, 'error': str(e)}

    def detect_model_performance_drift(self, model_name: str,
                                       current_metrics: Dict[str, float],
                                       baseline_metrics: Dict[str, float],
                                       model_type: str = 'classification') -> Dict[str, Any]:
        """
        Detect drift in model performance metrics

        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            model_type: 'classification' or 'regression'

        Returns:
            Dictionary with performance drift metrics
        """
        try:
            drift_detected = False
            drift_metrics = {}

            threshold = self.drift_thresholds['performance_threshold'][model_type]

            # Compare key metrics based on model type
            if model_type == 'classification':
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in current_metrics and metric in baseline_metrics:
                        # For classification metrics, lower is worse
                        change = current_metrics[metric] - \
                            baseline_metrics[metric]
                        pct_change = change / \
                            baseline_metrics[metric] if baseline_metrics[metric] != 0 else float(
                                'inf')

                        metric_drift = pct_change < -threshold
                        if metric_drift:
                            drift_detected = True

                        drift_metrics[metric] = {
                            'current': current_metrics[metric],
                            'baseline': baseline_metrics[metric],
                            'change': change,
                            'pct_change': pct_change,
                            'drift_detected': metric_drift
                        }
            else:  # regression
                for metric in ['mse', 'rmse', 'mae']:
                    if metric in current_metrics and metric in baseline_metrics:
                        # For error metrics, higher is worse
                        change = current_metrics[metric] - \
                            baseline_metrics[metric]
                        pct_change = change / \
                            baseline_metrics[metric] if baseline_metrics[metric] != 0 else float(
                                'inf')

                        metric_drift = pct_change > threshold
                        if metric_drift:
                            drift_detected = True

                        drift_metrics[metric] = {
                            'current': current_metrics[metric],
                            'baseline': baseline_metrics[metric],
                            'change': change,
                            'pct_change': pct_change,
                            'drift_detected': metric_drift
                        }

            # Store results in Redis
            if self.redis:
                try:
                    result_summary = {
                        'drift_detected': drift_detected,
                        'timestamp': time.time(),
                        'model_type': model_type
                    }
                    self.redis.hset(
                        f"drift:performance:{model_name}", mapping=result_summary)

                    # Store detailed results
                    self.redis.set(
                        f"drift:performance_details:{model_name}:{int(time.time())}",
                        json.dumps(drift_metrics)
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing performance drift results in Redis: {str(e)}")

            return {
                'drift_detected': drift_detected,
                'metrics': drift_metrics,
                'detection_timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error detecting performance drift: {str(e)}")
            return {'drift_detected': False, 'error': str(e)}


class ModelEvaluator:
    """
    Evaluates model performance and validates predictions
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client

    def evaluate_classification_model(self, model, model_type: str, X_test, y_test,
                                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a classification model

        Args:
            model: The model to evaluate
            model_type: Type of model ('xgboost', 'keras', 'sklearn')
            X_test: Test features
            y_test: Test targets (ground truth)
            feature_names: List of feature names

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Make predictions based on model type
            if model_type == 'xgboost':
                # For XGBoost, convert to DMatrix first
                dtest = xgb.DMatrix(X_test, label=y_test,
                                    feature_names=feature_names)
                y_pred_proba = model.predict(dtest)
                y_pred = (y_pred_proba > 0.5).astype(int)
            elif model_type == 'keras':
                # For Keras models
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
                # Handle multi-dim output
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                    y_pred_proba = np.max(y_pred_proba, axis=1)
            elif model_type == 'sklearn':
                # For scikit-learn models
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = y_pred  # No probability, just use predictions
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Calculate classification metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(
                y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(
                y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(
                y_test, y_pred, average='weighted', zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Classification report as text
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics['classification_report'] = report

            # Calculate feature importance if available
            metrics['feature_importance'] = self._get_feature_importance(
                model, model_type, feature_names)

            # Store in Redis if available
            if self.redis:
                try:
                    # Store main metrics (exclude large elements like feature_importance)
                    redis_metrics = {k: v for k, v in metrics.items()
                                     if k not in ['confusion_matrix', 'classification_report', 'feature_importance']}

                    model_key = f"model_evaluation:{model_type}:{uuid.uuid4()}"
                    self.redis.hset(model_key, mapping=redis_metrics)
                    # Expire after 7 days
                    self.redis.expire(model_key, 60*60*24*7)
                except Exception as e:
                    logger.error(
                        f"Error storing evaluation metrics in Redis: {str(e)}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating classification model: {str(e)}")
            return {'error': str(e)}

    def evaluate_regression_model(self, model, model_type: str, X_test, y_test,
                                  feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a regression model

        Args:
            model: The model to evaluate
            model_type: Type of model ('xgboost', 'keras', 'sklearn')
            X_test: Test features
            y_test: Test targets (ground truth)
            feature_names: List of feature names

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Make predictions based on model type
            if model_type == 'xgboost':
                # For XGBoost, convert to DMatrix first
                dtest = xgb.DMatrix(X_test, label=y_test,
                                    feature_names=feature_names)
                y_pred = model.predict(dtest)
            elif model_type == 'keras':
                # For Keras models
                y_pred = model.predict(X_test)
                # Handle multi-dim output
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    raise ValueError(
                        "Multi-output regression not supported for evaluation")
            elif model_type == 'sklearn':
                # For scikit-learn models
                y_pred = model.predict(X_test)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Calculate regression metrics
            metrics = {}
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)

            # Error distribution
            errors = y_test - y_pred
            metrics['mean_error'] = np.mean(errors)
            metrics['error_std'] = np.std(errors)

            # Calculate feature importance if available
            metrics['feature_importance'] = self._get_feature_importance(
                model, model_type, feature_names)

            # Store in Redis if available
            if self.redis:
                try:
                    # Store main metrics (exclude large elements like feature_importance)
                    redis_metrics = {k: v for k, v in metrics.items()
                                     if k != 'feature_importance'}

                    model_key = f"model_evaluation:{model_type}:{uuid.uuid4()}"
                    self.redis.hset(model_key, mapping=redis_metrics)
                    # Expire after 7 days
                    self.redis.expire(model_key, 60*60*24*7)
                except Exception as e:
                    logger.error(
                        f"Error storing evaluation metrics in Redis: {str(e)}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating regression model: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self, model, model_type: str,
                                feature_names: List[str] = None) -> Dict[str, float]:
        """Extract feature importance from model if available"""
        try:
            importance = {}

            if model_type == 'xgboost':
                # Get feature importance map
                importance_map = model.get_score(importance_type='gain')
                if feature_names:
                    importance = {name: importance_map.get(
                        name, 0) for name in feature_names}
                else:
                    importance = importance_map

            elif model_type == 'sklearn':
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    if feature_names and len(imp) == len(feature_names):
                        importance = {name: float(
                            imp[i]) for i, name in enumerate(feature_names)}
                    else:
                        importance = {f"feature_{i}": float(
                            imp[i]) for i in range(len(imp))}

            # Normalize importance values if not empty
            if importance:
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v/total for k, v in importance.items()}

            return importance

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return {}


class LearningEngine:
    """
    Main learning engine that coordinates continual learning and model training
    """

    def __init__(self, redis_client=None, ml_engine=None, data_pipeline=None, use_gpu=True):
        """
        Initialize the learning engine

        Args:
            redis_client: Redis client for communication and caching
            ml_engine: Reference to the ML engine
            data_pipeline: Reference to the data pipeline
            use_gpu: Whether to use GPU acceleration
        """
        self.redis = redis_client
        self.ml_engine = ml_engine
        self.data_pipeline = data_pipeline
        self.use_gpu = use_gpu

        # Configuration
        self.config = {
            'models_dir': os.environ.get('MODELS_DIR', './models'),
            'data_dir': os.environ.get('DATA_DIR', './data'),
            'training_schedule': {
                'daily_update': os.environ.get('DAILY_UPDATE_TIME', '23:30'),
                'full_retrain': os.environ.get('FULL_RETRAIN_TIME', '00:30'),
                'drift_check': os.environ.get('DRIFT_CHECK_TIME', '12:00')
            },
            'update_window': int(os.environ.get('UPDATE_WINDOW_DAYS', '5')),
            'performance_threshold': float(os.environ.get('PERFORMANCE_THRESHOLD', '0.8')),
            'max_versions': int(os.environ.get('MAX_MODEL_VERSIONS', '5')),
            'drift_detection': {
                'enabled': os.environ.get('DRIFT_DETECTION_ENABLED', 'true').lower() == 'true',
                'auto_retrain': os.environ.get('DRIFT_AUTO_RETRAIN', 'true').lower() == 'true',
                'min_samples': int(os.environ.get('DRIFT_MIN_SAMPLES', '1000'))
            },
            'gpu': {
                'memory_limit_mb': int(os.environ.get('TF_CUDA_HOST_MEM_LIMIT_IN_MB', '16000')),
                'tensorrt_precision': os.environ.get('TENSORRT_PRECISION_MODE', 'FP16'),
                'mixed_precision': os.environ.get('TF_MIXED_PRECISION', 'true').lower() == 'true',
                'memory_growth': os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'true').lower() == 'true',
                'xla_optimization': os.environ.get('TF_XLA_FLAGS', '').find('auto_jit=2') != -1
            }
        }

        # Initialize GPU if enabled
        if self.use_gpu:
            self._initialize_gpu()

        # Create component instances
        self.model_registry = ModelRegistry(
            models_dir=self.config['models_dir'],
            redis_client=redis_client
        )

        self.model_deployer = ModelDeployer(
            model_registry=self.model_registry,
            redis_client=redis_client
        )

        self.performance_tracker = ModelPerformanceTracker(
            redis_client=redis_client
        )

        self.drift_detector = DriftDetector(
            redis_client=redis_client
        )

        self.model_evaluator = ModelEvaluator(
            redis_client=redis_client
        )

        # State
        self.running = False
        self.threads = []

        # Ensure directories exist
        os.makedirs(self.config['models_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)

        logger.info(
            "Learning Engine initialized with GPU acceleration" if self.use_gpu else "Learning Engine initialized (CPU mode)")

    def _initialize_gpu(self):
        """Initialize GPU environment for optimal performance"""
        try:
            # Check if TensorFlow is available
            if tf is not None:
                # Configure TensorFlow for GPU
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"Found {len(gpus)} GPU(s): {gpus}")

                    # Configure memory growth to prevent OOM errors
                    if self.config['gpu']['memory_growth']:
                        for gpu in gpus:
                            try:
                                tf.config.experimental.set_memory_growth(
                                    gpu, True)
                                logger.info(
                                    f"Enabled memory growth for {gpu.name}")
                            except Exception as e:
                                logger.warning(
                                    f"Error setting memory growth: {e}")

                    # Set memory limit if specified
                    if self.config['gpu']['memory_limit_mb'] > 0:
                        try:
                            tf.config.set_logical_device_configuration(
                                gpus[0],
                                [tf.config.LogicalDeviceConfiguration(
                                    memory_limit=self.config['gpu']['memory_limit_mb'] * 1024 * 1024
                                )]
                            )
                            logger.info(
                                f"Set memory limit to {self.config['gpu']['memory_limit_mb']}MB")
                        except Exception as e:
                            logger.warning(f"Error setting memory limit: {e}")

                    # Enable mixed precision if configured
                    if self.config['gpu']['mixed_precision']:
                        try:
                            from tensorflow.keras.mixed_precision import set_global_policy
                            set_global_policy('mixed_float16')
                            logger.info(
                                "Enabled mixed precision (float16) for TensorFlow")
                        except Exception as e:
                            logger.warning(
                                f"Could not set mixed precision policy: {e}")

                    # Log TensorFlow device placement for debugging
                    tf.debugging.set_log_device_placement(False)

                    # Test GPU availability
                    with tf.device('/GPU:0'):
                        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                        c = tf.matmul(a, b)
                        logger.info(
                            f"TensorFlow GPU test successful: {c.numpy()}")
                else:
                    logger.warning("No GPU detected by TensorFlow")
                    self.use_gpu = False

            # Check if CuPy is available
            if CUPY_AVAILABLE and cp is not None:
                try:
                    num_gpus = cp.cuda.runtime.getDeviceCount()
                    if num_gpus > 0:
                        # Get device properties
                        for i in range(num_gpus):
                            props = cp.cuda.runtime.getDeviceProperties(i)
                            name = props["name"].decode()
                            logger.info(f"CuPy GPU {i}: {name}")

                        # Set up memory pool for better performance
                        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                        logger.info("CuPy memory pool configured")

                        # Test CuPy
                        a = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
                        b = cp.array([[5, 6], [7, 8]], dtype=cp.float32)
                        c = cp.matmul(a, b)
                        logger.info(
                            f"CuPy GPU test successful: {cp.asnumpy(c)}")
                    else:
                        logger.warning("No GPU detected by CuPy")
                except Exception as e:
                    logger.error(f"Error initializing CuPy: {e}")

            # Check if TensorRT is available
            if TENSORRT_AVAILABLE and trt is not None:
                try:
                    # Log TensorRT availability
                    logger.info(
                        f"TensorRT is available for model optimization (precision: {self.config['gpu']['tensorrt_precision']})")
                except Exception as e:
                    logger.error(f"Error checking TensorRT: {e}")

            # Log GPU memory information
            self._log_gpu_memory()

        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.use_gpu = False

    def _log_gpu_memory(self):
        """Log current GPU memory usage"""
        try:
            if CUPY_AVAILABLE and cp is not None:
                free, total = cp.cuda.runtime.memGetInfo()
                used = total - free
                logger.info(
                    f"GPU Memory: {used/(1024**2):.2f}GB used, {free/(1024**2):.2f}GB free, {total/(1024**2):.2f}GB total")

                # Record in Prometheus if available
                if PROMETHEUS_AVAILABLE:
                    try:
                        device_id = cp.cuda.Device().id
                        device_props = cp.cuda.runtime.getDeviceProperties(
                            device_id)
                        device_name = device_props["name"].decode()

                        GPU_MEMORY_USAGE.labels(device=device_name).set(used)
                    except Exception as prom_e:
                        logger.warning(
                            f"Error recording GPU memory in Prometheus: {prom_e}")

                # Store in Redis if available
                if self.redis:
                    self.redis.hset(
                        "system:gpu", "memory_used_mb", int(used/(1024*1024)))
                    self.redis.hset(
                        "system:gpu", "memory_free_mb", int(free/(1024*1024)))
                    self.redis.hset(
                        "system:gpu", "memory_total_mb", int(total/(1024*1024)))
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")

    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent fragmentation"""
        try:
            # Clear TensorFlow memory
            if tf is not None:
                tf.keras.backend.clear_session()
                logger.info("TensorFlow session cleared")

            # Clear CuPy memory pool
            if CUPY_AVAILABLE and cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
                if hasattr(cp.cuda, 'pinned_memory_pool'):
                    cp.cuda.pinned_memory_pool.free_all_blocks()
                logger.info("CuPy memory pool cleared")

            # Force garbage collection
            import gc
            gc.collect()

            # Log memory after clearing
            self._log_gpu_memory()

            return True
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
            return False

    def start(self):
        """Start the learning engine"""
        if self.running:
            logger.warning("Learning engine already running")
            return

        self.running = True
        logger.info("Starting learning engine")

        # Schedule tasks
        self._schedule_tasks()

        # Start scheduler thread
        scheduler_thread = threading.Thread(
            target=self._scheduler_thread, daemon=True)
        scheduler_thread.start()
        self.threads.append(scheduler_thread)

        # Start periodic drift detection if enabled
        if self.config['drift_detection']['enabled']:
            drift_thread = threading.Thread(
                target=self._drift_detection_thread, daemon=True)
            drift_thread.start()
            self.threads.append(drift_thread)

        logger.info("Learning engine started")

    def stop(self):
        """Stop the learning engine"""
        if not self.running:
            logger.warning("Learning engine not running")
            return

        logger.info("Stopping learning engine")
        self.running = False

        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)

        # Clean up GPU resources if using GPU
        if self.use_gpu:
            logger.info("Cleaning up GPU resources")
            self._clear_gpu_memory()

            # Additional TensorFlow cleanup
            if tf is not None:
                try:
                    # Reset TensorFlow state
                    tf.keras.backend.clear_session()

                    # Force garbage collection
                    import gc
                    gc.collect()

                    logger.info("TensorFlow resources cleaned up")
                except Exception as e:
                    logger.error(
                        f"Error cleaning up TensorFlow resources: {e}")

        logger.info("Learning engine stopped successfully")

    def _schedule_tasks(self):
        """Schedule tasks for model updates and retraining"""
        # Schedule daily model updates
        schedule.every().day.at(self.config['training_schedule']['daily_update']).do(
            self.daily_model_update)

        # Schedule full model retraining
        schedule.every().day.at(self.config['training_schedule']['full_retrain']).do(
            self.full_model_retrain)

        # Schedule drift checks
        if self.config['drift_detection']['enabled']:
            schedule.every().day.at(self.config['training_schedule']['drift_check']).do(
                self.check_for_drift)

        logger.info("Scheduled learning engine tasks")

    def _scheduler_thread(self):
        """Thread to run scheduled tasks"""
        logger.info("Starting scheduler thread")

        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()

                # Sleep for a bit
                time.sleep(30)

            except Exception as e:
                logger.error(
                    f"Error in scheduler thread: {str(e)}", exc_info=True)
                time.sleep(60)  # Back off on error

    def _drift_detection_thread(self):
        """Thread for periodic drift detection"""
        logger.info("Starting drift detection thread")

        # Initial delay to let the system stabilize
        time.sleep(300)

        while self.running:
            try:
                # Check for new data to analyze for drift
                self._check_for_data_drift()

                # Sleep between checks (every 2 hours)
                time.sleep(7200)

            except Exception as e:
                logger.error(
                    f"Error in drift detection thread: {str(e)}", exc_info=True)
                time.sleep(300)  # Back off on error

    def daily_model_update(self):
        """Perform daily incremental update of models"""
        logger.info("Starting daily model update")

        try:
            # Load recent data
            recent_data = self._load_recent_data()

            if recent_data is None or len(recent_data) == 0:
                logger.error("Failed to load recent data for model updating")
                return False

            # Get currently deployed models
            deployed_models = self.model_deployer.get_all_deployed_models()

            # Update each model
            updated_models = {}

            for model_name, version in deployed_models.items():
                try:
                    # Skip models that shouldn't be updated incrementally
                    if model_name == 'market_regime':
                        logger.info(
                            f"Skipping incremental update for {model_name} model")
                        continue

                    # Load the model
                    model, metadata = self.model_registry.load_model(
                        model_name, version)

                    if model is None:
                        logger.warning(
                            f"Could not load {model_name} model for updating")
                        continue

                    # Update the model
                    success, updated_model, metrics = self._update_model(
                        model_name, model, metadata.get('model_type', 'xgboost'), recent_data)

                    if success and updated_model:
                        # Register the updated model
                        new_version = self.model_registry.register_model(
                            model_name=model_name,
                            model=updated_model,
                            model_type=metadata.get('model_type', 'xgboost'),
                            metrics=metrics,
                            feature_names=metadata.get('feature_names'),
                            hyperparams=metadata.get('hyperparams')
                        )

                        # Track performance
                        self.performance_tracker.log_metrics(
                            model_name, new_version, metrics)

                        # Compare to baseline
                        if 'metrics' in metadata:
                            diff = self.performance_tracker.compare_to_baseline(
                                model_name, new_version)
                            logger.info(
                                f"Model {model_name} v{new_version} performance diff: {diff}")

                        # Deploy if better than current
                        if self._should_deploy_model(model_name, new_version, metrics):
                            self.model_deployer.deploy_model(
                                model_name, new_version)
                            logger.info(
                                f"Deployed new version of {model_name}: v{new_version}")

                        updated_models[model_name] = new_version
                        logger.info(
                            f"Successfully updated {model_name} model to v{new_version}")
                    else:
                        logger.warning(f"Failed to update {model_name} model")

                except Exception as e:
                    logger.error(
                        f"Error updating {model_name} model: {str(e)}", exc_info=True)

            logger.info(
                f"Daily model update completed: {len(updated_models)} models updated")
            return True

        except Exception as e:
            logger.error(
                f"Error in daily model update: {str(e)}", exc_info=True)
            return False

    def full_model_retrain(self):
        """Perform full retraining of all models"""
        logger.info("Starting full model retraining")

        try:
            # Use ML engine for full retraining if available
            if self.ml_engine:
                success = self.ml_engine.train_all_models()

                if success:
                    # Register newly trained models
                    self._register_ml_engine_models()
                    logger.info("Full model retraining completed successfully")
                    return True
                else:
                    logger.error("Full model retraining with ML engine failed")
                    return False
            else:
                # Fallback to internal training logic
                return self._train_all_models_internally()

        except Exception as e:
            logger.error(
                f"Error in full model retraining: {str(e)}", exc_info=True)
            return False

    def _register_ml_engine_models(self):
        """Register models trained by the ML engine"""
        try:
            if not self.ml_engine:
                return False

            # Check models directory for new models
            models_dir = self.config['models_dir']
            model_types = {
                'signal_detection': 'xgboost',
                'price_prediction': 'keras',
                'risk_assessment': 'sklearn',
                'exit_strategy': 'xgboost',
                'market_regime': 'sklearn'
            }

            for model_name, model_type in model_types.items():
                # Determine file extension based on model type
                if model_type == 'xgboost':
                    ext = 'xgb'
                elif model_type == 'keras':
                    ext = 'h5'
                elif model_type == 'sklearn':
                    ext = 'pkl'
                else:
                    ext = 'model'

                # Check if model file exists
                model_path = os.path.join(
                    models_dir, f"{model_name}_model.{ext}")
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}")
                    continue

                # Load model and metrics
                try:
                    if model_type == 'xgboost':
                        model = xgb.Booster()
                        model.load_model(model_path)
                    elif model_type == 'keras':
                        model = tf_load_model(model_path)
                    elif model_type == 'sklearn':
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        logger.warning(f"Unsupported model type: {model_type}")
                        continue

                    # Try to load metrics
                    metrics_path = os.path.join(
                        models_dir, f"{model_name}_metrics.json")
                    metrics = {}
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)

                    # Register the model
                    new_version = self.model_registry.register_model(
                        model_name=model_name,
                        model=model,
                        model_type=model_type,
                        metrics=metrics
                    )

                    # Deploy the model
                    self.model_deployer.deploy_model(model_name, new_version)
                    logger.info(
                        f"Registered and deployed {model_name} v{new_version} from ML engine")

                except Exception as e:
                    logger.error(
                        f"Error registering {model_name} model: {str(e)}")

            return True

        except Exception as e:
            logger.error(f"Error registering ML engine models: {str(e)}")
            return False

    def _train_all_models_internally(self):
        """Train all models using internal training logic"""
        # This would be a fallback implementation if ML engine is not available
        # In a real system, you would implement model-specific training logic here
        logger.warning(
            "Internal model training not implemented, requires ML engine")
        return False

    def _load_recent_data(self):
        """Load recent data for model updates"""
        try:
            if not self.data_pipeline:
                logger.error(
                    "Data pipeline not available for loading recent data")
                return None

            # Calculate date ranges
            end_date = datetime.datetime.now()
            start_date = end_date - \
                datetime.timedelta(days=self.config['update_window'])

            logger.info(f"Loading recent data from {start_date} to {end_date}")

            # Get active tickers
            active_tickers = self._get_active_tickers()

            if not active_tickers:
                logger.warning("No active tickers for recent data loading")
                return None

            logger.info(
                f"Loading recent data for {len(active_tickers)} tickers")

            # Load price data
            price_data = self.data_pipeline.load_price_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe='1m'
            )

            # Load options data
            options_data = self.data_pipeline.load_options_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date
            )

            # Load market data
            market_data = self.data_pipeline.load_market_data(
                start_date=start_date,
                end_date=end_date,
                symbols=['SPY', 'QQQ', 'IWM', 'VIX']
            )

            # Combine data
            combined_data = self._prepare_training_data(
                price_data=price_data,
                options_data=options_data,
                market_data=market_data
            )

            logger.info(
                f"Loaded {len(combined_data) if combined_data is not None else 0} recent samples for model updating")
            return combined_data

        except Exception as e:
            logger.error(f"Error loading recent data: {str(e)}", exc_info=True)
            return None

    def _get_active_tickers(self):
        """Get list of active tickers for training"""
        try:
            # Try to get from Redis if available
            if self.redis:
                active_tickers_str = self.redis.get("system:active_tickers")
                if active_tickers_str:
                    return json.loads(active_tickers_str)

            # Fallback to default tickers from configuration if ML engine is not available
            if not self.ml_engine:
                default_tickers = config['stock_selection']['universe']['default_tickers']
                logger.info(
                    f"Using default tickers from configuration: {default_tickers}")
                return default_tickers

            # Delegate to ML engine if available
            return self.ml_engine.get_active_tickers()

        except Exception as e:
            logger.error(f"Error getting active tickers: {str(e)}")
            # Return safe defaults
            return ['SPY', 'QQQ', 'IWM']

    def _prepare_training_data(self, price_data, options_data, market_data):
        """Prepare training data from raw inputs"""
        try:
            if self.ml_engine:
                # Use ML engine's data preparation if available
                return self.ml_engine.prepare_training_data(
                    price_data=price_data,
                    options_data=options_data,
                    market_data=market_data
                )
            else:
                # Simple fallback implementation
                logger.warning(
                    "ML engine not available for data preparation, using simple implementation")

                if price_data is None or len(price_data) == 0:
                    return None

                # Convert to DataFrame if it's not already
                if not isinstance(price_data, pd.DataFrame):
                    if isinstance(price_data, dict):
                        # Convert dictionary of ticker -> data to a single DataFrame
                        combined = []
                        for ticker, data in price_data.items():
                            if isinstance(data, pd.DataFrame):
                                data['ticker'] = ticker
                                combined.append(data)

                        if combined:
                            price_data = pd.concat(combined, ignore_index=True)
                        else:
                            return None
                    else:
                        logger.error(
                            f"Unsupported price data format: {type(price_data)}")
                        return None

                # Very basic feature generation
                # In a real system, this would be much more sophisticated
                return price_data

        except Exception as e:
            logger.error(
                f"Error preparing training data: {str(e)}", exc_info=True)
            return None

    def _update_model(self, model_name, model, model_type, recent_data):
        """Update a specific model with recent data"""
        try:
            logger.info(f"Updating {model_name} model of type {model_type}")

            # Prepare data based on model type
            features, target = self._prepare_model_specific_data(
                model_name, recent_data)

            if features is None or target is None or len(features) == 0 or len(target) == 0:
                logger.warning(
                    f"No valid data for updating {model_name} model")
                return False, None, {}

            # Apply different update approaches based on model type
            if model_type == 'xgboost':
                return self._update_xgboost_model(model_name, model, features, target)
            elif model_type == 'keras':
                return self._update_keras_model(model_name, model, features, target)
            elif model_type == 'sklearn':
                return self._update_sklearn_model(model_name, model, features, target)
            else:
                logger.warning(
                    f"Unsupported model type for incremental update: {model_type}")
                return False, None, {}

        except Exception as e:
            logger.error(
                f"Error updating {model_name} model: {str(e)}", exc_info=True)
            return False, None, {}

    def _prepare_model_specific_data(self, model_name, data):
        """Prepare data specifically for each model type"""
        if data is None:
            return None, None

        try:
            if self.ml_engine:
                # Use ML engine's data preparation if available
                if model_name == 'signal_detection':
                    return self.ml_engine.prepare_signal_detection_data(data)
                elif model_name == 'price_prediction':
                    return self.ml_engine.prepare_price_prediction_data(data)
                elif model_name == 'risk_assessment':
                    return self.ml_engine.prepare_risk_assessment_data(data)
                elif model_name == 'exit_strategy':
                    return self.ml_engine.prepare_exit_strategy_data(data)
                elif model_name == 'market_regime':
                    return self.ml_engine.prepare_market_regime_data(data)

            # Simple fallback implementation
            logger.warning(
                f"ML engine not available for {model_name} data preparation, using simple implementation")

            # Basic feature extraction - this is just a placeholder
            # In a real system, this would be much more sophisticated and model-specific
            if isinstance(data, pd.DataFrame):
                # For classification models
                if model_name in ['signal_detection', 'risk_assessment']:
                    # Simple example: use price-based features to predict direction
                    numeric_cols = data.select_dtypes(
                        include=['float64', 'int64']).columns
                    X = data[numeric_cols].fillna(0).values

                    # Create a simple binary target (up/down movement)
                    if 'price_close' in data.columns and 'price_open' in data.columns:
                        y = (data['price_close'] >
                             data['price_open']).astype(int).values
                    else:
                        # Fallback to random data for demonstration
                        y = np.random.randint(0, 2, size=len(data))

                    return X, y

                # For regression models
                elif model_name in ['price_prediction', 'exit_strategy']:
                    numeric_cols = data.select_dtypes(
                        include=['float64', 'int64']).columns
                    X = data[numeric_cols].fillna(0).values

                    # Create a simple regression target (next price)
                    if 'price_close' in data.columns:
                        # Shift to predict next value
                        y = data['price_close'].shift(
                            -1).fillna(method='ffill').values
                    else:
                        # Fallback to random data for demonstration
                        y = np.random.randn(len(data))

                    return X, y

                # For clustering models
                elif model_name == 'market_regime':
                    numeric_cols = data.select_dtypes(
                        include=['float64', 'int64']).columns
                    X = data[numeric_cols].fillna(0).values

                    # No target for unsupervised learning
                    return X, None

            logger.error(
                f"Could not prepare data for {model_name}, unsupported data format")
            return None, None

        except Exception as e:
            logger.error(
                f"Error preparing model-specific data for {model_name}: {str(e)}", exc_info=True)
            return None, None

    def _update_xgboost_model(self, model_name, model, features, target):
        """Update an XGBoost model incrementally with GPU acceleration"""
        try:
            # Clear GPU memory before training if using GPU
            if self.use_gpu:
                self._clear_gpu_memory()
                logger.info(
                    f"Cleared GPU memory before training {model_name} XGBoost model")

            # Create DMatrix
            dtrain = xgb.DMatrix(features, label=target)

            # Split data for validation
            val_split = 0.2
            split_idx = int(len(features) * (1 - val_split))

            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = target[:split_idx], target[split_idx:]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Configuration for incremental update
            params = {
                'objective': 'binary:logistic' if model_name == 'signal_detection' else 'reg:squarederror',
                'eval_metric': ['logloss', 'auc'] if model_name == 'signal_detection' else ['rmse', 'mae'],
                'eta': 0.01,  # Lower learning rate for incremental updates
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

            # Add GPU acceleration if available
            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                params['predictor'] = 'gpu_predictor'
                logger.info(
                    f"Using GPU acceleration for {model_name} XGBoost model")

            # Prepare evaluation list
            eval_list = [(dtrain, 'train'), (dval, 'validation')]

            # Reduced number of rounds for incremental update
            num_rounds = 50

            # Log start of training
            start_time = time.time()
            logger.info(
                f"Starting incremental training of {model_name} XGBoost model with {len(X_train)} samples")

            # Update model with early stopping
            updated_model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_rounds,
                evals=eval_list,
                early_stopping_rounds=5,
                xgb_model=model,
                verbose_eval=10
            )

            # Log training completion
            training_time = time.time() - start_time
            logger.info(
                f"XGBoost training completed in {training_time:.2f} seconds")

            # Log GPU memory after training
            if self.use_gpu:
                self._log_gpu_memory()

            # Evaluate updated model on validation set
            y_pred = updated_model.predict(dval)

            metrics = {}
            metrics['training_time'] = training_time

            if model_name == 'signal_detection':
                # Binary classification metrics
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics['accuracy'] = float(
                    accuracy_score(y_val, y_pred_binary))
                metrics['precision'] = float(precision_score(
                    y_val, y_pred_binary, zero_division=0))
                metrics['recall'] = float(recall_score(
                    y_val, y_pred_binary, zero_division=0))
                metrics['f1'] = float(
                    f1_score(y_val, y_pred_binary, zero_division=0))

                # Calculate AUC if possible
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics['auc'] = float(roc_auc_score(y_val, y_pred))
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")

                logger.info(
                    f"Updated {model_name} XGBoost model metrics: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, time={training_time:.2f}s")

                # Only accept if above threshold
                if metrics['accuracy'] < self.config['performance_threshold']:
                    logger.warning(
                        f"Updated {model_name} model accuracy below threshold: {metrics['accuracy']:.4f} < {self.config['performance_threshold']}"
                    )
                    return False, None, metrics
            else:
                # Regression metrics
                metrics['mse'] = float(mean_squared_error(y_val, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y_val, y_pred))

                # Calculate R² score
                metrics['r2'] = float(r2_score(y_val, y_pred))

                logger.info(
                    f"Updated {model_name} XGBoost model RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}, time={training_time:.2f}s")

            # Get feature importance
            importance = updated_model.get_score(importance_type='gain')
            metrics['feature_importance'] = {
                k: float(v) for k, v in importance.items()}

            # Get training history from model
            if hasattr(updated_model, 'eval_result'):
                metrics['training_history'] = {}
                for metric in updated_model.eval_result.keys():
                    for dataset in updated_model.eval_result[metric].keys():
                        key = f"{dataset}_{metric}"
                        metrics['training_history'][key] = [
                            float(x) for x in updated_model.eval_result[metric][dataset]]

            # Clear GPU memory after training
            if self.use_gpu:
                self._clear_gpu_memory()

            return True, updated_model, metrics

        except Exception as e:
            logger.error(
                f"Error updating XGBoost model {model_name}: {str(e)}", exc_info=True)

            # Clean up GPU memory on error
            if self.use_gpu:
                self._clear_gpu_memory()

            return False, None, {}

    def _update_keras_model(self, model_name, model, features, target):
        """Update a Keras model incrementally with GPU acceleration"""
        try:
            # Clear GPU memory before training if using GPU
            if self.use_gpu:
                self._clear_gpu_memory()
                logger.info(
                    f"Cleared GPU memory before training {model_name} model")

            # Configure for incremental training
            lr = 0.001  # Low learning rate for fine-tuning
            epochs = 5   # Few epochs for incremental update
            batch_size = 64 if self.use_gpu else 32  # Larger batch size for GPU

            # Enable mixed precision for better performance on GPU
            if self.use_gpu and self.config['gpu']['mixed_precision']:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info(
                        f"Enabled mixed precision for {model_name} model training")
                except Exception as e:
                    logger.warning(
                        f"Could not set mixed precision policy: {e}")

            # Compile model with custom parameters for incremental training
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy' if model_name == 'signal_detection' else 'mse',
                metrics=['accuracy', 'AUC'] if model_name == 'signal_detection' else [
                    'mae', 'mse']
            )

            # Create callbacks for training
            callbacks = [
                # Early stopping to prevent overfitting
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                ),
                # Reduce learning rate when plateauing
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=1,
                    min_lr=0.0001
                )
            ]

            # Add TensorBoard logging if in debug mode
            if logger.level <= logging.DEBUG:
                log_dir = os.path.join(self.config['models_dir'], 'logs', model_name,
                                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    profile_batch='500,520' if self.use_gpu else 0  # Only profile on GPU
                )
                callbacks.append(tensorboard_callback)
                logger.debug(f"TensorBoard logging enabled at {log_dir}")

            # Split data for validation
            val_split = 0.2
            split_idx = int(len(features) * (1 - val_split))

            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = target[:split_idx], target[split_idx:]

            # Convert inputs to TensorFlow tensors for better performance
            X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
            X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

            # Log start of training
            start_time = time.time()
            logger.info(
                f"Starting incremental training of {model_name} model with {len(X_train)} samples")

            # Train with GPU acceleration if available
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                history = model.fit(
                    X_train_tensor, y_train_tensor,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_tensor, y_val_tensor),
                    callbacks=callbacks,
                    verbose=1 if logger.level <= logging.INFO else 0
                )

            # Log training completion
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")

            # Log GPU memory after training
            if self.use_gpu:
                self._log_gpu_memory()

            # Evaluate updated model
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                evaluation = model.evaluate(
                    X_val_tensor, y_val_tensor, verbose=0)

            metrics = {}
            metrics['training_time'] = training_time

            if model_name == 'signal_detection':
                # Classification metrics
                with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                    y_pred_proba = model.predict(
                        X_val_tensor, batch_size=batch_size)

                y_pred = (y_pred_proba > 0.5).astype(int)
                metrics['loss'] = float(evaluation[0])
                metrics['accuracy'] = float(evaluation[1])

                # Add AUC if available (index 2)
                if len(evaluation) > 2:
                    metrics['auc'] = float(evaluation[2])

                metrics['precision'] = float(
                    precision_score(y_val, y_pred, zero_division=0))
                metrics['recall'] = float(
                    recall_score(y_val, y_pred, zero_division=0))
                metrics['f1'] = float(f1_score(y_val, y_pred, zero_division=0))

                logger.info(
                    f"Updated {model_name} model metrics: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, time={training_time:.2f}s")

                # Only accept if above threshold
                if metrics['accuracy'] < self.config['performance_threshold']:
                    logger.warning(
                        f"Updated {model_name} model accuracy below threshold: {metrics['accuracy']:.4f} < {self.config['performance_threshold']}"
                    )
                    return False, None, metrics
            else:
                # Regression metrics
                metrics['loss'] = float(evaluation[0])
                metrics['mae'] = float(evaluation[1])

                # Add MSE if available (index 2)
                if len(evaluation) > 2:
                    metrics['mse'] = float(evaluation[2])
                else:
                    # Calculate MSE manually
                    with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                        y_pred = model.predict(
                            X_val_tensor, batch_size=batch_size)
                    metrics['mse'] = float(mean_squared_error(y_val, y_pred))

                metrics['rmse'] = float(np.sqrt(metrics['mse']))

                logger.info(
                    f"Updated {model_name} model RMSE: {metrics['rmse']:.6f}, time={training_time:.2f}s")

            # Add training history
            metrics['training_history'] = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }

            # Add additional metrics if available
            for metric_name in ['accuracy', 'mae', 'mse', 'auc']:
                if metric_name in history.history:
                    metrics['training_history'][metric_name] = [
                        float(x) for x in history.history[metric_name]]
                if f'val_{metric_name}' in history.history:
                    metrics['training_history'][f'val_{metric_name}'] = [
                        float(x) for x in history.history[f'val_{metric_name}']]

            # Optimize with TensorRT if available and configured
            if self.use_gpu and TENSORRT_AVAILABLE and trt is not None:
                try:
                    logger.info(f"Optimizing {model_name} model with TensorRT")

                    # Save model to temporary file
                    temp_model_path = os.path.join(
                        self.config['models_dir'], f"{model_name}_temp.h5")
                    model.save(temp_model_path)

                    # Convert to SavedModel format (required for TensorRT)
                    saved_model_path = os.path.join(
                        self.config['models_dir'], f"{model_name}_saved_model")
                    tf.keras.models.save_model(model, saved_model_path)

                    # Configure TensorRT conversion
                    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                        max_workspace_size_bytes=8000000000,  # 8GB
                        precision_mode=self.config['gpu']['tensorrt_precision'],
                        maximum_cached_engines=100
                    )

                    # Create TensorRT converter
                    converter = trt.TrtGraphConverterV2(
                        input_saved_model_dir=saved_model_path,
                        conversion_params=conversion_params
                    )

                    # Convert model
                    converter.convert()

                    # Save converted model
                    trt_model_path = os.path.join(
                        self.config['models_dir'], f"{model_name}_trt")
                    converter.save(trt_model_path)

                    # Add TensorRT optimization info to metrics
                    metrics['tensorrt_optimized'] = True
                    metrics['tensorrt_precision'] = self.config['gpu']['tensorrt_precision']

                    logger.info(
                        f"Successfully optimized {model_name} model with TensorRT")

                    # Return the original model (TensorRT model is saved for inference)
                    # We don't return the TensorRT model directly as it's not compatible with further training
                except Exception as e:
                    logger.error(f"Error optimizing model with TensorRT: {e}")
                    logger.info("Falling back to non-optimized model")

            # Clear GPU memory after training
            if self.use_gpu:
                self._clear_gpu_memory()

            return True, model, metrics

        except Exception as e:
            logger.error(
                f"Error updating Keras model {model_name}: {str(e)}", exc_info=True)

            # Clean up GPU memory on error
            if self.use_gpu:
                self._clear_gpu_memory()

            return False, None, {}

    def _update_sklearn_model(self, model_name, model, features, target):
        """Update a scikit-learn model"""
        try:
            # For most scikit-learn models, incremental updates are not directly supported
            # We'll need to use a partial_fit method if available, or retrain the model

            metrics = {}

            if hasattr(model, 'partial_fit'):
                # For models that support partial_fit (e.g., SGDClassifier)
                model.partial_fit(features, target)
                logger.info(f"Updated {model_name} model using partial_fit")

                # Evaluate
                y_pred = model.predict(features)

                if model_name in ['signal_detection', 'risk_assessment']:
                    # Classification metrics
                    metrics['accuracy'] = float(accuracy_score(target, y_pred))
                    metrics['precision'] = float(
                        precision_score(target, y_pred, zero_division=0))
                    metrics['recall'] = float(recall_score(
                        target, y_pred, zero_division=0))
                    metrics['f1'] = float(
                        f1_score(target, y_pred, zero_division=0))

                    logger.info(
                        f"Updated {model_name} model metrics: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

                    # Only accept if above threshold
                    if metrics['accuracy'] < self.config['performance_threshold']:
                        logger.warning(
                            f"Updated {model_name} model accuracy below threshold: {metrics['accuracy']:.4f} < {self.config['performance_threshold']}"
                        )
                        return False, None, metrics
                else:
                    # Regression metrics
                    metrics['mse'] = float(mean_squared_error(target, y_pred))
                    metrics['rmse'] = float(np.sqrt(metrics['mse']))
                    metrics['mae'] = float(mean_absolute_error(target, y_pred))

                    logger.info(
                        f"Updated {model_name} model RMSE: {metrics['rmse']:.6f}")

                return True, model, metrics

            elif model_name == 'market_regime':
                # For clustering models like KMeans, we need special handling
                if isinstance(model, KMeans):
                    # Initialize with the previous centroids
                    prev_centroids = model.cluster_centers_
                    n_clusters = len(prev_centroids)

                    # Create a new model with the same parameters but initialized with previous centroids
                    new_model = KMeans(
                        n_clusters=n_clusters,
                        init=prev_centroids,
                        n_init=1,
                        max_iter=100
                    )

                    # Fit on new data
                    new_model.fit(features)

                    # Calculate metrics
                    metrics['inertia'] = float(new_model.inertia_)
                    metrics['centroid_shift'] = float(np.mean(np.linalg.norm(
                        new_model.cluster_centers_ - prev_centroids, axis=1)))

                    logger.info(
                        f"Updated {model_name} model with inertia: {metrics['inertia']:.4f}")

                    return True, new_model, metrics
                else:
                    logger.warning(
                        f"Unsupported model type for {model_name} update: {type(model)}")
                    return False, None, {}
            else:
                # For models that don't support partial_fit, we need to retrain
                # This isn't ideal for incremental learning, but some models require it

                # Create a new instance with the same parameters
                params = model.get_params()
                new_model = model.__class__(**params)

                # Train on new data
                new_model.fit(features, target)

                # Evaluate
                y_pred = new_model.predict(features)

                if model_name in ['signal_detection', 'risk_assessment']:
                    # Classification metrics
                    metrics['accuracy'] = float(accuracy_score(target, y_pred))
                    metrics['precision'] = float(
                        precision_score(target, y_pred, zero_division=0))
                    metrics['recall'] = float(recall_score(
                        target, y_pred, zero_division=0))
                    metrics['f1'] = float(
                        f1_score(target, y_pred, zero_division=0))

                    logger.info(
                        f"Retrained {model_name} model metrics: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

                    # Only accept if above threshold
                    if metrics['accuracy'] < self.config['performance_threshold']:
                        logger.warning(
                            f"Retrained {model_name} model accuracy below threshold: {metrics['accuracy']:.4f} < {self.config['performance_threshold']}"
                        )
                        return False, None, metrics
                else:
                    # Regression metrics
                    metrics['mse'] = float(mean_squared_error(target, y_pred))
                    metrics['rmse'] = float(np.sqrt(metrics['mse']))
                    metrics['mae'] = float(mean_absolute_error(target, y_pred))

                    logger.info(
                        f"Retrained {model_name} model RMSE: {metrics['rmse']:.6f}")

                return True, new_model, metrics

        except Exception as e:
            logger.error(
                f"Error updating scikit-learn model {model_name}: {str(e)}", exc_info=True)
            return False, None, {}

    def _should_deploy_model(self, model_name, version, metrics):
        """Determine if a new model version should be deployed"""
        try:
            # Get current deployed version
            current_version = self.model_deployer.deployed_models.get(
                model_name)

            # If no model is currently deployed, deploy this one
            if current_version is None:
                logger.info(
                    f"No current {model_name} model deployed, deploying v{version}")
                return True

            # Load current model's metrics
            current_metrics = None
            try:
                if self.redis:
                    metrics_data = self.redis.hgetall(
                        f"model:metrics:{model_name}:{current_version}")
                    if metrics_data:
                        current_metrics = {k: float(v) for k, v in metrics_data.items()
                                           if k not in ['timestamp', 'version']}

                if not current_metrics:
                    # Try to get from registry
                    _, metadata = self.model_registry.load_model(
                        model_name, current_version)
                    if metadata and 'metrics' in metadata:
                        current_metrics = metadata['metrics']
            except Exception as e:
                logger.error(f"Error getting current model metrics: {str(e)}")

            if not current_metrics:
                logger.warning(
                    f"No metrics for current {model_name} v{current_version}, deploying new version")
                return True

            # For classification models
            if model_name in ['signal_detection', 'risk_assessment']:
                # Compare key metrics like accuracy, f1
                for metric in ['accuracy', 'f1']:
                    if metric in metrics and metric in current_metrics:
                        # If new model is better by at least 1%
                        if metrics[metric] >= current_metrics[metric] * 1.01:
                            logger.info(
                                f"New {model_name} v{version} is better on {metric}: {metrics[metric]:.4f} vs {current_metrics[metric]:.4f}")
                            return True

            # For regression models
            elif model_name in ['price_prediction', 'exit_strategy']:
                # For error metrics like RMSE, lower is better
                for metric in ['rmse', 'mae']:
                    if metric in metrics and metric in current_metrics:
                        # If new model error is lower by at least 1%
                        if metrics[metric] <= current_metrics[metric] * 0.99:
                            logger.info(
                                f"New {model_name} v{version} is better on {metric}: {metrics[metric]:.4f} vs {current_metrics[metric]:.4f}")
                            return True

            # For market regime model, special case
            elif model_name == 'market_regime':
                # Lower inertia is better
                if 'inertia' in metrics and 'inertia' in current_metrics:
                    if metrics['inertia'] <= current_metrics['inertia'] * 0.95:
                        logger.info(
                            f"New {model_name} v{version} has better inertia: {metrics['inertia']:.4f} vs {current_metrics['inertia']:.4f}")
                        return True

            logger.info(
                f"New {model_name} v{version} not significantly better than current v{current_version}, not deploying")
            return False

        except Exception as e:
            logger.error(
                f"Error in deployment decision for {model_name} v{version}: {str(e)}")
            # Conservative approach: don't deploy on error
            return False

    def check_for_drift(self):
        """Scheduled task to check for drift in data and model performance"""
        logger.info("Performing scheduled drift check")

        try:
            # Get recent data
            recent_data = self._load_recent_data()

            if recent_data is None or len(recent_data) == 0:
                logger.error("Failed to load recent data for drift detection")
                return False

            # Check for data drift for each model type
            drift_detected = self._check_data_drift_for_all_models(recent_data)

            # Check for model performance drift
            perf_drift = self._check_model_performance_drift()

            # Combine results
            any_drift = drift_detected or perf_drift

            # If drift detected and auto-retrain is enabled, trigger retraining
            if any_drift and self.config['drift_detection']['auto_retrain']:
                logger.info("Drift detected, triggering full model retraining")
                self.full_model_retrain()

            return any_drift

        except Exception as e:
            logger.error(
                f"Error in scheduled drift check: {str(e)}", exc_info=True)
            return False

    def _check_data_drift_for_all_models(self, recent_data):
        """Check for data drift for all model types"""
        try:
            any_drift_detected = False

            # Get models to check
            deployed_models = self.model_deployer.get_all_deployed_models()

            for model_name in deployed_models:
                # Prepare model-specific data
                features, _ = self._prepare_model_specific_data(
                    model_name, recent_data)

                if features is None or len(features) == 0:
                    logger.warning(
                        f"No features for drift detection for {model_name}")
                    continue

                # Convert to DataFrame for drift detection
                if not isinstance(features, pd.DataFrame):
                    # Convert numpy array to DataFrame
                    features_df = pd.DataFrame(features)
                else:
                    features_df = features

                # Skip if too few samples
                if len(features_df) < self.config['drift_detection']['min_samples']:
                    logger.info(
                        f"Too few samples for drift detection for {model_name}: {len(features_df)} < {self.config['drift_detection']['min_samples']}")
                    continue

                # Detect drift
                drift_results = self.drift_detector.detect_drift(
                    f"model:{model_name}", features_df)

                if drift_results.get('drift_detected', False):
                    logger.warning(f"Data drift detected for {model_name}")
                    any_drift_detected = True

                    # Log detailed drift results
                    drift_features = drift_results.get('features', {})
                    drifted_features = [f for f, v in drift_features.items()
                                        if isinstance(v, dict) and v.get('drift_detected', False)]

                    logger.info(
                        f"Drifted features for {model_name}: {drifted_features}")
                else:
                    logger.info(f"No data drift detected for {model_name}")

            return any_drift_detected

        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}", exc_info=True)
            return False

    def _check_model_performance_drift(self):
        """Check for drift in model performance metrics"""
        try:
            any_drift_detected = False

            # Get deployed models
            deployed_models = self.model_deployer.get_all_deployed_models()

            for model_name, version in deployed_models.items():
                # Skip if we don't have baseline metrics
                if model_name not in self.performance_tracker.baseline_metrics:
                    logger.warning(
                        f"No baseline metrics for {model_name}, skipping performance drift check")
                    continue

                # Get current metrics
                current_metrics = None
                try:
                    if self.redis:
                        metrics_data = self.redis.hgetall(
                            f"model:metrics:{model_name}:{version}")
                        if metrics_data:
                            current_metrics = {k: float(v) for k, v in metrics_data.items()
                                               if k not in ['timestamp', 'version']}

                    if not current_metrics:
                        # Try to get from registry
                        _, metadata = self.model_registry.load_model(
                            model_name, version)
                        if metadata and 'metrics' in metadata:
                            current_metrics = metadata['metrics']
                except Exception as e:
                    logger.error(
                        f"Error getting current model metrics: {str(e)}")

                if not current_metrics:
                    logger.warning(
                        f"No metrics for {model_name} v{version}, skipping performance drift check")
                    continue

                # Determine model type
                model_type = 'classification' if model_name in [
                    'signal_detection', 'risk_assessment'] else 'regression'

                # Check for performance drift
                drift_results = self.drift_detector.detect_model_performance_drift(
                    model_name,
                    current_metrics,
                    self.performance_tracker.baseline_metrics[model_name],
                    model_type
                )

                if drift_results.get('drift_detected', False):
                    logger.warning(
                        f"Performance drift detected for {model_name} v{version}")
                    any_drift_detected = True

                    # Log detailed drift results
                    drift_metrics = drift_results.get('metrics', {})
                    for metric, details in drift_metrics.items():
                        if isinstance(details, dict) and details.get('drift_detected', False):
                            logger.info(
                                f"Drift in {metric} for {model_name}: {details.get('current', 0):.4f} vs baseline {details.get('baseline', 0):.4f} (change: {details.get('pct_change', 0):.2f}%)")
                else:
                    logger.info(
                        f"No performance drift detected for {model_name} v{version}")

            return any_drift_detected

        except Exception as e:
            logger.error(
                f"Error checking performance drift: {str(e)}", exc_info=True)
            return False

    def _check_for_data_drift(self):
        """Periodic non-scheduled check for data drift"""
        try:
            # Get latest market data
            if not self.data_pipeline:
                return

            # Get market data for the last few hours
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(hours=6)

            market_data = self.data_pipeline.load_market_data(
                start_date=start_date,
                end_date=end_date,
                symbols=['SPY', 'QQQ', 'IWM', 'VIX']
            )

            if market_data is None or len(market_data) == 0:
                return

            # Check for regime change
            regime_change = self._detect_market_regime_change(market_data)

            if regime_change:
                logger.warning("Market regime change detected")

                # Store event in Redis
                if self.redis:
                    self.redis.set("system:regime_change", time.time())
                    self.redis.publish("system:events", json.dumps({
                        "type": "regime_change",
                        "timestamp": time.time(),
                        "details": regime_change
                    }))

                # Trigger drift check
                self.check_for_drift()

        except Exception as e:
            logger.error(f"Error in periodic drift check: {str(e)}")

    def _detect_market_regime_change(self, market_data):
        """Detect change in market regime from market data"""
        try:
            # This implementation would use the market_regime model
            # to detect shifts in market conditions

            # Get the current market regime model
            model, metadata = self.model_deployer.get_deployed_model(
                'market_regime')

            if model is None:
                logger.warning("No market regime model deployed")
                return None

            # Prepare features for regime detection
            features, _ = self._prepare_model_specific_data(
                'market_regime', market_data)

            if features is None or len(features) == 0:
                return None

            # Make predictions
            if isinstance(model, KMeans):
                regimes = model.predict(features)

                # Check if the regime has changed
                current_regime = regimes[-1]
                prev_regime = regimes[0] if len(
                    regimes) > 1 else current_regime

                if current_regime != prev_regime:
                    # Regime change detected
                    return {
                        "previous_regime": int(prev_regime),
                        "current_regime": int(current_regime),
                        "timestamp": time.time()
                    }

            return None

        except Exception as e:
            logger.error(f"Error detecting market regime change: {str(e)}")
            return None

    def create_model_challenger(self, model_name, hyperparams=None):
        """Create a challenger model for A/B testing against the champion"""
        logger.info(f"Creating challenger model for {model_name}")

        try:
            # Get currently deployed model
            champion_model, metadata = self.model_deployer.get_deployed_model(
                model_name)

            if champion_model is None:
                logger.error(f"No deployed model found for {model_name}")
                return False

            # Load recent data for training
            recent_data = self._load_recent_data()

            if recent_data is None:
                logger.error(
                    "Failed to load recent data for challenger model training")
                return False

            # Prepare data
            features, target = self._prepare_model_specific_data(
                model_name, recent_data)

            if features is None or target is None:
                logger.error(
                    f"Failed to prepare data for {model_name} challenger")
                return False

            # Get model type
            model_type = metadata.get('model_type', 'xgboost')

            # Train the challenger with different hyperparameters
            if hyperparams is None:
                # Use default variation of hyperparameters
                if model_type == 'xgboost':
                    hyperparams = {
                        'max_depth': 7 if metadata.get('hyperparams', {}).get('max_depth', 6) == 6 else 6,
                        'eta': 0.05 if metadata.get('hyperparams', {}).get('eta', 0.1) == 0.1 else 0.1,
                        'subsample': 0.9 if metadata.get('hyperparams', {}).get('subsample', 0.8) == 0.8 else 0.8,
                        'objective': 'binary:logistic' if model_name == 'signal_detection' else 'reg:squarederror'
                    }
                elif model_type == 'keras':
                    # For Keras, we'd need a more complex approach to vary architecture
                    hyperparams = {'learning_rate': 0.001}
                else:
                    hyperparams = {}

            # Train model based on type
            challenger_model, metrics = None, {}

            if model_type == 'xgboost':
                dtrain = xgb.DMatrix(features, label=target)
                challenger_model = xgb.train(
                    hyperparams, dtrain, num_boost_round=100)

                # Evaluate
                y_pred = challenger_model.predict(dtrain)

                if model_name == 'signal_detection':
                    # Classification metrics
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    metrics['accuracy'] = float(
                        accuracy_score(target, y_pred_binary))
                    metrics['precision'] = float(precision_score(
                        target, y_pred_binary, zero_division=0))
                    metrics['recall'] = float(recall_score(
                        target, y_pred_binary, zero_division=0))
                    metrics['f1'] = float(
                        f1_score(target, y_pred_binary, zero_division=0))
                else:
                    # Regression metrics
                    metrics['mse'] = float(mean_squared_error(target, y_pred))
                    metrics['rmse'] = float(np.sqrt(metrics['mse']))
                    metrics['mae'] = float(mean_absolute_error(target, y_pred))

            elif model_type == 'sklearn':
                # Create new instance of same class with different hyperparams
                challenger_model = metadata.get(
                    'model_class', RandomForestClassifier)(**hyperparams)
                challenger_model.fit(features, target)

                # Evaluate
                y_pred = challenger_model.predict(features)

                if model_name in ['signal_detection', 'risk_assessment']:
                    # Classification metrics
                    metrics['accuracy'] = float(accuracy_score(target, y_pred))
                    metrics['precision'] = float(
                        precision_score(target, y_pred, zero_division=0))
                    metrics['recall'] = float(recall_score(
                        target, y_pred, zero_division=0))
                    metrics['f1'] = float(
                        f1_score(target, y_pred, zero_division=0))
                else:
                    # Regression metrics
                    metrics['mse'] = float(mean_squared_error(target, y_pred))
                    metrics['rmse'] = float(np.sqrt(metrics['mse']))
                    metrics['mae'] = float(mean_absolute_error(target, y_pred))

            # Register the challenger model
            if challenger_model is not None:
                new_version = self.model_registry.register_model(
                    model_name=f"{model_name}_challenger",
                    model=challenger_model,
                    model_type=model_type,
                    metrics=metrics,
                    hyperparams=hyperparams
                )

                logger.info(
                    f"Created challenger model {model_name}_challenger v{new_version}")

                # Store A/B test info in Redis
                if self.redis:
                    ab_test_info = {
                        "champion": model_name,
                        "challenger": f"{model_name}_challenger",
                        "champion_version": metadata.get('version', 0),
                        "challenger_version": new_version,
                        "start_time": time.time(),
                        "hyperparams": json.dumps(hyperparams),
                        "metrics": json.dumps(metrics)
                    }

                    self.redis.hset(
                        f"ab_test:{model_name}", mapping=ab_test_info)

                return True
            else:
                logger.error(
                    f"Failed to create challenger model for {model_name}")
                return False

        except Exception as e:
            logger.error(
                f"Error creating challenger model for {model_name}: {str(e)}", exc_info=True)
            return False


# Utility functions
def get_active_models():
    """Get list of active trading models"""
    return [
        'signal_detection',  # Detects potential trading signals
        'price_prediction',  # Predicts price movements
        'risk_assessment',   # Assesses trade risk
        'exit_strategy',     # Determines optimal exit points
        'market_regime'      # Identifies market regimes
    ]


class ContinualLearningSystem:
    """
    Wrapper class for the LearningEngine that provides the interface expected by unified_system.py
    """

    def __init__(self, redis_client=None, ml_engine=None, data_pipeline=None, use_gpu=True,
                 slack_webhook_url=None, slack_bot_token=None):
        """
        Initialize the continual learning system

        Args:
            redis_client: Redis client for communication and caching
            ml_engine: Reference to the ML engine
            data_pipeline: Reference to the data pipeline
            use_gpu: Whether to use GPU acceleration
            slack_webhook_url: Slack webhook URL for notifications
            slack_bot_token: Slack bot token for API access
        """
        # Initialize the underlying learning engine
        self.learning_engine = LearningEngine(
            redis_client=redis_client,
            ml_engine=ml_engine,
            data_pipeline=data_pipeline,
            use_gpu=use_gpu
        )

        # Initialize Slack reporter
        self.slack_reporter = SlackReporter(
            webhook_url=slack_webhook_url,
            bot_token=slack_bot_token
        )

        # Initialize Prometheus metrics if available
        self.metrics = {}
        if PROMETHEUS_AVAILABLE:
            # Training metrics
            self.metrics['training_duration'] = prom.Histogram(
                'continual_learning_training_duration_seconds',
                'Duration of model training in seconds',
                ['model_name', 'version']
            )

            self.metrics['model_accuracy'] = prom.Gauge(
                'continual_learning_model_accuracy',
                'Model accuracy metric',
                ['model_name', 'version']
            )

            self.metrics['model_f1'] = prom.Gauge(
                'continual_learning_model_f1_score',
                'Model F1 score metric',
                ['model_name', 'version']
            )

            # Drift detection metrics
            self.metrics['drift_score'] = prom.Gauge(
                'continual_learning_drift_score',
                'Data drift detection score',
                ['data_type']
            )

            self.metrics['retraining_count'] = prom.Counter(
                'continual_learning_retraining_count_total',
                'Total number of model retraining events',
                ['model_name', 'reason']
            )

            # Start Prometheus HTTP server if not already running
            try:
                prom.start_http_server(
                    int(os.environ.get('PROMETHEUS_PORT', 8000)))
                logging.getLogger('continual_learning').info(
                    f"Prometheus metrics server started on port {os.environ.get('PROMETHEUS_PORT', 8000)}")
            except Exception as e:
                logging.getLogger('continual_learning').warning(
                    f"Could not start Prometheus server: {str(e)}")

        logger.info("Continual Learning System initialized")

    def start(self):
        """Start the continual learning system"""
        self.learning_engine.start()

    def stop(self):
        """Stop the continual learning system"""
        self.learning_engine.stop()

    def daily_model_update(self):
        """Perform daily incremental update of models"""
        success = self.learning_engine.daily_model_update()

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and success:
            for model_name in get_active_models():
                # Get current version
                current_version = self.learning_engine.model_deployer.deployed_models.get(
                    model_name)
                if current_version:
                    # Get metrics
                    metrics_data = None
                    if self.learning_engine.redis:
                        metrics_data = self.learning_engine.redis.hgetall(
                            f"model:metrics:{model_name}:{current_version}")

                    if metrics_data:
                        # Update Prometheus metrics
                        if 'accuracy' in metrics_data:
                            self.metrics['model_accuracy'].labels(
                                model_name=model_name,
                                version=current_version
                            ).set(float(metrics_data['accuracy']))

                        if 'f1' in metrics_data:
                            self.metrics['model_f1'].labels(
                                model_name=model_name,
                                version=current_version
                            ).set(float(metrics_data['f1']))

        return success

    def full_model_retrain(self):
        """Perform full retraining of all models"""
        start_time = time.time()
        success = self.learning_engine.full_model_retrain()
        duration = time.time() - start_time

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and success:
            for model_name in get_active_models():
                self.metrics['training_duration'].labels(
                    model_name=model_name,
                    version='full_retrain'
                ).observe(duration)

                self.metrics['retraining_count'].labels(
                    model_name=model_name,
                    reason='scheduled'
                ).inc()

        # Send notification via Slack
        if success:
            self.slack_reporter.report_training_complete(
                training_time=duration,
                model_name='all_models',
                version='full_retrain'
            )

        return success

    def check_for_drift(self):
        """Scheduled task to check for drift in data and model performance"""
        drift_detected = self.learning_engine.check_for_drift()

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and drift_detected:
            for model_name in get_active_models():
                self.metrics['drift_score'].labels(
                    data_type=model_name
                ).set(1.0)  # Simple binary indicator

                # If drift triggered retraining
                if self.learning_engine.config['drift_detection']['auto_retrain']:
                    self.metrics['retraining_count'].labels(
                        model_name=model_name,
                        reason='drift'
                    ).inc()

        return drift_detected

    def create_model_challenger(self, model_name, hyperparams=None):
        """Create a challenger model for A/B testing against the champion"""
        return self.learning_engine.create_model_challenger(model_name, hyperparams)
