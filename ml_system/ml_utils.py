#!/usr/bin/env python3
"""
ML Utilities

This module provides utility functions for machine learning models:
1. Feature selection
2. Time series cross-validation
3. Drift detection
4. Hyperparameter optimization
5. Diagnostics
"""

import os
import re
import json
import time
import logging
import subprocess
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ks_2samp

# Import optuna for hyperparameter optimization if available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.getLogger('ml_utils').warning(
        "Optuna not available. Hyperparameter optimization will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_utils')


def select_features(X, y, problem_type='regression', method='importance', threshold=0.01, n_features=20):
    """Select most important features using specified method"""
    logger.info(f"Performing feature selection using {method} method")

    try:
        if method == 'importance':
            # Use a simple model to get feature importances
            if problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42)
            else:
                model = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=42)

            model.fit(X, y)
            importances = model.feature_importances_

            # Select features with importance above threshold
            selected_features = X.columns[importances > threshold]

            # Ensure we have at least 5 features
            if len(selected_features) < 5:
                # Take top 5 features by importance
                selected_features = X.columns[np.argsort(importances)[-5:]]

            logger.info(
                f"Selected {len(selected_features)} features using importance threshold")

            return X[selected_features]

        elif method == 'rfe':
            # Use Recursive Feature Elimination
            if problem_type == 'classification':
                estimator = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42)
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=42)

            selector = RFE(estimator, n_features_to_select=min(
                n_features, X.shape[1]), step=1)
            selector = selector.fit(X, y)

            selected_features = X.columns[selector.support_]
            logger.info(
                f"Selected {len(selected_features)} features using RFE")

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

            logger.info(
                f"Selected {len(selected_features)} features using mutual information")

            return X[selected_features]

        else:
            logger.warning(
                f"Unknown feature selection method: {method}. Using all features.")
            return X

    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return X


def create_time_series_splits(X, y, n_splits=5, embargo_size=10):
    """Create time series cross-validation splits with embargo period"""
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Apply embargo period to prevent data leakage
    purged_splits = []
    for train_idx, test_idx in tscv.split(X):
        # Apply embargo: remove samples at the end of train that are too close to test
        if embargo_size > 0:
            min_test_idx = min(test_idx)
            embargo_idx = range(
                max(min_test_idx - embargo_size, 0), min_test_idx)
            train_idx = np.setdiff1d(train_idx, embargo_idx)

        purged_splits.append((train_idx, test_idx))

    return purged_splits


def detect_feature_drift(current_data, reference_data, threshold=0.05):
    """Detect drift in feature distributions using KS test"""
    try:
        # Select numeric features only
        numeric_features = reference_data.select_dtypes(
            include=[np.number]).columns

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
                        drift_features[feature] = {'ks_statistic': float(
                            ks_statistic), 'p_value': float(p_value)}

        return drift_detected, drift_features

    except Exception as e:
        logger.error(f"Error detecting feature drift: {str(e)}")
        return False, {}


def optimize_hyperparameters(data, model_type, config, data_processor=None):
    """Run hyperparameter optimization for a specific model type"""
    if not OPTUNA_AVAILABLE:
        logger.warning(
            "Optuna not available. Skipping hyperparameter optimization.")
        return None

    try:
        logger.info(f"Optimizing hyperparameters for {model_type} model")

        if model_type == 'signal_detection':
            return optimize_signal_detection_hyperparams(data, config, data_processor)
        elif model_type == 'price_prediction':
            return optimize_price_prediction_hyperparams(data, config, data_processor)
        elif model_type == 'exit_strategy':
            return optimize_exit_strategy_hyperparams(data, config, data_processor)
        else:
            logger.warning(
                f"Hyperparameter optimization not implemented for {model_type}")
            return None

    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        return None


def optimize_signal_detection_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for signal detection model"""
    try:
        # Prepare data
        features, target = data_processor.prepare_signal_detection_data(data)

        if len(features) == 0 or len(target) == 0:
            logger.error(
                "No valid data for signal detection hyperparameter optimization")
            return None

        # Apply feature selection if enabled
        if config['feature_selection']['enabled']:
            features = select_features(
                features, target, 'classification',
                config['feature_selection']['method'],
                config['feature_selection']['threshold'],
                config['feature_selection']['n_features']
            )

        # Create time series splits for cross-validation
        splits = create_time_series_splits(
            features, target,
            config['time_series_cv']['n_splits'],
            config['time_series_cv']['embargo_size']
        )

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

            # Import XGBoost here to avoid issues if not available
            import xgboost as xgb

            for train_idx, test_idx in splits:
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                dtest = xgb.DMatrix(X_test_scaled, label=y_test)

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
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_test, y_pred)
                cv_scores.append(auc)

            return np.mean(cv_scores)

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        # Get best parameters
        best_params = study.best_params
        logger.info(
            f"Best hyperparameters for signal detection: {best_params}")

        # Save best parameters
        params_path = os.path.join(
            config['models_dir'], 'signal_detection_optimized_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f)

        return best_params

    except Exception as e:
        logger.error(
            f"Error optimizing signal detection hyperparameters: {str(e)}")
        return None


def optimize_price_prediction_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for price prediction model"""
    # Implementation would be similar to signal_detection but for LSTM model
    logger.warning(
        "Price prediction hyperparameter optimization not implemented")
    return None


def optimize_exit_strategy_hyperparams(data, config, data_processor):
    """Optimize hyperparameters for exit strategy model"""
    # Implementation would be similar to signal_detection but for regression
    logger.warning("Exit strategy hyperparameter optimization not implemented")
    return None


def run_diagnostics():
    """Run diagnostics to identify GPU configuration issues"""
    logger.info("=== Running GPU Diagnostics ===")

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
        results["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi"]).decode()
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
            logger.info(
                f"TensorFlow detected {len(gpus)} GPU(s): {results['tensorflow_gpu']}")
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
        logger.info(
            f"CUDA version: {results['cuda_version']}, CuPy version: {results['cupy_version']}")

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

                logger.info(
                    f"GH200 Benchmark: Matrix multiplication {size}x{size} took {latency:.2f} ms")
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


class GPUStatsTracker:
    """Track GPU statistics during model training"""

    def __init__(self, polling_interval=10.0):
        self.polling_interval = polling_interval
        self.stats = []
        self.running = False
        self.thread = None

    def start(self):
        """Start tracking GPU statistics"""
        if self.running:
            return

        self.running = True
        self.stats = []

        # Start tracking in a separate thread
        import threading
        self.thread = threading.Thread(target=self._track_stats)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop tracking GPU statistics and return results"""
        if not self.running:
            return self.stats

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        return self.stats

    def _track_stats(self):
        """Track GPU statistics in a loop"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            while self.running:
                timestamp = time.time()
                devices_stats = []

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    device_stats = {
                        "device_id": i,
                        "name": name,
                        "memory_used": memory.used,
                        "memory_total": memory.total,
                        "gpu_utilization": utilization.gpu,
                        "memory_utilization": utilization.memory
                    }

                    devices_stats.append(device_stats)

                self.stats.append({
                    "timestamp": timestamp,
                    "devices": devices_stats
                })

                time.sleep(self.polling_interval)

        except ImportError:
            logger.warning("pynvml not available, cannot track GPU statistics")
        except Exception as e:
            logger.error(f"Error tracking GPU statistics: {str(e)}")
        finally:
            self.running = False


class SlackReporter:
    """Report model training results to Slack"""

    def __init__(self, webhook_url=None, bot_token=None, channel='#system-notifications'):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel = channel

    def report_error(self, message, phase=None):
        """Report an error to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = f"❌ *Error*: {message}"
            if phase:
                text += f"\n*Phase*: {phase}"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")

    def report_model_metrics(self, model_name, metrics, training_time):
        """Report model metrics to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = f"✅ *{model_name.replace('_', ' ').title()} Model Trained*"
            text += f"\n*Training Time*: {training_time:.2f} seconds"

            # Add metrics
            text += "\n*Metrics*:"
            for key, value in metrics.items():
                if key != 'feature_importance' and key != 'training_history':
                    if isinstance(value, float):
                        text += f"\n- {key}: {value:.4f}"
                    else:
                        text += f"\n- {key}: {value}"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")

    def report_training_complete(self, total_time, models_status, gpu_stats=None):
        """Report training completion to Slack"""
        if not self.webhook_url and not self.bot_token:
            return

        try:
            import requests

            text = f"🎉 *Model Training Complete*"
            text += f"\n*Total Training Time*: {total_time:.2f} seconds"

            # Add model status
            text += "\n*Models Status*:"
            for model_name, status in models_status.items():
                if status.get('success', False):
                    text += f"\n- {model_name}: ✅"
                else:
                    text += f"\n- {model_name}: ❌ ({status.get('error', 'Unknown error')})"

            # Add GPU stats if available
            if gpu_stats:
                text += "\n*GPU Statistics*:"

                # Calculate average utilization
                avg_gpu_util = np.mean(
                    [stat['devices'][0]['gpu_utilization'] for stat in gpu_stats if stat['devices']])
                max_gpu_util = np.max(
                    [stat['devices'][0]['gpu_utilization'] for stat in gpu_stats if stat['devices']])
                avg_mem_util = np.mean(
                    [stat['devices'][0]['memory_utilization'] for stat in gpu_stats if stat['devices']])
                max_mem_used = np.max(
                    [stat['devices'][0]['memory_used'] for stat in gpu_stats if stat['devices']])

                text += f"\n- Average GPU Utilization: {avg_gpu_util:.1f}%"
                text += f"\n- Peak GPU Utilization: {max_gpu_util:.1f}%"
                text += f"\n- Average Memory Utilization: {avg_mem_util:.1f}%"
                text += f"\n- Peak Memory Used: {max_mem_used / (1024**2):.1f} MB"

            if self.webhook_url:
                payload = {
                    "text": text,
                    "channel": self.channel
                }

                requests.post(self.webhook_url, json=payload)
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "channel": self.channel,
                    "text": text
                }

                requests.post("https://slack.com/api/chat.postMessage",
                              headers=headers, json=payload)

        except Exception as e:
            logger.error(f"Error reporting to Slack: {str(e)}")
