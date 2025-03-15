#!/usr/bin/env python3
"""
Model Retrainer

This module provides functionality for retraining machine learning models
based on new data and performance feedback.
"""

import os
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Import project utilities
from src.utils.logging import get_logger
from src.utils.serialization import save_json, load_json
from src.utils.metrics import calculate_trading_metrics

logger = get_logger("model_retrainer")


class ModelRetrainer:
    """
    Retrains machine learning models based on new data and performance feedback.

    This class provides methods for:
    - Retraining models with new data
    - Adapting models to changing market conditions
    - Optimizing model hyperparameters
    - Validating retrained models
    - Managing model versioning
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the model retrainer.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        # Default configuration
        self.validation_split = self.config.get(
            "validation_split", 0.2
        )  # 20% validation
        self.min_improvement_threshold = self.config.get(
            "min_improvement_threshold", 0.05
        )  # 5% improvement
        self.max_retraining_attempts = self.config.get("max_retraining_attempts", 3)
        self.hyperparameter_tuning = self.config.get("hyperparameter_tuning", True)
        self.optimization_metric = self.config.get(
            "optimization_metric", "dollar_profit"
        )
        self.early_stopping = self.config.get("early_stopping", True)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 10)
        self.model_backup = self.config.get("model_backup", True)
        self.backup_dir = self.config.get("backup_dir", "data/model_backups")

        # Initialize state
        self.retraining_history = {}

        logger.info(
            f"Initialized ModelRetrainer with optimization_metric={self.optimization_metric}"
        )

    def retrain_model(
        self,
        model_registry: Any,
        features: pd.DataFrame,
        target: pd.Series,
        current_regime: str | None = None,
    ) -> Any | None:
        """
        Retrain a model with new data.

        Args:
            model_registry: Model registry containing models
            features: Feature data for training
            target: Target data for training
            current_regime: Current market regime (optional)

        Returns:
            Retrained model, or None if retraining failed
        """
        logger.info(f"Starting model retraining with {len(features)} samples")

        try:
            # Get current model
            current_model = self._get_current_model(model_registry, current_regime)

            if current_model is None:
                logger.warning("No current model available for retraining")
                return None

            # Get model metadata
            model_id = self._get_model_id(current_model)
            model_type = self._get_model_type(current_model)

            logger.info(f"Retraining model {model_id} of type {model_type}")

            # Split data for training and validation
            train_features, train_target, val_features, val_target = self._split_data(
                features, target
            )

            # Create new model instance
            new_model = self._create_model_instance(model_type, current_model)

            if new_model is None:
                logger.warning(
                    f"Failed to create new model instance of type {model_type}"
                )
                return None

            # Optimize hyperparameters if enabled
            if self.hyperparameter_tuning:
                new_model = self._optimize_hyperparameters(
                    new_model, train_features, train_target, val_features, val_target
                )

            # Train model
            trained_model = self._train_model(
                new_model, train_features, train_target, val_features, val_target
            )

            if trained_model is None:
                logger.warning("Model training failed")
                return None

            # Validate model
            validation_result = self._validate_model(
                trained_model, current_model, val_features, val_target
            )

            if not validation_result["is_improved"]:
                logger.warning(
                    f"Retrained model did not improve performance: {validation_result['message']}"
                )
                return None

            # Update model registry
            new_model_id = self._update_model_registry(
                model_registry, trained_model, validation_result, current_regime
            )

            if new_model_id:
                logger.info(
                    f"Model retraining successful, new model ID: {new_model_id}"
                )

                # Update retraining history
                self._update_retraining_history(
                    model_id, new_model_id, validation_result, current_regime
                )

                # Backup model if enabled
                if self.model_backup:
                    self._backup_model(trained_model, new_model_id)

                return trained_model
            else:
                logger.warning("Failed to update model registry")
                return None

        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            logger.error(traceback.format_exc())
            return None

    def _get_current_model(
        self, model_registry: Any, current_regime: str | None = None
    ) -> Any | None:
        """
        Get current model from registry.

        Args:
            model_registry: Model registry containing models
            current_regime: Current market regime (optional)

        Returns:
            Current model, or None if not available
        """
        try:
            # Check if model registry has get_model method
            if hasattr(model_registry, "get_model"):
                # Try to get regime-specific model
                if current_regime:
                    model = model_registry.get_model(regime=current_regime)
                    if model:
                        return model

                # Fall back to default model
                return model_registry.get_model()

            # Check if model registry has models attribute
            elif hasattr(model_registry, "models"):
                models = model_registry.models

                # Try to get regime-specific model
                if current_regime and current_regime in models:
                    return models[current_regime]

                # Try to get default model
                if "default" in models:
                    return models["default"]

                # Return first model
                if models:
                    return next(iter(models.values()))

            # Check if model registry is a dictionary
            elif isinstance(model_registry, dict) and "models" in model_registry:
                models = model_registry["models"]

                # Try to get regime-specific model
                if current_regime and current_regime in models:
                    return models[current_regime]

                # Try to get default model
                if "default" in models:
                    return models["default"]

                # Try to get primary model
                if "primary_model" in models:
                    return models["primary_model"]

                # Return first model
                if models:
                    return next(iter(models.values()))

            logger.warning("Could not find current model in registry")
            return None

        except Exception as e:
            logger.error(f"Error getting current model: {e}")
            return None

    def _get_model_id(self, model: Any) -> str:
        """
        Get model ID.

        Args:
            model: Model instance

        Returns:
            Model ID
        """
        # Check if model has id attribute
        if hasattr(model, "id"):
            return model.id

        # Check if model has model_id attribute
        elif hasattr(model, "model_id"):
            return model.model_id

        # Generate ID based on model type and timestamp
        model_type = self._get_model_type(model)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        return f"{model_type}_{timestamp}"

    def _get_model_type(self, model: Any) -> str:
        """
        Get model type.

        Args:
            model: Model instance

        Returns:
            Model type
        """
        # Check if model has type attribute
        if hasattr(model, "type"):
            return model.type

        # Check if model has model_type attribute
        elif hasattr(model, "model_type"):
            return model.model_type

        # Get type from class name
        return model.__class__.__name__

    def _split_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data for training and validation.

        Args:
            features: Feature data
            target: Target data

        Returns:
            Tuple of (train_features, train_target, val_features, val_target)
        """
        # Check if features has datetime index
        if isinstance(features.index, pd.DatetimeIndex):
            # Time-based split
            split_idx = int(len(features) * (1 - self.validation_split))

            train_features = features.iloc[:split_idx]
            train_target = target.iloc[:split_idx]
            val_features = features.iloc[split_idx:]
            val_target = target.iloc[split_idx:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split

            train_features, val_features, train_target, val_target = train_test_split(
                features, target, test_size=self.validation_split, random_state=42
            )

        return train_features, train_target, val_features, val_target

    def _create_model_instance(self, model_type: str, current_model: Any) -> Any | None:
        """
        Create a new model instance based on model type.

        Args:
            model_type: Type of model to create
            current_model: Current model instance

        Returns:
            New model instance, or None if creation failed
        """
        try:
            # Check if current model has clone method
            if hasattr(current_model, "clone"):
                return current_model.clone()

            # Check if current model is scikit-learn model
            if hasattr(current_model, "get_params"):
                from sklearn.base import clone

                return clone(current_model)

            # Check if current model is TensorFlow/Keras model
            if hasattr(current_model, "get_config") and hasattr(
                current_model, "from_config"
            ):
                config = current_model.get_config()
                return current_model.__class__.from_config(config)

            # Check if current model is PyTorch model
            if hasattr(current_model, "state_dict") and hasattr(
                current_model, "load_state_dict"
            ):
                import copy

                new_model = copy.deepcopy(current_model)
                new_model.load_state_dict(current_model.state_dict())
                return new_model

            # Create new instance based on model type
            if model_type == "RandomForestRegressor":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor()

            elif model_type == "XGBRegressor":
                from xgboost import XGBRegressor

                return XGBRegressor()
            elif model_type == "LGBMRegressor":
                try:
                    from lightgbm import LGBMRegressor  # type: ignore
                    return LGBMRegressor()
                except ImportError:
                    logger.warning("LightGBM is not installed. Please install it with 'pip install lightgbm'")
                    return None
                

            elif model_type == "LinearRegression":
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

            elif model_type == "LSTMModel":
                # Assuming LSTMModel is a custom class
                if hasattr(current_model, "__class__"):
                    return current_model.__class__()

            elif model_type == "AttentionModel":
                # Assuming AttentionModel is a custom class
                if hasattr(current_model, "__class__"):
                    return current_model.__class__()

            logger.warning(f"Unknown model type: {model_type}")
            return None

        except Exception as e:
            logger.error(f"Error creating model instance: {e}")
            return None

    def _optimize_hyperparameters(
        self,
        model: Any,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        val_features: pd.DataFrame,
        val_target: pd.Series,
    ) -> Any:
        """
        Optimize model hyperparameters.

        Args:
            model: Model instance
            train_features: Training features
            train_target: Training target
            val_features: Validation features
            val_target: Validation target

        Returns:
            Optimized model
        """
        try:
            # Check if model is scikit-learn compatible
            if hasattr(model, "get_params") and hasattr(model, "set_params"):
                # Define parameter grid based on model type
                model_type = self._get_model_type(model)
                param_grid = self._get_parameter_grid(model_type)

                if not param_grid:
                    logger.warning(
                        f"No parameter grid available for model type: {model_type}"
                    )
                    return model

                # Use GridSearchCV or RandomizedSearchCV
                from sklearn.model_selection import RandomizedSearchCV

                # Define scoring function based on optimization metric
                scoring = self._get_scoring_function()

                # Create search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=10,
                    scoring=scoring,
                    cv=3,
                    random_state=42,
                    n_jobs=-1,
                )

                # Fit search
                search.fit(train_features, train_target)

                # Return best model
                return search.best_estimator_

            # Check if model has hyperparameter_tuning method
            elif hasattr(model, "hyperparameter_tuning"):
                return model.hyperparameter_tuning(
                    train_features, train_target, val_features, val_target
                )

            # No hyperparameter optimization available
            logger.info(
                f"No hyperparameter optimization available for model type: {self._get_model_type(model)}"
            )
            return model

        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return model

    def _get_parameter_grid(self, model_type: str) -> dict[str, list[Any]]:
        """
        Get parameter grid for hyperparameter optimization.

        Args:
            model_type: Type of model

        Returns:
            Parameter grid
        """
        if model_type == "RandomForestRegressor":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        elif model_type == "XGBRegressor":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

        elif model_type == "LGBMRegressor":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

        elif model_type == "LinearRegression":
            return {}  # No hyperparameters for LinearRegression

        # Add more model types as needed

        return {}

    def _get_scoring_function(self) -> str | Callable:
        """
        Get scoring function for hyperparameter optimization.

        Returns:
            Scoring function or name
        """
        if self.optimization_metric == "r2":
            return "r2"

        elif self.optimization_metric == "mse":
            return "neg_mean_squared_error"

        elif self.optimization_metric == "mae":
            return "neg_mean_absolute_error"

        elif self.optimization_metric == "dollar_profit":
            # Custom scoring function for dollar profit
            def dollar_profit_scorer(estimator, X, y):
                # Make predictions
                y_pred = estimator.predict(X)

                # Calculate dollar profit (simplified)
                # Assuming positive prediction = long, negative prediction = short
                position = np.sign(y_pred)
                returns = position * y

                # Calculate total dollar profit
                dollar_profit = np.sum(returns)

                return dollar_profit

            return dollar_profit_scorer

        # Default to r2
        return "r2"

    def _train_model(
        self,
        model: Any,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        val_features: pd.DataFrame,
        val_target: pd.Series,
    ) -> Any | None:
        """
        Train model with early stopping if supported.

        Args:
            model: Model instance
            train_features: Training features
            train_target: Training target
            val_features: Validation features
            val_target: Validation target

        Returns:
            Trained model, or None if training failed
        """
        try:
            # Check if model is scikit-learn compatible
            if hasattr(model, "fit"):
                # Check if model supports early stopping
                if (
                    self.early_stopping
                    and hasattr(model, "fit")
                    and "early_stopping_rounds" in str(model.fit.__code__.co_varnames)
                ):
                    # XGBoost, LightGBM, etc.
                    model.fit(
                        train_features,
                        train_target,
                        eval_set=[(val_features, val_target)],
                        early_stopping_rounds=self.early_stopping_patience,
                        verbose=False,
                    )
                elif self.early_stopping and hasattr(model, "n_iter_no_change"):
                    # Some scikit-learn models
                    model.n_iter_no_change = self.early_stopping_patience
                    model.fit(train_features, train_target)
                else:
                    # Standard fit
                    model.fit(train_features, train_target)

                return model

            # Check if model has custom train method
            elif hasattr(model, "train"):
                return model.train(
                    train_features,
                    train_target,
                    val_features,
                    val_target,
                    early_stopping=self.early_stopping,
                    patience=self.early_stopping_patience,
                )

            logger.warning(
                f"Model does not have fit or train method: {self._get_model_type(model)}"
            )
            return None

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def _validate_model(
        self,
        new_model: Any,
        current_model: Any,
        val_features: pd.DataFrame,
        val_target: pd.Series,
    ) -> dict[str, Any]:
        """
        Validate new model against current model.

        Args:
            new_model: New model instance
            current_model: Current model instance
            val_features: Validation features
            val_target: Validation target

        Returns:
            Validation result dictionary
        """
        try:
            # Calculate performance metrics for new model
            new_metrics = self._calculate_performance_metrics(
                new_model, val_features, val_target
            )

            # Calculate performance metrics for current model
            current_metrics = self._calculate_performance_metrics(
                current_model, val_features, val_target
            )

            # Compare performance
            primary_metric = self.optimization_metric

            if (
                primary_metric not in new_metrics
                or primary_metric not in current_metrics
            ):
                return {
                    "is_improved": False,
                    "message": f"Primary metric {primary_metric} not available",
                    "new_metrics": new_metrics,
                    "current_metrics": current_metrics,
                }

            new_value = new_metrics[primary_metric]
            current_value = current_metrics[primary_metric]

            # Check if metric should be maximized or minimized
            maximize_metric = primary_metric not in [
                "mse",
                "mae",
                "rmse",
                "max_drawdown",
            ]

            if maximize_metric:
                improvement = (
                    (new_value - current_value) / abs(current_value)
                    if current_value != 0
                    else float("inf")
                )
                is_improved = (
                    new_value > current_value
                    and improvement >= self.min_improvement_threshold
                )
            else:
                improvement = (
                    (current_value - new_value) / abs(current_value)
                    if current_value != 0
                    else float("inf")
                )
                is_improved = (
                    new_value < current_value
                    and improvement >= self.min_improvement_threshold
                )

            return {
                "is_improved": is_improved,
                "improvement": improvement,
                "message": f"Improvement: {improvement:.2%}"
                if is_improved
                else f"Insufficient improvement: {improvement:.2%}",
                "new_metrics": new_metrics,
                "current_metrics": current_metrics,
            }

        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {
                "is_improved": False,
                "message": f"Error validating model: {e}",
                "new_metrics": {},
                "current_metrics": {},
            }

    def _calculate_performance_metrics(
        self, model: Any, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, float]:
        """
        Calculate performance metrics for model.

        Args:
            model: Model instance
            features: Feature data
            target: Target data

        Returns:
            Dictionary with performance metrics
        """
        try:
            # Make predictions
            if hasattr(model, "predict"):
                predictions = model.predict(features)
            else:
                logger.warning(
                    f"Model does not have predict method: {self._get_model_type(model)}"
                )
                return {}

            # Calculate regression metrics
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            metrics = {
                "mse": mean_squared_error(target, predictions),
                "rmse": np.sqrt(mean_squared_error(target, predictions)),
                "mae": mean_absolute_error(target, predictions),
                "r2": r2_score(target, predictions),
            }

            # Calculate trading-specific metrics
            trading_metrics = self._calculate_trading_metrics(target, predictions)
            metrics.update(trading_metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _calculate_trading_metrics(
        self, target: pd.Series, predictions: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Args:
            target: Target data
            predictions: Model predictions

        Returns:
            Dictionary with trading metrics
        """
        try:
            # Convert to numpy arrays
            y_true = target.values if isinstance(target, pd.Series) else target
            y_pred = predictions
            
            # Use the utility function from utils.metrics
            metrics = calculate_trading_metrics(y_true, y_pred, initial_capital=10000)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}

    def _update_model_registry(
        self,
        model_registry: Any,
        model: Any,
        validation_result: dict[str, Any],
        current_regime: str | None = None,
    ) -> str | None:
        """
        Update model registry with new model.

        Args:
            model_registry: Model registry
            model: New model
            validation_result: Validation result
            current_regime: Current market regime (optional)

        Returns:
            New model ID, or None if update failed
        """
        try:
            # Generate new model ID
            model_id = self._get_model_id(model)

            # Create model metadata
            metadata = {
                "model_id": model_id,
                "model_type": self._get_model_type(model),
                "training_time": datetime.now().isoformat(),
                "performance_metrics": validation_result["new_metrics"],
                "improvement": validation_result.get("improvement", 0),
                "regime": current_regime,
            }

            # Check if model registry has register_model method
            if hasattr(model_registry, "register_model"):
                # Register model
                if current_regime:
                    model_registry.register_model(
                        model, metadata=metadata, regime=current_regime
                    )
                else:
                    model_registry.register_model(model, metadata=metadata)

                return model_id

            # Check if model registry has models attribute
            elif hasattr(model_registry, "models"):
                # Add model to registry
                if current_regime:
                    model_registry.models[current_regime] = model
                else:
                    model_registry.models["default"] = model

                # Add metadata if supported
                if hasattr(model_registry, "metadata"):
                    model_registry.metadata[model_id] = metadata

                return model_id

            # Check if model registry is a dictionary
            elif isinstance(model_registry, dict) and "models" in model_registry:
                # Add model to registry
                if current_regime:
                    model_registry["models"][current_regime] = model
                else:
                    model_registry["models"]["primary_model"] = model

                # Add metadata if supported
                if "model_metadata" in model_registry:
                    model_registry["model_metadata"][model_id] = metadata

                return model_id

            logger.warning("Could not update model registry")
            return None

        except Exception as e:
            logger.error(f"Error updating model registry: {e}")
            return None

    def _update_retraining_history(
        self,
        old_model_id: str,
        new_model_id: str,
        validation_result: dict[str, Any],
        current_regime: str | None = None,
    ) -> None:
        """
        Update retraining history.

        Args:
            old_model_id: ID of old model
            new_model_id: ID of new model
            validation_result: Validation result
            current_regime: Current market regime (optional)
        """
        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "old_model_id": old_model_id,
            "new_model_id": new_model_id,
            "improvement": validation_result.get("improvement", 0),
            "metrics": {
                "old": validation_result.get("current_metrics", {}),
                "new": validation_result.get("new_metrics", {}),
            },
            "regime": current_regime,
        }

        # Add to history
        if old_model_id not in self.retraining_history:
            self.retraining_history[old_model_id] = []

        self.retraining_history[old_model_id].append(entry)

    def _backup_model(self, model: Any, model_id: str) -> bool:
        """
        Backup model to disk.

        Args:
            model: Model to backup
            model_id: Model ID

        Returns:
            True if backup was successful, False otherwise
        """
        try:
            # Create backup directory if it doesn't exist

            # Backup path
            backup_path = os.path.join(self.backup_dir, f"{model_id}.pkl")

            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            # Backup model
            if hasattr(model, "save"):
                # Model has save method
                model.save(backup_path)
            else:
                # Use pickle
                import pickle

                with open(backup_path, "wb") as f:
                    pickle.dump(model, f)

            logger.info(f"Model backup saved to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error backing up model: {e}")
            return False

    def get_retraining_history(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get retraining history.

        Returns:
            Dictionary mapping model IDs to retraining history
        """
        return self.retraining_history

    def save_retraining_history(self, file_path: str) -> bool:
        """
        Save retraining history to file.

        Args:
            file_path: Path to save history

        Returns:
            True if save was successful, False otherwise
        """
        try:
            save_json(self.retraining_history, file_path)
            logger.info(f"Retraining history saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving retraining history: {e}")
            return False

    def load_retraining_history(self, file_path: str) -> bool:
        """
        Load retraining history from file.

        Args:
            file_path: Path to load history from

        Returns:
            True if load was successful, False otherwise
        """
        try:
            self.retraining_history = load_json(file_path)
            logger.info(f"Retraining history loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading retraining history: {e}")
            return False
