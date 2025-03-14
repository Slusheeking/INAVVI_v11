#!/usr/bin/env python3
"""
XGBoost Model for Trading

This module implements an XGBoost model optimized for trading applications.
It includes custom dollar profit objective functions, feature importance tracking,
and hyperparameter optimization.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.model_training.optimization.dollar_profit_objective import (
    DollarProfitObjective,
    create_tensorflow_dollar_profit_loss as create_dollar_profit_objective,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("xgboost_model")


class XGBoostModel:
    """
    XGBoost model optimized for trading applications.

    This class implements an XGBoost model with custom dollar profit objective
    functions, feature importance tracking, and hyperparameter optimization.
    """

    def __init__(
        self,
        model_id: str | None = None,
        objective: str = "dollar_profit",
        use_gpu: bool = True,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        dollar_profit_params: dict[str, Any] | None = None,
    ):
        """
        Initialize the XGBoost model.

        Args:
            model_id: Unique identifier for the model
            objective: Objective function ('dollar_profit', 'reg:squarederror', 'binary:logistic')
            use_gpu: Whether to use GPU for training
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction required to make a further partition
            min_child_weight: Minimum sum of instance weight needed in a child
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            early_stopping_rounds: Validation metric needs to improve at least once in
                                  every early_stopping_rounds round(s) to continue training
            random_state: Random seed
            dollar_profit_params: Parameters for dollar profit objective function
        """
        self.model_id = (
            model_id or f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.objective = objective
        self.use_gpu = use_gpu
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.dollar_profit_params = dollar_profit_params or {}

        # Initialize model
        self.model = None
        self.feature_names: list[str] | None = None
        self.feature_importance: np.ndarray | None = None
        self.training_history: dict[str, Any] | None = None
        self.metadata = {
            "model_id": self.model_id,
            "model_type": "xgboost",
            "objective": self.objective,
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "gamma": self.gamma,
                "min_child_weight": self.min_child_weight,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_state": self.random_state,
                "dollar_profit_params": self.dollar_profit_params,
            },
            "performance": {},
            "feature_importance": {},
        }

        logger.info(f"Initialized XGBoostModel with ID {self.model_id}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        additional_train_data: dict[str, np.ndarray] | None = None,
        additional_val_data: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Names of features
            sample_weight_train: Sample weights for training data
            sample_weight_val: Sample weights for validation data
            additional_train_data: Additional data for training (e.g., price data for dollar profit)
            additional_val_data: Additional data for validation

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        # Store feature names
        self.feature_names = (
            feature_names
            if feature_names is not None
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )

        # Create validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state
            )
            if sample_weight_train is not None:
                _, sample_weight_val = train_test_split(
                    sample_weight_train, test_size=0.2, random_state=self.random_state
                )

        # Create DMatrix objects
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_names,
            weight=sample_weight_train,
        )
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self.feature_names,
            weight=sample_weight_val,
        )

        # Add additional data if provided
        if additional_train_data:
            for key, value in additional_train_data.items():
                dtrain.set_float_info(key, value)

        if additional_val_data:
            for key, value in additional_val_data.items():
                dval.set_float_info(key, value)

        # Set up parameters
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "alpha": self.reg_alpha,
            "lambda": self.reg_lambda,
            "random_state": self.random_state,
            "tree_method": "gpu_hist" if self.use_gpu else "hist",
            "predictor": "gpu_predictor" if self.use_gpu else "cpu_predictor",
        }

        # Set objective function
        if self.objective == "dollar_profit":
            # Create dollar profit objective
            obj = create_dollar_profit_objective(**self.dollar_profit_params)
            params["objective"] = "reg:squarederror"  # Base objective
        else:
            # Use standard XGBoost objective
            obj = None
            params["objective"] = self.objective

        # Set up evaluation metrics
        if self.objective == "binary:logistic":
            metrics = ["logloss", "auc", "error"]
            params["eval_metric"] = metrics
        else:
            metrics = ["rmse"]
            params["eval_metric"] = metrics

        # Train model
        evals_result: dict[str, dict[str, list[float]]] = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            obj=obj,
            evals_result=evals_result,
            verbose_eval=100,
        )

        # Store training history
        self.training_history = evals_result if evals_result else None

        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance()

        # Calculate performance metrics
        train_predictions = self.predict(X_train)
        val_predictions = self.predict(X_val)

        train_metrics = self._calculate_metrics(y_train, train_predictions)
        val_metrics = self._calculate_metrics(y_val, val_predictions)

        # Calculate dollar profit if additional data is provided
        if additional_train_data and "price" in additional_train_data:
            dollar_profit_obj = DollarProfitObjective(**self.dollar_profit_params)

            train_dollar_metrics = dollar_profit_obj.evaluate_dollar_profit(
                y_train,
                train_predictions,
                additional_train_data["price"],
                additional_train_data.get("atr"),
            )

            val_dollar_metrics = dollar_profit_obj.evaluate_dollar_profit(
                y_val,
                val_predictions,
                additional_val_data["price"]
                if additional_val_data and "price" in additional_val_data
                else None,
                additional_val_data.get("atr") if additional_val_data else None,
            )

            train_metrics.update(train_dollar_metrics)
            val_metrics.update(val_dollar_metrics)

        # Update metadata
        self.metadata["performance"] = {
            "train": train_metrics,
            "validation": val_metrics,
            "training_time": time.time() - start_time,
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
        }

        self.metadata["feature_importance"] = {
            name: float(importance)
            for name, importance in zip(self.feature_names, self.feature_importance)
        }

        self.metadata["updated_at"] = datetime.now().isoformat()

        logger.info(
            f"Trained XGBoostModel {self.model_id} in {time.time() - start_time:.2f} seconds"
        )
        logger.info(
            f"Best iteration: {self.model.best_iteration}, Best score: {self.model.best_score}"
        )

        return {
            "model_id": self.model_id,
            "training_time": time.time() - start_time,
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": dict(
                zip(self.feature_names, self.feature_importance.tolist())
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        # Make predictions
        predictions = self.model.predict(dmatrix)

        return predictions

    def save(self, directory: str) -> str:
        """
        Save the model to a directory.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save model
        model_path = os.path.join(directory, f"{self.model_id}.model")
        self.model.save_model(model_path)

        # Save metadata
        metadata_path = os.path.join(directory, f"{self.model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save feature names
        feature_path = os.path.join(directory, f"{self.model_id}_features.json")
        with open(feature_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        # Save training history
        if self.training_history:
            history_path = os.path.join(directory, f"{self.model_id}_history.json")
            with open(history_path, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {}
                for dataset, metrics in self.training_history.items():
                    history_dict[dataset] = {}
                    for metric, values in metrics.items():
                        history_dict[dataset][metric] = [float(v) for v in values]
                json.dump(history_dict, f, indent=2)

        logger.info(f"Saved XGBoostModel {self.model_id} to {directory}")

        return model_path

    def load(self, model_path: str, metadata_path: str | None = None) -> None:
        """
        Load the model from a file.

        Args:
            model_path: Path to the model file
            metadata_path: Path to the metadata file (optional)
        """
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

            # Update instance variables from metadata
            self.model_id = self.metadata.get("model_id", self.model_id)
            self.objective = self.metadata.get("objective", self.objective)

            params: dict[str, Any] = self.metadata.get("parameters", {})
            self.n_estimators = params.get("n_estimators", self.n_estimators)
            self.max_depth = params.get("max_depth", self.max_depth)
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.subsample = params.get("subsample", self.subsample)
            self.colsample_bytree = params.get(
                "colsample_bytree", self.colsample_bytree
            )
            self.gamma = params.get("gamma", self.gamma)
            self.min_child_weight = params.get(
                "min_child_weight", self.min_child_weight
            )
            self.reg_alpha = params.get("reg_alpha", self.reg_alpha)
            self.reg_lambda = params.get("reg_lambda", self.reg_lambda)
            self.dollar_profit_params = params.get(
                "dollar_profit_params", self.dollar_profit_params
            )

            # Load feature importance
            feature_importance_dict = self.metadata.get("feature_importance", {})
            if feature_importance_dict:
                self.feature_names = (
                    list(feature_importance_dict.keys())
                    if feature_importance_dict
                    else None
                )
                self.feature_importance = np.array(
                    list(feature_importance_dict.values())
                    if feature_importance_dict
                    else []
                )

        # Try to load feature names from separate file if not in metadata
        if self.feature_names is None:
            feature_path = os.path.join(
                os.path.dirname(model_path), f"{self.model_id}_features.json"
            )
            if os.path.exists(feature_path):
                with open(feature_path) as f:
                    self.feature_names = json.load(f)

        logger.info(f"Loaded XGBoostModel {self.model_id} from {model_path}")

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        if self.feature_importance is None:
            self.feature_importance = self._calculate_feature_importance()

        return dict(zip(self.feature_names, self.feature_importance.tolist()))

    def get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model metadata
        """
        return self.metadata

    def _calculate_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance.

        Returns:
            Array of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Get feature importance
        importance_type = (
            "gain"  # 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        )
        importance = self.model.get_score(importance_type=importance_type)

        # Convert to array
        importance_array = np.zeros(len(self.feature_names))
        for feature, score in importance.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                importance_array[idx] = score

        # Normalize
        if importance_array.sum() > 0:
            importance_array = importance_array / importance_array.sum()

        return importance_array

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        # Regression metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = np.mean(np.abs(y_true - y_pred))

        # Direction accuracy
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics["direction_accuracy"] = np.mean(direction_true == direction_pred)

        # Classification metrics if binary
        if (
            set(np.unique(y_true)).issubset({0, 1})
            and self.objective == "binary:logistic"
        ):
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
            metrics["precision"] = precision_score(y_true, y_pred_binary)
            metrics["recall"] = recall_score(y_true, y_pred_binary)
            metrics["f1"] = f1_score(y_true, y_pred_binary)

        return metrics


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create synthetic data
    X = np.random.rand(1000, 20)
    y = np.random.normal(0, 0.01, 1000)  # Small price changes
    price_data = np.random.uniform(100, 200, 1000)  # Price data
    atr_data = np.random.uniform(1, 5, 1000)  # ATR data

    # Create model
    model = XGBoostModel(
        objective="dollar_profit",
        use_gpu=True,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        dollar_profit_params={
            "position_sizing_method": "volatility_adjusted",
            "risk_per_trade": 0.01,
            "account_size": 100000.0,
        },
    )

    # Train model
    result = model.train(
        X_train=X,
        y_train=y,
        feature_names=[f"feature_{i}" for i in range(20)],
        additional_train_data={"price": price_data, "atr": atr_data},
    )

    print(f"Training result: {result}")

    # Make predictions
    predictions = model.predict(X[:10])
    print(f"Predictions: {predictions}")

    # Get feature importance
    importance = model.get_feature_importance()
    print(f"Feature importance: {importance}")

    # Save model
    temp_dir = os.path.join(tempfile.gettempdir(), "xgboost_models")
    model_path = model.save(temp_dir)
    print(f"Model saved to {model_path}")
