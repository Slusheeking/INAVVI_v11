"""
Model training for the Autonomous Trading System.

This package provides utilities for training, validating, and managing
machine learning models for the trading system.
"""

# Model implementations
from src.model_training.models.cnn_model import CNNModel
from src.model_training.models.lstm_model import LSTMModel
from src.model_training.models.xgboost_model import XGBoostModel
from src.model_training.models.inference import ModelInference

# Model registry
from src.model_training.registry.model_registry import (
    ModelRegistry,
    register_model,
    get_model,
    list_models,
    delete_model,
    get_latest_model,
    get_model_metadata,
    update_model_metadata,
)

# Model validation
from src.model_training.validation.model_validator import (
    ModelValidator,
    validate_model,
    cross_validate,
    calculate_validation_metrics,
    compare_models,
    generate_validation_report,
)

__all__ = [
    # Models
    "CNNModel",
    "LSTMModel",
    "XGBoostModel",
    "ModelInference",
    
    # Registry
    "ModelRegistry",
    "register_model",
    "get_model",
    "list_models",
    "delete_model",
    "get_latest_model",
    "get_model_metadata",
    "update_model_metadata",
    
    # Validation
    "ModelValidator",
    "validate_model",
    "cross_validate",
    "calculate_validation_metrics",
    "compare_models",
    "generate_validation_report",
]