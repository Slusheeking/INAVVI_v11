"""
Model registry for the Autonomous Trading System.

This module provides utilities for registering, versioning, and managing
machine learning models used in the trading system.
"""

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

__all__ = [
    "ModelRegistry",
    "register_model",
    "get_model",
    "list_models",
    "delete_model",
    "get_latest_model",
    "get_model_metadata",
    "update_model_metadata",
]