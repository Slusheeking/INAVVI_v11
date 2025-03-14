"""
Model Training Models Package

This package contains the machine learning models used for prediction in the autonomous trading system.
"""

from src.utils.logging.logger import get_logger
from src.model_training.models.xgboost_model import XGBoostModel
from src.model_training.models.lstm_model import LSTMModel
from src.model_training.models.cnn_model import CNNModel

# Set up logger for this module
logger = get_logger("model_training.models")
logger.info("Initializing model training models package")

__all__ = ["XGBoostModel", "LSTMModel", "CNNModel"]

logger.debug(f"Loaded models: {', '.join(__all__)}")