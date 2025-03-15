"""
Model Inference for the Autonomous Trading System.

This module provides functionality for model inference.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import time

from src.utils.logging import get_logger
from src.utils.serialization import (
    load_json,
    deserialize_model,
)
from src.utils.time import now

logger = get_logger(__name__)

class ModelInference:
    """
    Inference engine for model predictions.
    """
    
    def __init__(self, model_registry=None, model_path=None, model_type=None, feature_names=None, scaler=None):
        """
        Initialize the model inference engine.
        
        Args:
            model_registry: Model registry for loading models
            model_path (str): Path to the model file
            model_type (str): Type of model ('tensorflow', 'xgboost', 'sklearn')
            feature_names (list): List of feature names
            scaler (object): Scaler for feature normalization
        """
        self.model_registry = model_registry
        self.model_path = model_path
        self.model_type = model_type
        self.feature_names = feature_names
        self.scaler = scaler
        self.model = None
        self.current_model_id = None
        
        if model_path is not None and model_type is not None:
            self.load_model(model_path, model_type)
    
    def load_model(self, model_path, model_type):
        """
        Load the model with optimizations for NVIDIA containers.
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model ('tensorflow', 'xgboost', 'sklearn')
        """
        self.model_path = model_path
        self.model_type = model_type
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Check if running in NVIDIA container
            is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
            if is_nvidia_container:
                logger.info(f"Loading model in NVIDIA container: {model_type}")
            
            if model_type == 'tensorflow':
                # Configure TensorFlow for optimal inference
                try:
                    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled for inference")
                    
                    # Enable mixed precision for faster inference
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
                    # Set optimizations for inference
                    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
                    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
                    os.environ["TF_USE_CUDNN"] = "1"
                    
                    logger.info("TensorFlow optimized for inference")
                except Exception as e:
                    logger.warning(f"Could not fully optimize TensorFlow: {e}")
                
                # Load the model with optimized settings
                # Use our serialization utilities for loading TensorFlow models
                self.model = deserialize_model(model_path)
                
                # Optimize the loaded model for inference
                try:
                    # Run a warmup prediction to compile the model
                    input_shape = self.model.input_shape
                    if isinstance(input_shape, list):
                        # Multiple inputs
                        dummy_inputs = [np.zeros((1,) + shape[1:]) for shape in input_shape]
                    else:
                        # Single input
                        dummy_inputs = np.zeros((1,) + input_shape[1:])
                    
                    self.model.predict(dummy_inputs)
                    logger.info("Model warmed up for inference")
                except Exception as e:
                    logger.warning(f"Could not warm up model: {e}")
                
            elif model_type == 'xgboost':
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                
                # Set GPU predictor if available
                if is_nvidia_container:
                    try:
                        self.model.set_param({'predictor': 'gpu_predictor'})
                        logger.info("XGBoost GPU predictor enabled")
                    except Exception as e:
                        logger.warning(f"Could not set XGBoost GPU predictor: {e}")
                
            elif model_type == 'sklearn':
                self.model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_metadata(self, metadata_path):
        """
        Load model metadata.
        
        Args:
            metadata_path (str): Path to the metadata file
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            metadata = load_json(metadata_path)
            
            self.feature_names = metadata.get('feature_names')
            
            # Load scaler if available
            scaler_path = metadata.get('scaler_path')
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Metadata loaded from {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def preprocess_features(self, features):
        """
        Preprocess features before inference.
        
        Args:
            features (pd.DataFrame or np.ndarray): Features to preprocess
            
        Returns:
            np.ndarray: Preprocessed features
        """
        # Convert to DataFrame if numpy array
        if isinstance(features, np.ndarray):
            if self.feature_names is not None:
                features = pd.DataFrame(features, columns=self.feature_names)
            else:
                features = pd.DataFrame(features)
        
        # Check feature names
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder features to match expected order
            features = features[self.feature_names]
        
        # Apply scaling if available
        if self.scaler is not None:
            features_array = self.scaler.transform(features)
        else:
            features_array = features.values
        
        return features_array
    
    def predict(self, features):
        """
        Make predictions with optimizations for NVIDIA containers.
        
        Args:
            features (pd.DataFrame or np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess features
        start_time = time.time()
        features_array = self.preprocess_features(features)
        preprocess_time = time.time() - start_time
        
        # Make predictions
        try:
            inference_start = time.time()
            
            if self.model_type == 'tensorflow':
                # For TensorFlow models, use optimized prediction
                # Convert to tensor with correct dtype
                if isinstance(features_array, np.ndarray):
                    # Use batch prediction for better performance
                    batch_size = 32  # Optimal batch size for most GPUs
                    
                    # If small input, just predict directly
                    if len(features_array) <= batch_size:
                        predictions = self.model.predict(features_array, verbose=0)
                    else:
                        # For larger inputs, use batching for better memory usage
                        predictions = []
                        for i in range(0, len(features_array), batch_size):
                            batch = features_array[i:i+batch_size]
                            batch_preds = self.model.predict(batch, verbose=0)
                            predictions.append(batch_preds)
                        predictions = np.vstack(predictions)
                else:
                    predictions = self.model.predict(features_array, verbose=0)
                    
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                # For XGBoost, use GPU predictor if available
                dmatrix = xgb.DMatrix(features_array)
                
                # Check if running in NVIDIA container
                is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
                if is_nvidia_container:
                    # Use GPU predictor explicitly
                    try:
                        self.model.set_param({'predictor': 'gpu_predictor'})
                    except Exception as e:
                        logger.debug(f"Could not set XGBoost GPU predictor: {e}")
                
                predictions = self.model.predict(dmatrix)
                
            elif self.model_type == 'sklearn':
                predictions = self.model.predict(features_array)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            # Log performance metrics
            logger.debug(f"Prediction stats - Preprocess: {preprocess_time:.4f}s, Inference: {inference_time:.4f}s, Total: {total_time:.4f}s")
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, features):
        """
        Make probability predictions for classification with optimizations for NVIDIA containers.
        
        Args:
            features (pd.DataFrame or np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess features
        start_time = time.time()
        features_array = self.preprocess_features(features)
        preprocess_time = time.time() - start_time
        
        # Make predictions
        try:
            inference_start = time.time()
            
            if self.model_type == 'tensorflow':
                # For TensorFlow models, use optimized prediction
                if isinstance(features_array, np.ndarray):
                    # Use batch prediction for better performance
                    batch_size = 32  # Optimal batch size for most GPUs
                    
                    # If small input, just predict directly
                    if len(features_array) <= batch_size:
                        predictions = self.model.predict(features_array, verbose=0)
                    else:
                        # For larger inputs, use batching for better memory usage
                        predictions = []
                        for i in range(0, len(features_array), batch_size):
                            batch = features_array[i:i+batch_size]
                            batch_preds = self.model.predict(batch, verbose=0)
                            predictions.append(batch_preds)
                        predictions = np.vstack(predictions)
                else:
                    predictions = self.model.predict(features_array, verbose=0)
                    
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                # For XGBoost, use GPU predictor if available
                dmatrix = xgb.DMatrix(features_array)
                
                # Check if running in NVIDIA container
                is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
                if is_nvidia_container:
                    # Use GPU predictor explicitly
                    try:
                        self.model.set_param({'predictor': 'gpu_predictor'})
                    except Exception as e:
                        logger.debug(f"Could not set XGBoost GPU predictor: {e}")
                
                predictions = self.model.predict(dmatrix)
                
            elif self.model_type == 'sklearn':
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(features_array)
                else:
                    raise AttributeError("Model does not support probability predictions")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            # Log performance metrics
            logger.debug(f"Probability prediction stats - Preprocess: {preprocess_time:.4f}s, Inference: {inference_time:.4f}s, Total: {total_time:.4f}s")
            
            return predictions
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            raise
    
    def batch_predict(self, features_list):
        """
        Make predictions for multiple feature sets.
        
        Args:
            features_list (list): List of feature sets
            
        Returns:
            list: List of predictions
        """
        return [self.predict(features) for features in features_list]
        
    def load_model_from_registry(self, model_id=None, model_type=None, ticker=None, stage="production"):
        """
        Load a model from the model registry.
        
        Args:
            model_id (str): ID of the model to load
            model_type (str): Type of model to load
            ticker (str): Ticker symbol for model
            stage (str): Model stage ('production', 'staging', 'development')
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        if self.model_registry is None:
            logger.error("Model registry is required for loading models")
            return False
            
        try:
            # If model_id is provided, load specific model
            if model_id:
                model, metadata = self.model_registry.load_model(model_id, model_type)
                self.model = model
                self.model_type = metadata.metadata.get("model_type", model_type)
                self.feature_names = metadata.metadata.get("feature_names")
                self.current_model_id = model_id
                logger.info(f"Loaded model {model_id} from registry")
                return True
                
            # If ticker and model_type are provided, load latest model for ticker
            elif ticker and model_type:
                model_result = self.model_registry.get_latest_model(model_type, stage)
                if model_result:
                    model, metadata = model_result
                    self.model = model
                    self.model_type = model_type
                    self.feature_names = metadata.metadata.get("feature_names")
                    self.current_model_id = metadata.metadata.get("model_id")
                    logger.info(f"Loaded latest {model_type} model for {ticker} from registry")
                    return True
                else:
                    logger.warning(f"No {model_type} model found for {ticker} in {stage} stage")
                    return False
            else:
                logger.error("Either model_id or both ticker and model_type must be provided")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            return False
            
    def generate_signal(self, ticker, features, confidence_threshold=0.6):
        """
        Generate trading signal based on model prediction.
        
        Args:
            ticker (str): Ticker symbol
            features (pd.DataFrame or np.ndarray): Features for prediction
            confidence_threshold (float): Threshold for signal confidence
            
        Returns:
            dict: Signal information or None if no signal
        """
        if self.model is None:
            # Try to load model from registry if available
            if self.model_registry:
                success = self.load_model_from_registry(ticker=ticker, model_type="xgboost")
                if not success:
                    logger.warning(f"No model available for {ticker}")
                    return None
            else:
                logger.error("No model loaded and no model registry available")
                return None
                
        try:
            # Make prediction
            prediction = self.predict(features)
            
            # Get latest price from features
            if isinstance(features, pd.DataFrame) and 'close' in features.columns:
                price = features['close'].iloc[-1]
            else:
                # Default price if not available
                price = 0.0
                
            # Determine signal direction and confidence
            if self.model_type == 'xgboost' or self.model_type == 'sklearn':
                # For regression models, use the prediction value
                signal_value = float(prediction[-1]) if isinstance(prediction, np.ndarray) else float(prediction)
                
                # Determine direction based on prediction
                if signal_value > confidence_threshold:
                    direction = "buy"
                    confidence = signal_value
                elif signal_value < -confidence_threshold:
                    direction = "sell"
                    confidence = abs(signal_value)
                else:
                    # No signal if confidence is below threshold
                    return None
                    
            elif self.model_type == 'tensorflow':
                # For classification models, use probability
                signal_value = float(prediction[-1]) if isinstance(prediction, np.ndarray) else float(prediction)
                
                # Determine direction based on probability
                if signal_value > confidence_threshold:
                    direction = "buy"
                    confidence = signal_value
                elif signal_value < (1 - confidence_threshold):
                    direction = "sell"
                    confidence = 1 - signal_value
                else:
                    # No signal if confidence is below threshold
                    return None
            else:
                logger.warning(f"Unsupported model type for signal generation: {self.model_type}")
                return None
                
            # Create signal
            signal = {
                "ticker": ticker,
                "direction": direction,
                "confidence": confidence,
                "price": price,
                "timestamp": now().isoformat(),
                "model_id": self.current_model_id,
                "model_type": self.model_type
            }
            
            logger.info(f"Generated {direction} signal for {ticker} with confidence {confidence:.4f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return None
