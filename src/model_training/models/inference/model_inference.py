"""
Model Inference for the Autonomous Trading System.

This module provides functionality for model inference.
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import json

logger = logging.getLogger(__name__)

class ModelInference:
    """
    Inference engine for model predictions.
    """
    
    def __init__(self, model_path=None, model_type=None, feature_names=None, scaler=None):
        """
        Initialize the model inference engine.
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model ('tensorflow', 'xgboost', 'sklearn')
            feature_names (list): List of feature names
            scaler (object): Scaler for feature normalization
        """
        self.model_path = model_path
        self.model_type = model_type
        self.feature_names = feature_names
        self.scaler = scaler
        self.model = None
        
        if model_path is not None and model_type is not None:
            self.load_model(model_path, model_type)
    
    def load_model(self, model_path, model_type):
        """
        Load the model.
        
        Args:
            model_path (str): Path to the model file
            model_type (str): Type of model ('tensorflow', 'xgboost', 'sklearn')
        """
        self.model_path = model_path
        self.model_type = model_type
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            if model_type == 'tensorflow':
                self.model = tf.keras.models.load_model(model_path)
            elif model_type == 'xgboost':
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(model_path)
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
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
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
        Make predictions.
        
        Args:
            features (pd.DataFrame or np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess features
        features_array = self.preprocess_features(features)
        
        # Make predictions
        try:
            if self.model_type == 'tensorflow':
                predictions = self.model.predict(features_array)
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features_array)
                predictions = self.model.predict(dmatrix)
            elif self.model_type == 'sklearn':
                predictions = self.model.predict(features_array)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, features):
        """
        Make probability predictions for classification.
        
        Args:
            features (pd.DataFrame or np.ndarray): Features for prediction
            
        Returns:
            np.ndarray: Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess features
        features_array = self.preprocess_features(features)
        
        # Make predictions
        try:
            if self.model_type == 'tensorflow':
                predictions = self.model.predict(features_array)
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features_array)
                predictions = self.model.predict(dmatrix)
            elif self.model_type == 'sklearn':
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(features_array)
                else:
                    raise AttributeError("Model does not support probability predictions")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
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
