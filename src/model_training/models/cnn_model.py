"""
CNN Model for the Autonomous Trading System.

This module provides a Convolutional Neural Network (CNN) model for time series prediction.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class CNNModel:
    """
    Convolutional Neural Network (CNN) model for time series prediction.
    """
    
    def __init__(self, input_shape, output_dim=1, filters=64, kernel_size=3, 
                 pool_size=2, dense_units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, features)
            output_dim (int): Dimension of output (1 for regression, >1 for classification)
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of kernel in convolutional layers
            pool_size (int): Size of pooling window
            dense_units (int): Number of units in dense layers
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the CNN model.
        
        Returns:
            tf.keras.Model: Built model
        """
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv1D(filters=self.filters, 
                         kernel_size=self.kernel_size, 
                         activation='relu', 
                         input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Second convolutional layer
        model.add(Conv1D(filters=self.filters*2, 
                         kernel_size=self.kernel_size, 
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.dense_units // 2, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.output_dim == 1:
            model.add(Dense(1))  # Regression
        else:
            model.add(Dense(self.output_dim, activation='softmax'))  # Classification
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        if self.output_dim == 1:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
            patience=10, model_path=None, verbose=1):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of epochs
            batch_size (int): Batch size
            patience (int): Patience for early stopping
            model_path (str): Path to save the best model
            verbose (int): Verbosity level
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = []
        
        # Early stopping
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks.append(early_stopping)
        
        # Model checkpoint
        if model_path is not None:
            checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss' if X_val is not None else 'loss')
            callbacks.append(checkpoint)
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
        
    def summary(self):
        """
        Get model summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
