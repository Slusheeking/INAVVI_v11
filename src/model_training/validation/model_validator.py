"""
Model Validator for the Autonomous Trading System.

This module provides functionality for validating model performance.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validator for model performance.
    """
    
    def __init__(self):
        """
        Initialize the model validator.
        """
        self.metrics = {}
        
    def validate_regression(self, y_true, y_pred):
        """
        Validate regression model performance.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: Regression metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
        }
        
        self.metrics['regression'] = metrics
        return metrics
    
    def validate_classification(self, y_true, y_pred, average='weighted'):
        """
        Validate classification model performance.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            average (str): Averaging method for multi-class metrics
            
        Returns:
            dict: Classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.metrics['classification'] = metrics
        return metrics
    
    def validate_binary_classification(self, y_true, y_pred_proba, threshold=0.5):
        """
        Validate binary classification model performance.
        
        Args:
            y_true (np.ndarray): True values
            y_pred_proba (np.ndarray): Predicted probabilities
            threshold (float): Classification threshold
            
        Returns:
            dict: Binary classification metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'threshold': threshold
        }
        
        self.metrics['binary_classification'] = metrics
        return metrics
    
    def validate_time_series(self, y_true, y_pred, horizon=1):
        """
        Validate time series model performance.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            horizon (int): Forecast horizon
            
        Returns:
            dict: Time series metrics
        """
        # Basic regression metrics
        metrics = self.validate_regression(y_true, y_pred)
        
        # Direction accuracy
        if len(y_true) > horizon:
            true_direction = np.sign(y_true[horizon:] - y_true[:-horizon])
            pred_direction = np.sign(y_pred[horizon:] - y_true[:-horizon])
            direction_accuracy = np.mean(true_direction == pred_direction)
            metrics['direction_accuracy'] = direction_accuracy
        
        self.metrics['time_series'] = metrics
        return metrics
    
    def validate_trading_strategy(self, y_true, y_pred, prices):
        """
        Validate trading strategy performance.
        
        Args:
            y_true (np.ndarray): True values (1 for buy, -1 for sell, 0 for hold)
            y_pred (np.ndarray): Predicted values (1 for buy, -1 for sell, 0 for hold)
            prices (np.ndarray): Asset prices
            
        Returns:
            dict: Trading strategy metrics
        """
        # Calculate returns
        true_returns = np.zeros(len(prices) - 1)
        pred_returns = np.zeros(len(prices) - 1)
        
        for i in range(len(prices) - 1):
            price_return = (prices[i+1] - prices[i]) / prices[i]
            true_returns[i] = y_true[i] * price_return
            pred_returns[i] = y_pred[i] * price_return
        
        # Calculate metrics
        metrics = {
            'true_total_return': np.sum(true_returns),
            'pred_total_return': np.sum(pred_returns),
            'true_sharpe_ratio': np.mean(true_returns) / np.std(true_returns) if np.std(true_returns) > 0 else 0,
            'pred_sharpe_ratio': np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0,
            'true_max_drawdown': self._calculate_max_drawdown(true_returns),
            'pred_max_drawdown': self._calculate_max_drawdown(pred_returns),
            'signal_accuracy': np.mean(y_true == y_pred)
        }
        
        self.metrics['trading_strategy'] = metrics
        return metrics
    
    def _calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown.
        
        Args:
            returns (np.ndarray): Returns
            
        Returns:
            float: Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    def get_metrics(self, metric_type=None):
        """
        Get metrics.
        
        Args:
            metric_type (str): Type of metrics to get
            
        Returns:
            dict: Metrics
        """
        if metric_type is not None:
            if metric_type not in self.metrics:
                raise ValueError(f"Metric type {metric_type} not found")
            return self.metrics[metric_type]
        
        return self.metrics
    
    def print_metrics(self, metric_type=None):
        """
        Print metrics.
        
        Args:
            metric_type (str): Type of metrics to print
        """
        metrics = self.get_metrics(metric_type)
        
        if isinstance(metrics, dict):
            for metric_type, metric_values in metrics.items():
                print(f"\n{metric_type.upper()} METRICS:")
                for metric_name, metric_value in metric_values.items():
                    if isinstance(metric_value, list):
                        print(f"  {metric_name}:")
                        for item in metric_value:
                            print(f"    {item}")
                    else:
                        print(f"  {metric_name}: {metric_value}")
        else:
            print("\nMETRICS:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, list):
                    print(f"  {metric_name}:")
                    for item in metric_value:
                        print(f"    {item}")
                else:
                    print(f"  {metric_name}: {metric_value}")
