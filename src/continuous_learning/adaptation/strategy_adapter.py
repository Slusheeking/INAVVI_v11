"""
Strategy Adapter for the Autonomous Trading System.

This module provides functionality for adapting strategy parameters.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class StrategyAdapter:
    """
    Adapter for strategy parameters.
    """
    
    def __init__(self, strategy_params=None):
        """
        Initialize the strategy adapter.
        
        Args:
            strategy_params (dict): Initial strategy parameters
        """
        self.strategy_params = strategy_params or {}
        self.param_history = []
        self.performance_history = []
        self.adaptation_model = None
    
    def record_performance(self, params, performance_metrics):
        """
        Record strategy performance.
        
        Args:
            params (dict): Strategy parameters
            performance_metrics (dict): Performance metrics
        """
        self.param_history.append(params.copy())
        self.performance_history.append(performance_metrics.copy())
        
        logger.info(f"Recorded performance for parameters: {params}")
        logger.info(f"Performance metrics: {performance_metrics}")
    
    def train_adaptation_model(self, target_metric='sharpe_ratio', model_type='random_forest'):
        """
        Train a model to predict performance based on parameters.
        
        Args:
            target_metric (str): Target performance metric
            model_type (str): Type of model to use
            
        Returns:
            object: Trained model
        """
        if len(self.param_history) < 5:
            logger.warning("Not enough data to train adaptation model")
            return None
        
        # Prepare data
        X = pd.DataFrame(self.param_history)
        y = np.array([metrics.get(target_metric, 0) for metrics in self.performance_history])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Adaptation model trained. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        self.adaptation_model = model
        return model
    
    def suggest_parameters(self, param_ranges, n_samples=100, target_metric='sharpe_ratio'):
        """
        Suggest improved strategy parameters.
        
        Args:
            param_ranges (dict): Ranges for parameters
            n_samples (int): Number of parameter combinations to evaluate
            target_metric (str): Target performance metric
            
        Returns:
            dict: Suggested parameters
        """
        if self.adaptation_model is None:
            self.train_adaptation_model(target_metric)
            
            if self.adaptation_model is None:
                logger.warning("Could not train adaptation model. Using current parameters.")
                return self.strategy_params.copy()
        
        # Generate random parameter combinations
        param_samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    sample[param] = np.random.randint(min_val, max_val + 1)
                else:
                    sample[param] = min_val + np.random.random() * (max_val - min_val)
            param_samples.append(sample)
        
        # Predict performance for each combination
        X_samples = pd.DataFrame(param_samples)
        y_pred = self.adaptation_model.predict(X_samples)
        
        # Find best parameters
        best_idx = np.argmax(y_pred)
        best_params = param_samples[best_idx]
        
        logger.info(f"Suggested parameters: {best_params}")
        logger.info(f"Predicted {target_metric}: {y_pred[best_idx]:.4f}")
        
        return best_params
    
    def adapt_parameters(self, market_regime=None):
        """
        Adapt strategy parameters based on market regime.
        
        Args:
            market_regime (str): Current market regime
            
        Returns:
            dict: Adapted parameters
        """
        if market_regime is None:
            return self.strategy_params.copy()
        
        # Filter performance history by market regime
        regime_indices = [i for i, metrics in enumerate(self.performance_history) 
                         if metrics.get('market_regime') == market_regime]
        
        if not regime_indices:
            logger.warning(f"No data for market regime: {market_regime}")
            return self.strategy_params.copy()
        
        # Get best parameters for this regime
        regime_performances = [self.performance_history[i].get('sharpe_ratio', 0) for i in regime_indices]
        best_idx = regime_indices[np.argmax(regime_performances)]
        best_params = self.param_history[best_idx]
        
        logger.info(f"Adapted parameters for {market_regime}: {best_params}")
        
        return best_params
    
    def update_strategy_params(self, new_params):
        """
        Update strategy parameters.
        
        Args:
            new_params (dict): New parameters
            
        Returns:
            dict: Updated parameters
        """
        self.strategy_params.update(new_params)
        logger.info(f"Updated strategy parameters: {self.strategy_params}")
        return self.strategy_params
