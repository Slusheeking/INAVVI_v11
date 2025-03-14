"""
Feature Analyzer for the Autonomous Trading System.

This module provides functionality for analyzing feature importance and correlation.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """
    Analyzer for feature importance and correlation.
    """
    
    def __init__(self):
        """
        Initialize the feature analyzer.
        """
        self.feature_importances = {}
        self.feature_correlations = {}
        
    def analyze_feature_importance(self, X, y, method='random_forest'):
        """
        Analyze feature importance using various methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Method to use for feature importance analysis
                Options: 'random_forest', 'mutual_info'
                
        Returns:
            pd.Series: Feature importances
        """
        if method == 'random_forest':
            return self._random_forest_importance(X, y)
        elif method == 'mutual_info':
            return self._mutual_info_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _random_forest_importance(self, X, y):
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            pd.Series: Feature importances
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        self.feature_importances['random_forest'] = importances
        return importances
    
    def _mutual_info_importance(self, X, y):
        """
        Calculate feature importance using Mutual Information.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            pd.Series: Feature importances
        """
        importances = pd.Series(
            mutual_info_regression(X, y),
            index=X.columns
        )
        importances = importances.sort_values(ascending=False)
        self.feature_importances['mutual_info'] = importances
        return importances
    
    def analyze_feature_correlation(self, X, method='pearson'):
        """
        Analyze feature correlation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            method (str): Correlation method
                Options: 'pearson', 'spearman', 'kendall'
                
        Returns:
            pd.DataFrame: Correlation matrix
        """
        corr_matrix = X.corr(method=method)
        self.feature_correlations[method] = corr_matrix
        return corr_matrix
    
    def get_highly_correlated_features(self, threshold=0.8, method='pearson'):
        """
        Get highly correlated feature pairs.
        
        Args:
            threshold (float): Correlation threshold
            method (str): Correlation method
            
        Returns:
            list: List of tuples (feature1, feature2, correlation)
        """
        if method not in self.feature_correlations:
            raise ValueError(f"Correlation matrix for method {method} not found. Run analyze_feature_correlation first.")
        
        corr_matrix = self.feature_correlations[method]
        highly_correlated = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find feature pairs with correlation greater than threshold
        for col in upper.columns:
            for idx, value in upper[col].items():
                if abs(value) > threshold:
                    highly_correlated.append((idx, col, value))
        
        return highly_correlated
    
    def get_redundant_features(self, threshold=0.95, method='pearson'):
        """
        Get redundant features that can be removed.
        
        Args:
            threshold (float): Correlation threshold
            method (str): Correlation method
            
        Returns:
            list: List of features that can be removed
        """
        highly_correlated = self.get_highly_correlated_features(threshold, method)
        
        # Group by feature1
        feature_groups = {}
        for f1, f2, corr in highly_correlated:
            if f1 not in feature_groups:
                feature_groups[f1] = []
            feature_groups[f1].append(f2)
            
            if f2 not in feature_groups:
                feature_groups[f2] = []
            feature_groups[f2].append(f1)
        
        # Find redundant features
        redundant_features = []
        for feature, correlated_features in feature_groups.items():
            if feature not in redundant_features and len(correlated_features) > 0:
                redundant_features.extend(correlated_features)
        
        return list(set(redundant_features))
