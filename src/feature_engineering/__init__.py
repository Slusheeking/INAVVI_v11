"""
Feature Engineering Module

This module provides functionality for creating, storing, and analyzing features.
"""

# Import main classes from submodules
from src.feature_engineering.analysis.feature_analyzer import FeatureAnalyzer
from src.feature_engineering.pipeline.feature_pipeline import FeaturePipeline, FeatureEngineer
from src.feature_engineering.store.feature_store import FeatureStore, feature_store

# Define what should be available when importing from this module
__all__ = [
    # Main classes
    "FeatureAnalyzer",
    "FeaturePipeline",
    "FeatureEngineer",
    "FeatureStore",
    
    # Singleton instances
    "feature_store"
]