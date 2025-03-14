"""
Data Pipeline Module

This package provides the main pipeline for acquiring market data from various sources
and storing it in the database.
"""

from src.data_acquisition.pipeline.data_pipeline import DataPipeline

__all__ = ["DataPipeline"]