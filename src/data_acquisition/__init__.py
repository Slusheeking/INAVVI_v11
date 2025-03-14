"""
Data Acquisition Package

This package provides functionality for acquiring market data from various sources
and storing it in the database.
"""

# Import main components for easier access
from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from src.data_acquisition.collectors import (
    PriceCollector,
    QuoteCollector,
    TradeCollector,
    OptionsCollector,
    MultiTimeframeDataCollector,
)
from src.data_acquisition.pipeline.data_pipeline import DataPipeline
from src.data_acquisition.storage.timescale_storage import TimescaleStorage
from src.data_acquisition.transformation.data_transformer import DataTransformer

__all__ = [
    "PolygonClient",
    "UnusualWhalesClient",
    "PriceCollector",
    "QuoteCollector",
    "TradeCollector",
    "OptionsCollector",
    "MultiTimeframeDataCollector",
    "DataPipeline",
    "TimescaleStorage",
    "DataTransformer",
]