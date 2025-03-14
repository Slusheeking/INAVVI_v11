"""
Data Storage Module

This package provides functionality for storing market data in TimescaleDB.
"""

from src.data_acquisition.storage.timescale_storage import TimescaleStorage

__all__ = ["TimescaleStorage"]