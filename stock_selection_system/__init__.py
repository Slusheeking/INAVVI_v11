"""
Stock Selection System Package

This package contains modules for GPU-accelerated stock selection and analysis.
It includes components for:
- GPU-optimized data processing
- Real-time market data handling
- Stock selection algorithms
- Day trading strategies
- Integration with external APIs (Polygon.io, Unusual Whales)
"""

__version__ = "1.0.0"

from .gpu_stock_selection_core import GPUStockSelectionSystem
from .gpu_optimized_polygon_api_client import GPUPolygonAPIClient
from .gpu_optimized_polygon_websocket_client import GPUPolygonWebSocketClient
from .gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
from .day_trading_system import *
from .market_data_helpers import *
from .websocket_core import *
from .websocket_enhanced_stock_selection import *
