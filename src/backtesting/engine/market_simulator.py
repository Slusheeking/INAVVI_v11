"""
MarketSimulator: Component for simulating market data in backtests.

This module provides functionality for simulating market data feeds and
conditions during backtesting of trading strategies.
"""
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path('/home/ubuntu/INAVVI_v11-1/src/logs')
log_dir.mkdir(parents=True, exist_ok=True)

# Create file handler
log_file = log_dir / 'market_simulator.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

class MarketSimulator:
    """
    Simulator for market data in backtests.
    
    The MarketSimulator handles the retrieval and processing of historical
    market data for backtesting, including support for multiple data sources,
    timeframes, and asset types.
    """
    
    def __init__(self, 
                data_source: str = "timescaledb",
                start_date: Optional[Union[str, datetime.datetime]] = None,
                end_date: Optional[Union[str, datetime.datetime]] = None,
                timeframes: Optional[List[str]] = None,
                symbols: Optional[List[str]] = None,
                include_options: bool = False,
                include_futures: bool = False):
        """
        Initialize the market simulator.
        
        Args:
            data_source: Source of market data ('timescaledb', 'csv', 'api')
            start_date: Start date for the simulation
            end_date: End date for the simulation
            timeframes: List of timeframes to include
            symbols: List of symbols to include
            include_options: Whether to include options data
            include_futures: Whether to include futures data
        """
        self.data_source = data_source
        
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date
        
        if isinstance(end_date, str):
            self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = end_date
        
        self.timeframes = timeframes or ["1d"]
        self.symbols = symbols or []
        self.include_options = include_options
        self.include_futures = include_futures
        
        # Initialize data storage
        self.data = {}
        self.current_data = {}
        self.current_timestamp = None
        
        logger.info(f"MarketSimulator initialized with {data_source} data source")
    
    def load_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load market data for the simulation.
        
        Args:
            symbols: List of symbols to load data for (overrides instance symbols)
            
        Returns:
            Dict of market data by symbol and timeframe
        """
        symbols_to_load = symbols or self.symbols
        
        if not symbols_to_load:
            logger.warning("No symbols specified for data loading")
            return {}
        
        logger.info(f"Loading market data for {len(symbols_to_load)} symbols")
        
        # Load data based on the data source
        if self.data_source == "timescaledb":
            self._load_from_timescaledb(symbols_to_load)
        elif self.data_source == "csv":
            self._load_from_csv(symbols_to_load)
        elif self.data_source == "api":
            self._load_from_api(symbols_to_load)
        else:
            logger.error(f"Unsupported data source: {self.data_source}")
        
        return self.data
    
    def _load_from_timescaledb(self, symbols: List[str]) -> None:
        """
        Load market data from TimescaleDB.
        
        Args:
            symbols: List of symbols to load data for
        """
        # This is a placeholder for actual TimescaleDB data loading
        # In a real implementation, this would connect to a TimescaleDB instance
        # and execute queries to retrieve the data
        
        logger.info("Loading data from TimescaleDB (placeholder)")
        
        # For each symbol and timeframe, create a placeholder DataFrame
        for symbol in symbols:
            self.data[symbol] = {}
            
            for timeframe in self.timeframes:
                # Create a date range for the simulation period
                if self.start_date and self.end_date:
                    date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
                    
                    # Create a DataFrame with OHLCV data
                    df = pd.DataFrame({
                        'timestamp': date_range,
                        'open': np.zeros(len(date_range)),
                        'high': np.zeros(len(date_range)),
                        'low': np.zeros(len(date_range)),
                        'close': np.zeros(len(date_range)),
                        'volume': np.zeros(len(date_range)),
                    })
                    
                    df = df.set_index('timestamp')
                    
                    self.data[symbol][timeframe] = df
                else:
                    logger.warning(f"Start date or end date not specified for {symbol}")
    
    def _load_from_csv(self, symbols: List[str]) -> None:
        """
        Load market data from CSV files.
        
        Args:
            symbols: List of symbols to load data for
        """
        # This is a placeholder for actual CSV data loading
        # In a real implementation, this would read CSV files from a directory
        
        logger.info("Loading data from CSV files (placeholder)")
        
        # For each symbol and timeframe, create a placeholder DataFrame
        for symbol in symbols:
            self.data[symbol] = {}
            
            for timeframe in self.timeframes:
                # Create a date range for the simulation period
                if self.start_date and self.end_date:
                    date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
                    
                    # Create a DataFrame with OHLCV data
                    df = pd.DataFrame({
                        'timestamp': date_range,
                        'open': np.zeros(len(date_range)),
                        'high': np.zeros(len(date_range)),
                        'low': np.zeros(len(date_range)),
                        'close': np.zeros(len(date_range)),
                        'volume': np.zeros(len(date_range)),
                    })
                    
                    df = df.set_index('timestamp')
                    
                    self.data[symbol][timeframe] = df
                else:
                    logger.warning(f"Start date or end date not specified for {symbol}")
    
    def _load_from_api(self, symbols: List[str]) -> None:
        """
        Load market data from an API.
        
        Args:
            symbols: List of symbols to load data for
        """
        # This is a placeholder for actual API data loading
        # In a real implementation, this would make API calls to retrieve the data
        
        logger.info("Loading data from API (placeholder)")
        
        # For each symbol and timeframe, create a placeholder DataFrame
        for symbol in symbols:
            self.data[symbol] = {}
            
            for timeframe in self.timeframes:
                # Create a date range for the simulation period
                if self.start_date and self.end_date:
                    date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
                    
                    # Create a DataFrame with OHLCV data
                    df = pd.DataFrame({
                        'timestamp': date_range,
                        'open': np.zeros(len(date_range)),
                        'high': np.zeros(len(date_range)),
                        'low': np.zeros(len(date_range)),
                        'close': np.zeros(len(date_range)),
                        'volume': np.zeros(len(date_range)),
                    })
                    
                    df = df.set_index('timestamp')
                    
                    self.data[symbol][timeframe] = df
                else:
                    logger.warning(f"Start date or end date not specified for {symbol}")
    
    def get_data_iterator(self, timeframe: str = "1d") -> Iterator[Tuple[datetime.datetime, Dict[str, Dict[str, Any]]]]:
        """
        Get an iterator for stepping through market data.
        
        Args:
            timeframe: Timeframe to iterate through
            
        Yields:
            Tuple of (timestamp, market data)
        """
        if not self.data:
            logger.warning("No data loaded, call load_data() first")
            return
        
        # Get all timestamps across all symbols for the specified timeframe
        all_timestamps = set()
        for symbol, timeframes in self.data.items():
            if timeframe in timeframes:
                all_timestamps.update(timeframes[timeframe].index)
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Iterate through timestamps
        for timestamp in sorted_timestamps:
            # Skip timestamps outside the simulation period
            if self.start_date and timestamp < self.start_date:
                continue
            if self.end_date and timestamp > self.end_date:
                continue
            
            # Get market data for this timestamp
            market_data = {}
            for symbol, timeframes in self.data.items():
                if timeframe in timeframes and timestamp in timeframes[timeframe].index:
                    row = timeframes[timeframe].loc[timestamp]
                    
                    market_data[symbol] = {
                        'timestamp': timestamp,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'price': row['close'],  # Use close as the current price
                    }
            
            self.current_timestamp = timestamp
            self.current_data = market_data
            
            yield timestamp, market_data
    
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           lookback: int = 100,
                           fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get historical data for a symbol up to the current timestamp.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            lookback: Number of bars to look back
            fields: List of fields to include (default: all)
            
        Returns:
            DataFrame with historical data
        """
        if not self.current_timestamp:
            logger.warning("No current timestamp, call get_data_iterator() first")
            return pd.DataFrame()
        
        if symbol not in self.data or timeframe not in self.data[symbol]:
            logger.warning(f"No data available for {symbol} at {timeframe} timeframe")
            return pd.DataFrame()
        
        # Get data up to the current timestamp
        df = self.data[symbol][timeframe]
        df = df[df.index <= self.current_timestamp]
        
        # Apply lookback limit
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        # Filter fields if specified
        if fields:
            df = df[fields]
        
        return df
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price
        """
        if not self.current_data or symbol not in self.current_data:
            logger.warning(f"No current data available for {symbol}")
            return 0.0
        
        return self.current_data[symbol].get('price', 0.0)
    
    def get_current_bar(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current bar data for a symbol.
        
        Args:
            symbol: Symbol to get bar data for
            
        Returns:
            Dict with bar data
        """
        if not self.current_data or symbol not in self.current_data:
            logger.warning(f"No current data available for {symbol}")
            return {}
        
        return self.current_data[symbol]
    
    def get_current_timestamp(self) -> datetime.datetime:
        """
        Get the current timestamp.
        
        Returns:
            Current timestamp
        """
        return self.current_timestamp
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the simulation.
        
        Args:
            symbol: Symbol to add
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"Added symbol {symbol} to simulation")
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the simulation.
        
        Args:
            symbol: Symbol to remove
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.data:
                del self.data[symbol]
            logger.info(f"Removed symbol {symbol} from simulation")
    
    def set_timeframes(self, timeframes: List[str]) -> None:
        """
        Set the timeframes for the simulation.
        
        Args:
            timeframes: List of timeframes
        """
        self.timeframes = timeframes
        logger.info(f"Set timeframes to {timeframes}")
    
    def get_volume_data(self) -> Dict[str, float]:
        """
        Get volume data for all symbols at the current timestamp.
        
        Returns:
            Dict of volumes by symbol
        """
        volumes = {}
        
        if self.current_data:
            for symbol, data in self.current_data.items():
                volumes[symbol] = data.get('volume', 0.0)
        
        return volumes
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """
        Get market statistics at the current timestamp.
        
        Returns:
            Dict of market statistics
        """
        if not self.current_data:
            return {}
        
        # Calculate market statistics
        prices = [data.get('price', 0.0) for data in self.current_data.values()]
        volumes = [data.get('volume', 0.0) for data in self.current_data.values()]
        
        stats = {
            'timestamp': self.current_timestamp,
            'symbols_count': len(self.current_data),
            'avg_price': np.mean(prices) if prices else 0.0,
            'total_volume': np.sum(volumes) if volumes else 0.0,
            'price_range': (np.min(prices), np.max(prices)) if prices else (0.0, 0.0),
        }
        
        return stats