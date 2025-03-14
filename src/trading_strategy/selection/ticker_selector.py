"""
Ticker Selector Module

This module provides functionality for selecting tickers based on various criteria
such as market cap, volume, and volatility.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from autonomous_trading_system.src.data_acquisition.api.polygon_client import PolygonClient
from autonomous_trading_system.src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient

logger = logging.getLogger(__name__)

class TickerSelector:
    """Class for selecting tickers based on various criteria."""
    
    def __init__(self, polygon_client: PolygonClient):
        """
        Initialize the ticker selector.
        
        Args:
            polygon_client: Polygon API client
        """
        self.polygon_client = polygon_client
        
    async def select_tickers(self, limit: int = 10) -> List[str]:
        """
        Select tickers based on market cap and volume.
        
        Args:
            limit: Maximum number of tickers to select
            
        Returns:
            List of selected ticker symbols
        """
        try:
            # For now, return a fixed list of major tech stocks
            # In production, this would query Polygon's screener endpoints
            # to select stocks based on criteria
            tickers = [
                "AAPL",  # Apple
                "MSFT",  # Microsoft
                "GOOGL", # Google
                "AMZN",  # Amazon
                "META",  # Meta (Facebook)
                "NVDA",  # NVIDIA
                "TSLA",  # Tesla
                "AMD",   # AMD
                "INTC",  # Intel
                "CRM"    # Salesforce
            ]
            
            logger.info(f"Selected {len(tickers)} tickers")
            return tickers[:limit]
            
        except Exception as e:
            logger.error(f"Error selecting tickers: {e}")
            # Return a minimal set of tickers to allow testing to continue
            return ["AAPL", "MSFT", "GOOGL"]


class DynamicTickerSelector:
    """
    Advanced ticker selector that dynamically selects tickers based on real-time market conditions.
    
    This class uses Polygon API data to select tickers based on multiple criteria:
    1. Market capitalization
    2. Trading volume
    3. Price volatility
    4. Sector performance
    5. Liquidity
    """
    
    def __init__(self, polygon_client: Optional[PolygonClient] = None, unusual_whales_client: Optional[UnusualWhalesClient] = None, max_tickers: int = 20):
        """
        Initialize the dynamic ticker selector.
        
        Args:
            polygon_client: Polygon API client for market data access (optional)
            max_tickers: Maximum number of tickers to track
        """
        self.polygon_client = polygon_client
        self.unusual_whales_client = unusual_whales_client
        self.max_tickers = max_tickers
        self.market_data = {}
        self.opportunity_scores = {}
        self.price_data = {}
        self.ticker_details = {}
        
        # Load trading parameters from environment variables
        self.max_position_size = float(os.environ.get("MAX_POSITION_SIZE", 2500))
        self.risk_percentage = float(os.environ.get("RISK_PERCENTAGE", 0.02))
        self.max_positions = int(os.environ.get("MAX_POSITIONS", 50))
        self.max_total_position_value = 5000  # Maximum total position value
        self._ticker_universe = []  # Empty list to start with
        logger.info(f"Initialized DynamicTickerSelector with max_position_size={self.max_position_size}, risk_percentage={self.risk_percentage}, max_positions={self.max_positions}, max_total={self.max_total_position_value}")
        
    @property
    def ticker_universe(self) -> List[str]:
        """
        Get the current ticker universe.
        
        Returns:
            List of ticker symbols in the universe
        """
        return self._ticker_universe
    
    async def fetch_ticker_universe(self, market_type: str = "stocks", limit: int = 5000) -> List[str]:
        """
        Dynamically fetch a universe of tickers from Polygon API.
        
        Args:
            market_type: Type of market ('stocks', 'options', 'forex', 'crypto')
            limit: Maximum number of tickers to fetch
            
        Returns:
            List of ticker symbols
        """
        if not self.polygon_client:
            logger.warning("No Polygon client available to fetch ticker universe")
            return self._ticker_universe
            
        try:
            # Use Polygon's tickers endpoint to get active tickers
            # This would typically be implemented in the PolygonClient class
            # For now, we'll simulate the API call with a direct request
            
            # Parameters for filtering tickers
            params = {
                "market": market_type,
                "active": "true",
                "sort": "ticker",
                "order": "asc", 
                "limit": limit
            }
            
            # Make the request to Polygon API
            endpoint = "/v3/reference/tickers"
            result = await self.polygon_client._make_request_async(endpoint, self.polygon_client._add_api_key(params))
            
            # Extract tickers from the response
            tickers = []
            if "results" in result:
                for item in result["results"]:
                    ticker = item.get("ticker")
                    if ticker:
                        tickers.append(ticker)
                        # Store ticker details for later use
                        self.ticker_details[ticker] = item
                        
            logger.info(f"Fetched {len(tickers)} tickers from Polygon API")
            self._ticker_universe = tickers
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching ticker universe: {e}")
            return self._ticker_universe
    
    def update_market_data(self, market_data: dict) -> None:
        """
        Update the market data used for ticker selection.
        
        Args:
            market_data: Dictionary mapping ticker symbols to market data
                Each entry should have 'ohlcv' (DataFrame) and 'metadata' (dict) keys
        """
        self.market_data = market_data
        
        # Extract price data for each ticker
        for ticker, data in market_data.items():
            if 'ohlcv' in data and not data['ohlcv'].empty:
                # Get the latest price
                df = data['ohlcv']
                if 'close' in df.columns:
                    latest_price = df['close'].iloc[-1] if len(df) > 0 else None
                    if latest_price is not None:
                        self.price_data[ticker] = latest_price
        
        logger.info(f"Updated market data for {len(market_data)} tickers, price data for {len(self.price_data)} tickers")
        
        # Update ticker universe with the keys from market_data
        if market_data:
            # Only update ticker universe if it's empty
            if not self._ticker_universe:
                self._ticker_universe = list(market_data.keys())
            logger.info(f"Updated ticker universe with {len(self._ticker_universe)} tickers")
        
        # If ticker universe is still empty, use a default set for testing
        if not self._ticker_universe:
            self._ticker_universe = ["SPY"]  # At least include SPY as a fallback
        
    def calculate_opportunity_scores(self) -> dict:
        """
        Calculate opportunity scores for each ticker based on market data.
        
        Returns:
            Dictionary mapping ticker symbols to opportunity scores
        """
        scores = {}
        
        try:
            # If market data is empty, return empty scores
            if not self.market_data:
                return scores
                
            for ticker, data in self.market_data.items():
                if 'metadata' not in data:
                    continue
                    
                metadata = data['metadata']
                
                # Calculate opportunity score based on volume and volatility
                # If volume is not available in metadata, try to get it from ticker details
                volume = metadata.get('volume', 0)
                if volume == 0 and ticker in self.ticker_details:
                    # Try to get volume from ticker details
                    ticker_detail = self.ticker_details.get(ticker, {})
                    if 'last_trade' in ticker_detail:
                        volume = ticker_detail['last_trade'].get('v', 0)
                    
                # Calculate scores
                volume_score = min(1.0, metadata.get('volume', 0) / 1_000_000)
                volatility_score = min(1.0, metadata.get('atr_pct', 0) / 5.0)
                price_score = 0.0
                
                # Get the latest price for this ticker
                latest_price = self.price_data.get(ticker)
                
                # Calculate price score based on position sizing constraints
                if latest_price is not None and latest_price > 0:
                    # Calculate how many shares we can buy with max_position_size
                    max_shares = self.max_position_size / latest_price
                    
                    # Calculate the dollar risk per share
                    dollar_risk_per_share = latest_price * self.risk_percentage
                    
                    # Calculate total dollar risk for the position
                    total_dollar_risk = dollar_risk_per_share * max_shares
                    
                    # Score based on risk-reward ratio (higher is better)
                    if total_dollar_risk > 0:
                        price_score = min(1.0, self.max_position_size / (total_dollar_risk * 100))
                    
                    # Penalize if price is too high to buy a reasonable number of shares
                    if max_shares < 10:  # Arbitrary threshold for minimum shares
                        price_score *= 0.5
                
                score = (volume_score * 0.3 + volatility_score * 0.4 + price_score * 0.3)  # Weighted scoring
                
                scores[ticker] = score
            
            self.opportunity_scores = scores
            logger.info(f"Calculated opportunity scores for {len(scores)} tickers")
            
        except Exception as e:
            logger.error(f"Error calculating opportunity scores: {e}")
            
        return scores
    
    def select_active_tickers(self, min_score: float = 0.0, enforce_position_limits: bool = True) -> List[str]:
        """
        Select active tickers based on opportunity scores.
        
        Args:
            min_score: Minimum opportunity score to include a ticker
            
        Returns:
            List of selected ticker symbols
        """
        try:
            # Filter tickers by minimum score
            if not self.opportunity_scores:
                # Calculate scores if not already done
                self.calculate_opportunity_scores()
                
            qualified_tickers = [
                ticker for ticker, score in self.opportunity_scores.items()
                if score >= min_score
            ]
            
            # Sort by score in descending order
            sorted_tickers = sorted(
                qualified_tickers,
                key=lambda t: self.opportunity_scores.get(t, 0),
                reverse=True
            )
            
            if enforce_position_limits:
                # Apply position sizing constraints
                selected_tickers = []
                total_position_value = 0.0
                position_count = 0
                
                for ticker in sorted_tickers:
                    # Get the latest price
                    latest_price = self.price_data.get(ticker)
                    
                    if latest_price is None or latest_price <= 0:
                        continue
                    
                    # Calculate position size based on max_position_size
                    position_size = min(self.max_position_size, latest_price * 100)  # Limit to 100 shares or max_position_size
                    
                    # Check if adding this position would exceed max_total_position_value
                    if total_position_value + position_size > self.max_total_position_value:
                        # Skip if we've already reached the max total
                        if total_position_value >= self.max_total_position_value:
                            continue
                        
                        # Otherwise, adjust position size to fit within max_total_position_value
                        position_size = self.max_total_position_value - total_position_value
                    
                    # Add ticker if position size is reasonable
                    if position_size >= 100:  # Minimum position size of $100
                        selected_tickers.append(ticker)
                        total_position_value += position_size
                        position_count += 1
                        
                        # Stop if we've reached max positions
                        if position_count >= self.max_positions:
                            break
            else:
                # Limit to max_positions
                selected_tickers = sorted_tickers[:self.max_positions]
            
            logger.info(f"Selected {len(selected_tickers)} active tickers")
            return selected_tickers
        
        except Exception as e:
            logger.error(f"Error selecting active tickers: {e}")
            # Return a minimal set of tickers as fallback
            if self._ticker_universe:
                return self._ticker_universe[:min(self.max_positions, len(self._ticker_universe))]
            else:
                return ["SPY"]  # Default to SPY as absolute fallback
    
    async def select_tickers(self, limit: int = 20) -> List[str]:
        """
        Asynchronous method to select tickers (for compatibility with TickerSelector).
        
        Args:
            limit: Maximum number of tickers to select
            
        Returns:
            List of selected ticker symbols
        """
        active_tickers = self.select_active_tickers()
        return active_tickers[:limit]
