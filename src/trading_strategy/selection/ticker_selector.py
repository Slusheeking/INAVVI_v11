"""
Ticker Selector Module

This module provides functionality for selecting tickers based on various criteria
such as market cap, volume, and volatility.
"""

from typing import List, Optional
import os

from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from src.utils.logging import get_logger

logger = get_logger("trading_strategy.selection.ticker_selector")

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
        Initialize the dynamic ticker selector with dollar-based position limits.
        
        Args:
            polygon_client: Polygon API client for market data access (optional)
            unusual_whales_client: Unusual Whales API client for options data (optional)
            max_tickers: Maximum number of tickers to track
        """
        self.polygon_client = polygon_client
        self.unusual_whales_client = unusual_whales_client
        self.max_tickers = max_tickers
        self.market_data = {}
        self.opportunity_scores = {}
        self.price_data = {}
        self.ticker_details = {}
        
        # Get dollar-based position limits from API clients
        self.max_position_value = 2500.0  # Default $2500 per stock
        self.max_daily_value = 5000.0     # Default $5000 per day
        
        # If polygon_client is provided, use its limits
        if self.polygon_client:
            self.max_position_value = getattr(self.polygon_client, 'max_position_value', 2500.0)
            self.max_daily_value = getattr(self.polygon_client, 'max_daily_value', 5000.0)
        
        # Load trading parameters from environment variables
        self.risk_percentage = float(os.environ.get("RISK_PERCENTAGE", 0.02))
        self.max_positions = int(os.environ.get("MAX_POSITIONS", 50))
        self._ticker_universe = []  # Empty list to start with
        
        logger.info(f"Initialized DynamicTickerSelector with max_position_value=${self.max_position_value:.2f}, max_daily_value=${self.max_daily_value:.2f}, risk_percentage={self.risk_percentage}, max_positions={self.max_positions}")
        
    @property
    def ticker_universe(self) -> List[str]:
        """
        Get the current ticker universe.
        
        Returns:
            List of ticker symbols in the universe
        """
        return self._ticker_universe
    
    async def fetch_ticker_universe(self, market_type: str = "stocks", limit: int = 5000, price_min: float = 1.0, price_max: float = None) -> List[str]:
        """
        Dynamically fetch a universe of tickers from Polygon API with price filtering for dollar-based position limits.
        
        Args:
            market_type: Type of market ('stocks', 'options', 'forex', 'crypto')
            limit: Maximum number of tickers to fetch
            price_min: Minimum price for tickers (default: $1.0)
            price_max: Maximum price for tickers (default: None)
            
        Returns:
            List of ticker symbols
        """
        if not self.polygon_client:
            logger.warning("No Polygon client available to fetch ticker universe")
            return self._ticker_universe
            
        try:
            # Calculate maximum price based on dollar-based position limits
            # If we want to buy at least 10 shares, the max price would be max_position_value / 10
            if price_max is None:
                price_max = self.max_position_value / 10  # Default to allow at least 10 shares
            
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
            filtered_tickers = []
            if "results" in result:
                for item in result["results"]:
                    ticker = item.get("ticker")
                    if ticker:
                        tickers.append(ticker)
                        # Store ticker details for later use
                        self.ticker_details[ticker] = item
                        
                        # Get the last price if available
                        last_price = None
                        if "last_quote" in item and "p" in item["last_quote"]:
                            last_price = item["last_quote"]["p"]
                        elif "last_trade" in item and "p" in item["last_trade"]:
                            last_price = item["last_trade"]["p"]
                        
                        # Filter by price if available
                        if last_price is not None:
                            self.price_data[ticker] = last_price
                            
                            # Check if price is within our range
                            if price_min <= last_price <= price_max:
                                filtered_tickers.append(ticker)
                                
                                # Calculate position value
                                position_value = last_price * min(100, int(self.max_position_value / last_price))
                                
                                # Log the ticker with its price and potential position value
                                logger.debug(f"Ticker {ticker} with price ${last_price:.2f}, potential position value ${position_value:.2f}")
            
            # If we have filtered tickers, use them; otherwise use all tickers
            if filtered_tickers:
                logger.info(f"Fetched {len(tickers)} tickers, filtered to {len(filtered_tickers)} within price range ${price_min:.2f}-${price_max:.2f}")
                self._ticker_universe = filtered_tickers
                return filtered_tickers
            else:
                logger.info(f"Fetched {len(tickers)} tickers from Polygon API (no price filtering applied)")
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
        Calculate opportunity scores for each ticker based on market data and dollar-based position limits.
        
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
                position_limit_score = 0.0
                
                # Get the latest price for this ticker
                latest_price = self.price_data.get(ticker)
                
                # Calculate price score based on position sizing constraints
                if latest_price is not None and latest_price > 0:
                    # Calculate how many shares we can buy with max_position_value
                    max_shares = self.max_position_value / latest_price
                    
                    # Calculate the dollar risk per share
                    dollar_risk_per_share = latest_price * self.risk_percentage
                    
                    # Calculate total dollar risk for the position
                    total_dollar_risk = dollar_risk_per_share * max_shares
                    
                    # Score based on risk-reward ratio (higher is better)
                    if total_dollar_risk > 0:
                        price_score = min(1.0, self.max_position_value / (total_dollar_risk * 100))
                    
                    # Calculate position limit score based on how well the stock fits within our dollar limits
                    # Higher score for stocks that allow for a reasonable number of shares within our limits
                    if max_shares >= 100:  # Ideal: can buy at least 100 shares
                        position_limit_score = 1.0
                    elif max_shares >= 10:  # Good: can buy at least 10 shares
                        position_limit_score = 0.8
                    elif max_shares >= 1:   # Acceptable: can buy at least 1 share
                        position_limit_score = 0.5
                    else:                   # Poor: can't even buy 1 share
                        position_limit_score = 0.0
                    
                    # Check if this position can be taken based on dollar limits
                    position_value = latest_price * min(100, max_shares)  # Standard lot size or max shares
                    
                    # Check with API clients if this position can be taken
                    can_take_position = True
                    if self.polygon_client and hasattr(self.polygon_client, 'can_take_position'):
                        can_take_position = self.polygon_client.can_take_position(ticker, position_value)
                        
                        # If we can't take the position, reduce the score
                        if not can_take_position:
                            position_limit_score *= 0.5
                
                # Weighted scoring with position limit consideration
                score = (
                    volume_score * 0.25 +
                    volatility_score * 0.35 +
                    price_score * 0.2 +
                    position_limit_score * 0.2
                )
                
                scores[ticker] = score
            
            self.opportunity_scores = scores
            logger.info(f"Calculated opportunity scores for {len(scores)} tickers with dollar-based position limits")
            
        except Exception as e:
            logger.error(f"Error calculating opportunity scores: {e}")
            
        return scores
    
    def select_active_tickers(self, min_score: float = 0.0, enforce_position_limits: bool = True, max_tickers: int = None) -> List[str]:
        """
        Select active tickers based on opportunity scores and dollar-based position limits.
        
        Args:
            min_score: Minimum opportunity score to include a ticker
            enforce_position_limits: Whether to enforce position limits
            max_tickers: Maximum number of tickers to return (defaults to self.max_positions)
            
        Returns:
            List of selected ticker symbols
        """
        try:
            # Use provided max_tickers or default to self.max_positions
            max_tickers = max_tickers or self.max_positions
            
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
                # Apply dollar-based position limits
                selected_tickers = []
                total_position_value = 0.0
                position_count = 0
                
                for ticker in sorted_tickers:
                    # Get the latest price
                    latest_price = self.price_data.get(ticker)
                    
                    if latest_price is None or latest_price <= 0:
                        continue
                    
                    # Calculate position value based on price and standard lot size
                    # Use a reasonable number of shares based on price
                    shares = max(1, min(100, int(self.max_position_value / latest_price)))
                    position_value = latest_price * shares
                    
                    # Check if this position can be taken based on dollar limits
                    can_take_position = True
                    
                    # Check with Polygon client if available
                    if self.polygon_client and hasattr(self.polygon_client, 'can_take_position'):
                        can_take_position = self.polygon_client.can_take_position(ticker, position_value)
                    
                    # If we can't take the position with Polygon, check with Unusual Whales
                    if not can_take_position and self.unusual_whales_client and hasattr(self.unusual_whales_client, 'can_take_position'):
                        can_take_position = self.unusual_whales_client.can_take_position(ticker, position_value)
                    
                    # If we can't take the position with either client, check our own limits
                    if not can_take_position:
                        # Check if adding this position would exceed max_daily_value
                        if total_position_value + position_value > self.max_daily_value:
                            # Skip if we've already reached the max total
                            if total_position_value >= self.max_daily_value:
                                continue
                            
                            # Otherwise, adjust position value to fit within max_daily_value
                            available_value = self.max_daily_value - total_position_value
                            shares = max(1, int(available_value / latest_price))
                            position_value = latest_price * shares
                        
                        # Check if position value exceeds max_position_value
                        if position_value > self.max_position_value:
                            shares = max(1, int(self.max_position_value / latest_price))
                            position_value = latest_price * shares
                    
                    # Add ticker if position value is reasonable
                    if position_value >= 100:  # Minimum position value of $100
                        selected_tickers.append(ticker)
                        total_position_value += position_value
                        position_count += 1
                        
                        # Update position tracking in API clients
                        if self.polygon_client and hasattr(self.polygon_client, 'update_position_tracking'):
                            self.polygon_client.update_position_tracking(ticker, position_value)
                        
                        if self.unusual_whales_client and hasattr(self.unusual_whales_client, 'update_position_tracking'):
                            self.unusual_whales_client.update_position_tracking(ticker, position_value)
                        
                        # Log the selected ticker with position details
                        logger.info(f"Selected ticker {ticker} with price ${latest_price:.2f}, shares {shares}, position value ${position_value:.2f}")
                        
                        # Stop if we've reached max tickers
                        if position_count >= max_tickers:
                            break
            else:
                # Limit to max_tickers without enforcing position limits
                selected_tickers = sorted_tickers[:max_tickers]
            
            logger.info(f"Selected {len(selected_tickers)} active tickers with total position value ${total_position_value:.2f}")
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
