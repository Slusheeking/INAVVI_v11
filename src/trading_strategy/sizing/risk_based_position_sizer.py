"""
Risk-Based Position Sizer

This module provides the RiskBasedPositionSizer class for calculating position sizes
based on risk parameters, including the 2% risk rule, ATR-based stops, and volatility
adjustments.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class RiskBasedPositionSizer:
    """
    Calculates position sizes based on risk parameters.

    The position sizing methodology follows these principles:
    1. Risk a fixed percentage of account value per trade (typically 1-2%)
    2. Use ATR-based stop losses to determine risk per share
    3. Adjust position sizes based on volatility
    4. Apply position size constraints (min/max)
    5. Boost allocation for focus tickers

    This approach ensures consistent risk exposure across different market conditions
    and ticker characteristics.
    """

    def __init__(
        self,
        account_size: float,
        risk_percentage: float = 0.02,
        max_position_size: float | None = None,
        min_position_size: float = 1000.0,
        max_positions: int = 20,
        focus_allocation_boost: float = 1.5,
        polygon_client = None,
        unusual_whales_client = None,
        max_position_value: float = 2500.0,
        max_daily_value: float = 5000.0,
    ):
        """
        Initialize the RiskBasedPositionSizer with dollar-based position limits.

        Args:
            account_size: Total account value
            risk_percentage: Percentage of account to risk per trade (0.02 = 2%)
            max_position_size: Maximum position size in dollars
            min_position_size: Minimum position size in dollars
            max_positions: Maximum number of concurrent positions
            focus_allocation_boost: Allocation boost for focus tickers
            polygon_client: Polygon API client for position limit checks
            unusual_whales_client: Unusual Whales API client for position limit checks
            max_position_value: Maximum position value per stock in dollars
            max_daily_value: Maximum total position value per day in dollars
        """
        self.account_size = account_size
        self.risk_percentage = risk_percentage
        self.min_position_size = min_position_size
        self.max_positions = max_positions
        self.focus_allocation_boost = focus_allocation_boost
        self.polygon_client = polygon_client
        self.unusual_whales_client = unusual_whales_client
        
        # Set dollar-based position limits
        self.max_position_value = max_position_value
        self.max_daily_value = max_daily_value
        
        # If polygon_client is provided, use its limits
        if self.polygon_client:
            self.max_position_value = getattr(self.polygon_client, 'max_position_value', max_position_value)
            self.max_daily_value = getattr(self.polygon_client, 'max_daily_value', max_daily_value)
        
        # Set max_position_size to the smaller of the provided value or max_position_value
        if max_position_size is None:
            self.max_position_size = min(account_size * 0.05, self.max_position_value)
        else:
            self.max_position_size = min(max_position_size, self.max_position_value)

        # Calculate maximum risk amount per trade
        self.max_risk_amount = account_size * risk_percentage

        logger.info(
            f"Initialized RiskBasedPositionSizer with account_size=${account_size:.2f}, "
            f"risk_percentage={risk_percentage:.1%}, max_risk_amount=${self.max_risk_amount:.2f}, "
            f"max_position_value=${self.max_position_value:.2f}, max_daily_value=${self.max_daily_value:.2f}"
        )

    def calculate_atr_based_stop(
        self,
        ticker: str,
        entry_price: float,
        atr: float,
        direction: str = "long",
        atr_multiplier: float = 2.0,
    ) -> float:
        """
        Calculate stop loss price based on Average True Range (ATR).

        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            atr: Average True Range value
            direction: Trade direction ('long' or 'short')
            atr_multiplier: Multiplier for ATR to determine stop distance

        Returns:
            Stop loss price
        """
        if direction.lower() == "long":
            stop_price = entry_price - (atr * atr_multiplier)
        else:
            stop_price = entry_price + (atr * atr_multiplier)

        # Ensure stop price is positive
        stop_price = max(0.01, stop_price)

        return stop_price

    def calculate_volatility_adjustment(
        self, ticker: str, atr: float, price: float
    ) -> float:
        """
        Calculate volatility adjustment factor for position sizing.

        Args:
            ticker: Ticker symbol
            atr: Average True Range value
            price: Current price of the asset

        Returns:
            Volatility adjustment factor (typically 0.5-1.5)
        """
        # Calculate ATR as percentage of price
        atr_percentage = atr / price

        # Typical ATR percentage is around 1-2% for stocks
        # Adjust factor based on deviation from typical
        if atr_percentage < 0.01:  # Low volatility
            adjustment = 1.5  # Increase position size
        elif atr_percentage > 0.03:  # High volatility
            adjustment = 0.5  # Decrease position size
        else:
            # Linear scaling between 0.5 and 1.5
            adjustment = 1.5 - ((atr_percentage - 0.01) / 0.02)

        return adjustment

    def calculate_position_size(
        self,
        ticker: str,
        entry_price: float,
        stop_price: float,
        conviction_score: float = 1.0,
        volatility_adjustment: float = 1.0,
        is_focus_ticker: bool = False,
        direction: str = "long",
    ) -> dict[str, Any]:
        """
        Calculate position size based on risk parameters and dollar-based position limits.

        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            stop_price: Stop loss price for the position
            conviction_score: Conviction score (0.5-1.5)
            volatility_adjustment: Volatility adjustment factor
            is_focus_ticker: Whether the ticker is in the focus universe
            direction: Trade direction ('long' or 'short')

        Returns:
            Dictionary with position sizing information
        """
        # Validate inputs
        if entry_price <= 0 or stop_price <= 0:
            raise ValueError("Entry and stop prices must be positive")

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            raise ValueError("Stop distance cannot be zero")

        # Calculate stop distance as percentage
        stop_percentage = stop_distance / entry_price

        # Calculate risk amount in dollars (adjusted for volatility)
        risk_amount = self.max_risk_amount * volatility_adjustment

        # Calculate shares based on risk
        shares = risk_amount / stop_distance

        # Apply conviction multiplier (0.5-1.5)
        conviction_score = max(0.5, min(1.5, conviction_score))
        shares = shares * conviction_score

        # Apply focus universe boost if applicable
        if is_focus_ticker:
            shares = shares * self.focus_allocation_boost

        # Calculate position value
        position_value = shares * entry_price

        # Check if this position can be taken based on dollar limits
        can_take_position = True
        position_limit_reason = ""
        
        # Check with Polygon client if available
        if self.polygon_client and hasattr(self.polygon_client, 'can_take_position'):
            can_take_position = self.polygon_client.can_take_position(ticker, position_value)
            if not can_take_position:
                position_limit_reason = "Exceeds Polygon client position limits"
        
        # If we can't take the position with Polygon, check with Unusual Whales
        if not can_take_position and self.unusual_whales_client and hasattr(self.unusual_whales_client, 'can_take_position'):
            can_take_position = self.unusual_whales_client.can_take_position(ticker, position_value)
            if not can_take_position:
                position_limit_reason = "Exceeds Unusual Whales client position limits"
        
        # Apply dollar-based position limits
        if position_value > self.max_position_value:
            shares = self.max_position_value / entry_price
            position_value = shares * entry_price
            logger.info(f"Reduced position size for {ticker} to respect max_position_value=${self.max_position_value:.2f}")
        
        # Apply maximum position constraint from account size
        if position_value > self.max_position_size:
            shares = self.max_position_size / entry_price
            position_value = shares * entry_price

        # Apply minimum position constraint
        if position_value < self.min_position_size:
            if self.min_position_size / entry_price < 1:
                # Can't buy fractional shares, so position is too small
                return {
                    "ticker": ticker,
                    "shares": 0,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "position_value": 0.0,
                    "risk_amount": 0.0,
                    "stop_percentage": stop_percentage,
                    "viable": False,
                    "reason": "Position size below minimum",
                }
            shares = self.min_position_size / entry_price
            position_value = shares * entry_price

        # Round down to whole shares
        shares = math.floor(shares)

        # Recalculate position value with whole shares
        position_value = shares * entry_price

        # Calculate actual risk amount
        actual_risk = position_value * (stop_distance / entry_price)

        # Check if position is viable (at least 1 share and within position limits)
        viable = shares >= 1 and can_take_position
        
        # Determine reason if not viable
        if not viable:
            if shares < 1:
                reason = "Not enough shares to trade"
            elif not can_take_position:
                reason = position_limit_reason
            else:
                reason = "Unknown reason"
        else:
            reason = ""

        result = {
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "position_value": position_value,
            "risk_amount": actual_risk,
            "stop_percentage": stop_percentage,
            "viable": viable,
            "is_focus_ticker": is_focus_ticker,
            "reason": reason,
            "direction": direction,
        }

        # Update position tracking in API clients if viable
        if viable:
            if self.polygon_client and hasattr(self.polygon_client, 'update_position_tracking'):
                self.polygon_client.update_position_tracking(ticker, position_value)
            
            if self.unusual_whales_client and hasattr(self.unusual_whales_client, 'update_position_tracking'):
                self.unusual_whales_client.update_position_tracking(ticker, position_value)
            
            logger.info(f"Position calculated for {ticker}: {shares} shares at ${entry_price:.2f} (${position_value:.2f})")

        return result

    def allocate_capital(
        self,
        opportunities: list[dict[str, Any]],
        current_positions: list[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Allocate capital across multiple trading opportunities.

        Args:
            opportunities: List of trading opportunities with position sizing information
            current_positions: List of current positions

        Returns:
            List of allocated positions
        """
        if current_positions is None:
            current_positions = []

        # Calculate available capital
        capital_used = sum(pos.get("position_value", 0) for pos in current_positions)
        available_capital = self.account_size - capital_used

        # Calculate available position slots
        available_positions = self.max_positions - len(current_positions)

        if available_positions <= 0 or available_capital <= 0:
            return []

        # Sort opportunities by conviction score (highest first)
        # Prioritize focus tickers over non-focus tickers
        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: (
                # First sort by focus status (True comes before False)
                not x.get("is_focus_ticker", False),
                # Then sort by conviction score (higher is better)
                -x.get("conviction_score", 0),
                # Finally sort by risk/reward ratio (lower is better)
                x.get("risk_amount", float("inf")) / max(x.get("position_value", 1), 1),
            ),
        )

        # Allocate capital to opportunities
        allocated_positions = []
        remaining_capital = available_capital

        for opportunity in sorted_opportunities:
            if len(allocated_positions) >= available_positions:
                break

            # Skip non-viable opportunities
            if not opportunity.get("viable", False):
                continue

            position_value = opportunity.get("position_value", 0)

            # Skip if position value is zero or not enough capital
            if position_value <= 0 or position_value > remaining_capital:
                continue

            # Allocate capital to this position
            allocated_positions.append(opportunity)
            remaining_capital -= position_value

        logger.info(
            f"Allocated capital to {len(allocated_positions)} positions, "
            f"using ${available_capital - remaining_capital:.2f} of ${available_capital:.2f} available"
        )

        return allocated_positions

    def update_account_size(self, new_account_size: float) -> None:
        """
        Update the account size and recalculate dependent parameters.

        Args:
            new_account_size: New account size in dollars
        """
        self.account_size = new_account_size
        self.max_risk_amount = new_account_size * self.risk_percentage

        # Update max position size if it was using the default
        if self.max_position_size == self.account_size * 0.05:
            self.max_position_size = new_account_size * 0.05

        logger.info(
            f"Updated account size to ${new_account_size:.2f}, "
            f"max_risk_amount=${self.max_risk_amount:.2f}"
        )
        
    def calculate_position_size_from_signal(
        self,
        ticker: str,
        signal_strength: float,
        volatility: float,
        current_price: float,
        is_focus_ticker: bool = False,
        direction: str = "long",
    ) -> float:
        """
        Calculate position size based on signal strength, volatility, and dollar-based position limits.
        
        Args:
            ticker: Ticker symbol
            signal_strength: Strength of the trading signal (0.0-1.0)
            volatility: Volatility measure (ATR as percentage of price)
            current_price: Current price of the asset
            is_focus_ticker: Whether the ticker is in the focus universe
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Position size in dollars
        """
        # Validate inputs
        if current_price <= 0:
            raise ValueError("Current price must be positive")
            
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
            
        # Calculate ATR in dollars
        atr = volatility * current_price
        
        # Calculate stop price based on ATR (2x ATR for stop distance)
        if direction.lower() == "long":
            stop_price = current_price - (atr * 2.0)
        else:
            stop_price = current_price + (atr * 2.0)
        
        # Ensure stop price is positive
        stop_price = max(0.01, stop_price)
        
        # Use signal strength as conviction score
        conviction_score = max(0.5, min(1.5, signal_strength))
        
        # Calculate volatility adjustment
        volatility_adjustment = self.calculate_volatility_adjustment(ticker, atr, current_price)
        
        # Check if this position can be taken based on dollar limits before calculating full position
        position_value = min(self.max_position_value, current_price * 100)  # Estimate with 100 shares or max position value
        can_take_position = True
        
        # Check with Polygon client if available
        if self.polygon_client and hasattr(self.polygon_client, 'can_take_position'):
            can_take_position = self.polygon_client.can_take_position(ticker, position_value)
        
        # If we can't take the position with Polygon, check with Unusual Whales
        if not can_take_position and self.unusual_whales_client and hasattr(self.unusual_whales_client, 'can_take_position'):
            can_take_position = self.unusual_whales_client.can_take_position(ticker, position_value)
        
        # If we can't take the position at all, return 0
        if not can_take_position:
            logger.warning(f"Cannot take position for {ticker} due to dollar-based position limits")
            return 0.0
        
        # Calculate position size using the existing method with dollar-based position limits
        position_info = self.calculate_position_size(
            ticker=ticker,
            entry_price=current_price,
            stop_price=stop_price,
            conviction_score=conviction_score,
            volatility_adjustment=volatility_adjustment,
            is_focus_ticker=is_focus_ticker,
            direction=direction
        )
        
        # Log the position sizing decision
        if position_info["viable"]:
            logger.info(f"Signal-based position for {ticker}: ${position_info['position_value']:.2f} ({position_info['shares']} shares)")
        else:
            logger.warning(f"Signal-based position for {ticker} not viable: {position_info['reason']}")
        
        return position_info["position_value"]
        
    def calculate_position_sizes(self, signals: dict[str, dict[str, Any]]) -> dict[str, float]:
        """
        Calculate position sizes for multiple signals with dollar-based position limits.
        
        Args:
            signals: Dictionary mapping ticker symbols to signal data
                Each signal should have 'signal_strength', 'volatility', and 'current_price' keys
                Optionally can include 'is_focus_ticker' and 'direction' keys
                
        Returns:
            Dictionary mapping ticker symbols to position sizes in dollars
        """
        position_sizes = {}
        total_position_value = 0.0
        
        # Sort signals by strength (highest first) to prioritize stronger signals
        sorted_tickers = sorted(
            signals.keys(),
            key=lambda t: signals[t].get("signal_strength", 0),
            reverse=True
        )
        
        for ticker in sorted_tickers:
            signal = signals[ticker]
            
            # Check if adding this position would exceed max_daily_value
            if total_position_value >= self.max_daily_value:
                logger.warning(f"Skipping position for {ticker} as max_daily_value (${self.max_daily_value:.2f}) would be exceeded")
                position_sizes[ticker] = 0.0
                continue
            
            # Get optional parameters
            is_focus_ticker = signal.get("is_focus_ticker", False)
            direction = signal.get("direction", "long")
            
            # Calculate position size
            position_value = self.calculate_position_size_from_signal(
                ticker,
                signal["signal_strength"],
                signal["volatility"],
                signal["current_price"],
                is_focus_ticker,
                direction
            )
            
            # Add to total if position is viable
            if position_value > 0:
                # Check if adding this would exceed max_daily_value
                if total_position_value + position_value > self.max_daily_value:
                    # Adjust position size to fit within max_daily_value
                    available_value = self.max_daily_value - total_position_value
                    if available_value >= 100:  # Minimum $100 position
                        position_value = available_value
                        logger.info(f"Adjusted position for {ticker} to ${position_value:.2f} to fit within max_daily_value")
                    else:
                        position_value = 0.0
                        logger.warning(f"Skipping position for {ticker} as remaining daily value (${available_value:.2f}) is too small")
                
                total_position_value += position_value
            
            position_sizes[ticker] = position_value
        
        logger.info(f"Calculated position sizes for {len(position_sizes)} signals, total value: ${total_position_value:.2f}/{self.max_daily_value:.2f}")
        return position_sizes
