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
    ):
        """
        Initialize the RiskBasedPositionSizer.

        Args:
            account_size: Total account value
            risk_percentage: Percentage of account to risk per trade (0.02 = 2%)
            max_position_size: Maximum position size in dollars
            min_position_size: Minimum position size in dollars
            max_positions: Maximum number of concurrent positions
            focus_allocation_boost: Allocation boost for focus tickers
        """
        self.account_size = account_size
        self.risk_percentage = risk_percentage
        self.max_position_size = max_position_size or (
            account_size * 0.05
        )  # Default to 5% of account
        self.min_position_size = min_position_size
        self.max_positions = max_positions
        self.focus_allocation_boost = focus_allocation_boost

        # Calculate maximum risk amount per trade
        self.max_risk_amount = account_size * risk_percentage

        logger.info(
            f"Initialized RiskBasedPositionSizer with account_size=${account_size:.2f}, "
            f"risk_percentage={risk_percentage:.1%}, max_risk_amount=${self.max_risk_amount:.2f}"
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
    ) -> dict[str, Any]:
        """
        Calculate position size based on risk parameters.

        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            stop_price: Stop loss price for the position
            conviction_score: Conviction score (0.5-1.5)
            volatility_adjustment: Volatility adjustment factor
            is_focus_ticker: Whether the ticker is in the focus universe

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

        # Apply maximum position constraint
        if position_value > self.max_position_size:
            shares = self.max_position_size / entry_price
            position_value = self.max_position_size

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

        # Check if position is viable (at least 1 share)
        viable = shares >= 1
        reason = "" if viable else "Not enough shares to trade"

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
        }

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
    ) -> float:
        """
        Calculate position size based on signal strength and volatility.
        
        Args:
            ticker: Ticker symbol
            signal_strength: Strength of the trading signal (0.0-1.0)
            volatility: Volatility measure (ATR as percentage of price)
            current_price: Current price of the asset
            is_focus_ticker: Whether the ticker is in the focus universe
            
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
        stop_price = current_price - (atr * 2.0)
        
        # Ensure stop price is positive
        stop_price = max(0.01, stop_price)
        
        # Use signal strength as conviction score
        conviction_score = max(0.5, min(1.5, signal_strength))
        
        # Calculate volatility adjustment
        volatility_adjustment = self.calculate_volatility_adjustment(ticker, atr, current_price)
        
        # Calculate position size using the existing method
        position_info = self.calculate_position_size(
            ticker=ticker,
            entry_price=current_price,
            stop_price=stop_price,
            conviction_score=conviction_score,
            volatility_adjustment=volatility_adjustment,
            is_focus_ticker=is_focus_ticker
        )
        
        return position_info["position_value"]
        
    def calculate_position_sizes(self, signals: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate position sizes for multiple signals."""
        position_sizes = {}
        for ticker, signal in signals.items():
            position_sizes[ticker] = self.calculate_position_size_from_signal(ticker, signal["signal_strength"], signal["volatility"], signal["current_price"])
        return position_sizes
