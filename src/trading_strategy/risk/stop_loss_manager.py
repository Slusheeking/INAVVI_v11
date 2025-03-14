"""
Stop Loss Manager

This module provides the StopLossManager class for managing stop losses for positions,
including fixed stops, trailing stops, and time-based stops.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple



logger = logging.getLogger(__name__)


class StopLossManager:
    """
    Manages stop losses for positions.

    The stop loss management approach includes:
    1. Fixed stop losses based on ATR or percentage
    2. Trailing stops that move with favorable price movement
    3. Time-based stops that exit positions after a specified time period
    4. Volatility-adjusted stops that adapt to changing market conditions

    This approach provides robust protection against excessive losses while allowing
    positions to capture profits.
    """

    def __init__(
        self,
        default_stop_loss_pct: float = 0.02,  # 2% default stop loss
        trailing_stop_activation_pct: float = 0.01,  # 1% profit to activate trailing stop
        trailing_stop_distance_pct: float = 0.01,  # 1% trailing stop distance
        time_stop_days: int = 10,  # 10-day time stop
        atr_multiplier: float = 2.0,  # 2x ATR for stop distance
        use_atr_stops: bool = True,  # Use ATR-based stops when available
        min_stop_distance_pct: float = 0.005,  # 0.5% minimum stop distance
    ):
        """
        Initialize the StopLossManager.

        Args:
            default_stop_loss_pct: Default stop loss percentage
            trailing_stop_activation_pct: Profit percentage to activate trailing stop
            trailing_stop_distance_pct: Trailing stop distance percentage
            time_stop_days: Number of days for time-based stop
            atr_multiplier: Multiplier for ATR-based stops
            use_atr_stops: Whether to use ATR-based stops when available
            min_stop_distance_pct: Minimum stop distance percentage
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.time_stop_days = time_stop_days
        self.atr_multiplier = atr_multiplier
        self.use_atr_stops = use_atr_stops
        self.min_stop_distance_pct = min_stop_distance_pct

        # Store position stop levels
        self.position_stops: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized StopLossManager with default_stop_loss_pct={default_stop_loss_pct:.1%}, "
            f"trailing_stop_activation_pct={trailing_stop_activation_pct:.1%}, "
            f"trailing_stop_distance_pct={trailing_stop_distance_pct:.1%}"
        )

    def calculate_stop_loss_from_volatility(
        self,
        ticker: str,
        entry_price: float,
        direction: str,
        volatility: float,
    ) -> float:
        """
        Calculate a simple stop loss price based on entry price and volatility.
        
        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            direction: Trade direction ('long' or 'short')
            volatility: Volatility measure (as percentage of price)
            
        Returns:
            Stop loss price
        """
        # Calculate ATR from volatility
        atr = entry_price * volatility
        
        # Use the existing method with the calculated ATR
        position_type = "long" if direction.lower() == "long" else "short"
        return self.calculate_stop_loss(entry_price, position_type, atr=atr)
        
    def is_stop_loss_triggered(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        direction: str,
        stop_loss_price: float,
    ) -> bool:
        """
        Check if a stop loss is triggered based on the current price.
        
        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            current_price: Current price of the asset
            direction: Trade direction ('long' or 'short')
            stop_loss_price: Stop loss price
            
        Returns:
            True if stop loss is triggered, False otherwise
        """
        if direction.lower() == "long":
            return current_price <= stop_loss_price
        else:
            return current_price >= stop_loss_price
        
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_type: str,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss price for a position.

        Args:
            entry_price: Entry price for the position
            position_type: Position type ('long' or 'short')
            volatility: Volatility as percentage of price
            atr: Average True Range value

        Returns:
            Stop loss price
        """
        if position_type.lower() == "long":
            # For long positions, stop loss is below entry price
            if self.use_atr_stops and atr is not None:
                # Use ATR-based stop loss if available
                stop_distance = atr * self.atr_multiplier
            else:
                # Use percentage-based stop loss
                stop_distance = entry_price * self.default_stop_loss_pct

            # Ensure minimum stop distance
            min_distance = entry_price * self.min_stop_distance_pct
            stop_distance = max(stop_distance, min_distance)

            return entry_price - stop_distance
        else:
            # For short positions, stop loss is above entry price
            if self.use_atr_stops and atr is not None:
                stop_distance = atr * self.atr_multiplier
            else:
                stop_distance = entry_price * self.default_stop_loss_pct

            # Ensure minimum stop distance
            min_distance = entry_price * self.min_stop_distance_pct
            stop_distance = max(stop_distance, min_distance)

            return entry_price + stop_distance

    def initialize_position_stop(
        self,
        position_id: str,
        symbol: str,
        entry_price: float,
        position_type: str,
        quantity: float,
        entry_time: Optional[datetime] = None,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Initialize stop loss for a new position.

        Args:
            position_id: Position ID
            symbol: Ticker symbol
            entry_price: Entry price for the position
            position_type: Position type ('long' or 'short')
            quantity: Position quantity
            entry_time: Entry time for the position
            atr: Average True Range value

        Returns:
            Dictionary with stop loss information
        """
        # Calculate initial stop loss price
        stop_price = self.calculate_stop_loss(entry_price, position_type, atr=atr)

        # Calculate stop distance percentage
        stop_distance_pct = abs(stop_price - entry_price) / entry_price

        # Calculate trailing stop level (None until activated)
        trailing_stop_level = None

        # Calculate time stop
        if entry_time is None:
            entry_time = datetime.now()
        time_stop = entry_time + timedelta(days=self.time_stop_days)

        # Create stop loss record
        stop_loss = {
            "position_id": position_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "position_type": position_type,
            "quantity": quantity,
            "entry_time": entry_time,
            "initial_stop_price": stop_price,
            "current_stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "trailing_stop_level": trailing_stop_level,
            "time_stop": time_stop,
            "atr": atr,
            "last_updated": datetime.now(),
        }

        # Store stop loss record
        self.position_stops[position_id] = stop_loss

        logger.info(
            f"Initialized stop loss for {symbol} {position_type} position at {entry_price:.2f}: "
            f"stop={stop_price:.2f} ({stop_distance_pct:.1%}), time_stop={time_stop}"
        )

        return stop_loss

    def update_position_stop(
        self,
        position_id: str,
        current_price: float,
        current_time: Optional[datetime] = None,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update stop loss for an existing position.

        Args:
            position_id: Position ID
            current_price: Current price of the asset
            current_time: Current time
            atr: Updated Average True Range value

        Returns:
            Updated stop loss information
        """
        # Get position stop record
        stop_loss = self.position_stops.get(position_id)
        if not stop_loss:
            logger.warning(f"No stop loss record found for position {position_id}")
            return {}

        # Extract position information
        symbol = stop_loss["symbol"]
        entry_price = stop_loss["entry_price"]
        position_type = stop_loss["position_type"]
        current_stop_price = stop_loss["current_stop_price"]
        trailing_stop_level = stop_loss["trailing_stop_level"]

        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()

        # Update ATR if provided
        if atr is not None:
            stop_loss["atr"] = atr

        # Calculate unrealized profit/loss
        if position_type == "long":
            unrealized_pl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pl_pct = (entry_price - current_price) / entry_price

        # Check if trailing stop should be activated
        if (
            trailing_stop_level is None
            and unrealized_pl_pct >= self.trailing_stop_activation_pct
        ):
            # Activate trailing stop
            if position_type == "long":
                trailing_stop_level = current_price * (
                    1 - self.trailing_stop_distance_pct
                )
            else:
                trailing_stop_level = current_price * (
                    1 + self.trailing_stop_distance_pct
                )

            stop_loss["trailing_stop_level"] = trailing_stop_level

            logger.info(
                f"Activated trailing stop for {symbol} {position_type} position: "
                f"price={current_price:.2f}, trailing_stop={trailing_stop_level:.2f}"
            )

        # Update trailing stop level if already activated
        elif trailing_stop_level is not None:
            if position_type == "long" and current_price > trailing_stop_level / (
                1 - self.trailing_stop_distance_pct
            ):
                # Update trailing stop for long position
                trailing_stop_level = current_price * (
                    1 - self.trailing_stop_distance_pct
                )
                stop_loss["trailing_stop_level"] = trailing_stop_level

                logger.info(
                    f"Updated trailing stop for {symbol} long position: "
                    f"price={current_price:.2f}, trailing_stop={trailing_stop_level:.2f}"
                )
            elif position_type == "short" and current_price < trailing_stop_level / (
                1 + self.trailing_stop_distance_pct
            ):
                # Update trailing stop for short position
                trailing_stop_level = current_price * (
                    1 + self.trailing_stop_distance_pct
                )
                stop_loss["trailing_stop_level"] = trailing_stop_level

                logger.info(
                    f"Updated trailing stop for {symbol} short position: "
                    f"price={current_price:.2f}, trailing_stop={trailing_stop_level:.2f}"
                )

        # Determine current stop price (max of initial stop and trailing stop)
        if trailing_stop_level is not None:
            if position_type == "long":
                current_stop_price = max(current_stop_price, trailing_stop_level)
            else:
                current_stop_price = min(current_stop_price, trailing_stop_level)

        # Update stop loss record
        stop_loss["current_stop_price"] = current_stop_price
        stop_loss["last_updated"] = current_time

        return stop_loss

    def check_stop_triggered(
        self,
        position_id: str,
        current_price: float,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if stop loss is triggered for a position.

        Args:
            position_id: Position ID
            current_price: Current price of the asset
            current_time: Current time

        Returns:
            Tuple of (is_triggered, trigger_reason, stop_loss_info)
        """
        # Get position stop record
        stop_loss = self.position_stops.get(position_id)
        if not stop_loss:
            logger.warning(f"No stop loss record found for position {position_id}")
            return False, "", {}

        # Extract position information
        symbol = stop_loss["symbol"]
        position_type = stop_loss["position_type"]
        current_stop_price = stop_loss["current_stop_price"]
        time_stop = stop_loss["time_stop"]

        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()

        # Check price stop
        price_stop_triggered = False
        if position_type == "long" and current_price <= current_stop_price:
            price_stop_triggered = True
        elif position_type == "short" and current_price >= current_stop_price:
            price_stop_triggered = True

        # Check time stop
        time_stop_triggered = current_time >= time_stop

        # Determine if any stop is triggered
        is_triggered = price_stop_triggered or time_stop_triggered

        # Determine trigger reason
        trigger_reason = ""
        if price_stop_triggered:
            if current_stop_price == stop_loss["initial_stop_price"]:
                trigger_reason = "initial_stop"
            else:
                trigger_reason = "trailing_stop"
        elif time_stop_triggered:
            trigger_reason = "time_stop"

        if is_triggered:
            logger.info(
                f"Stop loss triggered for {symbol} {position_type} position: "
                f"price={current_price:.2f}, stop={current_stop_price:.2f}, reason={trigger_reason}"
            )

        return is_triggered, trigger_reason, stop_loss

    def remove_position_stop(self, position_id: str) -> None:
        """
        Remove stop loss record for a position.

        Args:
            position_id: Position ID
        """
        if position_id in self.position_stops:
            stop_loss = self.position_stops[position_id]
            symbol = stop_loss["symbol"]

            del self.position_stops[position_id]

            logger.info(f"Removed stop loss record for {symbol} position {position_id}")

    def get_all_position_stops(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all position stop records.

        Returns:
            Dictionary mapping position IDs to stop loss records
        """
        return self.position_stops 

    def get_position_stop(self, position_id: str) -> Dict[str, Any]:
        """
        Get stop loss record for a position.

        Args:
            position_id: Position ID

        Returns:
            Stop loss record for the position
        """
        return self.position_stops.get(position_id, {})
