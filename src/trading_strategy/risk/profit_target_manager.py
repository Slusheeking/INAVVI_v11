"""
Profit Target Manager

This module provides the ProfitTargetManager class for managing profit targets for
positions, including fixed targets, trailing targets, and time-based targets.
"""

import math
from datetime import datetime, timedelta
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("trading_strategy.risk.profit_target_manager")


class ProfitTargetManager:
    """
    Manages profit targets for positions.

    The profit target management approach includes:
    1. Fixed profit targets based on R-multiples or percentage
    2. Trailing targets that lock in profits as price moves favorably
    3. Time-based targets that exit positions after a specified time period
    4. Volatility-adjusted targets that adapt to changing market conditions
    5. Partial profit taking at multiple levels

    This approach provides a systematic way to capture profits while allowing
    positions to run for larger gains.
    """

    def __init__(
        self,
        default_profit_target_pct: float = 0.05,  # 5% default profit target
        trailing_target_activation_pct: float = 0.03,  # 3% profit to activate trailing target
        trailing_target_distance_pct: float = 0.02,  # 2% trailing target distance
        time_target_days: int = 5,  # 5-day time target
        r_multiple_targets: list[float] = [1.0, 2.0, 3.0],  # R-multiple targets
        partial_profit_levels: list[tuple[float, float]] = [
            (0.5, 0.33),
            (1.0, 0.33),
            (2.0, 0.34),
        ],  # (R-multiple, percentage)
        use_volatility_adjustment: bool = True,  # Adjust targets based on volatility
        min_profit_target_pct: float = 0.01,  # 1% minimum profit target
    ):
        """
        Initialize the ProfitTargetManager.

        Args:
            default_profit_target_pct: Default profit target percentage
            trailing_target_activation_pct: Profit percentage to activate trailing target
            trailing_target_distance_pct: Trailing target distance percentage
            time_target_days: Number of days for time-based target
            r_multiple_targets: List of R-multiple targets
            partial_profit_levels: List of (R-multiple, percentage) tuples for partial profit taking
            use_volatility_adjustment: Whether to adjust targets based on volatility
            min_profit_target_pct: Minimum profit target percentage
        """
        self.default_profit_target_pct = default_profit_target_pct
        self.trailing_target_activation_pct = trailing_target_activation_pct
        self.trailing_target_distance_pct = trailing_target_distance_pct
        self.time_target_days = time_target_days
        self.r_multiple_targets = r_multiple_targets
        self.partial_profit_levels = partial_profit_levels
        self.use_volatility_adjustment = use_volatility_adjustment
        self.min_profit_target_pct = min_profit_target_pct

        # Store position profit targets
        self.position_targets: dict[str, dict[str, Any]] = {}

        logger.info(
            f"Initialized ProfitTargetManager with default_profit_target_pct={default_profit_target_pct:.1%}, "
            f"trailing_target_activation_pct={trailing_target_activation_pct:.1%}, "
            f"r_multiple_targets={r_multiple_targets}"
        )

    def calculate_simple_profit_target(
        self,
        ticker: str,
        entry_price: float,
        direction: str,
        volatility: float,
    ) -> float:
        """
        Calculate a simple profit target price based on entry price and volatility.
        
        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            direction: Trade direction ('long' or 'short')
            volatility: Volatility measure (as percentage of price)
            
        Returns:
            Profit target price
        """
        # Calculate target percentage
        target_percentage = self.default_profit_target_pct
        
        # Adjust target based on volatility if enabled
        if self.use_volatility_adjustment:
            volatility_adjustment = min(max(volatility / 0.02, 0.5), 2.0)
            target_percentage = target_percentage * volatility_adjustment
            
        # Calculate target price based on direction
        target_price = entry_price * (1 + target_percentage) if direction.lower() == "long" else entry_price * (1 - target_percentage)
        return target_price
        
    def is_profit_target_reached(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
        direction: str,
        profit_target_price: float,
    ) -> bool:
        """
        Check if a profit target is reached based on the current price.
        
        Args:
            ticker: Ticker symbol
            entry_price: Entry price for the position
            current_price: Current price of the asset
            direction: Trade direction ('long' or 'short')
            profit_target_price: Profit target price
            
        Returns:
            True if profit target is reached, False otherwise
        """
        if direction.lower() == "long":
            return current_price >= profit_target_price
        else:
            return current_price <= profit_target_price
        
    def calculate_profit_target(
        self,
        entry_price: float,
        position_type: str,
        stop_loss_price: float,
        volatility: float | None = None,
        atr: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate profit target for a position.

        Args:
            entry_price: Entry price for the position
            position_type: Position type ('long' or 'short')
            stop_loss_price: Stop loss price for the position
            volatility: Volatility as percentage of price
            atr: Average True Range value

        Returns:
            Dictionary with profit target information
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)

        # Calculate R-multiple targets
        r_multiple_prices = {}
        for r in self.r_multiple_targets:
            if position_type.lower() == "long":
                # For long positions, profit target is above entry price
                r_multiple_prices[r] = entry_price + (risk_per_share * r)
            else:
                # For short positions, profit target is below entry price
                r_multiple_prices[r] = entry_price - (risk_per_share * r)

        # Calculate percentage-based target
        target_percentage = self.default_profit_target_pct

        # Adjust target based on volatility if enabled
        if self.use_volatility_adjustment and volatility is not None:
            # Higher volatility = higher target
            volatility_adjustment = min(max(volatility / 0.02, 0.5), 2.0)
            target_percentage = target_percentage * volatility_adjustment

        # Ensure minimum target percentage
        target_percentage = max(target_percentage, self.min_profit_target_pct)

        # Calculate percentage-based target price
        if position_type.lower() == "long":
            percentage_target_price = entry_price * (1 + target_percentage)
        else:
            percentage_target_price = entry_price * (1 - target_percentage)

        # Calculate partial profit targets
        partial_targets = []
        for r_multiple, percentage in self.partial_profit_levels:
            if position_type.lower() == "long":
                price = entry_price + (risk_per_share * r_multiple)
            else:
                price = entry_price - (risk_per_share * r_multiple)

            partial_targets.append(
                {
                    "r_multiple": r_multiple,
                    "percentage": percentage,
                    "price": price,
                    "executed": False,
                }
            )

        # Sort partial targets by price (ascending for short, descending for long)
        partial_targets.sort(
            key=lambda x: x["price"], reverse=(position_type.lower() == "long")
        )

        # Compile profit target information
        profit_target = {
            "entry_price": entry_price,
            "position_type": position_type,
            "stop_loss_price": stop_loss_price,
            "risk_per_share": risk_per_share,
            "r_multiple_targets": r_multiple_prices,
            "percentage_target": target_percentage,
            "percentage_target_price": percentage_target_price,
            "partial_targets": partial_targets,
            "trailing_target_level": None,  # Will be set when activated
            "trailing_target_price": None,  # Will be set when activated
            "time_target": None,  # Will be set when position is initialized
            "atr": atr,
            "volatility": volatility,
        }

        return profit_target

    def initialize_position_target(
        self,
        position_id: str,
        symbol: str,
        entry_price: float,
        position_type: str,
        stop_loss_price: float,
        quantity: float,
        entry_time: datetime | None = None,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Initialize profit target for a new position.

        Args:
            position_id: Position ID
            symbol: Ticker symbol
            entry_price: Entry price for the position
            position_type: Position type ('long' or 'short')
            stop_loss_price: Stop loss price for the position
            quantity: Position quantity
            entry_time: Entry time for the position
            atr: Average True Range value
            volatility: Volatility as percentage of price

        Returns:
            Dictionary with profit target information
        """
        # Calculate profit target
        profit_target = self.calculate_profit_target(
            entry_price=entry_price,
            position_type=position_type,
            stop_loss_price=stop_loss_price,
            volatility=volatility,
            atr=atr,
        )

        # Calculate time target
        if entry_time is None:
            entry_time = datetime.now()
        time_target = entry_time + timedelta(days=self.time_target_days)

        # Add position information
        profit_target.update(
            {
                "position_id": position_id,
                "symbol": symbol,
                "quantity": quantity,
                "entry_time": entry_time,
                "time_target": time_target,
                "last_updated": datetime.now(),
                "partial_executions": [],
            }
        )

        # Store profit target
        self.position_targets[position_id] = profit_target

        logger.info(
            f"Initialized profit target for {symbol} {position_type} position at {entry_price:.2f}: "
            f"percentage_target={profit_target['percentage_target']:.1%}, "
            f"target_price={profit_target['percentage_target_price']:.2f}, "
            f"time_target={time_target}"
        )

        return profit_target

    def update_position_target(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime | None = None,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Update profit target for an existing position.

        Args:
            position_id: Position ID
            current_price: Current price of the asset
            current_time: Current time
            atr: Updated Average True Range value
            volatility: Updated volatility as percentage of price

        Returns:
            Updated profit target information
        """
        # Get position target
        profit_target = self.position_targets.get(position_id)
        if not profit_target:
            logger.warning(f"No profit target found for position {position_id}")
            return {}

        # Extract position information
        symbol = profit_target["symbol"]
        entry_price = profit_target["entry_price"]
        position_type = profit_target["position_type"]
        trailing_target_level = profit_target["trailing_target_level"]
        trailing_target_price = profit_target["trailing_target_price"]

        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()

        # Update ATR and volatility if provided
        if atr is not None:
            profit_target["atr"] = atr
        if volatility is not None:
            profit_target["volatility"] = volatility

        # Calculate unrealized profit/loss
        if position_type == "long":
            unrealized_pl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pl_pct = (entry_price - current_price) / entry_price

        # Check if trailing target should be activated
        if (
            trailing_target_level is None
            and unrealized_pl_pct >= self.trailing_target_activation_pct
        ):
            # Activate trailing target
            if position_type == "long":
                trailing_target_price = current_price * (
                    1 - self.trailing_target_distance_pct
                )
            else:
                trailing_target_price = current_price * (
                    1 + self.trailing_target_distance_pct
                )

            profit_target["trailing_target_level"] = current_price
            profit_target["trailing_target_price"] = trailing_target_price

            logger.info(
                f"Activated trailing target for {symbol} {position_type} position: "
                f"price={current_price:.2f}, trailing_target={trailing_target_price:.2f}"
            )

        # Update trailing target level if already activated
        elif trailing_target_level is not None:
            if position_type == "long" and current_price > trailing_target_level:
                # Update trailing target for long position
                new_trailing_target_price = current_price * (
                    1 - self.trailing_target_distance_pct
                )

                # Only update if new trailing target is higher
                if new_trailing_target_price > trailing_target_price:
                    profit_target["trailing_target_level"] = current_price
                    profit_target["trailing_target_price"] = new_trailing_target_price

                    logger.info(
                        f"Updated trailing target for {symbol} long position: "
                        f"price={current_price:.2f}, trailing_target={new_trailing_target_price:.2f}"
                    )
            elif position_type == "short" and current_price < trailing_target_level:
                # Update trailing target for short position
                new_trailing_target_price = current_price * (
                    1 + self.trailing_target_distance_pct
                )

                # Only update if new trailing target is lower
                if new_trailing_target_price < trailing_target_price:
                    profit_target["trailing_target_level"] = current_price
                    profit_target["trailing_target_price"] = new_trailing_target_price

                    logger.info(
                        f"Updated trailing target for {symbol} short position: "
                        f"price={current_price:.2f}, trailing_target={new_trailing_target_price:.2f}"
                    )

        # Check partial profit targets
        for i, target in enumerate(profit_target["partial_targets"]):
            if target["executed"]:
                continue

            if (position_type == "long" and current_price >= target["price"]) or (
                position_type == "short" and current_price <= target["price"]
            ):
                # Mark target as executed
                profit_target["partial_targets"][i]["executed"] = True
                profit_target["partial_targets"][i]["execution_price"] = current_price
                profit_target["partial_targets"][i]["execution_time"] = current_time

                # Add to partial executions
                profit_target["partial_executions"].append(
                    {
                        "r_multiple": target["r_multiple"],
                        "percentage": target["percentage"],
                        "target_price": target["price"],
                        "execution_price": current_price,
                        "execution_time": current_time,
                    }
                )

                logger.info(
                    f"Partial profit target reached for {symbol} {position_type} position: "
                    f"r_multiple={target['r_multiple']}, percentage={target['percentage']:.1%}, "
                    f"price={current_price:.2f}"
                )

        # Update last updated time
        profit_target["last_updated"] = current_time

        return profit_target

    def check_target_triggered(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Check if profit target is triggered for a position.

        Args:
            position_id: Position ID
            current_price: Current price of the asset
            current_time: Current time

        Returns:
            Tuple of (is_triggered, trigger_reason, profit_target_info)
        """
        # Get position target
        profit_target = self.position_targets.get(position_id)
        if not profit_target:
            logger.warning(f"No profit target found for position {position_id}")
            return False, "", {}

        # Extract position information
        symbol = profit_target["symbol"]
        position_type = profit_target["position_type"]
        percentage_target_price = profit_target["percentage_target_price"]
        trailing_target_price = profit_target["trailing_target_price"]
        time_target = profit_target["time_target"]

        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()

        # Check percentage target
        percentage_target_triggered = False
        if position_type == "long" and current_price >= percentage_target_price:
            percentage_target_triggered = True
        elif position_type == "short" and current_price <= percentage_target_price:
            percentage_target_triggered = True

        # Check trailing target
        trailing_target_triggered = False
        if trailing_target_price is not None:
            if position_type == "long" and current_price <= trailing_target_price:
                trailing_target_triggered = True
            elif position_type == "short" and current_price >= trailing_target_price:
                trailing_target_triggered = True

        # Check time target
        time_target_triggered = current_time >= time_target if time_target else False

        # Determine if any target is triggered
        is_triggered = (
            percentage_target_triggered
            or trailing_target_triggered
            or time_target_triggered
        )

        # Determine trigger reason
        trigger_reason = ""
        if percentage_target_triggered:
            trigger_reason = "percentage_target"
        elif trailing_target_triggered:
            trigger_reason = "trailing_target"
        elif time_target_triggered:
            trigger_reason = "time_target"

        if is_triggered:
            logger.info(
                f"Profit target triggered for {symbol} {position_type} position: "
                f"price={current_price:.2f}, reason={trigger_reason}"
            )

        return is_triggered, trigger_reason, profit_target

    def get_partial_exit_quantity(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Get quantity to exit for partial profit taking.

        Args:
            position_id: Position ID
            current_price: Current price of the asset
            current_time: Current time

        Returns:
            Tuple of (exit_quantity, target_info)
        """
        # Get position target
        profit_target = self.position_targets.get(position_id)
        if not profit_target:
            logger.warning(f"No profit target found for position {position_id}")
            return 0.0, {}

        # Extract position information
        position_type = profit_target["position_type"]
        quantity = profit_target["quantity"]
        partial_targets = profit_target["partial_targets"]

        logger.debug(f"Processing partial exit for {profit_target['symbol']} with quantity {quantity}")
        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()

        # Check for partial targets
        for i, target in enumerate(partial_targets):
            if target["executed"]:
                continue

            if (position_type == "long" and current_price >= target["price"]) or (
                position_type == "short" and current_price <= target["price"]
            ):
                # Calculate exit quantity
                exit_quantity = quantity * target["percentage"]

                # Round to whole shares
                exit_quantity = math.floor(exit_quantity)

                # Ensure at least 1 share
                exit_quantity = max(1, exit_quantity)

                # Ensure not more than remaining quantity
                remaining_quantity = quantity - sum(
                    execution.get("quantity", 0)
                    for execution in profit_target.get("partial_executions", [])
                )
                exit_quantity = min(exit_quantity, remaining_quantity)

                # Create target info
                target_info = {
                    "r_multiple": target["r_multiple"],
                    "percentage": target["percentage"],
                    "target_price": target["price"],
                    "execution_price": current_price,
                    "execution_time": current_time,
                    "quantity": exit_quantity,
                }

                return exit_quantity, target_info

        return 0.0, {}

    def remove_position_target(self, position_id: str) -> None:
        """
        Remove profit target for a position.

        Args:
            position_id: Position ID
        """
        if position_id in self.position_targets:
            profit_target = self.position_targets[position_id]
            symbol = profit_target["symbol"]
            logger.debug(f"Removing profit target for symbol {symbol}, position {position_id}")

            del self.position_targets[position_id]

            logger.info(f"Removed profit target for {symbol} position {position_id}")

    def get_all_position_targets(self) -> dict[str, dict[str, Any]]:
        """
        Get all position profit targets.

        Returns:
            Dictionary mapping position IDs to profit target records
        """
        return self.position_targets

    def get_position_target(self, position_id: str) -> dict[str, Any]:
        """
        Get profit target for a position.

        Args:
            position_id: Position ID

        Returns:
            Profit target record for the position
        """
        return self.position_targets.get(position_id, {})
