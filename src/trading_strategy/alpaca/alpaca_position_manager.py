"""
Alpaca Position Manager

This module provides the AlpacaPositionManager class for managing positions through the Alpaca API,
including position tracking, updates, and risk management.
"""

from datetime import datetime
from typing import Any

from src.trading_strategy.alpaca.alpaca_client import AlpacaClient
from src.trading_strategy.risk.profit_target_manager import ProfitTargetManager
from src.trading_strategy.risk.stop_loss_manager import StopLossManager
from src.trading_strategy.signals.peak_detector import PeakDetector
from src.utils.logging import get_logger

logger = get_logger("trading_strategy.alpaca.alpaca_position_manager")


class AlpacaPositionManager:
    """
    Manages positions through the Alpaca API.

    The AlpacaPositionManager class provides functionality for:
    1. Position tracking and updates
    2. Stop loss and profit target management
    3. Position risk monitoring
    4. Position performance tracking
    5. Position exit decision making

    This class integrates with the AlpacaIntegration class for API access and
    with risk management components for stop loss and profit target management.
    """

    def __init__(
        self,
        alpaca_integration = None,
        alpaca_client: AlpacaClient = None,
        stop_loss_manager: StopLossManager | None = None,
        profit_target_manager: ProfitTargetManager | None = None,
        peak_detector: PeakDetector | None = None,
        max_positions: int = 20,
        max_position_value_pct: float = 0.05,  # 5% of account value
        max_sector_exposure_pct: float = 0.25,  # 25% of account value
        max_drawdown_pct: float = 0.02,  # 2% max drawdown per position
        use_trailing_stops: bool = True,
        use_profit_targets: bool = True,
        use_peak_detection: bool = True,
        position_sync_enabled: bool = True,
        position_update_interval: int = 60,  # seconds
    ):
        """
        Initialize the AlpacaPositionManager.

        Args:
            alpaca_integration: AlpacaIntegration instance (either this or alpaca_client must be provided)
            alpaca_client: AlpacaClient instance (either this or alpaca_integration must be provided)
            stop_loss_manager: StopLossManager instance
            profit_target_manager: ProfitTargetManager instance
            peak_detector: PeakDetector instance
            max_positions: Maximum number of positions
            max_position_value_pct: Maximum position value as percentage of account value
            max_sector_exposure_pct: Maximum sector exposure as percentage of account value
            max_drawdown_pct: Maximum drawdown percentage per position
            use_trailing_stops: Whether to use trailing stops
            use_profit_targets: Whether to use profit targets
            use_peak_detection: Whether to use peak detection for exits
            position_sync_enabled: Whether to enable position synchronization
            position_update_interval: Position update interval in seconds
        """
        # Handle either alpaca_integration or alpaca_client
        if alpaca_integration is not None:
            self.alpaca = alpaca_integration
        elif alpaca_client is not None:
            # Create a simple adapter that provides the methods we need
            self.alpaca = alpaca_client
            # Add get_positions method if it doesn't exist
            self.alpaca.get_positions = getattr(self.alpaca, 'get_positions', self.alpaca.get_positions)
        else:
            raise ValueError("Either alpaca_integration or alpaca_client must be provided")
            
        self.stop_loss_manager = stop_loss_manager
        self.profit_target_manager = profit_target_manager
        self.peak_detector = peak_detector

        self.max_positions = max_positions
        self.max_position_value_pct = max_position_value_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.use_trailing_stops = use_trailing_stops
        self.use_profit_targets = use_profit_targets
        self.use_peak_detection = use_peak_detection
        self.position_sync_enabled = position_sync_enabled
        self.position_update_interval = position_update_interval

        # Position tracking
        self.positions: dict[str, dict[str, Any]] = {}
        self.position_history: dict[str, list[dict[str, Any]]] = {}
        self.position_performance: dict[str, dict[str, Any]] = {}

        # Order tracking
        self.orders: dict[str, dict[str, Any]] = {}
        self.order_history: dict[str, list[dict[str, Any]]] = {}

        # Sector exposure tracking
        self.sector_exposure: dict[str, float] = {}

        # Last update time
        self.last_update_time = datetime.min

        logger.info(
            f"Initialized AlpacaPositionManager with max_positions={max_positions}, "
            f"max_position_value_pct={max_position_value_pct:.1%}, "
            f"max_drawdown_pct={max_drawdown_pct:.1%}"
        )

    def update_positions(self, force_update: bool = False) -> dict[str, dict[str, Any]]:
        """
        Update position information from Alpaca.

        Args:
            force_update: Whether to force update regardless of update interval

        Returns:
            Dictionary of updated positions
        """
        # Check if update is needed
        current_time = datetime.now()
        if (
            not force_update
            and (current_time - self.last_update_time).total_seconds()
            < self.position_update_interval
        ):
            return self.positions

        # Get account information
        account = self.alpaca.get_account()

        # Get current positions from Alpaca
        alpaca_positions = self.alpaca.get_positions()

        # Update positions
        for position in alpaca_positions:
            symbol = position["symbol"]

            # Check if position is new
            is_new_position = symbol not in self.positions

            # Get previous position data if exists
            prev_position = self.positions.get(symbol, {})

            # Update position
            self.positions[symbol] = position

            # Add additional information
            self.positions[symbol]["account_value"] = float(account["equity"])
            self.positions[symbol]["position_value_pct"] = position[
                "market_value"
            ] / float(account["equity"])
            self.positions[symbol]["entry_time"] = prev_position.get(
                "entry_time", current_time
            )
            self.positions[symbol]["last_update_time"] = current_time

            # Calculate drawdown
            entry_price = float(position.get("avg_entry_price", 1) or 1)  # Default to 1 to avoid division by zero
            current_price = float(position.get("current_price", 0) or 0)
            if position["side"] == "long":
                drawdown_pct = max(0, (entry_price - current_price) / entry_price)
            else:
                drawdown_pct = max(0, (current_price - entry_price) / entry_price)

            self.positions[symbol]["drawdown_pct"] = drawdown_pct

            # Check stop loss and profit target if available
            if self.stop_loss_manager and symbol in self.positions:
                stop_price = self.stop_loss_manager.get_stop_loss_price(symbol)
                if stop_price:
                    self.positions[symbol]["stop_price"] = stop_price

            if self.profit_target_manager and symbol in self.positions:
                target_info = self.profit_target_manager.get_position_target(symbol)
                if target_info:
                    self.positions[symbol]["profit_target"] = target_info.get(
                        "percentage_target_price"
                    )
                    self.positions[symbol]["trailing_target"] = target_info.get(
                        "trailing_target_price"
                    )

            # Add to position history
            if symbol not in self.position_history:
                self.position_history[symbol] = []

            self.position_history[symbol].append(
                {
                    "timestamp": current_time,
                    "symbol": symbol,
                    "qty": position.get("qty", 0),
                    "avg_entry_price": position.get("avg_entry_price", 0),
                    "current_price": position.get("current_price", 0),
                    "market_value": position.get("market_value", 0),
                    "unrealized_pl": position.get("unrealized_pl", 0),
                    "unrealized_plpc": position.get("unrealized_plpc", 0),
                    "side": position.get("side", "long"),
                    "drawdown_pct": drawdown_pct,
                }
            )

            # Update position performance
            self._update_position_performance(symbol)

            # Log new position
            if is_new_position:
                logger.info(
                    f"New position: {symbol} {position.get('side', 'long')} {position.get('qty', 0)} @ {position.get('avg_entry_price', 0)}"
                )

        # Check for closed positions
        closed_symbols = set(self.positions.keys()) - {
            position["symbol"] for position in alpaca_positions
        }
        for symbol in closed_symbols:
            logger.info(f"Position closed: {symbol}")

            # Move to position history if not already there
            if symbol not in self.position_history:
                self.position_history[symbol] = []

            # Add final position state to history
            if symbol in self.positions:
                self.position_history[symbol].append(
                    {
                        "timestamp": current_time,
                        "symbol": symbol,
                        "qty": self.positions[symbol].get("qty", 0),
                        "avg_entry_price": self.positions[symbol].get("avg_entry_price", 0),
                        "current_price": self.positions[symbol].get("current_price", 0),
                        "market_value": self.positions[symbol].get("market_value", 0),
                        "unrealized_pl": self.positions[symbol].get("unrealized_pl", 0),
                        "unrealized_plpc": self.positions[symbol].get("unrealized_plpc", 0),
                        "side": self.positions[symbol].get("side", "long"),
                        "drawdown_pct": self.positions[symbol].get("drawdown_pct", 0),
                        "closed": True,
                    }
                )

            # Remove from current positions
            if symbol in self.positions:
                del self.positions[symbol]

        # Update sector exposure
        self._update_sector_exposure()

        # Update last update time
        self.last_update_time = current_time

        return self.positions

    def _update_position_performance(self, symbol: str) -> None:
        """
        Update position performance metrics.

        Args:
            symbol: Symbol to update performance for
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Initialize performance record if not exists
        if symbol not in self.position_performance:
            self.position_performance[symbol] = {
                "symbol": symbol,
                "entry_time": position.get("entry_time", datetime.now()),
                "entry_price": float(position.get("avg_entry_price", 0)),
                "side": position.get("side", "long"),
                "max_price": float(position.get("current_price", 0)),
                "min_price": float(position.get("current_price", 0)),
                "max_profit": float(position.get("unrealized_pl", 0)),
                "max_profit_pct": float(position.get("unrealized_plpc", 0)),
                "max_drawdown": position.get("drawdown_pct", 0),
                "current_price": float(position.get("current_price", 0)),
                "current_profit": float(position.get("unrealized_pl", 0)),
                "current_profit_pct": float(position.get("unrealized_plpc", 0)),
                "current_drawdown": position.get("drawdown_pct", 0),
                "holding_period_days": 0,
            }
            return

        # Get current performance
        perf = self.position_performance[symbol]

        # Update max and min prices
        if position.get("side", "long") == "long":
            perf["max_price"] = max(perf["max_price"], float(position.get("current_price", 0)))
            perf["min_price"] = min(perf["min_price"], float(position.get("current_price", 0)))
        else:
            perf["max_price"] = min(perf["max_price"], float(position.get("current_price", 0)))
            perf["min_price"] = max(perf["min_price"], float(position.get("current_price", 0)))

        # Update max profit
        if float(position.get("unrealized_pl", 0)) > perf["max_profit"]:
            perf["max_profit"] = float(position.get("unrealized_pl", 0))
            perf["max_profit_pct"] = float(position.get("unrealized_plpc", 0))

        # Update max drawdown
        if position.get("drawdown_pct", 0) > perf["max_drawdown"]:
            perf["max_drawdown"] = position.get("drawdown_pct", 0)

        # Update current values
        perf["current_price"] = float(position.get("current_price", 0))
        perf["current_profit"] = float(position.get("unrealized_pl", 0))
        perf["current_profit_pct"] = float(position.get("unrealized_plpc", 0))
        perf["current_drawdown"] = position.get("drawdown_pct", 0)

        # Update holding period
        entry_time = perf["entry_time"]
        current_time = datetime.now()
        perf["holding_period_days"] = (current_time - entry_time).total_seconds() / (
            24 * 60 * 60
        )

    def _update_sector_exposure(self) -> None:
        """
        Update sector exposure based on current positions.
        """
        # Reset sector exposure
        self.sector_exposure = {}

        # Get account value
        account = self.alpaca.get_account()
        account_value = float(account["equity"])

        # Calculate sector exposure
        for symbol, position in self.positions.items():
            # Get sector for symbol (placeholder - would need to integrate with a data provider)
            sector = self._get_sector_for_symbol(symbol)

            # Skip if no sector
            if not sector:
                continue

            # Get position value
            position_value = float(position["market_value"])

            # Add to sector exposure
            if sector not in self.sector_exposure:
                self.sector_exposure[sector] = 0

            self.sector_exposure[sector] += position_value

        # Convert to percentages
        for sector in self.sector_exposure:
            self.sector_exposure[sector] = self.sector_exposure[sector] / account_value

    def _get_sector_for_symbol(self, symbol: str) -> str | None:
        """
        Get sector for a symbol.
        
        This method should be implemented to use a proper market data provider
        to get the actual sector information for a given symbol.

        Args:
            symbol: Symbol to get sector for

        Returns:
            Sector or None if not available or if the symbol is invalid
        """
        if not symbol:
            return None
        
        # TODO: Implement integration with a market data provider to get actual sector data
        return "Unknown"

    def check_position_exits(self) -> list[dict[str, Any]]:
        """
        Check if any positions should be exited.

        Returns:
            List of positions that should be exited
        """
        # Update positions
        self.update_positions(force_update=True)

        # Positions to exit
        positions_to_exit = []

        # Check each position
        for symbol, position in self.positions.items():
            exit_reasons = []

            # Check max drawdown
            if position.get("drawdown_pct", 0) >= self.max_drawdown_pct:
                exit_reasons.append("max_drawdown")

            # Check stop loss if available
            if self.stop_loss_manager and "stop_price" in position:
                stop_price = position["stop_price"]
                current_price = float(position["current_price"])

                if position["side"] == "long" and current_price <= stop_price:
                    exit_reasons.append("stop_loss")
                elif position["side"] == "short" and current_price >= stop_price:
                    exit_reasons.append("stop_loss")

            # Check profit target if available
            if self.use_profit_targets and "profit_target" in position:
                profit_target = position["profit_target"]
                current_price = float(position["current_price"])

                if position["side"] == "long" and current_price >= profit_target:
                    exit_reasons.append("profit_target")
                elif position["side"] == "short" and current_price <= profit_target:
                    exit_reasons.append("profit_target")

            # Check trailing target if available
            if self.use_trailing_stops and "trailing_target" in position:
                trailing_target = position["trailing_target"]
                current_price = float(position["current_price"])

                if position["side"] == "long" and current_price <= trailing_target:
                    exit_reasons.append("trailing_stop")
                elif position["side"] == "short" and current_price >= trailing_target:
                    exit_reasons.append("trailing_stop")

            # Check peak detection if available
            if self.use_peak_detection and self.peak_detector:
                # Get historical prices
                bars = self.alpaca.get_bars(symbol, "15Min", limit=100)
                if symbol in bars and not bars[symbol].empty:
                    df = bars[symbol]
                    prices = df["close"].values

                    # Detect peaks
                    exit_points = self.peak_detector.identify_exit_points(
                        prices=prices, direction=position["side"]
                    )

                    # Check if we should exit based on peak detection
                    if exit_points.get("most_recent_exit_point"):
                        exit_reasons.append("peak_detection")

            # If any exit reasons, add to list
            if exit_reasons:
                positions_to_exit.append(
                    {
                        "symbol": symbol,
                        "position": position,
                        "exit_reasons": exit_reasons,
                    }
                )

        return positions_to_exit

    def exit_position(
        self,
        symbol: str,
        qty: float | None = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Exit a position.

        Args:
            symbol: Symbol to exit
            qty: Quantity to exit (if None, exit entire position)
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price: Limit price for limit and stop-limit orders

        Returns:
            Order information or None if position does not exist
        """
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} does not exist")
            return None

        position = self.positions[symbol]

        # Determine quantity to exit
        if qty is None:
            qty = abs(float(position.get("qty", 0)))

        # Determine side (opposite of position side)
        side = "sell" if position.get("side", "long") == "long" else "buy"

        # Submit order
        order = self.alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
        )

        # Log exit
        logger.info(
            f"Exiting position {symbol} {position.get('side', 'long')} {qty} @ "
            f"{limit_price if limit_price else 'market'}: {order['id']}"
        )

        # Add to order tracking
        self.orders[order["id"]] = order

        # Add to order history
        if symbol not in self.order_history:
            self.order_history[symbol] = []

        self.order_history[symbol].append(order)

        return order

    def exit_positions(
        self,
        positions_to_exit: list[dict[str, Any]],
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> list[dict[str, Any]]:
        """
        Exit multiple positions.

        Args:
            positions_to_exit: List of positions to exit
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')

        Returns:
            List of orders
        """
        orders = []

        for position_info in positions_to_exit:
            symbol = position_info["symbol"]

            # Exit position
            order = self.exit_position(
                symbol=symbol, order_type=order_type, time_in_force=time_in_force
            )

            if order:
                orders.append(order)

        return orders

    def get_position_count(self) -> int:
        """
        Get number of open positions.

        Returns:
            Number of open positions
        """
        return len(self.positions)

    def get_positions(self) -> dict[str, dict[str, Any]]:
        """
        Get all positions.

        Returns:
            Dictionary of positions
        """
        return self.positions

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Symbol to get position for

        Returns:
            Position information or None if no position exists
        """
        return self.positions.get(symbol)

    def get_position_pl(self, symbol: str) -> float:
        """
        Get position profit/loss for a specific symbol.

        Args:
            symbol: Symbol to get profit/loss for

        Returns:
            Position profit/loss or 0 if no position exists
        """
        position = self.get_position(symbol)
        if position:
            return float(position.get("unrealized_pl", 0))
        return 0.0

    def get_total_position_value(self) -> float:
        """
        Get total value of all positions.

        Returns:
            Total position value
        """
        return sum(
            float(position["market_value"]) for position in self.positions.values()
        )
        
    def get_position_value(self, symbol: str) -> float:
        """
        Get position value for a specific symbol.

        Args:
            symbol: Symbol to get value for

        Returns:
            Position value or 0 if no position exists
        """
        position = self.get_position(symbol)
        return float(position.get("market_value", 0)) if position else 0.0

    def get_position_exposure(self) -> float:
        """
        Get total position exposure as percentage of account value.

        Returns:
            Total position exposure
        """
        # Get account value
        account = self.alpaca.get_account()
        account_value = float(account["equity"])

        # Calculate exposure
        position_value = self.get_total_position_value()

        return position_value / account_value if account_value > 0 else 0
        
    def get_total_portfolio_value(self) -> float:
        """
        Get total portfolio value.

        Returns:
            Total portfolio value
        """
        # Get account value
        account = self.alpaca.get_account()

        return float(account["equity"])

    def get_sector_exposure(self, sector: str) -> float:
        """
        Get exposure to a specific sector as percentage of account value.

        Args:
            sector: Sector to get exposure for

        Returns:
            Sector exposure
        """
        return self.sector_exposure.get(sector, 0)

    def can_enter_position(
        self, symbol: str, side: str, qty: float, price: float
    ) -> tuple[bool, str]:
        """
        Check if a new position can be entered.

        Args:
            symbol: Symbol to enter
            side: Position side ('long' or 'short')
            qty: Quantity to enter
            price: Entry price

        Returns:
            Tuple of (can_enter, reason)
        """
        # Update positions
        self.update_positions()

        # Check if position already exists
        if symbol in self.positions:
            existing_side = self.positions[symbol]["side"]
            if existing_side != side:
                return (
                    False,
                    f"Position already exists with opposite side: {existing_side}",
                )

        # Check max positions
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            return (
                False,
                f"Maximum positions reached: {len(self.positions)}/{self.max_positions}",
            )

        # Get account value
        account = self.alpaca.get_account()
        account_value = float(account["equity"])

        # Calculate position value
        position_value = qty * price

        # Check max position value
        max_position_value = account_value * self.max_position_value_pct
        if position_value > max_position_value:
            return (
                False,
                f"Position value exceeds maximum: ${position_value:.2f} > ${max_position_value:.2f}",
            )

        # Check sector exposure
        sector = self._get_sector_for_symbol(symbol)
        if sector:
            current_sector_exposure = self.get_sector_exposure(sector)
            new_sector_exposure = current_sector_exposure + (
                position_value / account_value
            )

            if new_sector_exposure > self.max_sector_exposure_pct:
                return (
                    False,
                    f"Sector exposure exceeds maximum: {new_sector_exposure:.1%} > {self.max_sector_exposure_pct:.1%}",
                )

        return True, ""

    def get_position_performance_metrics(self) -> dict[str, dict[str, Any]]:
        """
        Get performance metrics for all positions.

        Returns:
            Dictionary mapping symbols to performance metrics
        """
        # Update positions
        self.update_positions()

        # Return performance metrics
        return self.position_performance

    def get_position_history(self, symbol: str) -> list[dict[str, Any]]:
        """
        Get history for a specific position.

        Args:
            symbol: Symbol to get history for

        Returns:
            List of position history records
        """
        return self.position_history.get(symbol, [])

    def get_order_history(self, symbol: str) -> list[dict[str, Any]]:
        """
        Get order history for a specific symbol.

        Args:
            symbol: Symbol to get order history for

        Returns:
            List of order history records
        """
        return self.order_history.get(symbol, [])
