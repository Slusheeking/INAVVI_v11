"""
Order Generator

This module provides the OrderGenerator class for generating orders based on
trading signals, position sizing, and risk parameters.
"""

import uuid
from datetime import datetime
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("trading_strategy.execution.order_generator")


class OrderGenerator:
    """
    Generates orders based on trading signals, position sizing, and risk parameters.

    The order generation process includes:
    1. Validating trading signals
    2. Calculating position sizes based on risk parameters
    3. Determining order types (market, limit, stop, etc.)
    4. Setting time-in-force parameters
    5. Generating client order IDs
    6. Applying order constraints

    This class serves as the bridge between trading signals and order execution.
    """

    def __init__(
        self,
        position_sizer,
        order_type_selector,
        default_time_in_force: str = "day",
        max_orders_per_batch: int = 10,
        min_order_value: float = 100.0,
        client_order_id_prefix: str = "ATS",
    ):
        """
        Initialize the OrderGenerator.

        Args:
            position_sizer: Position sizer instance
            order_type_selector: Order type selector instance
            default_time_in_force: Default time-in-force parameter
            max_orders_per_batch: Maximum number of orders per batch
            min_order_value: Minimum order value in dollars
            client_order_id_prefix: Prefix for client order IDs
        """
        self.position_sizer = position_sizer
        self.order_type_selector = order_type_selector
        self.default_time_in_force = default_time_in_force
        self.max_orders_per_batch = max_orders_per_batch
        self.min_order_value = min_order_value
        self.client_order_id_prefix = client_order_id_prefix

        logger.info(
            f"Initialized OrderGenerator with max_orders_per_batch={max_orders_per_batch}, "
            f"min_order_value=${min_order_value:.2f}"
        )

    def generate_order(
        self,
        ticker: str,
        quantity: float,
        side: str,
        order_type: str,
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: str = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a simple order with the specified parameters.
        
        Args:
            ticker: Ticker symbol
            quantity: Order quantity
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force parameter
            extended_hours: Whether to allow extended hours trading
            
        Returns:
            Order object
        """
        # Generate client order ID
        client_order_id = self._generate_client_order_id(ticker, side)
        
        # Create order object
        order = {
            "symbol": ticker,
            "qty": quantity,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force or self.default_time_in_force,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "client_order_id": client_order_id,
            "extended_hours": extended_hours,
        }
        
        return order
        
    def generate_orders(self, orders_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Generate multiple orders from a list of order data.
        
        Args:
            orders_data: List of order data dictionaries
            
        Returns:
            List of order objects
        """
        orders = []
        
        for order_data in orders_data:
            order = self.generate_order(
                ticker=order_data.get("ticker"),
                quantity=order_data.get("quantity"),
                side=order_data.get("side"),
                order_type=order_data.get("order_type"),
                limit_price=order_data.get("limit_price")
            )
            orders.append(order)
            
        return orders
        
    def generate_order_from_signal(
        self,
        signal: dict[str, Any],
        account_info: dict[str, Any],
        risk_params: dict[str, Any],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate an order based on a trading signal.

        Args:
            signal: Trading signal with symbol, direction, etc.
            account_info: Account information including balance
            risk_params: Risk parameters for position sizing
            market_data: Current market data for the symbol

        Returns:
            Order object with symbol, quantity, order type, etc.
        """
        # Extract signal information
        symbol = signal.get("symbol")
        direction = signal.get("direction", "long")
        confidence = signal.get("confidence", 1.0)

        # Validate signal
        if not symbol:
            logger.warning("Signal missing symbol, skipping order generation")
            return {}

        # Get current price and volatility
        current_price = market_data.get("last_price", 0)
        atr = market_data.get("atr", 0)

        # Validate price and ATR
        if current_price <= 0 or atr <= 0:
            logger.warning(
                f"Invalid price or ATR for {symbol}, skipping order generation"
            )
            return {}

        # Calculate stop loss price
        stop_loss_price = self._calculate_stop_loss_price(
            current_price, direction, atr, risk_params
        )

        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            ticker=symbol,
            entry_price=current_price,
            stop_price=stop_loss_price,
            conviction_score=confidence,
            volatility_adjustment=self._calculate_volatility_adjustment(
                atr, current_price
            ),
            is_focus_ticker=signal.get("is_focus_ticker", False),
        )

        # Check if position is viable
        if not position_size.get("viable", False):
            logger.info(
                f"Position for {symbol} not viable: {position_size.get('reason')}"
            )
            return {}

        # Determine order type
        (
            order_type,
            limit_price,
            stop_price,
        ) = self.order_type_selector.select_order_type(
            symbol=symbol,
            direction=direction,
            current_price=current_price,
            atr=atr,
            market_data=market_data,
        )

        # Determine side (buy/sell)
        side = "buy" if direction == "long" else "sell"

        # Generate client order ID
        client_order_id = self._generate_client_order_id(symbol, side)

        # Create order object
        order = {
            "symbol": symbol,
            "quantity": position_size.get("shares", 0),
            "side": side,
            "type": order_type,
            "time_in_force": self.default_time_in_force,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "client_order_id": client_order_id,
            "extended_hours": False,
            "position_value": position_size.get("position_value", 0),
            "risk_amount": position_size.get("risk_amount", 0),
            "stop_loss_price": stop_loss_price,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "signal_id": signal.get("id", ""),
            "metadata": {
                "atr": atr,
                "volatility_adjustment": position_size.get(
                    "volatility_adjustment", 1.0
                ),
                "is_focus_ticker": position_size.get("is_focus_ticker", False),
                "conviction_score": confidence,
            },
        }

        # Validate order
        if not self._validate_order(order):
            logger.warning(f"Order validation failed for {symbol}")
            return {}

        logger.info(
            f"Generated {order_type} order for {symbol}: {side} {order['quantity']} shares "
            f"at ${current_price:.2f}, position value=${order['position_value']:.2f}"
        )

        return order

    def generate_orders_batch(
        self,
        signals: list[dict[str, Any]],
        account_info: dict[str, Any],
        risk_params: dict[str, Any],
        market_data: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate orders for a batch of trading signals.

        Args:
            signals: List of trading signals
            account_info: Account information including balance
            risk_params: Risk parameters for position sizing
            market_data: Dictionary mapping symbols to market data

        Returns:
            List of order objects
        """
        orders = []

        # Sort signals by priority (confidence, is_focus_ticker)
        sorted_signals = sorted(
            signals,
            key=lambda x: (x.get("is_focus_ticker", False), x.get("confidence", 0)),
            reverse=True,
        )

        # Limit to max_orders_per_batch
        signals_to_process = sorted_signals[: self.max_orders_per_batch]

        # Generate orders for each signal
        for signal in signals_to_process:
            symbol = signal.get("symbol")

            # Skip signals without symbol
            if not symbol:
                continue

            # Get market data for symbol
            symbol_market_data = market_data.get(symbol, {})

            # Generate order
            order = self.generate_order_from_signal(
                signal, account_info, risk_params, symbol_market_data
            )

            # Add valid orders to batch
            if order:
                orders.append(order)

        logger.info(
            f"Generated {len(orders)} orders from {len(signals_to_process)} signals"
        )

        return orders

    def _calculate_stop_loss_price(
        self,
        current_price: float,
        direction: str,
        atr: float,
        risk_params: dict[str, Any],
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            current_price: Current price of the asset
            direction: Trade direction ('long' or 'short')
            atr: Average True Range value
            risk_params: Risk parameters

        Returns:
            Stop loss price
        """
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)

        if direction == "long":
            stop_price = current_price - (atr * atr_multiplier)
        else:
            stop_price = current_price + (atr * atr_multiplier)

        # Ensure stop price is positive
        stop_price = max(0.01, stop_price)

        return stop_price

    def _calculate_volatility_adjustment(self, atr: float, price: float) -> float:
        """
        Calculate volatility adjustment factor.

        Args:
            atr: Average True Range value
            price: Current price of the asset

        Returns:
            Volatility adjustment factor
        """
        # Calculate ATR as percentage of price
        atr_percentage = atr / price if price > 0 else 0

        # Adjust factor based on deviation from typical
        if atr_percentage < 0.01:  # Low volatility
            adjustment = 1.5  # Increase position size
        elif atr_percentage > 0.03:  # High volatility
            adjustment = 0.5  # Decrease position size
        else:
            # Linear scaling between 0.5 and 1.5
            adjustment = 1.5 - ((atr_percentage - 0.01) / 0.02)

        return adjustment

    def _generate_client_order_id(self, symbol: str, side: str) -> str:
        """
        Generate a unique client order ID.

        Args:
            symbol: Ticker symbol
            side: Order side ('buy' or 'sell')

        Returns:
            Client order ID
        """
        # Generate a UUID
        order_uuid = str(uuid.uuid4()).replace("-", "")[:12]

        # Create client order ID with prefix, symbol, side, and UUID
        client_order_id = f"{self.client_order_id_prefix}_{symbol}_{side}_{order_uuid}"

        return client_order_id

    def _validate_order(self, order: dict[str, Any]) -> bool:
        """
        Validate an order before submission.

        Args:
            order: Order object

        Returns:
            True if order is valid, False otherwise
        """
        # Check required fields
        required_fields = ["symbol", "quantity", "side", "type"]
        for field in required_fields:
            if field not in order or not order[field]:
                logger.warning(f"Order missing required field: {field}")
                return False

        # Check quantity
        quantity = order.get("quantity", 0)
        if quantity <= 0:
            logger.warning(f"Invalid order quantity: {quantity}")
            return False

        # Check position value
        position_value = order.get("position_value", 0)
        if position_value < self.min_order_value:
            logger.warning(
                f"Order value (${position_value:.2f}) below minimum (${self.min_order_value:.2f})"
            )
            return False

        # Check limit price for limit orders
        if order.get("type") in ["limit", "stop_limit"] and not order.get(
            "limit_price"
        ):
            logger.warning("Limit order missing limit price")
            return False

        # Check stop price for stop orders
        if order.get("type") in ["stop", "stop_limit"] and not order.get("stop_price"):
            logger.warning("Stop order missing stop price")
            return False

        return True
