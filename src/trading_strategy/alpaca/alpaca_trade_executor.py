"""
Alpaca Trade Executor

This module provides the AlpacaTradeExecutor class for executing trades through the
Alpaca API, monitoring order status, and handling execution reports.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class AlpacaTradeExecutor:
    """
    Executes trades through the Alpaca API.

    The trade execution process includes:
    1. Submitting orders to the Alpaca API
    2. Monitoring order status
    3. Handling execution reports
    4. Managing order cancellations and replacements
    5. Tracking execution quality metrics

    This class serves as the interface between the trading strategy and the Alpaca API.
    """

    def __init__(
        self,
        alpaca_client,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        execution_timeout: float = 30.0,
        max_concurrent_orders: int = 10,
    ):
        """
        Initialize the AlpacaTradeExecutor.

        Args:
            alpaca_client: Alpaca API client instance
            max_retries: Maximum number of retries for failed orders
            retry_delay: Delay between retries in seconds
            execution_timeout: Timeout for order execution in seconds
            max_concurrent_orders: Maximum number of concurrent orders
        """
        self.alpaca_client = alpaca_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.execution_timeout = execution_timeout
        self.max_concurrent_orders = max_concurrent_orders

        # Order tracking
        self.pending_orders: dict[str, dict[str, Any]] = {}
        self.completed_orders: dict[str, dict[str, Any]] = {}
        self.execution_reports: dict[str, list[dict[str, Any]]] = {}

        # Thread pool for concurrent order execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_orders)

        logger.info(
            f"Initialized AlpacaTradeExecutor with max_retries={max_retries}, "
            f"execution_timeout={execution_timeout}s"
        )

    def submit_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """
        Submit an order to Alpaca.

        Args:
            order: Order dictionary with symbol, quantity, side, etc.

        Returns:
            Dictionary with order status and information
        """
        # Extract order parameters
        symbol = order.get("symbol")
        quantity = order.get("quantity", 0)
        side = order.get("side", "buy")
        order_type = order.get("type", "market")
        time_in_force = order.get("time_in_force", "day")
        limit_price = order.get("limit_price")
        stop_price = order.get("stop_price")
        client_order_id = order.get("client_order_id")
        extended_hours = order.get("extended_hours", False)

        # Validate order
        if not symbol or quantity <= 0:
            logger.warning(
                f"Invalid order parameters: symbol={symbol}, quantity={quantity}"
            )
            return {
                "status": "rejected",
                "reason": "Invalid order parameters",
                "order": order,
            }

        # Submit order to Alpaca
        try:
            # Submit order
            alpaca_order = self.alpaca_client.submit_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                extended_hours=extended_hours,
            )

            # Create order status
            order_status = {
                "status": "submitted",
                "order_id": alpaca_order.get("id"),
                "client_order_id": alpaca_order.get("client_order_id"),
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "submitted_at": alpaca_order.get("submitted_at"),
                "alpaca_order": alpaca_order,
            }

            # Add to pending orders
            self.pending_orders[alpaca_order.get("id")] = order_status

            logger.info(
                f"Submitted {order_type} order for {symbol}: "
                f"{side} {quantity} shares, order_id={alpaca_order.get('id')}"
            )

            return order_status

        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            return {"status": "error", "reason": str(e), "order": order}

    def submit_orders_batch(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Submit a batch of orders to Alpaca.

        Args:
            orders: List of order dictionaries

        Returns:
            List of order status dictionaries
        """
        # Limit to max_concurrent_orders
        if len(orders) > self.max_concurrent_orders:
            logger.warning(
                f"Limiting batch to {self.max_concurrent_orders} orders "
                f"(requested {len(orders)})"
            )
            orders = orders[: self.max_concurrent_orders]

        # Submit orders concurrently
        futures = [self.executor.submit(self.submit_order, order) for order in orders]

        # Wait for all orders to complete
        order_statuses = [future.result() for future in futures]

        logger.info(f"Submitted batch of {len(orders)} orders")

        return order_statuses

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get the status of an order.

        Args:
            order_id: Order ID

        Returns:
            Dictionary with order status and information
        """
        # Check if order is in pending orders
        if order_id in self.pending_orders:
            try:
                # Get order from Alpaca
                alpaca_order = self.alpaca_client.get_order(order_id)

                # Update order status
                order_status = self.pending_orders[order_id].copy()
                order_status["status"] = alpaca_order.get("status")
                order_status["filled_qty"] = alpaca_order.get("filled_qty", 0)
                order_status["filled_quantity"] = alpaca_order.get("filled_qty", 0)  # For backward compatibility
                order_status["filled_avg_price"] = alpaca_order.get("filled_avg_price")
                order_status["updated_at"] = alpaca_order.get("updated_at")

                # Check if order is complete
                if alpaca_order.get("status") in [
                    "filled",
                    "canceled",
                    "expired",
                    "rejected",
                ]:
                    # Move to completed orders
                    self.completed_orders[order_id] = order_status
                    del self.pending_orders[order_id]

                    # Add execution report
                    self._add_execution_report(order_id, alpaca_order)

                return order_status

            except Exception as e:
                logger.error(f"Error getting order status for {order_id}: {e}")
                return {"status": "error", "reason": str(e), "order_id": order_id}

        # Check if order is in completed orders
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]

        # Order not found
        else:
            logger.warning(f"Order {order_id} not found")
            return {"status": "not_found", "order_id": order_id}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            Dictionary with cancellation status
        """
        # Check if order is in pending orders
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found or already completed")
            return {"status": "not_found", "order_id": order_id}

        try:
            # Cancel order in Alpaca
            self.alpaca_client.cancel_order(order_id)

            # Update order status
            order_status = self.get_order_status(order_id)

            logger.info(f"Canceled order {order_id}")

            return {
                "status": "canceled",
                "order_id": order_id,
                "order_status": order_status,
            }

        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return {"status": "error", "reason": str(e), "order_id": order_id}

    def cancel_all_orders(self) -> dict[str, Any]:
        """
        Cancel all pending orders.

        Returns:
            Dictionary with cancellation status
        """
        try:
            # Cancel all orders in Alpaca
            self.alpaca_client.cancel_all_orders()

            # Get pending order IDs
            pending_order_ids = list(self.pending_orders.keys())

            # Update order statuses
            for order_id in pending_order_ids:
                self.get_order_status(order_id)

            logger.info(f"Canceled all orders ({len(pending_order_ids)} pending)")

            return {"status": "success", "canceled_orders": len(pending_order_ids)}

        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return {"status": "error", "reason": str(e)}

    def wait_for_order_completion(
        self, order_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """
        Wait for an order to complete.

        Args:
            order_id: Order ID
            timeout: Timeout in seconds (defaults to execution_timeout)

        Returns:
            Dictionary with order status
        """
        # Set timeout
        if timeout is None:
            timeout = self.execution_timeout

        # Check if order is already completed
        if order_id in self.completed_orders:
            return self.completed_orders[order_id]

        # Check if order is pending
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found")
            return {"status": "not_found", "order_id": order_id}

        # Wait for order completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Get order status
            order_status = self.get_order_status(order_id)

            # Check if order is complete
            if order_status.get("status") in [
                "filled",
                "canceled",
                "expired",
                "rejected",
            ]:
                return order_status

            # Wait before checking again
            time.sleep(0.5)

        # Timeout reached
        logger.warning(f"Timeout waiting for order {order_id} to complete")
        return {
            "status": "timeout",
            "order_id": order_id,
            "order_status": self.get_order_status(order_id),
        }

    def get_execution_report(self, order_id: str) -> list[dict[str, Any]]:
        """
        Get execution report for an order.

        Args:
            order_id: Order ID

        Returns:
            List of execution reports
        """
        return self.execution_reports.get(order_id, [])

    def get_all_execution_reports(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get all execution reports.

        Returns:
            Dictionary mapping order IDs to execution reports
        """
        return self.execution_reports

    def _add_execution_report(
        self, order_id: str, alpaca_order: dict[str, Any]
    ) -> None:
        """
        Add execution report for an order.

        Args:
            order_id: Order ID
            alpaca_order: Alpaca order information
        """
        # Create execution report
        execution_report = {
            "order_id": order_id,
            "client_order_id": alpaca_order.get("client_order_id"),
            "symbol": alpaca_order.get("symbol"),
            "side": alpaca_order.get("side"),
            "type": alpaca_order.get("type"),
            "quantity": float(alpaca_order.get("qty", 0)),
            "filled_quantity": float(alpaca_order.get("filled_qty", 0)),
            "filled_avg_price": float(alpaca_order.get("filled_avg_price", 0))
            if alpaca_order.get("filled_avg_price")
            else None,
            "status": alpaca_order.get("status"),
            "created_at": alpaca_order.get("created_at"),
            "submitted_at": alpaca_order.get("submitted_at"),
            "filled_at": alpaca_order.get("filled_at"),
            "canceled_at": alpaca_order.get("canceled_at"),
            "expired_at": alpaca_order.get("expired_at"),
            "failed_at": alpaca_order.get("failed_at"),
            "timestamp": datetime.now().isoformat(),
        }

        # Add to execution reports
        if order_id not in self.execution_reports:
            self.execution_reports[order_id] = []

        self.execution_reports[order_id].append(execution_report)
