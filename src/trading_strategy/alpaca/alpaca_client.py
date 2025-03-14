"""
Alpaca API Client

This module provides a client for interacting with the Alpaca API for trading operations.

Capabilities:
- Authentication
- Accounts
- Assets
- Corporate Actions
- Orders
- Positions
- Portfolio History
- Watchlists
- Account Configurations
- Account Activities
- Calendar
- Clock
"""

import logging
import os
from datetime import datetime
from typing import Any

from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for interacting with the Alpaca API for trading operations."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the Alpaca API client.

        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY environment variable)
            api_secret: Alpaca API secret (defaults to ALPACA_API_SECRET environment variable)
            base_url: Alpaca API base URL (defaults to ALPACA_API_ENDPOINT environment variable)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.base_url = base_url or os.getenv(
            "ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")

        # Initialize REST API client
        self.api = REST(
            key_id=self.api_key, secret_key=self.api_secret, base_url=self.base_url
        )

        # Initialize streaming client
        self.stream = Stream(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            data_feed="iex",  # Use 'sip' for paid subscription
        )

    # Account Information
    def get_account(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary with account information
        """
        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_equity": float(account.last_equity),
                "daytrade_count": account.daytrade_count,
                "last_maintenance_margin": float(account.last_maintenance_margin),
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "regt_buying_power": float(account.regt_buying_power),
            }
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            raise

    # Positions
    def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of positions
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "side": "long" if float(position.qty) > 0 else "short",
                }
                for position in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """
        Get position for a specific symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Position information or None if no position exists
        """
        try:
            position = self.api.get_position(symbol)
            return {
                "symbol": position.symbol,
                "quantity": float(position.qty),
                "entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": float(position.market_value),
                "cost_basis": float(position.cost_basis),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "side": "long" if float(position.qty) > 0 else "short",
            }
        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Error getting position for {symbol}: {e}")
            raise

    # Orders
    def get_orders(
        self,
        status: str | None = None,
        limit: int = 100,
        after: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get orders.

        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum number of orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp

        Returns:
            List of orders
        """
        try:
            # Convert datetime objects to strings
            if isinstance(after, datetime):
                after = after.isoformat()
            if isinstance(until, datetime):
                until = until.isoformat()

            orders = self.api.list_orders(
                status=status, limit=limit, after=after, until=until
            )

            return [
                {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "time_in_force": order.time_in_force,
                    "limit_price": float(order.limit_price)
                    if order.limit_price
                    else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "quantity": float(order.qty),
                    "filled_quantity": float(order.filled_qty),
                    "status": order.status,
                    "created_at": order.created_at,
                    "updated_at": order.updated_at,
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "expired_at": order.expired_at,
                    "canceled_at": order.canceled_at,
                    "failed_at": order.failed_at,
                    "filled_avg_price": float(order.filled_avg_price)
                    if order.filled_avg_price
                    else None,
                    "extended_hours": order.extended_hours,
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """
        Submit an order.

        Args:
            symbol: Ticker symbol
            quantity: Order quantity
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop_price: Stop price (required for 'stop' and 'stop_limit' orders)
            client_order_id: Client order ID
            extended_hours: Whether to allow trading during extended hours

        Returns:
            Order information
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                extended_hours=extended_hours,
            )

            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "quantity": float(order.qty),
                "filled_quantity": float(order.filled_qty),
                "status": order.status,
                "created_at": order.created_at,
                "updated_at": order.updated_at,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "expired_at": order.expired_at,
                "canceled_at": order.canceled_at,
                "failed_at": order.failed_at,
                "filled_avg_price": float(order.filled_avg_price)
                if order.filled_avg_price
                else None,
                "extended_hours": order.extended_hours,
            }
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            raise

    # Order Management
    def get_order(self, order_id: str) -> dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order information
        """
        try:
            order = self.api.get_order(order_id)

            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "quantity": float(order.qty),
                "filled_quantity": float(order.filled_qty),
                "status": order.status,
                "created_at": order.created_at,
                "updated_at": order.updated_at,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "expired_at": order.expired_at,
                "canceled_at": order.canceled_at,
                "failed_at": order.failed_at,
                "filled_avg_price": float(order.filled_avg_price)
                if order.filled_avg_price
                else None,
                "extended_hours": order.extended_hours,
            }
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel order by ID.

        Args:
            order_id: Order ID
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} canceled")
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders canceled")
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            raise

    # Trade Data
    def get_latest_trades(
        self, symbols: str | list[str], limit: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get latest trades for one or more symbols.

        Args:
            symbols: Ticker symbol or list of symbols
            limit: Maximum number of trades per symbol

        Returns:
            Dictionary mapping symbols to lists of trades
        """
        try:
            # Convert to list if single symbol
            if isinstance(symbols, str):
                symbols = [symbols]

            result = {}
            for symbol in symbols:
                trades = self.api.get_latest_trades(symbol, limit)
                result[symbol] = [
                    {
                        "timestamp": trade.t,
                        "price": float(trade.p),
                        "size": float(trade.s),
                        "exchange": trade.x,
                        "trade_id": trade.i,
                        "tape": trade.z,
                    }
                    for trade in trades
                ]

            return result
        except Exception as e:
            logger.error(f"Error getting latest trades: {e}")
            raise

    # Quote Data
    def get_latest_quotes(
        self, symbols: str | list[str], limit: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get latest quotes for one or more symbols.

        Args:
            symbols: Ticker symbol or list of symbols
            limit: Maximum number of quotes per symbol

        Returns:
            Dictionary mapping symbols to lists of quotes
        """
        try:
            # Convert to list if single symbol
            if isinstance(symbols, str):
                symbols = [symbols]

            result = {}
            for symbol in symbols:
                quotes = self.api.get_latest_quotes(symbol, limit)
                result[symbol] = [
                    {
                        "timestamp": quote.t,
                        "bid_price": float(quote.p),
                        "ask_price": float(quote.P),
                        "bid_size": float(quote.s),
                        "ask_size": float(quote.S),
                        "exchange": quote.x,
                        "conditions": quote.c,
                    }
                    for quote in quotes
                ]

            return result
        except Exception as e:
            logger.error(f"Error getting latest quotes: {e}")
            raise

    # Market Clock
    def get_clock(self) -> dict[str, Any]:
        """
        Get market clock information.

        Returns:
            Dictionary with market clock information
        """
        try:
            clock = self.api.get_clock()
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
        except Exception as e:
            logger.error(f"Error getting market clock: {e}")
            raise

    # Market Calendar
    def get_calendar(
        self,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get market calendar.

        Args:
            start: Start date
            end: End date

        Returns:
            List of market calendar days
        """
        try:
            calendar = self.api.get_calendar(start=start, end=end)
            return [
                {
                    "date": day.date,
                    "open": day.open,
                    "close": day.close,
                    "session_open": day.session_open,
                    "session_close": day.session_close,
                }
                for day in calendar
            ]
        except Exception as e:
            logger.error(f"Error getting market calendar: {e}")
            raise

    # Assets
    def get_assets(
        self, status: str = "active", asset_class: str = "us_equity"
    ) -> list[dict[str, Any]]:
        """
        Get assets.

        Args:
            status: Asset status ('active', 'inactive')
            asset_class: Asset class ('us_equity', 'crypto')

        Returns:
            List of assets
        """
        try:
            assets = self.api.list_assets(status=status, asset_class=asset_class)
            return [
                {
                    "id": asset.id,
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "status": asset.status,
                    "tradable": asset.tradable,
                    "marginable": asset.marginable,
                    "shortable": asset.shortable,
                    "easy_to_borrow": asset.easy_to_borrow,
                    "fractionable": asset.fractionable,
                }
                for asset in assets
            ]
        except Exception as e:
            logger.error(f"Error getting assets: {e}")
            raise

    def get_asset(self, symbol: str) -> dict[str, Any]:
        """
        Get asset by symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Asset information
        """
        try:
            asset = self.api.get_asset(symbol)
            return {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "status": asset.status,
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
            }
        except Exception as e:
            logger.error(f"Error getting asset {symbol}: {e}")
            raise

    # Watchlists
    def get_watchlists(self) -> list[dict[str, Any]]:
        """
        Get all watchlists.

        Returns:
            List of watchlists
        """
        try:
            watchlists = self.api.get_watchlists()
            return [
                {
                    "id": watchlist.id,
                    "name": watchlist.name,
                    "account_id": watchlist.account_id,
                    "created_at": watchlist.created_at,
                    "updated_at": watchlist.updated_at,
                }
                for watchlist in watchlists
            ]
        except Exception as e:
            logger.error(f"Error getting watchlists: {e}")
            raise

    def get_watchlist(self, watchlist_id: str) -> dict[str, Any]:
        """
        Get a specific watchlist.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            Watchlist information
        """
        try:
            watchlist = self.api.get_watchlist(watchlist_id)
            return {
                "id": watchlist.id,
                "name": watchlist.name,
                "account_id": watchlist.account_id,
                "created_at": watchlist.created_at,
                "updated_at": watchlist.updated_at,
                "assets": [
                    {"id": asset.id, "symbol": asset.symbol, "name": asset.name}
                    for asset in watchlist.assets
                ],
            }
        except Exception as e:
            logger.error(f"Error getting watchlist {watchlist_id}: {e}")
            raise

    def create_watchlist(self, name: str, symbols: list[str] = None) -> dict[str, Any]:
        """
        Create a new watchlist.

        Args:
            name: Watchlist name
            symbols: List of symbols to add to the watchlist

        Returns:
            Created watchlist information
        """
        try:
            watchlist = self.api.create_watchlist(name, symbols or [])
            return {
                "id": watchlist.id,
                "name": watchlist.name,
                "account_id": watchlist.account_id,
                "created_at": watchlist.created_at,
                "updated_at": watchlist.updated_at,
            }
        except Exception as e:
            logger.error(f"Error creating watchlist {name}: {e}")
            raise

    def update_watchlist(self, watchlist_id: str, name: str = None, symbols: list[str] = None) -> dict[str, Any]:
        """
        Update a watchlist.

        Args:
            watchlist_id: Watchlist ID
            name: New watchlist name
            symbols: New list of symbols

        Returns:
            Updated watchlist information
        """
        try:
            watchlist = self.api.update_watchlist(watchlist_id, name, symbols)
            return {
                "id": watchlist.id,
                "name": watchlist.name,
                "account_id": watchlist.account_id,
                "created_at": watchlist.created_at,
                "updated_at": watchlist.updated_at,
            }
        except Exception as e:
            logger.error(f"Error updating watchlist {watchlist_id}: {e}")
            raise

    def delete_watchlist(self, watchlist_id: str) -> None:
        """
        Delete a watchlist.

        Args:
            watchlist_id: Watchlist ID
        """
        try:
            self.api.delete_watchlist(watchlist_id)
            logger.info(f"Watchlist {watchlist_id} deleted")
        except Exception as e:
            logger.error(f"Error deleting watchlist {watchlist_id}: {e}")
            raise

    # Account Activities
    def get_account_activities(
        self,
        activity_types: str | list[str] = None,
        start: str | datetime = None,
        end: str | datetime = None,
        direction: str = "desc",
        page_size: int = 100,
        page_token: str = None,
    ) -> list[dict[str, Any]]:
        """
        Get account activities.

        Args:
            activity_types: Activity types to filter by
            start: Start date/time
            end: End date/time
            direction: Sort direction ('asc' or 'desc')
            page_size: Number of activities per page
            page_token: Page token for pagination

        Returns:
            List of account activities
        """
        try:
            activities = self.api.get_activities(
                activity_types=activity_types,
                start=start,
                end=end,
                direction=direction,
                page_size=page_size,
                page_token=page_token,
            )
            return [
                {
                    "id": activity.id,
                    "activity_type": activity.activity_type,
                    "transaction_time": activity.transaction_time,
                    "type": activity.type,
                    "price": float(activity.price) if activity.price else None,
                    "qty": float(activity.qty) if activity.qty else None,
                    "side": activity.side,
                    "symbol": activity.symbol,
                    "leaves_qty": float(activity.leaves_qty) if activity.leaves_qty else None,
                    "order_id": activity.order_id,
                    "cum_qty": float(activity.cum_qty) if activity.cum_qty else None,
                }
                for activity in activities
            ]
        except Exception as e:
            logger.error(f"Error getting account activities: {e}")
            raise

    # Portfolio History
    def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
        date_end: str | datetime = None,
        extended_hours: bool = False,
    ) -> dict[str, Any]:
        """
        Get portfolio history.

        Args:
            period: Time period to get history for ('1D', '1W', '1M', '3M', '1A', 'all')
            timeframe: Resolution of data ('1Min', '5Min', '15Min', '1H', '1D')
            date_end: End date for the history
            extended_hours: Whether to include extended hours

        Returns:
            Portfolio history data
        """
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe,
                date_end=date_end,
                extended_hours=extended_hours,
            )
            return {
                "timestamp": history.timestamp,
                "equity": history.equity,
                "profit_loss": history.profit_loss,
                "profit_loss_pct": history.profit_loss_pct,
                "base_value": history.base_value,
                "timeframe": history.timeframe,
            }
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            raise

    # Account Configurations
    def get_account_configurations(self) -> dict[str, Any]:
        """
        Get account configurations.

        Returns:
            Account configurations
        """
        try:
            config = self.api.get_account_configurations()
            return {
                "dtbp_check": config.dtbp_check,
                "no_shorting": config.no_shorting,
                "suspend_trade": config.suspend_trade,
                "trade_confirm_email": config.trade_confirm_email,
            }
        except Exception as e:
            logger.error(f"Error getting account configurations: {e}")
            raise

    def update_account_configurations(
        self,
        dtbp_check: str = None,
        no_shorting: bool = None,
        suspend_trade: bool = None,
        trade_confirm_email: str = None,
    ) -> dict[str, Any]:
        """
        Update account configurations.

        Args:
            dtbp_check: Day trading buying power check ('both', 'entry', 'exit', 'none')
            no_shorting: Whether to disable shorting
            suspend_trade: Whether to suspend trading
            trade_confirm_email: Trade confirmation email setting ('all', 'none')

        Returns:
            Updated account configurations
        """
        try:
            params = {}
            if dtbp_check is not None:
                params["dtbp_check"] = dtbp_check
            if no_shorting is not None:
                params["no_shorting"] = no_shorting
            if suspend_trade is not None:
                params["suspend_trade"] = suspend_trade
            if trade_confirm_email is not None:
                params["trade_confirm_email"] = trade_confirm_email

            config = self.api.update_account_configurations(**params)
            return {
                "dtbp_check": config.dtbp_check,
                "no_shorting": config.no_shorting,
                "suspend_trade": config.suspend_trade,
                "trade_confirm_email": config.trade_confirm_email,
            }
        except Exception as e:
            logger.error(f"Error updating account configurations: {e}")
            raise

    # Corporate Actions
    def get_corporate_actions(
        self,
        action_type: str = None,
        symbol: str = None,
        start: str | datetime = None,
        end: str | datetime = None,
    ) -> list[dict[str, Any]]:
        """
        Get corporate actions.

        Args:
            action_type: Type of corporate action
            symbol: Symbol to filter by
            start: Start date
            end: End date

        Returns:
            List of corporate actions
        """
        try:
            actions = self.api.get_corporate_actions(
                action_type=action_type,
                symbol=symbol,
                start=start,
                end=end,
            )
            return [
                {
                    "id": action.id,
                    "symbol": action.symbol,
                    "action_type": action.action_type,
                    "date": action.date,
                    "amount": float(action.amount) if action.amount else None,
                    "ratio": action.ratio,
                }
                for action in actions
            ]
        except Exception as e:
            logger.error(f"Error getting corporate actions: {e}")
            raise

    def close(self) -> None:
        """
        Close all connections and release resources.
        
        This method should be called when the client is no longer needed.
        """
        self.api._session.close()
        logger.info("AlpacaClient resources released")
