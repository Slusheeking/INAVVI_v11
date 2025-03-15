"""
Options Data Collector

This module provides a collector for gathering options data from various sources.
Note: Options data is collected from Polygon.io as Alpaca does not provide options data in the free tier.
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
from src.data_acquisition.api.polygon_client import PolygonClient

# Import pandas and numpy directly
import pandas as pd

# Import logging utility
from src.utils.logging import get_logger

# Set up logger
logger = get_logger("data_acquisition.collectors.options_collector")


class OptionsCollector:
    """Collects options data from Polygon.io."""

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the options collector.

        Args:
            polygon_client: Polygon.io API client
            config: Configuration dictionary with options like max_threads
        """
        self.polygon = polygon_client or PolygonClient()
        self.config = config or {}

        # Set up thread pool for parallel data collection
        self.max_threads = self.config.get("max_threads", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)

        logger.info(f"Initialized OptionsCollector with {self.max_threads} threads")

    def collect_options_chain(
        self,
        underlyings: List[str],
        expiration_date: Optional[Union[str, datetime, date]] = None,
        strike_price: Optional[float] = None,
        contract_type: Optional[str] = None,  # 'call' or 'put'
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Collect options chains for multiple underlying assets.

        Args:
            underlyings: List of underlying asset symbols
            expiration_date: Optional expiration date filter (YYYY-MM-DD or datetime)
            strike_price: Optional strike price filter
            contract_type: Optional contract type filter ('call' or 'put')

        Returns:
            Dictionary mapping underlying symbols to lists of option contracts
        """
        logger.info(f"Collecting options chains for {len(underlyings)} underlyings")

        # Collect options chains for each underlying in parallel
        futures = {}
        results = {}

        for underlying in underlyings:
            future = self.executor.submit(
                self._collect_underlying_options_chain,
                underlying,
                expiration_date,
                strike_price,
                contract_type,
            )
            futures[future] = underlying

        # Process results as they complete
        for future in as_completed(futures):
            underlying = futures[future]
            try:
                options_chain = future.result()
                if options_chain:
                    results[underlying] = options_chain
            except Exception as e:
                logger.error(f"Error collecting options chain for {underlying}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(
            f"Completed options chain collection for {len(results)} underlyings"
        )
        return results

    def _collect_underlying_options_chain(
        self,
        underlying: str,
        expiration_date: Optional[Union[str, datetime, date]] = None,
        strike_price: Optional[float] = None,
        contract_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect options chain for a single underlying asset.

        Args:
            underlying: Underlying asset symbol
            expiration_date: Optional expiration date filter (YYYY-MM-DD or datetime)
            strike_price: Optional strike price filter
            contract_type: Optional contract type filter ('call' or 'put')

        Returns:
            List of option contracts
        """
        try:
            logger.debug(f"Collecting options chain for {underlying}")

            # Get options chain from Polygon API
            options_chain = self.polygon.get_options_chain(
                underlying, expiration_date, strike_price, contract_type
            )

            if not options_chain:
                logger.warning(f"No options found for {underlying}")
                return []

            logger.debug(f"Collected {len(options_chain)} options for {underlying}")
            return options_chain

        except Exception as e:
            logger.error(f"Error collecting options chain for {underlying}: {e}")
            logger.debug(traceback.format_exc())
            return []

    def collect_options_expirations(
        self, underlyings: List[str]
    ) -> dict[str, list[str]]:
        """
        Collect available expiration dates for multiple underlying assets.

        Args:
            underlyings: List of underlying asset symbols

        Returns:
            Dictionary mapping underlying symbols to lists of expiration dates
        """
        logger.info(
            f"Collecting options expirations for {len(underlyings)} underlyings"
        )

        # First collect all options chains
        all_chains = self.collect_options_chain(underlyings)

        # Extract unique expiration dates for each underlying
        expirations = {}

        for underlying, options_chain in all_chains.items():
            if options_chain:
                # Extract expiration dates
                exp_dates = sorted(
                    {
                        option.get("expiration_date")
                        for option in options_chain
                        if option.get("expiration_date")
                    }
                )

                if exp_dates:
                    expirations[underlying] = exp_dates
                    logger.debug(
                        f"Found {len(exp_dates)} expiration dates for {underlying}"
                    )
                else:
                    logger.debug(f"No expiration dates found for {underlying}")

        logger.info(
            f"Completed options expiration collection for {len(expirations)} underlyings"
        )
        return expirations

    def collect_options_strikes(
        self, underlyings: List[str], expiration_date: Union[str, datetime, date]
    ) -> dict[str, dict[str, list[float]]]:
        """
        Collect available strike prices for multiple underlying assets for a specific expiration date.

        Args:
            underlyings: List of underlying asset symbols
            expiration_date: Expiration date filter (YYYY-MM-DD or datetime)

        Returns:
            Dictionary mapping underlying symbols to dictionaries of contract types and strike prices
        """
        logger.info(
            f"Collecting options strikes for {len(underlyings)} underlyings for expiration {expiration_date}"
        )

        # Ensure expiration date is in the correct format
        if isinstance(expiration_date, datetime):
            exp_date_str = expiration_date.strftime("%Y-%m-%d")
        elif isinstance(expiration_date, date):
            exp_date_str = expiration_date.strftime("%Y-%m-%d")
        else:
            exp_date_str = expiration_date

        # Collect options chains for the specific expiration date
        all_chains = self.collect_options_chain(underlyings, exp_date_str)

        # Extract unique strike prices for each underlying, separated by call/put
        strikes = {}

        for underlying, options_chain in all_chains.items():
            if options_chain:
                # Separate calls and puts
                call_strikes = sorted(
                    {
                        option.get("strike_price")
                        for option in options_chain
                        if option.get("strike_price")
                        and option.get("contract_type") == "call"
                    }
                )

                put_strikes = sorted(
                    {
                        option.get("strike_price")
                        for option in options_chain
                        if option.get("strike_price")
                        and option.get("contract_type") == "put"
                    }
                )

                if call_strikes or put_strikes:
                    strikes[underlying] = {"call": call_strikes, "put": put_strikes}
                    logger.debug(
                        f"Found {len(call_strikes)} call strikes and {len(put_strikes)} put strikes for {underlying}"
                    )
                else:
                    logger.debug(f"No strike prices found for {underlying}")

        logger.info(
            f"Completed options strike collection for {len(strikes)} underlyings"
        )
        return strikes

    def collect_options_prices(
        self, option_symbols: List[str], date_to_collect: Union[str, datetime, date]
    ) -> dict[str, pd.DataFrame]:
        """
        Collect historical prices for specific option contracts.

        Args:
            option_symbols: List of option contract symbols (e.g., 'O:AAPL230616C00150000')
            date_to_collect: Date to collect prices for

        Returns:
            Dictionary mapping option symbols to DataFrames with price data
        """
        logger.info(
            f"Collecting options prices for {len(option_symbols)} contracts on {date_to_collect}"
        )

        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime("%Y-%m-%d")
        else:
            date_str = date_to_collect

        # Collect prices for each option contract in parallel
        futures = {}
        results = {}

        for option_symbol in option_symbols:
            future = self.executor.submit(
                self._collect_option_prices, option_symbol, date_str
            )
            futures[future] = option_symbol

        # Process results as they complete
        for future in as_completed(futures):
            option_symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[option_symbol] = df
            except Exception as e:
                logger.error(f"Error collecting prices for {option_symbol}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed options price collection for {len(results)} contracts")
        return results

    def _collect_option_prices(self, option_symbol: str, date_str: str) -> pd.DataFrame:
        """
        Collect historical prices for a single option contract.

        Args:
            option_symbol: Option contract symbol
            date_str: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with price data
        """
        try:
            logger.debug(f"Collecting prices for {option_symbol} on {date_str}")

            # Get aggregates from Polygon API
            # Note: For options, we use the same aggregates endpoint but with the option symbol
            df = self.polygon.get_aggregates(
                option_symbol, 1, "minute", date_str, date_str, adjusted="true"
            )

            if df.empty:
                logger.warning(f"No prices found for {option_symbol} on {date_str}")
                return df

            logger.debug(
                f"Collected {len(df)} price points for {option_symbol} on {date_str}"
            )
            return df

        except Exception as e:
            logger.error(
                f"Error collecting prices for {option_symbol} on {date_str}: {e}"
            )
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Error collecting prices for {option_symbol} on {date_str}: {e}"
            )
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
                
    def collect_options_greeks(
        self,
        option_symbols: List[str],
        underlying_prices: Dict[str, float],
        risk_free_rate: float = 0.05,
        pricing_date: Optional[Union[str, datetime, date]] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate option Greeks for a list of option contracts.

        Args:
            option_symbols: List of option contract symbols
            underlying_prices: Dictionary mapping underlying symbols to current prices
            risk_free_rate: Risk-free interest rate (annual, decimal)
            pricing_date: Date for pricing (defaults to today)

        Returns:
            Dictionary mapping option symbols to dictionaries of Greeks
        """
        logger.info(f"Calculating Greeks for {len(option_symbols)} option contracts")

        # Set pricing date to today if not provided
        if pricing_date is None:
            pricing_date = datetime.now().date()
        elif isinstance(pricing_date, str):
            pricing_date = datetime.strptime(pricing_date, "%Y-%m-%d").date()
        elif isinstance(pricing_date, datetime):
            pricing_date = pricing_date.date()

        # Collect options chains to get contract details
        # First, extract underlying symbols from option symbols
        # Option symbols format: O:AAPL230616C00150000
        underlyings = set()
        for option_symbol in option_symbols:
            parts = option_symbol.split(":")
            if len(parts) > 1:
                # Extract ticker from option symbol
                ticker_part = parts[1]
                # Find where the date part starts (first digit)
                for i, char in enumerate(ticker_part):
                    if char.isdigit():
                        underlying = ticker_part[:i]
                        underlyings.add(underlying)
                        break

        # Collect options chains for all underlyings
        all_chains = self.collect_options_chain(list(underlyings))

        # Calculate Greeks for each option
        greeks = {}

        for option_symbol in option_symbols:
            try:
                # Parse option symbol to extract details
                parts = option_symbol.split(":")
                if len(parts) <= 1:
                    logger.warning(f"Invalid option symbol format: {option_symbol}")
                    continue

                ticker_part = parts[1]

                # Find where the date part starts (first digit)
                for i, char in enumerate(ticker_part):
                    if char.isdigit():
                        underlying = ticker_part[:i]
                        # Extract the rest of the symbol (not used currently)
                        break
                else:
                    logger.warning(
                        f"Could not parse underlying from option symbol: {option_symbol}"
                    )
                    continue

                # Find contract in the options chain
                contract = None
                if underlying in all_chains:
                    for opt in all_chains[underlying]:
                        if opt.get("ticker") == option_symbol:
                            contract = opt
                            break

                if not contract:
                    logger.warning(f"Contract details not found for {option_symbol}")
                    continue

                # Extract contract details
                strike = contract.get("strike_price")
                expiration = contract.get("expiration_date")
                contract_type = contract.get("contract_type", "").lower()

                if not all(
                    [strike, expiration, contract_type, underlying in underlying_prices]
                ):
                    logger.warning(f"Missing required data for {option_symbol}")
                    continue

                # Calculate time to expiration in years
                exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                days_to_expiry = (exp_date - pricing_date).days
                if days_to_expiry <= 0:
                    logger.warning(
                        f"Option {option_symbol} has expired or expires today"
                    )
                    continue

                t = days_to_expiry / 365.0

                # Get underlying price
                s = underlying_prices[underlying]

                # Get implied volatility (if available) or use a default
                iv = contract.get("implied_volatility")
                if iv is None:
                    # If IV is not available, try to estimate from recent prices
                    # This is a simplified approach - in a real system, you'd use a more sophisticated IV calculation
                    iv = 0.3  # Default to 30% volatility

                # Calculate option price and Greeks using Black-Scholes model
                # Note: This is a simplified implementation
                # In a production system, you'd use a more sophisticated model and library
                from scipy.stats import norm
                import math

                # Black-Scholes formula
                d1 = (math.log(s / strike) + (risk_free_rate + 0.5 * iv**2) * t) / (
                    iv * math.sqrt(t)
                )
                d2 = d1 - iv * math.sqrt(t)

                if contract_type == "call":
                    price = s * norm.cdf(d1) - strike * math.exp(
                        -risk_free_rate * t
                    ) * norm.cdf(d2)
                    delta = norm.cdf(d1)
                    gamma = norm.pdf(d1) / (s * iv * math.sqrt(t))
                    theta = -(s * iv * norm.pdf(d1)) / (
                        2 * math.sqrt(t)
                    ) - risk_free_rate * strike * math.exp(
                        -risk_free_rate * t
                    ) * norm.cdf(
                        d2
                    )
                    vega = (
                        s * math.sqrt(t) * norm.pdf(d1) / 100
                    )  # Divided by 100 to get the change per 1% change in IV
                    rho = (
                        strike * t * math.exp(-risk_free_rate * t) * norm.cdf(d2) / 100
                    )  # Divided by 100 to get the change per 1% change in interest rate
                else:  # put
                    price = strike * math.exp(-risk_free_rate * t) * norm.cdf(
                        -d2
                    ) - s * norm.cdf(-d1)
                    delta = norm.cdf(d1) - 1
                    gamma = norm.pdf(d1) / (s * iv * math.sqrt(t))
                    theta = -(s * iv * norm.pdf(d1)) / (
                        2 * math.sqrt(t)
                    ) + risk_free_rate * strike * math.exp(
                        -risk_free_rate * t
                    ) * norm.cdf(
                        -d2
                    )
                    vega = s * math.sqrt(t) * norm.pdf(d1) / 100
                    rho = (
                        -strike
                        * t
                        * math.exp(-risk_free_rate * t)
                        * norm.cdf(-d2)
                        / 100
                    )

                # Store Greeks
                greeks[option_symbol] = {
                    "price": price,
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "rho": rho,
                    "implied_volatility": iv,
                    "underlying_price": s,
                    "strike": strike,
                    "days_to_expiry": days_to_expiry,
                    "contract_type": contract_type,
                }

                logger.debug(f"Calculated Greeks for {option_symbol}")

            except Exception as e:
                logger.error(f"Error calculating Greeks for {option_symbol}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Completed Greeks calculation for {len(greeks)} option contracts")
        return greeks
