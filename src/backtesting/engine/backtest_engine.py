"""
BacktestEngine: Core component that orchestrates the backtesting process.

This module provides the main engine for running backtests of trading strategies
using historical market data. It coordinates the interaction between market simulation,
execution simulation, and strategy evaluation.
"""

import datetime
import importlib
import json
import uuid
from pathlib import Path
import os
from typing import Any, Dict, List, Optional
import pandas as pd

from ...config.backtesting_config import BacktestingConfig
from ..analysis.backtest_analyzer import (
    BacktestAnalyzer,
)
from ..analysis.strategy_evaluator import (
    StrategyEvaluator,
)
from .execution_simulator import (
    ExecutionSimulator,
)
from .market_simulator import (
    MarketSimulator,
)
from ...utils.time.market_calendar import MarketCalendar
from ...utils.logging.logger import setup_logger


class BacktestEngine:
    """
    Main engine for running backtests of trading strategies.

    The BacktestEngine coordinates the interaction between market data, strategy signals,
    order execution, and performance analysis to provide a comprehensive backtesting
    environment.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        execution_model: str = "realistic",
        data_source: str = "timescaledb",
        timeframes: Optional[List[str]] = None,
        commission_model: Optional[Dict[str, Any]] = None,
        slippage_model: Optional[Dict[str, Any]] = None,
        market_impact_model: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the BacktestEngine with configuration parameters.

        Args:
            start_date: Start date for the backtest in YYYY-MM-DD format
            end_date: End date for the backtest in YYYY-MM-DD format
            initial_capital: Initial capital for the backtest
            execution_model: Execution model type ('perfect', 'next_bar', 'realistic')
            data_source: Source of market data ('timescaledb', 'csv', 'api')
            timeframes: List of timeframes to include in the backtest
            commission_model: Configuration for commission model
            slippage_model: Configuration for slippage model
            market_impact_model: Configuration for market impact model
            log_level: Logging level
        """
        # Set up logging
        self.logger = setup_logger(__name__, log_level)

        # Store configuration
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.execution_model = execution_model
        self.data_source = data_source
        self.timeframes = timeframes or ["1d"]

        # Set up transaction cost models
        self.commission_model = commission_model or {
            "type": "percentage",
            "value": 0.001,  # 0.1% commission by default
        }
        self.slippage_model = slippage_model or {
            "type": "fixed",
            "value": 0.0001,  # 1 basis point by default
        }
        self.market_impact_model = market_impact_model or {
            "type": "linear",
            "factor": 0.1,
        }

        # Initialize components
        self.market_simulator = MarketSimulator(
            data_source=self.data_source,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframes=self.timeframes,
        )

        self.execution_simulator = ExecutionSimulator(
            execution_model=self.execution_model,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            market_impact_model=self.market_impact_model,
        )

        self.strategy_evaluator = StrategyEvaluator()
        self.analyzer = BacktestAnalyzer()

        # Initialize state
        self.universe = []
        self.strategy = None
        self.strategy_config = {}
        self.results = None
        self.portfolio = {
            "cash": initial_capital,
            "positions": {},
            "equity": initial_capital,
            "history": [],
        }

        # Generate a unique ID for this backtest
        self.backtest_id = str(uuid.uuid4())

        self.logger.info(f"BacktestEngine initialized with ID {self.backtest_id} for period {start_date} to {end_date}")

    def set_universe(self, symbols: List[str]) -> None:
        """
        Set the universe of tradable symbols for the backtest.

        Args:
            symbols: List of ticker symbols
        """
        self.universe = symbols
        self.logger.info(f"Universe set with {len(symbols)} symbols: {', '.join(symbols[:5])}{' and more' if len(symbols) > 5 else ''}")

        # Load market data for the universe
        self.market_simulator.load_data(symbols)

    def load_strategy(self, strategy_name: str, version: str = "latest") -> Any:
        """
        Load a trading strategy by name and version.

        Args:
            strategy_name: Name of the strategy
            version: Version of the strategy

        Returns:
            The loaded strategy object
        """
        # Construct the strategy path
        strategy_module_path = f"src.trading_strategy.strategies.{strategy_name.lower()}"

        try:
            # Import the strategy module
            strategy_module = importlib.import_module(strategy_module_path)

            # Load the strategy class
            strategy_class = getattr(strategy_module, strategy_name)

            # Load strategy configuration
            # Try multiple possible locations for the config file
            # First, try to get the project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            
            possible_paths = [
                # Project root / configs / strategies
                project_root / "configs" / "strategies" / f"{strategy_name.lower()}_{version}.json",
                # Project root / src / config
                project_root / "src" / "config" / f"strategy_{strategy_name.lower()}.json",
                # Current directory / configs / strategies
                Path(os.getcwd()) / "configs" / "strategies" / f"{strategy_name.lower()}_{version}.json",
                # From backtesting config
                Path(BacktestingConfig().get_full_config().get("strategy_config_path", "")) / f"{strategy_name.lower()}_{version}.json"
            ]
            
            config_path = next((p for p in possible_paths if p.exists()), possible_paths[0])
            
            self.logger.debug(f"Looking for strategy config at: {config_path.absolute()}")
            self.logger.debug(f"Current working directory: {Path.cwd()}")
            if config_path.exists():
                with open(config_path) as f:
                    self.strategy_config = json.load(f)
            else:
                self.logger.warning(
                    f"Strategy configuration file not found: {config_path}"
                )
                self.logger.warning(f"Tried paths: {[str(p) for p in possible_paths]}")
                self.strategy_config = {}

            # Initialize the strategy
            self.strategy = strategy_class(self.strategy_config)
            self.logger.info(
                f"Strategy {strategy_name} (version {version}) loaded successfully"
            )

            return self.strategy

        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load strategy {strategy_name}: {str(e)}")
            raise ValueError(f"Strategy {strategy_name} could not be loaded: {str(e)}")

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest with the configured settings.

        Returns:
            Dictionary containing backtest results
        """
        if not self.universe:
            self.logger.error(
                "Universe not set. Call set_universe() before running the backtest."
            )
            raise ValueError("Universe not set")

        if not self.strategy:
            self.logger.error(
                "Strategy not loaded. Call load_strategy() before running the backtest."
            )
            raise ValueError("Strategy not loaded")

        self.logger.info(
            f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}"
        )

        # Initialize results containers
        trades = []
        portfolio_history = []
        signals = []

        # Get the market calendar (default to NYSE)
        market_calendar = MarketCalendar(exchange="NYSE")

        # Convert string dates to datetime objects
        start_date_dt = self.start_date
        end_date_dt = self.end_date

        # Get all trading days in the backtest period
        trading_days = market_calendar.get_trading_days(start_date_dt, end_date_dt)

        # Main backtest loop
        for day in trading_days:
            self.logger.debug(f"Processing day: {day}")

            # Get market data for the current day
            try:
                self.logger.debug(f"Fetching market data for {day}")
            except Exception as e:
                self.logger.error(f"Error logging day: {str(e)}")
            market_data = self.market_simulator.get_data_for_date(day)

            if not market_data:
                self.logger.warning(f"No market data available for {day}")
                continue

            # Generate trading signals
            try:
                self.logger.debug(f"Generating signals for {day}")
                day_signals = self.strategy.generate_signals(market_data, self.portfolio)
                signals.extend(day_signals)
                if day_signals:
                    self.logger.info(f"Generated {len(day_signals)} signals for {day}")
            except Exception as e:
                self.logger.error(f"Error generating signals for {day}: {str(e)}")
                day_signals = []

            # Execute signals
            try:
                self.logger.debug(f"Executing signals for {day}")
                day_trades = self.execution_simulator.execute_signals(
                    signals=day_signals, market_data=market_data, portfolio=self.portfolio
                )
                if day_trades:
                    self.logger.info(f"Executed {len(day_trades)} trades for {day}")
                    for trade in day_trades:
                        self.logger.debug(f"Trade: {trade['symbol']} {trade['quantity']} @ {trade['price']}")
            except Exception as e:
                self.logger.error(f"Error executing signals for {day}: {str(e)}")
                self.logger.error(f"Portfolio state: Cash={self.portfolio['cash']}, Equity={self.portfolio['equity']}")
                day_trades = []
                
            trades.extend(day_trades)

            # Update portfolio
            self._update_portfolio(day_trades, market_data, day)

            # Record portfolio state
            portfolio_snapshot = self._get_portfolio_snapshot(day)
            portfolio_history.append(portfolio_snapshot)

        # Compile results
        self.results = {
            "backtest_id": self.backtest_id,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_capital": self.initial_capital,
            "final_equity": self.portfolio["equity"],
            "trades": trades,
            "signals": signals,
            "portfolio_history": portfolio_history,
            "universe": self.universe,
            "strategy": self.strategy.__class__.__name__,
            "strategy_config": self.strategy_config,
            "execution_model": self.execution_model,
            "commission_model": self.commission_model,
            "slippage_model": self.slippage_model,
            "market_impact_model": self.market_impact_model,
        }

        # Analyze results
        self.results["metrics"] = self.analyzer.calculate_metrics(self.results)

        self.logger.info(
            f"Backtest completed. Initial capital: ${self.initial_capital:.2f}, Final equity: ${self.portfolio['equity']:.2f}, Return: {((self.portfolio['equity'] / self.initial_capital) - 1) * 100:.2f}%"
        )

        return self.results

    def get_analyzer(self) -> BacktestAnalyzer:
        """
        Get the backtest analyzer instance.

        Returns:
            The BacktestAnalyzer instance
        """
        return self.analyzer

    def save_results(self, output_dir: str) -> str:
        """
        Save backtest results to disk.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to the saved results file
        """
        if not self.results:
            self.logger.error("No results to save. Run the backtest first.")
            raise ValueError("No results to save")

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results to JSON file
        results_file = output_path / f"backtest_{self.backtest_id}.json"

        # Convert complex objects to serializable format
        serializable_results = self._prepare_results_for_serialization(self.results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        return str(results_file)

    def run_walk_forward(
        self, training_window: int, testing_window: int, step_size: int = None
    ) -> Dict[str, Any]:
        """
        Run a walk-forward optimization backtest.

        Args:
            training_window: Number of days for the training window
            testing_window: Number of days for the testing window
            step_size: Number of days to step forward (defaults to testing_window)

        Returns:
            Dictionary containing walk-forward backtest results
        """
        if step_size is None:
            step_size = testing_window

        # Get the market calendar
        market_calendar = MarketCalendar(exchange="NYSE")

        # Convert string dates to datetime objects
        start_date_dt = self.start_date
        end_date_dt = self.end_date

        # Get all trading days in the backtest period
        all_trading_days = market_calendar.get_trading_days(
            start_date_dt, end_date_dt
        )

        if len(all_trading_days) < training_window + testing_window:
            self.logger.error(
                f"Insufficient data for walk-forward testing. Need at least {training_window + testing_window} trading days."
            )
            raise ValueError("Insufficient data for walk-forward testing")

        walk_forward_results = []

        # Walk forward through the data
        for i in range(
            0, len(all_trading_days) - (training_window + testing_window) + 1, step_size
        ):
            # Define training and testing periods
            train_start_idx = i
            train_end_idx = i + training_window - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(
                test_start_idx + testing_window - 1, len(all_trading_days) - 1
            )

            train_start = all_trading_days[train_start_idx]
            train_end = all_trading_days[train_end_idx]
            test_start = all_trading_days[test_start_idx]
            test_end = all_trading_days[test_end_idx]

            self.logger.info(
                f"Walk-forward iteration: Training {train_start} to {train_end}, Testing {test_start} to {test_end}"
            )

            # Train the strategy
            self.strategy.train(
                market_data=self.market_simulator.get_data_for_period(
                    train_start, train_end
                ),
                universe=self.universe,
            )

            # Run backtest on the test period
            test_engine = BacktestEngine(
                start_date=test_start,
                end_date=test_end,
                initial_capital=self.initial_capital,
                execution_model=self.execution_model,
                data_source=self.data_source,
                timeframes=self.timeframes,
                commission_model=self.commission_model,
                slippage_model=self.slippage_model,
                market_impact_model=self.market_impact_model,
            )
            test_engine.set_universe(self.universe)
            test_engine.strategy = self.strategy  # Use the trained strategy

            # Run the test period backtest
            test_results = test_engine.run()

            # Store the results for this walk-forward iteration
            walk_forward_results.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "results": test_results,
                }
            )

        # Compile overall walk-forward results
        combined_results = self._combine_walk_forward_results(walk_forward_results)

        return combined_results

    def run_monte_carlo(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations on the backtest results.

        Args:
            num_simulations: Number of Monte Carlo simulations to run

        Returns:
            Dictionary containing Monte Carlo simulation results
        """
        if not self.results:
            self.logger.error("No results to analyze. Run the backtest first.")
            raise ValueError("No results to analyze")

        # Get the trades from the backtest results
        trades = self.results["trades"]

        if not trades:
            self.logger.warning("No trades to analyze for Monte Carlo simulation")
            return {"error": "No trades to analyze"}

        # Run Monte Carlo simulation
        mc_results = self.analyzer.run_monte_carlo_simulation(
            trades=trades,
            initial_capital=self.initial_capital,
            num_simulations=num_simulations,
        )

        return mc_results

    def _update_portfolio(
        self, trades: List[Dict], market_data: Dict, date: str
    ) -> bool:
        """
        Update the portfolio based on executed trades and current market data.

        Args:
            trades: List of executed trades
            market_data: Current market data
            date: Current date
            
        Returns:
            bool: True if portfolio was updated successfully, False otherwise
        """
        if not trades:
            return True
            
        # Process trades with validation
        for trade in trades:
            symbol = trade["symbol"]
            quantity = trade["quantity"]
            price = trade["price"]
            commission = trade["commission"]

            # Update cash
            try:
                trade_value = quantity * price
            except TypeError as e:
                self.logger.error(f"Error calculating trade value: {str(e)}")
                self.logger.error(f"Trade details: symbol={symbol}, quantity={quantity}, price={price}")
                return False
                
            self.portfolio["cash"] -= trade_value + commission

            # Update positions
            if symbol not in self.portfolio["positions"]:
                self.portfolio["positions"][symbol] = {
                    "quantity": 0,
                    "cost_basis": 0,
                    "market_value": 0,
                }

            current_position = self.portfolio["positions"][symbol]

            if quantity > 0:  # Buy
                # Update cost basis
                try:
                    total_cost = (
                        current_position["quantity"] * current_position["cost_basis"]
                    )
                    new_total_cost = total_cost + (quantity * price)
                    new_total_quantity = current_position["quantity"] + quantity

                    if new_total_quantity > 0:
                        current_position["cost_basis"] = new_total_cost / new_total_quantity
                except (TypeError, ZeroDivisionError) as e:
                    self.logger.error(f"Error updating cost basis: {str(e)}")
                    self.logger.error(
                        f"Position details: quantity={current_position['quantity']}, "
                        f"cost_basis={current_position['cost_basis']}, new_quantity={quantity}, price={price}"
                    )

                current_position["quantity"] += quantity

            else:  # Sell
                current_position[
                    "quantity"
                ] += quantity  # quantity is negative for sells

                # If position is closed, reset cost basis
                if current_position["quantity"] == 0:
                    current_position["cost_basis"] = 0

        # Update market values
        portfolio_value = self.portfolio["cash"]

        for symbol, position in self.portfolio["positions"].items():
            if position["quantity"] != 0:
                # Get latest price
                try:
                    if symbol in market_data:
                        # Find the first timeframe with data
                        for timeframe in self.timeframes:
                            if timeframe in market_data[symbol] and isinstance(market_data[symbol][timeframe], pd.DataFrame):
                                if not market_data[symbol][timeframe].empty and "close" in market_data[symbol][timeframe].columns:
                                    latest_price = market_data[symbol][timeframe]["close"].iloc[-1]
                                    position["market_value"] = position["quantity"] * latest_price
                                    portfolio_value += position["market_value"]
                                    break
                except Exception as e:
                    self.logger.error(f"Error updating market value for {symbol}: {str(e)}")
                    self.logger.error(f"Position: {position}, Market data keys: {list(market_data.keys() if market_data else [])}")

        # Update portfolio equity
        self.portfolio["equity"] = portfolio_value

        # Record portfolio state
        self.portfolio["history"].append(
            {"date": date, "cash": self.portfolio["cash"], "equity": portfolio_value}
        )
        
        return True

    def _get_portfolio_snapshot(self, date: str) -> Dict[str, Any]:
        """
        Get a snapshot of the portfolio state for a given date.

        Args:
            date: Date for the snapshot

        Returns:
            Dictionary containing portfolio state
        """
        positions = []

        for symbol, position in self.portfolio["positions"].items():
            if position["quantity"] != 0:
                positions.append(
                    {
                        "symbol": symbol,
                        "quantity": position["quantity"],
                        "cost_basis": position["cost_basis"],
                        "market_value": position["market_value"],
                    }
                )

        return {
            "date": date,
            "cash": self.portfolio["cash"],
            "equity": self.portfolio["equity"],
            "positions": positions,
        }

    def _prepare_results_for_serialization(self, results: Dict) -> Dict:
        """
        Prepare results dictionary for JSON serialization.

        Args:
            results: Results dictionary

        Returns:
            Serializable results dictionary
        """
        serializable = {}

        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable[key] = value
            elif isinstance(value, (datetime.datetime, datetime.date)):
                serializable[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                serializable[key] = [
                    self._prepare_results_for_serialization(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            elif isinstance(value, dict):
                serializable[key] = self._prepare_results_for_serialization(value)
            elif hasattr(value, "__dict__"):
                serializable[key] = str(value)
            else:
                serializable[key] = str(value)

        return serializable

    def _combine_walk_forward_results(self, walk_forward_results: List[Dict]) -> Dict:
        """
        Combine results from multiple walk-forward iterations.

        Args:
            walk_forward_results: List of walk-forward iteration results

        Returns:
            Combined walk-forward results
        """
        if not walk_forward_results:
            return {}

        # Extract test period results
        test_results = [wf["results"] for wf in walk_forward_results]

        # Combine portfolio histories
        combined_portfolio_history = []
        for result in test_results:
            combined_portfolio_history.extend(result["portfolio_history"])

        # Sort by date
        combined_portfolio_history.sort(key=lambda x: x["date"])

        # Combine trades
        combined_trades = []
        for result in test_results:
            combined_trades.extend(result["trades"])

        # Sort by date
        combined_trades.sort(key=lambda x: x["timestamp"])

        # Calculate overall metrics
        overall_metrics = self.analyzer.calculate_metrics(
            {
                "initial_capital": self.initial_capital,
                "portfolio_history": combined_portfolio_history,
                "trades": combined_trades,
            }
        )

        # Create combined results
        combined_results = {
            "backtest_id": f"wf_{self.backtest_id}",
            "start_date": walk_forward_results[0]["train_start"],
            "end_date": walk_forward_results[-1]["test_end"],
            "initial_capital": self.initial_capital,
            "final_equity": combined_portfolio_history[-1]["equity"]
            if combined_portfolio_history
            else self.initial_capital,
            "trades": combined_trades,
            "portfolio_history": combined_portfolio_history,
            "metrics": overall_metrics,
            "walk_forward_iterations": walk_forward_results,
            "universe": self.universe,
            "strategy": self.strategy.__class__.__name__,
            "strategy_config": self.strategy_config,
            "execution_model": self.execution_model,
            "commission_model": self.commission_model,
            "slippage_model": self.slippage_model,
            "market_impact_model": self.market_impact_model,
        }

        return combined_results
