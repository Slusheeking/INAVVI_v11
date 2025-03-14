#!/usr/bin/env python3
"""
Performance Analyzer

This module provides functionality for analyzing the performance of trading strategies
and models to identify areas for improvement and adaptation.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

# Fix for matplotlib import issue
try:
    # First, try to import matplotlib._docstring as docstring to fix the missing module
    import sys
    import matplotlib
    if not hasattr(matplotlib, 'docstring') and hasattr(matplotlib, '_docstring'):
        sys.modules['matplotlib.docstring'] = matplotlib._docstring
        matplotlib.docstring = matplotlib._docstring
except Exception:
    pass  # Silently handle any issues with the matplotlib fix
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("performance_analyzer")


class PerformanceAnalyzer:
    """
    Analyzes the performance of trading strategies and models to identify areas for improvement.

    This class provides methods for:
    - Calculating performance metrics
    - Analyzing drawdowns
    - Identifying regime-specific performance
    - Analyzing trade attribution
    - Detecting performance drift
    - Generating performance reports
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the performance analyzer.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        # Default configuration
        self.lookback_window = self.config.get("lookback_window", 60)  # 60 days
        self.min_trades_for_analysis = self.config.get("min_trades_for_analysis", 30)
        self.performance_metrics = self.config.get(
            "performance_metrics",
            [
                "total_return",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "avg_profit_per_trade",
                "avg_holding_time",
            ],
        )
        self.benchmark_ticker = self.config.get("benchmark_ticker", "SPY")
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2% annual
        self.drawdown_threshold = self.config.get("drawdown_threshold", 0.05)  # 5%
        self.drift_detection_window = self.config.get(
            "drift_detection_window", 20
        )  # 20 days
        self.drift_significance_level = self.config.get(
            "drift_significance_level", 0.05
        )  # 5% significance

        logger.info(
            f"Initialized PerformanceAnalyzer with lookback_window={self.lookback_window}"
        )

    def analyze_performance(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
        model_predictions: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Analyze trading performance.

        Args:
            trades: DataFrame with trade information
            equity_curve: DataFrame with equity curve
            market_data: DataFrame with market data (optional)
            model_predictions: DataFrame with model predictions (optional)

        Returns:
            Dictionary with performance analysis
        """
        logger.info(f"Analyzing performance for {len(trades)} trades")

        # Check if we have enough trades
        if len(trades) < self.min_trades_for_analysis:
            logger.warning(
                f"Not enough trades for analysis: {len(trades)} < {self.min_trades_for_analysis}"
            )
            return {
                "status": "insufficient_data",
                "message": f"Not enough trades for analysis: {len(trades)} < {self.min_trades_for_analysis}",
                "timestamp": datetime.now().isoformat(),
            }

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            trades, equity_curve, market_data
        )

        # Analyze drawdowns
        drawdown_analysis = self._analyze_drawdowns(equity_curve)

        # Analyze trade attribution
        trade_attribution = self._analyze_trade_attribution(trades, model_predictions)

        # Detect performance drift
        drift_analysis = self._detect_performance_drift(trades, equity_curve)

        # Identify regime-specific performance
        regime_performance = (
            self._analyze_regime_performance(trades, market_data)
            if market_data is not None
            else None
        )

        # Analyze model performance if predictions are available
        model_performance = (
            self._analyze_model_performance(trades, model_predictions)
            if model_predictions is not None
            else None
        )

        # Compile analysis results
        analysis_results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "drawdown_analysis": drawdown_analysis,
            "trade_attribution": trade_attribution,
            "drift_analysis": drift_analysis,
            "regime_performance": regime_performance,
            "model_performance": model_performance,
            "improvement_recommendations": self._generate_recommendations(
                performance_metrics,
                drawdown_analysis,
                trade_attribution,
                drift_analysis,
                regime_performance,
                model_performance,
            ),
        }

        logger.info(
            f"Performance analysis completed: Sharpe ratio = {performance_metrics.get('sharpe_ratio', 'N/A')}"
        )

        return analysis_results

    def _calculate_performance_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            trades: DataFrame with trade information
            equity_curve: DataFrame with equity curve
            market_data: DataFrame with market data (optional)

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        # Ensure equity_curve has datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            if "timestamp" in equity_curve.columns:
                equity_curve = equity_curve.set_index("timestamp")
            elif "date" in equity_curve.columns:
                equity_curve = equity_curve.set_index("date")

        # Calculate returns
        if "returns" not in equity_curve.columns:
            equity_curve["returns"] = equity_curve["equity"].pct_change()

        # Calculate total return
        initial_equity = equity_curve["equity"].iloc[0]
        final_equity = equity_curve["equity"].iloc[-1]
        metrics["total_return"] = (final_equity - initial_equity) / initial_equity

        # Calculate annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            metrics["annualized_return"] = (1 + metrics["total_return"]) ** (
                365 / days
            ) - 1
        else:
            metrics["annualized_return"] = 0

        # Calculate Sharpe ratio
        daily_returns = equity_curve["returns"].dropna()
        if len(daily_returns) > 0:
            excess_returns = (
                daily_returns - self.risk_free_rate / 252
            )  # Daily risk-free rate
            metrics["sharpe_ratio"] = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0
                else 0
            )
        else:
            metrics["sharpe_ratio"] = 0

        # Calculate Sortino ratio
        if len(daily_returns) > 0:
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = (
                downside_returns.std() * np.sqrt(252)
                if len(downside_returns) > 0
                else 0
            )
            metrics["sortino_ratio"] = (
                daily_returns.mean() * 252 / downside_deviation
                if downside_deviation > 0
                else 0
            )
        else:
            metrics["sortino_ratio"] = 0

        # Calculate maximum drawdown
        if "drawdown" not in equity_curve.columns:
            equity_curve["drawdown"] = (
                1 - equity_curve["equity"] / equity_curve["equity"].cummax()
            )
        metrics["max_drawdown"] = equity_curve["drawdown"].max()

        # Calculate Calmar ratio
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / metrics["max_drawdown"]
            if metrics["max_drawdown"] > 0
            else 0
        )

        # Calculate trade-based metrics
        if len(trades) > 0:
            # Win rate
            metrics["win_rate"] = len(trades[trades["profit"] > 0]) / len(trades)

            # Profit factor
            gross_profit = trades[trades["profit"] > 0]["profit"].sum()
            gross_loss = abs(trades[trades["profit"] < 0]["profit"].sum())
            metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Average profit per trade
            metrics["avg_profit_per_trade"] = trades["profit"].mean()

            # Average holding time
            if "exit_time" in trades.columns and "entry_time" in trades.columns:
                holding_times = (
                    trades["exit_time"] - trades["entry_time"]
                ).dt.total_seconds() / 3600  # Hours
                metrics["avg_holding_time"] = holding_times.mean()

            # Average profit per hour
            if "avg_holding_time" in metrics and metrics["avg_holding_time"] > 0:
                metrics["avg_profit_per_hour"] = (
                    metrics["avg_profit_per_trade"] / metrics["avg_holding_time"]
                )

            # Maximum consecutive wins/losses
            trades["win"] = trades["profit"] > 0
            win_streak = (
                trades["win"]
                .groupby((trades["win"] != trades["win"].shift()).cumsum())
                .cumcount()
                + 1
            )
            metrics["max_consecutive_wins"] = (
                win_streak[trades["win"]].max()
                if len(win_streak[trades["win"]]) > 0
                else 0
            )
            metrics["max_consecutive_losses"] = (
                win_streak[~trades["win"]].max()
                if len(win_streak[~trades["win"]]) > 0
                else 0
            )

        # Calculate benchmark comparison if market data is available
        if market_data is not None and self.benchmark_ticker in market_data.columns:
            # Align dates
            benchmark_data = market_data[self.benchmark_ticker].reindex(
                equity_curve.index
            )
            benchmark_returns = benchmark_data.pct_change().dropna()

            if len(benchmark_returns) > 0:
                # Calculate beta
                if len(daily_returns) == len(benchmark_returns):
                    covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    metrics["beta"] = (
                        covariance / benchmark_variance if benchmark_variance > 0 else 0
                    )

                # Calculate alpha
                if "beta" in metrics:
                    metrics["alpha"] = (
                        metrics["annualized_return"] - self.risk_free_rate
                    ) - metrics["beta"] * (
                        benchmark_returns.mean() * 252 - self.risk_free_rate
                    )

                # Calculate information ratio
                tracking_error = (daily_returns - benchmark_returns).std() * np.sqrt(
                    252
                )
                metrics["information_ratio"] = (
                    (daily_returns.mean() - benchmark_returns.mean())
                    * 252
                    / tracking_error
                    if tracking_error > 0
                    else 0
                )

        return metrics

    def _analyze_drawdowns(self, equity_curve: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze drawdowns.

        Args:
            equity_curve: DataFrame with equity curve

        Returns:
            Dictionary with drawdown analysis
        """
        # Calculate drawdowns if not already calculated
        if "drawdown" not in equity_curve.columns:
            equity_curve["drawdown"] = (
                1 - equity_curve["equity"] / equity_curve["equity"].cummax()
            )

        # Find drawdown periods
        is_drawdown = equity_curve["drawdown"] >= self.drawdown_threshold
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)

        # Get drawdown start and end indices
        drawdown_start_indices = equity_curve.index[drawdown_starts]
        drawdown_end_indices = equity_curve.index[drawdown_ends]

        # If we're currently in a drawdown, add the last date as an end
        if len(drawdown_start_indices) > len(drawdown_end_indices):
            drawdown_end_indices = drawdown_end_indices.append(
                pd.Index([equity_curve.index[-1]])
            )

        # Create list of drawdown periods
        drawdown_periods = []

        for i in range(len(drawdown_start_indices)):
            if i < len(drawdown_end_indices):
                start_date = drawdown_start_indices[i]
                end_date = drawdown_end_indices[i]

                # Get drawdown data for this period
                period_data = equity_curve.loc[start_date:end_date]

                # Calculate drawdown metrics
                max_drawdown = period_data["drawdown"].max()
                duration = (end_date - start_date).days
                recovery = (
                    period_data["equity"].iloc[-1] / period_data["equity"].iloc[0] - 1
                )

                drawdown_periods.append(
                    {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "duration": duration,
                        "max_drawdown": float(max_drawdown),
                        "recovery": float(recovery),
                        "is_active": end_date == equity_curve.index[-1]
                        and period_data["drawdown"].iloc[-1] >= self.drawdown_threshold,
                    }
                )

        # Calculate drawdown statistics
        if drawdown_periods:
            avg_drawdown = np.mean(
                [period["max_drawdown"] for period in drawdown_periods]
            )
            avg_duration = np.mean([period["duration"] for period in drawdown_periods])
            max_duration = np.max([period["duration"] for period in drawdown_periods])
            drawdown_frequency = (
                len(drawdown_periods)
                / (equity_curve.index[-1] - equity_curve.index[0]).days
                * 365
            )  # Annualized
        else:
            avg_drawdown = 0
            avg_duration = 0
            max_duration = 0
            drawdown_frequency = 0

        # Compile drawdown analysis
        drawdown_analysis = {
            "drawdown_periods": drawdown_periods,
            "statistics": {
                "avg_drawdown": float(avg_drawdown),
                "avg_duration": float(avg_duration),
                "max_duration": float(max_duration),
                "drawdown_frequency": float(drawdown_frequency),
                "current_drawdown": float(equity_curve["drawdown"].iloc[-1]),
            },
        }

        return drawdown_analysis

    def _analyze_trade_attribution(
        self, trades: pd.DataFrame, model_predictions: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """
        Analyze trade attribution.

        Args:
            trades: DataFrame with trade information
            model_predictions: DataFrame with model predictions (optional)

        Returns:
            Dictionary with trade attribution analysis
        """
        attribution = {}

        # Analyze by ticker
        if "ticker" in trades.columns:
            ticker_performance = trades.groupby("ticker").agg(
                {
                    "profit": ["sum", "mean", "count"],
                    "entry_price": "mean",
                    "exit_price": "mean",
                }
            )

            # Calculate win rate by ticker
            ticker_win_rate = trades.groupby("ticker")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare ticker attribution
            ticker_attribution = []
            for ticker in ticker_performance.index:
                ticker_data = ticker_performance.loc[ticker]
                ticker_attribution.append(
                    {
                        "ticker": ticker,
                        "total_profit": float(ticker_data["profit"]["sum"]),
                        "avg_profit": float(ticker_data["profit"]["mean"]),
                        "trade_count": int(ticker_data["profit"]["count"]),
                        "win_rate": float(ticker_win_rate.get(ticker, 0)),
                        "avg_entry_price": float(ticker_data["entry_price"]["mean"]),
                        "avg_exit_price": float(ticker_data["exit_price"]["mean"]),
                    }
                )

            # Sort by total profit
            ticker_attribution.sort(key=lambda x: x["total_profit"], reverse=True)
            attribution["by_ticker"] = ticker_attribution

        # Analyze by strategy
        if "strategy" in trades.columns:
            strategy_performance = trades.groupby("strategy").agg(
                {"profit": ["sum", "mean", "count"]}
            )

            # Calculate win rate by strategy
            strategy_win_rate = trades.groupby("strategy")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare strategy attribution
            strategy_attribution = []
            for strategy in strategy_performance.index:
                strategy_data = strategy_performance.loc[strategy]
                strategy_attribution.append(
                    {
                        "strategy": strategy,
                        "total_profit": float(strategy_data["profit"]["sum"]),
                        "avg_profit": float(strategy_data["profit"]["mean"]),
                        "trade_count": int(strategy_data["profit"]["count"]),
                        "win_rate": float(strategy_win_rate.get(strategy, 0)),
                    }
                )

            # Sort by total profit
            strategy_attribution.sort(key=lambda x: x["total_profit"], reverse=True)
            attribution["by_strategy"] = strategy_attribution

        # Analyze by time of day
        if "entry_time" in trades.columns:
            # Extract hour of day
            trades["hour_of_day"] = trades["entry_time"].dt.hour

            # Group by hour of day
            hour_performance = trades.groupby("hour_of_day").agg(
                {"profit": ["sum", "mean", "count"]}
            )

            # Calculate win rate by hour
            hour_win_rate = trades.groupby("hour_of_day")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare hour attribution
            hour_attribution = []
            for hour in range(24):
                if hour in hour_performance.index:
                    hour_data = hour_performance.loc[hour]
                    hour_attribution.append(
                        {
                            "hour": hour,
                            "total_profit": float(hour_data["profit"]["sum"]),
                            "avg_profit": float(hour_data["profit"]["mean"]),
                            "trade_count": int(hour_data["profit"]["count"]),
                            "win_rate": float(hour_win_rate.get(hour, 0)),
                        }
                    )
                else:
                    hour_attribution.append(
                        {
                            "hour": hour,
                            "total_profit": 0.0,
                            "avg_profit": 0.0,
                            "trade_count": 0,
                            "win_rate": 0.0,
                        }
                    )

            attribution["by_hour_of_day"] = hour_attribution

        # Analyze by day of week
        if "entry_time" in trades.columns:
            # Extract day of week
            trades["day_of_week"] = trades["entry_time"].dt.dayofweek

            # Group by day of week
            day_performance = trades.groupby("day_of_week").agg(
                {"profit": ["sum", "mean", "count"]}
            )

            # Calculate win rate by day
            day_win_rate = trades.groupby("day_of_week")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare day attribution
            day_attribution = []
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            for day in range(7):
                if day in day_performance.index:
                    day_data = day_performance.loc[day]
                    day_attribution.append(
                        {
                            "day": day_names[day],
                            "total_profit": float(day_data["profit"]["sum"]),
                            "avg_profit": float(day_data["profit"]["mean"]),
                            "trade_count": int(day_data["profit"]["count"]),
                            "win_rate": float(day_win_rate.get(day, 0)),
                        }
                    )
                else:
                    day_attribution.append(
                        {
                            "day": day_names[day],
                            "total_profit": 0.0,
                            "avg_profit": 0.0,
                            "trade_count": 0,
                            "win_rate": 0.0,
                        }
                    )

            attribution["by_day_of_week"] = day_attribution

        # Analyze by holding time
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            # Calculate holding time in hours
            trades["holding_time"] = (
                trades["exit_time"] - trades["entry_time"]
            ).dt.total_seconds() / 3600

            # Create holding time bins
            bins = [0, 1, 4, 8, 24, 72, float("inf")]
            labels = ["<1h", "1-4h", "4-8h", "8-24h", "1-3d", ">3d"]
            trades["holding_time_bin"] = pd.cut(
                trades["holding_time"], bins=bins, labels=labels
            )

            # Group by holding time bin
            holding_performance = trades.groupby("holding_time_bin").agg(
                {"profit": ["sum", "mean", "count"]}
            )

            # Calculate win rate by holding time
            holding_win_rate = trades.groupby("holding_time_bin")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare holding time attribution
            holding_attribution = []
            for bin_label in labels:
                if bin_label in holding_performance.index:
                    bin_data = holding_performance.loc[bin_label]
                    holding_attribution.append(
                        {
                            "holding_time": bin_label,
                            "total_profit": float(bin_data["profit"]["sum"]),
                            "avg_profit": float(bin_data["profit"]["mean"]),
                            "trade_count": int(bin_data["profit"]["count"]),
                            "win_rate": float(holding_win_rate.get(bin_label, 0)),
                        }
                    )
                else:
                    holding_attribution.append(
                        {
                            "holding_time": bin_label,
                            "total_profit": 0.0,
                            "avg_profit": 0.0,
                            "trade_count": 0,
                            "win_rate": 0.0,
                        }
                    )

            attribution["by_holding_time"] = holding_attribution

        # Analyze by model confidence if predictions are available
        if model_predictions is not None and "confidence" in model_predictions.columns:
            # Merge trades with predictions
            if "prediction_id" in trades.columns:
                merged_data = trades.merge(
                    model_predictions,
                    left_on="prediction_id",
                    right_index=True,
                    how="left",
                )
            else:
                # Try to merge on timestamp
                merged_data = trades.merge(
                    model_predictions,
                    left_on="entry_time",
                    right_index=True,
                    how="left",
                )

            if "confidence" in merged_data.columns:
                # Create confidence bins
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
                merged_data["confidence_bin"] = pd.cut(
                    merged_data["confidence"], bins=bins, labels=labels
                )

                # Group by confidence bin
                confidence_performance = merged_data.groupby("confidence_bin").agg(
                    {"profit": ["sum", "mean", "count"]}
                )

                # Calculate win rate by confidence
                confidence_win_rate = merged_data.groupby("confidence_bin")[
                    "profit"
                ].apply(lambda x: (x > 0).mean())

                # Prepare confidence attribution
                confidence_attribution = []
                for bin_label in labels:
                    if bin_label in confidence_performance.index:
                        bin_data = confidence_performance.loc[bin_label]
                        confidence_attribution.append(
                            {
                                "confidence": bin_label,
                                "total_profit": float(bin_data["profit"]["sum"]),
                                "avg_profit": float(bin_data["profit"]["mean"]),
                                "trade_count": int(bin_data["profit"]["count"]),
                                "win_rate": float(
                                    confidence_win_rate.get(bin_label, 0)
                                ),
                            }
                        )
                    else:
                        confidence_attribution.append(
                            {
                                "confidence": bin_label,
                                "total_profit": 0.0,
                                "avg_profit": 0.0,
                                "trade_count": 0,
                                "win_rate": 0.0,
                            }
                        )

                attribution["by_model_confidence"] = confidence_attribution

        return attribution

    def _detect_performance_drift(
        self, trades: pd.DataFrame, equity_curve: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Detect performance drift.

        Args:
            trades: DataFrame with trade information
            equity_curve: DataFrame with equity curve

        Returns:
            Dictionary with drift analysis
        """
        drift_analysis = {"detected": False, "metrics": {}}

        # Check if we have enough data
        if len(trades) < 2 * self.drift_detection_window:
            return {
                "detected": False,
                "message": f"Not enough trades for drift detection: {len(trades)} < {2 * self.drift_detection_window}",
                "metrics": {},
            }

        # Sort trades by time
        if "entry_time" in trades.columns:
            trades = trades.sort_values("entry_time")

        # Split trades into recent and historical
        recent_trades = trades.iloc[-self.drift_detection_window :]
        historical_trades = trades.iloc[: -self.drift_detection_window]

        # Calculate metrics for both periods
        metrics_to_check = ["win_rate", "avg_profit_per_trade", "profit_factor"]

        for metric in metrics_to_check:
            if metric == "win_rate":
                recent_value = (recent_trades["profit"] > 0).mean()
                historical_value = (historical_trades["profit"] > 0).mean()
            elif metric == "avg_profit_per_trade":
                recent_value = recent_trades["profit"].mean()
                historical_value = historical_trades["profit"].mean()
            elif metric == "profit_factor":
                recent_profit = recent_trades[recent_trades["profit"] > 0][
                    "profit"
                ].sum()
                recent_loss = abs(
                    recent_trades[recent_trades["profit"] < 0]["profit"].sum()
                )
                historical_profit = historical_trades[historical_trades["profit"] > 0][
                    "profit"
                ].sum()
                historical_loss = abs(
                    historical_trades[historical_trades["profit"] < 0]["profit"].sum()
                )

                recent_value = (
                    recent_profit / recent_loss if recent_loss > 0 else float("inf")
                )
                historical_value = (
                    historical_profit / historical_loss
                    if historical_loss > 0
                    else float("inf")
                )

            # Calculate percent change
            if historical_value != 0:
                percent_change = (
                    (recent_value - historical_value) / abs(historical_value) * 100
                )
            else:
                percent_change = 0 if recent_value == 0 else float("inf")

            # Perform statistical test
            if metric == "win_rate":
                # Use proportion z-test
                p_value = self._proportion_z_test(
                    (historical_trades["profit"] > 0).sum(),
                    len(historical_trades),
                    (recent_trades["profit"] > 0).sum(),
                    len(recent_trades),
                )
            else:
                # Use t-test
                t_stat, p_value = stats.ttest_ind(
                    historical_trades["profit"],
                    recent_trades["profit"],
                    equal_var=False,
                )

            # Check if drift is significant
            is_significant = p_value < self.drift_significance_level

            # Store results
            drift_analysis["metrics"][metric] = {
                "historical_value": float(historical_value),
                "recent_value": float(recent_value),
                "percent_change": float(percent_change),
                "p_value": float(p_value),
                "is_significant": bool(is_significant),
            }

            # Update overall drift detection
            if is_significant:
                drift_analysis["detected"] = True

        # Check for equity curve drift
        if len(equity_curve) > 2 * self.drift_detection_window:
            # Calculate daily returns
            if "returns" not in equity_curve.columns:
                equity_curve["returns"] = equity_curve["equity"].pct_change()

            # Split into recent and historical
            recent_returns = (
                equity_curve["returns"].iloc[-self.drift_detection_window :].dropna()
            )
            historical_returns = (
                equity_curve["returns"].iloc[: -self.drift_detection_window].dropna()
            )

            # Calculate Sharpe ratio for both periods
            recent_sharpe = (
                recent_returns.mean() / recent_returns.std() * np.sqrt(252)
                if recent_returns.std() > 0
                else 0
            )
            historical_sharpe = (
                historical_returns.mean() / historical_returns.std() * np.sqrt(252)
                if historical_returns.std() > 0
                else 0
            )

            # Calculate percent change
            if historical_sharpe != 0:
                percent_change = (
                    (recent_sharpe - historical_sharpe) / abs(historical_sharpe) * 100
                )
            else:
                percent_change = 0 if recent_sharpe == 0 else float("inf")

            # Perform bootstrap test for Sharpe ratio
            p_value = self._bootstrap_sharpe_test(
                historical_returns.values, recent_returns.values
            )

            # Check if drift is significant
            is_significant = p_value < self.drift_significance_level

            # Store results
            drift_analysis["metrics"]["sharpe_ratio"] = {
                "historical_value": float(historical_sharpe),
                "recent_value": float(recent_sharpe),
                "percent_change": float(percent_change),
                "p_value": float(p_value),
                "is_significant": bool(is_significant),
            }

            # Update overall drift detection
            if is_significant:
                drift_analysis["detected"] = True

        return drift_analysis

    def _proportion_z_test(self, count1: int, n1: int, count2: int, n2: int) -> float:
        """
        Perform a two-proportion z-test.

        Args:
            count1: Number of successes in first sample
            n1: Size of first sample
            count2: Number of successes in second sample
            n2: Size of second sample

        Returns:
            p-value
        """
        p1 = count1 / n1
        p2 = count2 / n2
        p_pooled = (count1 + count2) / (n1 + n2)

        # Calculate z-statistic
        z = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return p_value

    def _bootstrap_sharpe_test(
        self, returns1: np.ndarray, returns2: np.ndarray, n_bootstraps: int = 1000
    ) -> float:
        """
        Perform a bootstrap test for the difference in Sharpe ratios.

        Args:
            returns1: Array of returns for first period
            returns2: Array of returns for second period
            n_bootstraps: Number of bootstrap samples

        Returns:
            p-value
        """
        # Calculate observed Sharpe ratios
        sharpe1 = returns1.mean() / returns1.std() if returns1.std() > 0 else 0
        sharpe2 = returns2.mean() / returns2.std() if returns2.std() > 0 else 0
        observed_diff = abs(sharpe1 - sharpe2)

        # Combine returns
        all_returns = np.concatenate([returns1, returns2])
        n1 = len(returns1)
        n2 = len(returns2)

        # Bootstrap
        count_greater = 0
        for _ in range(n_bootstraps):
            # Shuffle returns
            np.random.shuffle(all_returns)

            # Split into two samples
            boot_returns1 = all_returns[:n1]
            boot_returns2 = all_returns[n1 : n1 + n2]

            # Calculate Sharpe ratios
            boot_sharpe1 = (
                boot_returns1.mean() / boot_returns1.std()
                if boot_returns1.std() > 0
                else 0
            )
            boot_sharpe2 = (
                boot_returns2.mean() / boot_returns2.std()
                if boot_returns2.std() > 0
                else 0
            )
            boot_diff = abs(boot_sharpe1 - boot_sharpe2)

            # Count number of bootstrap differences greater than observed
            if boot_diff >= observed_diff:
                count_greater += 1

        # Calculate p-value
        p_value = count_greater / n_bootstraps

        return p_value

    def _analyze_regime_performance(
        self, trades: pd.DataFrame, market_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze performance across different market regimes.

        Args:
            trades: DataFrame with trade information
            market_data: DataFrame with market data

        Returns:
            Dictionary with regime performance analysis
        """
        # Check if regime information is available
        if "regime" not in market_data.columns:
            return None

        # Ensure trades has entry_time column
        if "entry_time" not in trades.columns:
            return None

        # Map trades to regimes
        trades_with_regime = trades.copy()
        trades_with_regime["regime"] = trades_with_regime["entry_time"].apply(
            lambda x: market_data.loc[market_data.index <= x, "regime"].iloc[-1]
            if any(market_data.index <= x)
            else None
        )

        # Filter out trades without regime
        trades_with_regime = trades_with_regime.dropna(subset=["regime"])

        # Group by regime
        regime_performance = trades_with_regime.groupby("regime").agg(
            {"profit": ["sum", "mean", "count"]}
        )

        # Calculate win rate by regime
        regime_win_rate = trades_with_regime.groupby("regime")["profit"].apply(
            lambda x: (x > 0).mean()
        )

        # Prepare regime performance
        regime_analysis = []
        for regime in regime_performance.index:
            regime_data = regime_performance.loc[regime]
            regime_analysis.append(
                {
                    "regime": regime,
                    "total_profit": float(regime_data["profit"]["sum"]),
                    "avg_profit": float(regime_data["profit"]["mean"]),
                    "trade_count": int(regime_data["profit"]["count"]),
                    "win_rate": float(regime_win_rate.get(regime, 0)),
                }
            )

        # Sort by total profit
        regime_analysis.sort(key=lambda x: x["total_profit"], reverse=True)

        return {"regime_performance": regime_analysis}

    def _analyze_model_performance(
        self, trades: pd.DataFrame, model_predictions: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze model prediction performance.

        Args:
            trades: DataFrame with trade information
            model_predictions: DataFrame with model predictions

        Returns:
            Dictionary with model performance analysis
        """
        # Check if we have prediction data
        if model_predictions is None or len(model_predictions) == 0:
            return None

        # Merge trades with predictions
        if "prediction_id" in trades.columns:
            merged_data = trades.merge(
                model_predictions, left_on="prediction_id", right_index=True, how="left"
            )
        else:
            # Try to merge on timestamp
            merged_data = trades.merge(
                model_predictions, left_on="entry_time", right_index=True, how="left"
            )

        # Check if merge was successful
        if len(merged_data) == 0:
            return None

        # Calculate prediction accuracy
        if (
            "predicted_direction" in merged_data.columns
            and "actual_direction" in merged_data.columns
        ):
            accuracy = (
                merged_data["predicted_direction"] == merged_data["actual_direction"]
            ).mean()
        elif "prediction" in merged_data.columns and "profit" in merged_data.columns:
            # Assume prediction > 0 means up, < 0 means down
            accuracy = (
                (merged_data["prediction"] > 0) == (merged_data["profit"] > 0)
            ).mean()
        else:
            accuracy = None

        # Calculate correlation between prediction and actual profit
        if "prediction" in merged_data.columns and "profit" in merged_data.columns:
            correlation = merged_data[["prediction", "profit"]].corr().iloc[0, 1]
        else:
            correlation = None

        # Calculate profit by prediction strength
        if "prediction" in merged_data.columns:
            # Create prediction strength bins
            merged_data["abs_prediction"] = merged_data["prediction"].abs()
            bins = [0, 0.001, 0.002, 0.005, 0.01, float("inf")]
            labels = ["0-0.1%", "0.1-0.2%", "0.2-0.5%", "0.5-1%", ">1%"]
            merged_data["prediction_bin"] = pd.cut(
                merged_data["abs_prediction"], bins=bins, labels=labels
            )

            # Group by prediction bin
            prediction_performance = merged_data.groupby("prediction_bin").agg(
                {"profit": ["sum", "mean", "count"]}
            )

            # Calculate win rate by prediction bin
            prediction_win_rate = merged_data.groupby("prediction_bin")["profit"].apply(
                lambda x: (x > 0).mean()
            )

            # Prepare prediction performance
            prediction_analysis = []
            for bin_label in labels:
                if bin_label in prediction_performance.index:
                    bin_data = prediction_performance.loc[bin_label]
                    prediction_analysis.append(
                        {
                            "prediction_strength": bin_label,
                            "total_profit": float(bin_data["profit"]["sum"]),
                            "avg_profit": float(bin_data["profit"]["mean"]),
                            "trade_count": int(bin_data["profit"]["count"]),
                            "win_rate": float(prediction_win_rate.get(bin_label, 0)),
                        }
                    )
                else:
                    prediction_analysis.append(
                        {
                            "prediction_strength": bin_label,
                            "total_profit": 0.0,
                            "avg_profit": 0.0,
                            "trade_count": 0,
                            "win_rate": 0.0,
                        }
                    )
        else:
            prediction_analysis = None

        # Compile model performance analysis
        model_analysis = {
            "accuracy": float(accuracy) if accuracy is not None else None,
            "correlation": float(correlation) if correlation is not None else None,
            "prediction_performance": prediction_analysis,
        }

        return model_analysis

    def _generate_recommendations(
        self,
        performance_metrics: dict[str, float],
        drawdown_analysis: dict[str, Any],
        trade_attribution: dict[str, Any],
        drift_analysis: dict[str, Any],
        regime_performance: dict[str, Any] | None = None,
        model_performance: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """
        Generate improvement recommendations based on performance analysis.

        Args:
            performance_metrics: Dictionary with performance metrics
            drawdown_analysis: Dictionary with drawdown analysis
            trade_attribution: Dictionary with trade attribution analysis
            drift_analysis: Dictionary with drift analysis
            regime_performance: Dictionary with regime performance analysis (optional)
            model_performance: Dictionary with model performance analysis (optional)

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check Sharpe ratio
        if performance_metrics.get("sharpe_ratio", 0) < 1.0:
            recommendations.append(
                {
                    "category": "risk_adjusted_return",
                    "issue": "Low Sharpe ratio",
                    "recommendation": "Consider adjusting position sizing or implementing tighter stop losses to improve risk-adjusted returns.",
                }
            )

        # Check maximum drawdown
        if performance_metrics.get("max_drawdown", 0) > 0.2:
            recommendations.append(
                {
                    "category": "risk_management",
                    "issue": "High maximum drawdown",
                    "recommendation": "Implement more aggressive drawdown controls or consider using a dynamic position sizing approach that reduces exposure during drawdowns.",
                }
            )

        # Check win rate
        if performance_metrics.get("win_rate", 0) < 0.4:
            recommendations.append(
                {
                    "category": "strategy",
                    "issue": "Low win rate",
                    "recommendation": "Review entry criteria to improve signal quality or consider implementing a higher profit target to loss ratio to compensate for the low win rate.",
                }
            )

        # Check profit factor
        if performance_metrics.get("profit_factor", 0) < 1.2:
            recommendations.append(
                {
                    "category": "strategy",
                    "issue": "Low profit factor",
                    "recommendation": "Focus on reducing the size of losing trades or increasing the size of winning trades to improve the profit factor.",
                }
            )

        # Check for performance drift
        if drift_analysis.get("detected", False):
            drift_metrics = drift_analysis.get("metrics", {})

            for metric, data in drift_metrics.items():
                if (
                    data.get("is_significant", False)
                    and data.get("percent_change", 0) < 0
                ):
                    recommendations.append(
                        {
                            "category": "performance_drift",
                            "issue": f"Significant decline in {metric}",
                            "recommendation": "Review recent market conditions and consider adjusting strategy parameters or temporarily reducing position sizes until performance stabilizes.",
                        }
                    )

        # Check trade attribution by ticker
        if "by_ticker" in trade_attribution:
            # Find tickers with negative performance
            negative_tickers = [
                t
                for t in trade_attribution["by_ticker"]
                if t["total_profit"] < 0 and t["trade_count"] >= 5
            ]

            if negative_tickers:
                ticker_list = ", ".join([t["ticker"] for t in negative_tickers[:3]])
                recommendations.append(
                    {
                        "category": "ticker_selection",
                        "issue": f'Poor performance on specific tickers ({ticker_list}{"..." if len(negative_tickers) > 3 else ""})',
                        "recommendation": "Consider excluding these tickers from your trading universe or developing ticker-specific parameters.",
                    }
                )

        # Check trade attribution by time of day
        if "by_hour_of_day" in trade_attribution:
            # Find hours with negative performance
            negative_hours = [
                h
                for h in trade_attribution["by_hour_of_day"]
                if h["total_profit"] < 0 and h["trade_count"] >= 5
            ]

            if negative_hours:
                hour_list = ", ".join([f"{h['hour']}:00" for h in negative_hours[:3]])
                recommendations.append(
                    {
                        "category": "time_of_day",
                        "issue": f'Poor performance during specific hours ({hour_list}{"..." if len(negative_hours) > 3 else ""})',
                        "recommendation": "Consider restricting trading during these hours or developing time-specific parameters.",
                    }
                )

        # Check trade attribution by holding time
        if "by_holding_time" in trade_attribution:
            # Find optimal holding time
            holding_times = trade_attribution["by_holding_time"]
            optimal_holding = max(
                holding_times,
                key=lambda x: x["avg_profit"]
                if x["trade_count"] >= 5
                else -float("inf"),
            )

            # Find suboptimal holding times with significant trade count
            suboptimal_holdings = [
                h
                for h in holding_times
                if h["avg_profit"] < optimal_holding["avg_profit"] * 0.5
                and h["trade_count"] >= 5
            ]

            if suboptimal_holdings:
                recommendations.append(
                    {
                        "category": "holding_time",
                        "issue": "Suboptimal holding times for some trades",
                        "recommendation": f"Consider optimizing exit timing. The most profitable holding time is {optimal_holding['holding_time']}.",
                    }
                )

        # Check regime performance
        if (
            regime_performance is not None
            and "regime_performance" in regime_performance
        ):
            regimes = regime_performance["regime_performance"]

            # Find regimes with negative performance
            negative_regimes = [
                r for r in regimes if r["total_profit"] < 0 and r["trade_count"] >= 5
            ]

            if negative_regimes:
                regime_list = ", ".join([r["regime"] for r in negative_regimes[:3]])
                recommendations.append(
                    {
                        "category": "market_regime",
                        "issue": f'Poor performance during specific market regimes ({regime_list}{"..." if len(negative_regimes) > 3 else ""})',
                        "recommendation": "Consider developing regime-specific strategies or reducing exposure during these market conditions.",
                    }
                )

        # Check model performance
        if model_performance is not None:
            # Check prediction accuracy
            if (
                model_performance.get("accuracy", 0) is not None
                and model_performance.get("accuracy", 0) < 0.55
            ):
                recommendations.append(
                    {
                        "category": "model",
                        "issue": "Low prediction accuracy",
                        "recommendation": "Review model features and consider retraining with more recent data or exploring alternative model architectures.",
                    }
                )

            # Check correlation between prediction and profit
            if (
                model_performance.get("correlation", 0) is not None
                and model_performance.get("correlation", 0) < 0.2
            ):
                recommendations.append(
                    {
                        "category": "model",
                        "issue": "Low correlation between predictions and actual profits",
                        "recommendation": "Consider optimizing the model for dollar profit rather than directional accuracy, or review the position sizing approach.",
                    }
                )

        return recommendations

    def generate_performance_report(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
        model_predictions: pd.DataFrame | None = None,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Args:
            trades: DataFrame with trade information
            equity_curve: DataFrame with equity curve
            market_data: DataFrame with market data (optional)
            model_predictions: DataFrame with model predictions (optional)
            save_path: Path to save the report (optional)

        Returns:
            Dictionary with performance report
        """
        # Analyze performance
        analysis_results = self.analyze_performance(
            trades, equity_curve, market_data, model_predictions
        )

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_results": analysis_results,
            "summary": self._generate_report_summary(analysis_results),
        }

        # Save report if path is provided
        if save_path:
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {save_path}")
            
            # Also save plot data
            plot_path = save_path.rsplit('.', 1)[0] + '_plot' if '.' in save_path else save_path + '_plot'
            self.plot_performance(equity_curve, trades, market_data, save_path=plot_path)
            logger.info(f"Plot data saved alongside performance report")

        return report

    def _generate_report_summary(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a summary of the performance report.

        Args:
            analysis_results: Dictionary with performance analysis results

        Returns:
            Dictionary with report summary
        """
        summary = {}

        # Check if analysis was successful
        if analysis_results.get("status") != "success":
            return {
                "status": analysis_results.get("status"),
                "message": analysis_results.get("message", "Analysis failed"),
            }

        # Extract key metrics
        metrics = analysis_results.get("performance_metrics", {})
        summary["key_metrics"] = {
            "total_return": metrics.get("total_return", 0),
            "annualized_return": metrics.get("annualized_return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
        }

        # Extract drawdown information
        drawdown = analysis_results.get("drawdown_analysis", {}).get("statistics", {})
        summary["drawdown"] = {
            "current_drawdown": drawdown.get("current_drawdown", 0),
            "avg_drawdown": drawdown.get("avg_drawdown", 0),
            "avg_duration": drawdown.get("avg_duration", 0),
        }

        # Extract drift information
        drift = analysis_results.get("drift_analysis", {})
        summary["drift"] = {
            "detected": drift.get("detected", False),
            "metrics": {
                k: v.get("percent_change", 0)
                for k, v in drift.get("metrics", {}).items()
            },
        }

        # Extract top recommendations
        recommendations = analysis_results.get("improvement_recommendations", [])
        summary["top_recommendations"] = recommendations[:3] if recommendations else []

        # Extract best and worst performers
        attribution = analysis_results.get("trade_attribution", {})

        if "by_ticker" in attribution and attribution["by_ticker"]:
            tickers = attribution["by_ticker"]
            summary["best_tickers"] = tickers[:3] if tickers else []
            summary["worst_tickers"] = (
                sorted(tickers, key=lambda x: x["total_profit"])[:3] if tickers else []
            )

        if "by_hour_of_day" in attribution and attribution["by_hour_of_day"]:
            hours = attribution["by_hour_of_day"]
            summary["best_hours"] = (
                sorted(hours, key=lambda x: x["avg_profit"], reverse=True)[:3]
                if hours
                else []
            )
            summary["worst_hours"] = (
                sorted(hours, key=lambda x: x["avg_profit"])[:3] if hours else []
            )

        return summary

    def plot_performance(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame | None = None, 
        market_data: pd.DataFrame | None = None, 
        save_path: str | None = None,
    ) -> None:
        """
        Plot performance metrics or save data for plotting.
        
        This version doesn't directly use matplotlib to avoid dependency issues.
        Instead, it saves the plot data to a JSON file that can be visualized 
        by an external plotting tool or script.

        Args:
            equity_curve: DataFrame with equity curve
            trades: DataFrame with trade information (optional)
            market_data: DataFrame with market data (optional)
            save_path: Path to save the plot data (optional)
        """
        try:
            # Create a dictionary to store plot data
            plot_data = {"timestamp": datetime.now().isoformat()}
            
            # Prepare equity curve data
            plot_data["equity_curve"] = {
                "dates": [d.isoformat() for d in equity_curve.index],
                "equity": equity_curve["equity"].tolist()
            }
            
            # Prepare drawdown data
            if "drawdown" not in equity_curve.columns:
                equity_curve["drawdown"] = (
                    1 - equity_curve["equity"] / equity_curve["equity"].cummax()
                )
            
            plot_data["drawdown"] = {
                "dates": [d.isoformat() for d in equity_curve.index],
                "values": (equity_curve["drawdown"] * 100).tolist()
            }
            
            # Prepare returns data
            if "returns" not in equity_curve.columns:
                equity_curve["returns"] = equity_curve["equity"].pct_change()
                
            plot_data["returns"] = {
                "dates": [d.isoformat() for d in equity_curve.index],
                "values": (equity_curve["returns"] * 100).tolist()
            }
            
            # Save plot data
            if save_path:
                import json
                json_path = save_path.rsplit('.', 1)[0] + '.json' if '.' in save_path else save_path + '.json'
                with open(json_path, 'w') as f:
                    json.dump(plot_data, f, indent=2)
                logger.info(f"Performance plot data saved to {json_path}")
                logger.info("Note: Use an external plotting tool to visualize this data, or reinstall compatible matplotlib.")

        except Exception as e:
            logger.error(f"Error preparing performance plot data: {e}")


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd

    # Generate sample data
    np.random.seed(42)

    # Generate equity curve
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    equity = [100000]
    for i in range(1, 100):
        daily_return = np.random.normal(0.001, 0.01)
        equity.append(equity[-1] * (1 + daily_return))

    equity_curve = pd.DataFrame({"date": dates, "equity": equity}).set_index("date")

    # Generate trades
    trades = []
    for i in range(50):
        entry_time = dates[np.random.randint(0, 90)]
        exit_time = entry_time + timedelta(days=np.random.randint(1, 5))
        profit = np.random.normal(100, 500)
        trades.append(
            {
                "trade_id": i,
                "ticker": np.random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": np.random.uniform(100, 200),
                "exit_price": np.random.uniform(100, 200),
                "profit": profit,
                "strategy": np.random.choice(
                    ["momentum", "mean_reversion", "trend_following"]
                ),
            }
        )

    trades_df = pd.DataFrame(trades)

    # Create analyzer
    analyzer = PerformanceAnalyzer()

    # Analyze performance
    results = analyzer.analyze_performance(trades_df, equity_curve)

    # Print results
    print(json.dumps(results, indent=2))

    # Plot performance
    analyzer.plot_performance(equity_curve, trades_df, save_path="performance_plot")
