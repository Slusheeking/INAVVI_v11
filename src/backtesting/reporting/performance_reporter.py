"""
Performance report generation for backtesting results.

This module provides functionality to generate detailed performance reports
from backtesting results, including key metrics, visualizations, and analysis.
"""

import base64
import io
import json
import logging
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import visualization libraries, but make them optional
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Create placeholders for matplotlib and seaborn
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False
from scipy import stats


class PerformanceReport:
    """
    Generates detailed reports of backtest results.

    The PerformanceReport class generates comprehensive performance reports
    from backtest results, including visualizations and summary statistics.
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the PerformanceReport.

        Args:
            log_level: Logging level
        """
        # Set up logging
        try:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(getattr(logging, log_level))
        except AttributeError:
            # Fallback to INFO if invalid log level provided
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

        # Initialize state
        self.results = None
        self.metrics = None
        self.figures = {}

        # Set up plotting style
        if VISUALIZATION_AVAILABLE:
            plt.style.use("seaborn-v0_8-darkgrid")

        self.logger.info("PerformanceReport initialized")

    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        format: str = "html",
        include_trades: bool = True,
        include_positions: bool = True,
    ) -> str:
        """
        Generate a performance report from backtest results.
        
        Args:
            results: Backtest results dictionary
            output_dir: Directory to save the report
            format: Report format ('html', 'jupyter', 'slack')
            include_trades: Whether to include detailed trade information
            include_positions: Whether to include detailed position information

        Returns:
            Path to the generated report
        """
        # Validate inputs
        if not isinstance(results, dict):
            error_msg = f"Results must be a dictionary, got {type(results)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        if not isinstance(output_dir, str):
            error_msg = f"Output directory must be a string, got {type(output_dir)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
            
        if format not in ["html", "jupyter", "slack"]:
            error_msg = f"Unsupported report format: {format}. Must be one of: html, jupyter, slack"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.results = results
        self.metrics = results.get("metrics", {})

        # Create output directory if it doesn't exist
        output_path = self._create_output_directory(output_dir)

        # Generate all figures
        self._generate_all_figures()

        # Generate report based on format
        if format == "html":
            report_path = self._generate_html_report(
                output_path, include_trades, include_positions
            )
        elif format == "slack":
            report_path = self._send_to_slack(
                output_path, include_trades, include_positions
            )
        elif format == "jupyter":
            report_path = self._generate_jupyter_report(
                output_path, include_trades, include_positions
            )
        else:
            self.logger.error(f"Unsupported report format: {format}")
            raise ValueError(f"Unsupported report format: {format}")

        self.logger.info(f"Generated {format} report at {report_path}")

        return str(report_path)

    def _create_output_directory(self, output_dir: str) -> Path:
        """
        Create the output directory if it doesn't exist.
        
        Args:
            output_dir: Directory path as string
            
        Returns:
            Path object for the created directory
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_path}")
            return output_path
        except (PermissionError, OSError) as e:
            self.logger.error(f"Failed to create output directory {output_dir}: {str(e)}")
            raise

    def _generate_all_figures(self) -> None:
        """
        Generate all figures for the performance report.
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping figure generation.")
            return None
            
        # Extract portfolio history
        portfolio_history = self.results.get("portfolio_history", [])

        if not portfolio_history:
            self.logger.warning("No portfolio history available for visualization")
            return None

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df.set_index("date", inplace=True)
        portfolio_df.sort_index(inplace=True)

        # Generate equity curve
        self.logger.debug("Generating equity curve figure")
        self.figures["equity_curve"] = self._generate_equity_curve(portfolio_df)

        # Generate drawdown chart
        self.figures["drawdown_chart"] = self._generate_drawdown_chart(portfolio_df)

        # Generate returns distribution
        self.figures["returns_distribution"] = self._generate_returns_distribution(
            portfolio_df
        )

        # Generate monthly returns heatmap
        self.figures["monthly_returns"] = self._generate_monthly_returns_heatmap(
            portfolio_df
        )

        # Generate trade analysis
        trades = self.results.get("trades", [])
        if trades:
            self.logger.debug(f"Generating trade analysis figures for {len(trades)} trades")
            self.figures["trade_analysis"] = self._generate_trade_analysis(trades)
            self.figures["cumulative_trades"] = self._generate_cumulative_trades(trades)
        self.logger.info(f"Generated {len(self.figures)} figures for performance report")

    def _generate_equity_curve(self, portfolio_df: pd.DataFrame) -> Optional[Any]:
        """
        Generate equity curve figure.

        Args:
            portfolio_df: DataFrame of portfolio history

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping equity curve generation.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot equity curve
        ax.plot(
            portfolio_df.index,
            portfolio_df["equity"],
            label="Portfolio Equity",
            linewidth=2,
        )

        # Add initial capital line
        initial_capital = self.results.get(
            "initial_capital", portfolio_df["equity"].iloc[0]
        )
        ax.axhline(
            y=initial_capital, color="r", linestyle="--", label="Initial Capital"
        )

        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Portfolio Equity Curve")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig

    def _generate_drawdown_chart(self, portfolio_df: pd.DataFrame) -> Optional[Any]:
        """
        Generate drawdown chart figure.

        Args:
            portfolio_df: DataFrame of portfolio history

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping drawdown chart generation.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate drawdown
        equity = portfolio_df["equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color="r", alpha=0.3)
        ax.plot(drawdown.index, drawdown, color="r", linewidth=1)

        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Portfolio Drawdown")

        # Add max drawdown line
        max_drawdown = drawdown.min()
        ax.axhline(
            y=max_drawdown,
            color="black",
            linestyle="--",
            label=f"Max Drawdown: {max_drawdown:.2%}",
        )

        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig

    def _generate_returns_distribution(self, portfolio_df: pd.DataFrame) -> Optional[Any]:
        """
        Generate returns distribution figure.

        Args:
            portfolio_df: DataFrame of portfolio history

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping returns distribution generation.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate daily returns
        returns = portfolio_df["equity"].pct_change().dropna()

        # Plot histogram
        sns.histplot(returns, bins=50, kde=True, ax=ax)

        # Add normal distribution
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        ax.plot(x, y, "r--", linewidth=2, label="Normal Distribution")

        # Add mean and zero lines
        ax.axvline(
            x=returns.mean(),
            color="g",
            linestyle="-",
            label=f"Mean: {returns.mean():.4%}",
        )
        ax.axvline(x=0, color="black", linestyle="-", label="Zero")

        # Format x-axis
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.2%}"))

        # Add labels and title
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.set_title("Daily Returns Distribution")

        # Add statistics
        stats_text = (
            f"Mean: {returns.mean():.4%}\n"
            f"Std Dev: {returns.std():.4%}\n"
            f"Skewness: {returns.skew():.4f}\n"
            f"Kurtosis: {returns.kurtosis():.4f}\n"
            f"Positive Days: {(returns > 0).mean():.2%}\n"
            f"Negative Days: {(returns < 0).mean():.2%}"
        )

        # Add text box with statistics
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig

    def _generate_monthly_returns_heatmap(self, portfolio_df: pd.DataFrame) -> Optional[Any]:
        """
        Generate monthly returns heatmap figure.

        Args:
            portfolio_df: DataFrame of portfolio history

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping monthly returns heatmap generation.")
            return None
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate daily returns
        returns = portfolio_df["equity"].pct_change().dropna()

        # Create a DataFrame with year and month
        returns_df = pd.DataFrame(returns)
        returns_df.index = pd.MultiIndex.from_arrays(
            [returns_df.index.year, returns_df.index.month], names=["Year", "Month"]
        )

        # Calculate monthly returns
        monthly_returns = returns_df.groupby(level=[0, 1]).apply(
            lambda x: (1 + x).prod() - 1
        )

        # Pivot the data for the heatmap
        monthly_returns_pivot = monthly_returns.unstack(level=0)

        # Create the heatmap
        sns.heatmap(
            monthly_returns_pivot,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            center=0,
            linewidths=1,
            ax=ax,
        )

        # Add labels and title
        ax.set_title("Monthly Returns Heatmap")
        ax.set_xlabel("Year")
        ax.set_ylabel("Month")

        # Replace month numbers with names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax.set_yticklabels(month_names)

        fig.tight_layout()
        return fig

    def _generate_trade_analysis(self, trades: List[Dict[str, Any]]) -> Optional[Any]:
        """
        Generate trade analysis figure.

        Args:
            trades: List of executed trades

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping trade analysis generation.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate trade P&L
        trade_pnl = []

        for trade in trades:
            # Extract trade details
            price = trade["price"]
            quantity = trade["quantity"]

            # Calculate P&L
            pnl = price * quantity
            trade_pnl.append(pnl)

        # Plot trade P&L
        colors = ["g" if pnl > 0 else "r" for pnl in trade_pnl]
        ax.bar(range(len(trade_pnl)), trade_pnl, color=colors)

        # Add labels and title
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("P&L ($)")
        ax.set_title("Trade P&L Analysis")

        # Add grid
        ax.grid(True)

        # Add statistics
        win_rate = np.mean([pnl > 0 for pnl in trade_pnl])
        avg_win = (
            np.mean([pnl for pnl in trade_pnl if pnl > 0])
            if any(pnl > 0 for pnl in trade_pnl)
            else 0
        )
        avg_loss = (
            np.mean([pnl for pnl in trade_pnl if pnl < 0])
            if any(pnl < 0 for pnl in trade_pnl)
            else 0
        )

        # Calculate profit factor
        winning_sum = sum([pnl for pnl in trade_pnl if pnl > 0])
        losing_sum = sum([pnl for pnl in trade_pnl if pnl < 0])
        profit_factor = (
            abs(winning_sum / losing_sum) if losing_sum != 0 else float("inf")
        )
        profit_factor_str = (
            f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
        )

        stats_text = (
            f"Number of Trades: {len(trade_pnl)}\n"
            f"Win Rate: {win_rate:.2%}\n"
            f"Average Win: ${avg_win:.2f}\n"
            f"Average Loss: ${avg_loss:.2f}\n"
            f"Profit Factor: {profit_factor_str}"
        )

        # Add text box with statistics
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        fig.tight_layout()
        return fig

    def _generate_cumulative_trades(self, trades: List[Dict[str, Any]]) -> Optional[Any]:
        """
        Generate cumulative trades figure.

        Args:
            trades: List of executed trades

        Returns:
            Matplotlib Figure object
        """
        # Check if visualization libraries are available
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available. Skipping cumulative trades generation.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate trade P&L
        trade_pnl = []
        timestamps = []

        for trade in trades:
            # Extract trade details
            price = trade["price"]
            quantity = trade["quantity"]
            timestamp = pd.to_datetime(trade["timestamp"])

            # Calculate P&L
            pnl = price * quantity

            trade_pnl.append(pnl)
            timestamps.append(timestamp)

        # Create DataFrame
        trade_df = pd.DataFrame({"timestamp": timestamps, "pnl": trade_pnl})
        trade_df.set_index("timestamp", inplace=True)
        trade_df.sort_index(inplace=True)

        # Calculate cumulative P&L
        trade_df["cumulative_pnl"] = trade_df["pnl"].cumsum()

        # Plot cumulative P&L
        ax.plot(trade_df.index, trade_df["cumulative_pnl"], linewidth=2)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("Cumulative Trade P&L")

        ax.grid(True)
        fig.tight_layout()
        return fig

    def _generate_html_report(
        self, output_path: Path, include_trades: bool, include_positions: bool
    ) -> Path:
        """
        Generate an HTML performance report.

        Args:
            output_path: Directory to save the report
            include_trades: Whether to include detailed trade information
            include_positions: Whether to include detailed position information

        Returns:
            Path to the generated report
        """
        # Create report file path
        try:
            backtest_id = self.results.get('backtest_id')
            if not backtest_id:
                backtest_id = "unknown"
                self.logger.warning("No backtest_id found in results, using 'unknown'")
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Error accessing backtest_id: {str(e)}, using 'unknown'")
            backtest_id = "unknown"
        report_file = output_path / f"backtest_report_{backtest_id}.html"

        # Convert figures to base64 for embedding in HTML
        figure_html = {}

        for name, fig in self.figures.items():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            figure_html[
                name
            ] = f'<img src="data:image/png;base64,{img_str}" alt="{name}" />'

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-group {{ margin-bottom: 30px; }}
                .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .figure {{ margin-bottom: 30px; text-align: center; }}
                .figure img {{ max-width: 100%; height: auto; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}

                // Open the first tab by default
                window.onload = function() {{
                    document.getElementsByClassName("tablinks")[0].click();
                }};
            </script>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Strategy:</strong> {self.results.get('strategy', 'Unknown')}</p>
                <p><strong>Period:</strong> {self.results.get('start_date', 'Unknown')} to {self.results.get('end_date', 'Unknown')}</p>
                <p><strong>Initial Capital:</strong> ${self.results.get('initial_capital', 0):,.2f}</p>
                <p><strong>Final Equity:</strong> ${self.results.get('final_equity', 0):,.2f}</p>
                <p><strong>Total Return:</strong> {self.metrics.get('total_return', 0) * 100:.2f}%</p>
                <p><strong>Annualized Return:</strong> {self.metrics.get('annualized_return', 0) * 100:.2f}%</p>
                <p><strong>Sharpe Ratio:</strong> {self.metrics.get('sharpe_ratio', 0):.2f}</p>
                <p><strong>Max Drawdown:</strong> {self.metrics.get('max_drawdown', 0) * 100:.2f}%</p>
                <p><strong>Win Rate:</strong> {self.metrics.get('win_rate', 0) * 100:.2f}%</p>
            </div>

            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Performance')">Performance</button>
                <button class="tablinks" onclick="openTab(event, 'Risk')">Risk</button>
                <button class="tablinks" onclick="openTab(event, 'Returns')">Returns</button>
                <button class="tablinks" onclick="openTab(event, 'Trades')">Trades</button>
                <button class="tablinks" onclick="openTab(event, 'Metrics')">Metrics</button>
            </div>

            <div id="Performance" class="tabcontent">
                <h2>Performance Charts</h2>

                <div class="figure">
                    <h3>Equity Curve</h3>
                    {figure_html.get('equity_curve', '<p>No equity curve available</p>')}
                </div>

                <div class="figure">
                    <h3>Cumulative Trades</h3>
                    {figure_html.get('cumulative_trades', '<p>No cumulative trades chart available</p>')}
                </div>

                <div class="figure">
                    <h3>Monthly Returns Heatmap</h3>
                    {figure_html.get('monthly_returns', '<p>No monthly returns heatmap available</p>')}
                </div>
            </div>

            <div id="Risk" class="tabcontent">
                <h2>Risk Charts</h2>

                <div class="figure">
                    <h3>Drawdown Chart</h3>
                    {figure_html.get('drawdown_chart', '<p>No drawdown chart available</p>')}
                </div>
            </div>

            <div id="Returns" class="tabcontent">
                <h2>Returns Analysis</h2>

                <div class="figure">
                    <h3>Returns Distribution</h3>
                    {figure_html.get('returns_distribution', '<p>No returns distribution chart available</p>')}
                </div>
            </div>

            <div id="Trades" class="tabcontent">
                <h2>Trade Analysis</h2>

                <div class="figure">
                    <h3>Trade P&L Analysis</h3>
                    {figure_html.get('trade_analysis', '<p>No trade analysis chart available</p>')}
                </div>
            </div>

            <div id="Metrics" class="tabcontent">
                <h2>Performance Metrics</h2>
        """

        # Add metrics tables
        if self.metrics:
            # Group metrics
            metric_groups = {
                "Return Metrics": [
                    "total_return",
                    "annualized_return",
                    "cagr",
                    "daily_return_mean",
                    "daily_return_std",
                    "positive_days",
                    "negative_days",
                ],
                "Risk Metrics": [
                    "volatility",
                    "downside_deviation",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "max_drawdown",
                    "calmar_ratio",
                    "var_95",
                    "cvar_95",
                ],
                "Trade Metrics": [
                    "num_trades",
                    "num_winning_trades",
                    "num_losing_trades",
                    "win_rate",
                    "profit_factor",
                    "avg_trade",
                    "avg_win",
                    "avg_loss",
                    "largest_win",
                    "largest_loss",
                    "gross_profit",
                    "gross_loss",
                    "net_profit",
                ],
            }

            for group_name, metric_names in metric_groups.items():
                html_content += f"""
                <h3>{group_name}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                """

                for metric in metric_names:
                    if metric in self.metrics:
                        value = self.metrics[metric]

                        # Format value based on metric type
                        if metric in [
                            "total_return",
                            "annualized_return",
                            "cagr",
                            "daily_return_mean",
                            "daily_return_std",
                            "positive_days",
                            "negative_days",
                            "max_drawdown",
                        ]:
                            formatted_value = f"{value * 100:.2f}%"
                        elif metric in [
                            "volatility",
                            "downside_deviation",
                            "var_95",
                            "cvar_95",
                        ]:
                            formatted_value = f"{value * 100:.2f}%"
                        elif metric in [
                            "sharpe_ratio",
                            "sortino_ratio",
                            "calmar_ratio",
                            "win_rate",
                            "profit_factor",
                        ]:
                            formatted_value = f"{value:.2f}"
                        elif metric in [
                            "avg_trade",
                            "avg_win",
                            "avg_loss",
                            "largest_win",
                            "largest_loss",
                            "gross_profit",
                            "gross_loss",
                            "net_profit",
                        ]:
                            formatted_value = f"${value:.2f}"
                        else:
                            formatted_value = str(value)

                        html_content += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{formatted_value}</td>
                        </tr>
                        """

                html_content += "</table>"

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write HTML content to file
        try:
            with open(report_file, "w") as f:
                f.write(html_content)
            self.logger.debug(f"HTML report written to {report_file}")
        except (PermissionError, IOError) as e:
            self.logger.error(f"Failed to write HTML report to {report_file}: {str(e)}")
            raise

        return report_file

    def _send_to_slack(
        self, output_path: Path, include_trades: bool, include_positions: bool
    ) -> Path:
        """
        Send an HTML report to Slack.

        Args:
            output_path: Directory to save the report
            include_trades: Whether to include detailed trade information
            include_positions: Whether to include detailed position information
            
        Returns:
            Path to the generated HTML report that was sent to Slack
        """
        # First generate the HTML report
        html_report = self._generate_html_report(
            output_path, include_trades, include_positions
        )
        
        # Get Slack webhook URL from environment variable
        # Use the #reports channel webhook URL by default
        webhook_url = None

        # Get webhook URL for reports channel from environment variable
        webhook_url = os.environ.get("SLACK_REPORTS_WEBHOOK_URL")
        if webhook_url:
            self.logger.info("Using reports channel webhook URL from environment variable")
        else:
            self.logger.warning("SLACK_REPORTS_WEBHOOK_URL not found in environment variables")

        bot_token = os.environ.get("SLACK_BOT_TOKEN")
        # Default channel for reports
        channel = "#reports"

        # Read the HTML file
        try:
            # No need to read the HTML file, just send the link to Slack
            # Send to Slack
            strategy_name = self.results.get("strategy", "Unknown")
            backtest_id = self.results.get("backtest_id", "unknown")
            message = f"*Backtest Report: {strategy_name}*\nID: {backtest_id}\nReport available at: {html_report}"
            
            payload = {
                "text": f"Backtest Report: {strategy_name} (ID: {backtest_id})",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*Backtest Report: {strategy_name}*\nID: {backtest_id}"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"Report available at: {html_report}"}}
                ]
            }
            
            if webhook_url:
                response = requests.post(webhook_url, json=payload)
                if response.status_code == 200:
                    self.logger.info("Successfully sent report notification to Slack #reports channel")
                else:
                    self.logger.warning(f"Failed to send to Slack via webhook: {response.status_code} - {response.text}")
                    # Fall back to bot token if webhook fails
                    if bot_token:
                        self.logger.info("Trying to send via Slack bot token")
                        headers = {"Authorization": f"Bearer {bot_token}"}
                        bot_payload = {
                            "channel": channel,
                            "text": message
                        }
                        bot_response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=bot_payload)
                        if bot_response.status_code == 200 and bot_response.json().get("ok"):
                            self.logger.info("Successfully sent report notification to Slack via bot token")
                        else:
                            self.logger.error(f"Failed to send to Slack via bot token: {bot_response.status_code} - {bot_response.text}")
            else:
                self.logger.warning("No Slack webhook URL or bot token provided. Skipping Slack notification.")
                self.logger.info(f"Report generated at: {html_report}")
                
            return html_report
        except Exception as e:
            self.logger.error(f"Error sending report to Slack: {str(e)}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            raise

    def _generate_jupyter_report(
        self, output_path: Path, include_trades: bool, include_positions: bool
    ) -> Path:
        """
        Generate a Jupyter notebook performance report.

        Args:
            output_path: Directory to save the report
            include_trades: Whether to include detailed trade information
            include_positions: Whether to include detailed position information

        Returns:
            Path to the generated report
        """
        # Create report file path
        try:
            backtest_id = self.results.get('backtest_id')
            if not backtest_id:
                backtest_id = "unknown"
                self.logger.warning("No backtest_id found in results, using 'unknown'")
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Error accessing backtest_id: {str(e)}, using 'unknown'")
            backtest_id = "unknown"
        report_file = output_path / f"backtest_report_{backtest_id}.ipynb"

        # Create notebook content
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        # Add title
        notebook["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Backtest Performance Report\n",
                    "\n",
                    f"**Strategy:** {self.results.get('strategy', 'Unknown')}\n",
                    f"**Period:** {self.results.get('start_date', 'Unknown')} to {self.results.get('end_date', 'Unknown')}\n",
                    f"**Initial Capital:** ${self.results.get('initial_capital', 0):,.2f}\n",
                    f"**Final Equity:** ${self.results.get('final_equity', 0):,.2f}\n",
                    f"**Total Return:** {self.metrics.get('total_return', 0) * 100:.2f}%\n",
                    f"**Sharpe Ratio:** {self.metrics.get('sharpe_ratio', 0):.2f}\n",
                    f"**Max Drawdown:** {self.metrics.get('max_drawdown', 0) * 100:.2f}%\n",
                ],
            }
        )

        # Add import cell
        notebook["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "%matplotlib inline\n",
                    "plt.style.use('seaborn-v0_8-darkgrid')\n",
                    "plt.rcParams['figure.figsize'] = [12, 8]\n",
                ],
                "execution_count": None,
                "outputs": [],
            }
        )

        # Write notebook to file
        try:
            with open(report_file, "w") as f:
                json.dump(notebook, f, indent=2)
            self.logger.debug(f"Jupyter notebook report written to {report_file}")
        except (PermissionError, IOError) as e:
            self.logger.error(f"Failed to write Jupyter notebook report to {report_file}: {str(e)}")
            raise

        return report_file