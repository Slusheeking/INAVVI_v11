"""
BacktestAnalyzer: Component for analyzing backtest results.

This module provides functionality for analyzing and interpreting the results
of backtests, including performance metrics, trade statistics, and risk measures.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.backtesting.analysis.strategy_analyzer import StrategyAnalyzer

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path('/home/ubuntu/INAVVI_v11-1/src/logs')
log_dir.mkdir(parents=True, exist_ok=True)

# Create file handler
log_file = log_dir / 'backtest_analyzer.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)
logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """
    Analyzer for backtest results.
    
    The BacktestAnalyzer processes raw backtest data to extract meaningful
    performance metrics, trade statistics, and risk measures to evaluate
    trading strategies.
    """
    
    def __init__(self):
        """Initialize the backtest analyzer."""
        self.strategy_analyzer = StrategyAnalyzer()
        self.results = {}
        
    def analyze_backtest(self, 
                         backtest_id: str,
                         trades: List[Dict[str, Any]], 
                         portfolio_history: List[Dict[str, Any]],
                         market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Analyze backtest results.
        
        Args:
            backtest_id: Unique identifier for the backtest
            trades: List of executed trades
            portfolio_history: Historical portfolio values
            market_data: Market data used in the backtest
            
        Returns:
            Dict containing analysis results
        """
        logger.info(f"Analyzing backtest {backtest_id}")
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            # Ensure datetime columns are properly formatted
            for col in ['entry_time', 'exit_time']:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col])
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            if 'timestamp' in portfolio_df.columns:
                portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
                portfolio_df = portfolio_df.set_index('timestamp')
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(trades_df, portfolio_df)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trades_df)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_df)
        
        # Calculate market correlation if market data is provided
        market_correlation = None
        if market_data:
            market_correlation = self._calculate_market_correlation(portfolio_df, market_data)
        
        # Compile results
        results = {
            'backtest_id': backtest_id,
            'performance_metrics': performance_metrics,
            'trade_statistics': trade_stats,
            'risk_metrics': risk_metrics,
        }
        
        if market_correlation:
            results['market_correlation'] = market_correlation
        
        self.results[backtest_id] = results
        return results
    
    def _calculate_performance_metrics(self, 
                                      trades_df: pd.DataFrame, 
                                      portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            trades_df: DataFrame of trades
            portfolio_df: DataFrame of portfolio history
            
        Returns:
            Dict of performance metrics
        """
        metrics = {}
        
        if not portfolio_df.empty and 'equity' in portfolio_df.columns:
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['equity'].pct_change()
            
            # Calculate cumulative returns
            initial_equity = portfolio_df['equity'].iloc[0]
            final_equity = portfolio_df['equity'].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            
            # Calculate annualized return
            days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
            else:
                annualized_return = 0
            
            # Calculate volatility
            daily_returns = portfolio_df['returns'].dropna()
            if not daily_returns.empty:
                volatility = daily_returns.std()
                annualized_volatility = volatility * np.sqrt(252)  # Assuming 252 trading days
            else:
                volatility = 0
                annualized_volatility = 0
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility
            else:
                sharpe_ratio = 0
            
            # Calculate drawdown
            portfolio_df['peak'] = portfolio_df['equity'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['equity'] - portfolio_df['peak']) / portfolio_df['peak']
            max_drawdown = portfolio_df['drawdown'].min()
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
            }
        
        return metrics
    
    def _calculate_trade_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trade statistics from backtest results.
        
        Args:
            trades_df: DataFrame of trades
            
        Returns:
            Dict of trade statistics
        """
        stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_holding_period': 0,
        }
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            # Calculate basic trade statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average profit and loss
            avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Calculate profit factor
            total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate average holding period
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / (60 * 60 * 24)  # in days
                avg_holding_period = trades_df['holding_period'].mean()
            else:
                avg_holding_period = 0
            
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period,
            }
        
        return stats
    
    def _calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk metrics from backtest results.
        
        Args:
            portfolio_df: DataFrame of portfolio history
            
        Returns:
            Dict of risk metrics
        """
        metrics = {}
        
        if not portfolio_df.empty and 'equity' in portfolio_df.columns:
            # Calculate returns
            if 'returns' not in portfolio_df.columns:
                portfolio_df['returns'] = portfolio_df['equity'].pct_change()
            
            returns = portfolio_df['returns'].dropna()
            
            if not returns.empty:
                # Calculate downside deviation
                negative_returns = returns[returns < 0]
                downside_deviation = negative_returns.std() if not negative_returns.empty else 0
                
                # Calculate Sortino ratio (assuming risk-free rate of 0)
                annualized_return = (1 + returns.mean()) ** 252 - 1
                sortino_ratio = annualized_return / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
                
                # Calculate Value at Risk (VaR)
                var_95 = np.percentile(returns, 5)
                
                # Calculate Conditional VaR (CVaR) / Expected Shortfall
                cvar_95 = returns[returns <= var_95].mean()
                
                # Calculate maximum drawdown duration
                if 'drawdown' not in portfolio_df.columns:
                    portfolio_df['peak'] = portfolio_df['equity'].cummax()
                    portfolio_df['drawdown'] = (portfolio_df['equity'] - portfolio_df['peak']) / portfolio_df['peak']
                
                # Find drawdown periods
                in_drawdown = False
                drawdown_start = None
                drawdown_periods = []
                
                for date, row in portfolio_df.iterrows():
                    if row['drawdown'] < 0 and not in_drawdown:
                        in_drawdown = True
                        drawdown_start = date
                    elif row['drawdown'] == 0 and in_drawdown:
                        in_drawdown = False
                        drawdown_periods.append((drawdown_start, date))
                
                # Calculate max drawdown duration in days
                if drawdown_periods:
                    drawdown_durations = [(end - start).days for start, end in drawdown_periods]
                    max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
                else:
                    max_drawdown_duration = 0
                
                metrics = {
                    'downside_deviation': downside_deviation,
                    'sortino_ratio': sortino_ratio,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'max_drawdown_duration': max_drawdown_duration,
                }
        
        return metrics
    
    def _calculate_market_correlation(self, 
                                     portfolio_df: pd.DataFrame, 
                                     market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate correlation between strategy returns and market returns.
        
        Args:
            portfolio_df: DataFrame of portfolio history
            market_data: Dict of market data DataFrames
            
        Returns:
            Dict of correlation metrics
        """
        correlations = {}
        
        if not portfolio_df.empty and 'returns' in portfolio_df.columns:
            portfolio_returns = portfolio_df['returns'].dropna()
            
            for market_name, market_df in market_data.items():
                if 'close' in market_df.columns:
                    market_df['returns'] = market_df['close'].pct_change()
                    
                    # Align dates
                    aligned_data = pd.concat([portfolio_returns, market_df['returns']], axis=1, join='inner')
                    aligned_data.columns = ['portfolio', 'market']
                    
                    if not aligned_data.empty:
                        correlation = aligned_data['portfolio'].corr(aligned_data['market'])
                        correlations[market_name] = correlation
        
        return correlations
    
    def compare_backtests(self, backtest_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            backtest_ids: List of backtest IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = {}
        
        for backtest_id in backtest_ids:
            if backtest_id in self.results:
                result = self.results[backtest_id]
                
                # Extract key metrics for comparison
                metrics = {}
                metrics.update(result.get('performance_metrics', {}))
                
                trade_stats = result.get('trade_statistics', {})
                metrics['win_rate'] = trade_stats.get('win_rate', 0)
                metrics['profit_factor'] = trade_stats.get('profit_factor', 0)
                
                risk_metrics = result.get('risk_metrics', {})
                metrics['sortino_ratio'] = risk_metrics.get('sortino_ratio', 0)
                
                comparison_data[backtest_id] = metrics
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, backtest_id: str) -> str:
        """
        Generate a comprehensive report for a backtest.
        
        Args:
            backtest_id: ID of the backtest to report on
            
        Returns:
            String containing the report
        """
        if backtest_id not in self.results:
            return f"No results found for backtest {backtest_id}"
        
        result = self.results[backtest_id]
        
        # Format the report
        report = f"BACKTEST REPORT: {backtest_id}\n"
        report += "=" * 50 + "\n\n"
        
        # Performance metrics
        report += "PERFORMANCE METRICS\n"
        report += "-" * 20 + "\n"
        perf_metrics = result.get('performance_metrics', {})
        for key, value in perf_metrics.items():
            if 'return' in key or 'drawdown' in key:
                report += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Trade statistics
        report += "TRADE STATISTICS\n"
        report += "-" * 20 + "\n"
        trade_stats = result.get('trade_statistics', {})
        for key, value in trade_stats.items():
            if key == 'win_rate':
                report += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
            elif isinstance(value, float):
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        report += "\n"
        
        # Risk metrics
        report += "RISK METRICS\n"
        report += "-" * 20 + "\n"
        risk_metrics = result.get('risk_metrics', {})
        for key, value in risk_metrics.items():
            if 'var' in key or 'cvar' in key:
                report += f"{key.upper()}: {value:.4f}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "\n"
        
        # Market correlation
        market_corr = result.get('market_correlation', {})
        if market_corr:
            report += "MARKET CORRELATION\n"
            report += "-" * 20 + "\n"
            for market, corr in market_corr.items():
                report += f"{market}: {corr:.4f}\n"
        
        return report