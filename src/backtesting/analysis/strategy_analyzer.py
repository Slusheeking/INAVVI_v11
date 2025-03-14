"""
Strategy Analyzer for the Autonomous Trading System.

This module provides functionality for analyzing strategy performance.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

logger = logging.getLogger(__name__)

class StrategyAnalyzer:
    """
    Analyzer for strategy performance.
    """
    
    def __init__(self, initial_capital=100000):
        """
        Initialize the strategy analyzer.
        
        Args:
            initial_capital (float): Initial capital
        """
        self.initial_capital = initial_capital
        self.performance_metrics = {}
    
    def calculate_returns(self, positions_df):
        """
        Calculate returns from positions.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            
        Returns:
            pd.DataFrame: DataFrame with returns
        """
        # Ensure positions_df has required columns
        required_columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity', 'symbol']
        missing_columns = set(required_columns) - set(positions_df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in positions_df: {missing_columns}")
        
        # Calculate P&L for each position
        positions_df['pnl'] = (positions_df['exit_price'] - positions_df['entry_price']) * positions_df['quantity']
        positions_df['return'] = positions_df['pnl'] / (positions_df['entry_price'] * positions_df['quantity'])
        positions_df['holding_period'] = (positions_df['exit_time'] - positions_df['entry_time']).dt.total_seconds() / (60 * 60 * 24)  # in days
        
        return positions_df
    
    def calculate_equity_curve(self, positions_df):
        """
        Calculate equity curve.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            
        Returns:
            pd.DataFrame: DataFrame with equity curve
        """
        # Ensure positions_df has required columns
        if 'pnl' not in positions_df.columns:
            positions_df = self.calculate_returns(positions_df)
        
        # Sort positions by exit time
        positions_df = positions_df.sort_values('exit_time')
        
        # Calculate cumulative P&L and equity
        positions_df['cumulative_pnl'] = positions_df['pnl'].cumsum()
        positions_df['equity'] = self.initial_capital + positions_df['cumulative_pnl']
        
        # Calculate drawdowns
        positions_df['peak'] = positions_df['equity'].cummax()
        positions_df['drawdown'] = (positions_df['equity'] - positions_df['peak']) / positions_df['peak']
        
        return positions_df
    
    def calculate_performance_metrics(self, positions_df, risk_free_rate=0.0):
        """
        Calculate performance metrics.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            risk_free_rate (float): Risk-free rate (annualized)
            
        Returns:
            dict: Performance metrics
        """
        # Ensure positions_df has required columns
        if 'equity' not in positions_df.columns:
            positions_df = self.calculate_equity_curve(positions_df)
        
        # Calculate returns
        positions_df['daily_return'] = positions_df['equity'].pct_change()
        
        # Calculate metrics
        total_days = (positions_df['exit_time'].max() - positions_df['entry_time'].min()).total_seconds() / (60 * 60 * 24)
        total_return = (positions_df['equity'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        
        daily_returns = positions_df['daily_return'].dropna().values
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate drawdown metrics
        max_drawdown = positions_df['drawdown'].min()
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        in_drawdown = False
        
        for dd in positions_df['drawdown'].values:
            if dd < 0:
                in_drawdown = True
                current_drawdown_duration += 1
            else:
                if in_drawdown:
                    max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
                    current_drawdown_duration = 0
                    in_drawdown = False
        
        # Calculate win/loss metrics
        wins = positions_df[positions_df['pnl'] > 0]
        losses = positions_df[positions_df['pnl'] < 0]
        
        win_rate = len(wins) / len(positions_df) if len(positions_df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf')
        
        # Calculate position metrics
        avg_holding_period = positions_df['holding_period'].mean()
        
        # Store metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'total_trades': len(positions_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses)
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def plot_equity_curve(self, positions_df, figsize=(12, 8)):
        """
        Plot equity curve.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Ensure positions_df has required columns
        if 'equity' not in positions_df.columns:
            positions_df = self.calculate_equity_curve(positions_df)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(positions_df['exit_time'], positions_df['equity'], label='Equity')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(positions_df['exit_time'], positions_df['drawdown'], 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_return_distribution(self, positions_df, figsize=(12, 8)):
        """
        Plot return distribution.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Ensure positions_df has required columns
        if 'return' not in positions_df.columns:
            positions_df = self.calculate_returns(positions_df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot return distribution
        returns = positions_df['return'].values
        ax.hist(returns, bins=50, alpha=0.5, label='Returns')
        
        # Plot normal distribution
        x = np.linspace(min(returns), max(returns), 100)
        ax.plot(x, norm.pdf(x, np.mean(returns), np.std(returns)) * len(returns) * (max(returns) - min(returns)) / 50,
                'r-', label='Normal Distribution')
        
        ax.set_title('Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_performance_report(self, positions_df, risk_free_rate=0.0):
        """
        Generate performance report.
        
        Args:
            positions_df (pd.DataFrame): DataFrame with position data
            risk_free_rate (float): Risk-free rate (annualized)
            
        Returns:
            str: Performance report
        """
        # Calculate metrics
        metrics = self.calculate_performance_metrics(positions_df, risk_free_rate)
        
        # Generate report
        report = "PERFORMANCE REPORT\n"
        report += "=================\n\n"
        
        report += "RETURN METRICS\n"
        report += f"Total Return: {metrics['total_return']:.2%}\n"
        report += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        report += f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
        report += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n"
        
        report += "DRAWDOWN METRICS\n"
        report += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"Maximum Drawdown Duration: {metrics['max_drawdown_duration']} days\n\n"
        
        report += "TRADE METRICS\n"
        report += f"Total Trades: {metrics['total_trades']}\n"
        report += f"Win Rate: {metrics['win_rate']:.2%}\n"
        report += f"Average Win: ${metrics['avg_win']:.2f}\n"
        report += f"Average Loss: ${metrics['avg_loss']:.2f}\n"
        report += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        report += f"Average Holding Period: {metrics['avg_holding_period']:.2f} days\n"
        
        return report
