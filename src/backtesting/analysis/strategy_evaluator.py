"""
StrategyEvaluator: Component for evaluating trading strategies.

This module provides functionality for evaluating trading strategies based on
various performance metrics, risk measures, and market conditions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable

from src.utils.logging import get_logger
from src.utils.metrics import calculate_trading_metrics

# Set up logging
logger = get_logger(__name__)

class StrategyEvaluator:
    """
    Evaluator for trading strategies.
    
    The StrategyEvaluator assesses trading strategies based on their performance
    across different market conditions, time periods, and parameter settings.
    """
    
    def __init__(self):
        """Initialize the strategy evaluator."""
        self.evaluation_results = {}
        
    def evaluate_strategy(self, 
                          strategy_name: str,
                          trades: List[Dict[str, Any]], 
                          portfolio_history: List[Dict[str, Any]],
                          market_data: Optional[Dict[str, pd.DataFrame]] = None,
                          parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a trading strategy.
        
        Args:
            strategy_name: Name of the strategy
            trades: List of executed trades
            portfolio_history: Historical portfolio values
            market_data: Market data used in the strategy
            parameters: Strategy parameters
            
        Returns:
            Dict containing evaluation results
        """
        logger.info(f"Evaluating strategy: {strategy_name}")
        
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
        
        # Evaluate market condition performance
        market_condition_performance = None
        if market_data:
            market_condition_performance = self._evaluate_market_conditions(trades_df, market_data)
        
        # Evaluate parameter sensitivity
        parameter_sensitivity = None
        if parameters:
            parameter_sensitivity = self._evaluate_parameter_sensitivity(parameters, performance_metrics)
        
        # Evaluate strategy robustness
        robustness_metrics = self._evaluate_robustness(trades_df, portfolio_df)
        
        # Compile results
        results = {
            'strategy_name': strategy_name,
            'performance_metrics': performance_metrics,
            'robustness_metrics': robustness_metrics,
        }
        
        if market_condition_performance:
            results['market_condition_performance'] = market_condition_performance
        
        if parameter_sensitivity:
            results['parameter_sensitivity'] = parameter_sensitivity
        
        self.evaluation_results[strategy_name] = results
        return results
    
    def _calculate_performance_metrics(self,
                                       trades_df: pd.DataFrame,
                                       portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for strategy evaluation.
        
        Args:
            trades_df: DataFrame of trades
            portfolio_df: DataFrame of portfolio history
            
        Returns:
            Dict of performance metrics
        """
        metrics = {}
        
        if not portfolio_df.empty and 'equity' in portfolio_df.columns:
            # Ensure returns are calculated
            if 'returns' not in portfolio_df.columns:
                portfolio_df['returns'] = portfolio_df['equity'].pct_change()
            
            # Use the metrics utility to calculate performance metrics
            basic_metrics = calculate_trading_metrics(
                portfolio_df['equity'],
                portfolio_df['returns'].dropna()
            )
            
            # Add basic metrics to our metrics dictionary
            metrics.update(basic_metrics)
            
            # Calculate additional strategy evaluation specific metrics
            daily_returns = portfolio_df['returns'].dropna()
            
            # Calculate Calmar ratio if not already calculated
            if 'calmar_ratio' not in metrics and 'max_drawdown' in metrics and metrics['max_drawdown'] < 0:
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            elif 'calmar_ratio' not in metrics:
                metrics['calmar_ratio'] = float('inf')
            
            # Calculate Omega ratio
            threshold = 0  # Can be adjusted
            returns_above_threshold = daily_returns[daily_returns > threshold]
            returns_below_threshold = daily_returns[daily_returns <= threshold]
            
            if len(returns_below_threshold) > 0 and abs(returns_below_threshold.sum()) > 0:
                metrics['omega_ratio'] = returns_above_threshold.sum() / abs(returns_below_threshold.sum())
            else:
                metrics['omega_ratio'] = float('inf')
        
        return metrics
    
    def _evaluate_market_conditions(self, 
                                   trades_df: pd.DataFrame, 
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate strategy performance under different market conditions.
        
        Args:
            trades_df: DataFrame of trades
            market_data: Dict of market data DataFrames
            
        Returns:
            Dict of market condition performance metrics
        """
        market_condition_performance = {}
        
        if not trades_df.empty and 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
            for market_name, market_df in market_data.items():
                if 'close' in market_df.columns:
                    # Calculate market returns
                    market_df['returns'] = market_df['close'].pct_change()
                    
                    # Define market conditions
                    market_df['volatility'] = market_df['returns'].rolling(window=20).std()
                    market_df['trend'] = market_df['close'].rolling(window=50).mean() - market_df['close'].rolling(window=200).mean()
                    
                    # Classify market conditions
                    market_df['market_condition'] = 'neutral'
                    market_df.loc[market_df['trend'] > 0, 'market_condition'] = 'uptrend'
                    market_df.loc[market_df['trend'] < 0, 'market_condition'] = 'downtrend'
                    market_df.loc[market_df['volatility'] > market_df['volatility'].quantile(0.8), 'market_condition'] = 'high_volatility'
                    
                    # Merge trades with market conditions
                    trades_with_conditions = pd.DataFrame()
                    for _, trade in trades_df.iterrows():
                        entry_date = trade['entry_time']
                        if entry_date in market_df.index:
                            trade_condition = market_df.loc[entry_date, 'market_condition']
                            trade_data = trade.copy()
                            trade_data['market_condition'] = trade_condition
                            trades_with_conditions = pd.concat([trades_with_conditions, pd.DataFrame([trade_data])])
                    
                    # Calculate performance by market condition
                    if not trades_with_conditions.empty:
                        condition_performance = {}
                        for condition in trades_with_conditions['market_condition'].unique():
                            condition_trades = trades_with_conditions[trades_with_conditions['market_condition'] == condition]
                            
                            total_trades = len(condition_trades)
                            winning_trades = len(condition_trades[condition_trades['pnl'] > 0])
                            win_rate = winning_trades / total_trades if total_trades > 0 else 0
                            
                            avg_pnl = condition_trades['pnl'].mean() if total_trades > 0 else 0
                            
                            condition_performance[condition] = {
                                'total_trades': total_trades,
                                'winning_trades': winning_trades,
                                'win_rate': win_rate,
                                'avg_pnl': avg_pnl,
                            }
                        
                        market_condition_performance[market_name] = condition_performance
        
        return market_condition_performance
    
    def _evaluate_parameter_sensitivity(self, 
                                       parameters: Dict[str, Any], 
                                       performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate sensitivity of strategy performance to parameter changes.
        
        Args:
            parameters: Strategy parameters
            performance_metrics: Performance metrics
            
        Returns:
            Dict of parameter sensitivity metrics
        """
        # This is a placeholder for parameter sensitivity analysis
        # In a real implementation, this would involve running the strategy with different
        # parameter values and analyzing how performance changes
        
        sensitivity = {}
        for param_name, param_value in parameters.items():
            # Simulate sensitivity by creating a random impact factor
            # In a real implementation, this would be based on actual performance differences
            impact_factor = np.random.uniform(0.5, 1.5)
            
            sensitivity[param_name] = {
                'value': param_value,
                'impact_factor': impact_factor,
                'impact_on_return': impact_factor * 0.1,  # Simulated impact on returns
                'impact_on_sharpe': impact_factor * 0.05,  # Simulated impact on Sharpe ratio
            }
        
        return sensitivity
    
    def _evaluate_robustness(self, 
                            trades_df: pd.DataFrame, 
                            portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate robustness of the strategy.
        
        Args:
            trades_df: DataFrame of trades
            portfolio_df: DataFrame of portfolio history
            
        Returns:
            Dict of robustness metrics
        """
        robustness_metrics = {}
        
        if not portfolio_df.empty and 'equity' in portfolio_df.columns:
            # Calculate returns consistency
            if 'returns' not in portfolio_df.columns:
                portfolio_df['returns'] = portfolio_df['equity'].pct_change()
            
            returns = portfolio_df['returns'].dropna()
            
            if not returns.empty:
                # Calculate percentage of positive periods
                positive_periods = (returns > 0).mean()
                
                # Calculate longest streak of positive and negative returns
                positive_streak = self._calculate_longest_streak(returns > 0)
                negative_streak = self._calculate_longest_streak(returns < 0)
                
                # Calculate return consistency (ratio of positive to negative returns)
                positive_sum = returns[returns > 0].sum()
                negative_sum = abs(returns[returns < 0].sum())
                return_consistency = positive_sum / negative_sum if negative_sum > 0 else float('inf')
                
                # Calculate recovery efficiency
                if 'drawdown' not in portfolio_df.columns:
                    portfolio_df['peak'] = portfolio_df['equity'].cummax()
                    portfolio_df['drawdown'] = (portfolio_df['equity'] - portfolio_df['peak']) / portfolio_df['peak']
                
                # Find drawdown periods
                drawdown_periods = []
                in_drawdown = False
                drawdown_start = None
                drawdown_depth = 0
                
                for date, row in portfolio_df.iterrows():
                    if row['drawdown'] < 0 and not in_drawdown:
                        in_drawdown = True
                        drawdown_start = date
                        drawdown_depth = row['drawdown']
                    elif row['drawdown'] < drawdown_depth and in_drawdown:
                        drawdown_depth = row['drawdown']
                    elif row['drawdown'] == 0 and in_drawdown:
                        in_drawdown = False
                        recovery_time = (date - drawdown_start).days
                        drawdown_periods.append((drawdown_depth, recovery_time))
                
                # Calculate recovery efficiency
                if drawdown_periods:
                    recovery_efficiencies = [abs(depth / recovery_time) if recovery_time > 0 else float('inf') 
                                           for depth, recovery_time in drawdown_periods]
                    avg_recovery_efficiency = np.mean([e for e in recovery_efficiencies if e != float('inf')])
                else:
                    avg_recovery_efficiency = float('inf')
                
                robustness_metrics = {
                    'positive_periods_ratio': positive_periods,
                    'longest_positive_streak': positive_streak,
                    'longest_negative_streak': negative_streak,
                    'return_consistency': return_consistency,
                    'avg_recovery_efficiency': avg_recovery_efficiency,
                }
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            # Calculate trade consistency
            trade_pnls = trades_df['pnl'].values
            
            if len(trade_pnls) > 0:
                # Calculate percentage of profitable trades
                profitable_trades_ratio = (trade_pnls > 0).mean()
                
                # Calculate longest streak of profitable and unprofitable trades
                profitable_streak = self._calculate_longest_streak(trade_pnls > 0)
                unprofitable_streak = self._calculate_longest_streak(trade_pnls <= 0)
                
                # Calculate average consecutive profit/loss
                avg_consecutive_profit = self._calculate_avg_consecutive_value(trade_pnls, lambda x: x > 0)
                avg_consecutive_loss = self._calculate_avg_consecutive_value(trade_pnls, lambda x: x <= 0)
                
                trade_metrics = {
                    'profitable_trades_ratio': profitable_trades_ratio,
                    'longest_profitable_streak': profitable_streak,
                    'longest_unprofitable_streak': unprofitable_streak,
                    'avg_consecutive_profit': avg_consecutive_profit,
                    'avg_consecutive_loss': avg_consecutive_loss,
                }
                
                robustness_metrics.update(trade_metrics)
        
        return robustness_metrics
    
    def _calculate_longest_streak(self, condition_series: np.ndarray) -> int:
        """
        Calculate the longest streak of True values in a boolean array.
        
        Args:
            condition_series: Boolean array
            
        Returns:
            Length of the longest streak
        """
        if len(condition_series) == 0:
            return 0
        
        current_streak = 0
        max_streak = 0
        
        for condition in condition_series:
            if condition:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_avg_consecutive_value(self, values: np.ndarray, condition: Callable[[float], bool]) -> float:
        """
        Calculate the average value during consecutive periods meeting a condition.
        
        Args:
            values: Array of values
            condition: Function that returns True/False for each value
            
        Returns:
            Average value during consecutive periods
        """
        if len(values) == 0:
            return 0
        
        consecutive_sums = []
        current_sum = 0
        
        for value in values:
            if condition(value):
                current_sum += value
            else:
                if current_sum != 0:
                    consecutive_sums.append(current_sum)
                    current_sum = 0
        
        if current_sum != 0:
            consecutive_sums.append(current_sum)
        
        return np.mean(consecutive_sums) if consecutive_sums else 0
    
    def compare_strategies(self, strategy_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategy_names: List of strategy names to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = {}
        
        for strategy_name in strategy_names:
            if strategy_name in self.evaluation_results:
                result = self.evaluation_results[strategy_name]
                
                # Extract key metrics for comparison
                metrics = {}
                metrics.update(result.get('performance_metrics', {}))
                metrics.update(result.get('robustness_metrics', {}))
                
                comparison_data[strategy_name] = metrics
        
        return pd.DataFrame(comparison_data)
    
    def generate_evaluation_report(self, strategy_name: str) -> str:
        """
        Generate a comprehensive evaluation report for a strategy.
        
        Args:
            strategy_name: Name of the strategy to report on
            
        Returns:
            String containing the evaluation report
        """
        if strategy_name not in self.evaluation_results:
            return f"No evaluation results found for strategy {strategy_name}"
        
        result = self.evaluation_results[strategy_name]
        
        # Format the report
        report = f"STRATEGY EVALUATION REPORT: {strategy_name}\n"
        report += "=" * 50 + "\n\n"
        
        # Performance metrics
        report += "PERFORMANCE METRICS\n"
        report += "-" * 20 + "\n"
        perf_metrics = result.get('performance_metrics', {})
        for key, value in perf_metrics.items():
            if 'return' in key or 'drawdown' in key:
                report += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
            elif isinstance(value, float) and not np.isinf(value):
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        report += "\n"
        
        # Robustness metrics
        report += "ROBUSTNESS METRICS\n"
        report += "-" * 20 + "\n"
        rob_metrics = result.get('robustness_metrics', {})
        for key, value in rob_metrics.items():
            if 'ratio' in key:
                report += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
            elif isinstance(value, float) and not np.isinf(value):
                report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        report += "\n"
        
        # Market condition performance
        market_cond = result.get('market_condition_performance', {})
        if market_cond:
            report += "MARKET CONDITION PERFORMANCE\n"
            report += "-" * 20 + "\n"
            for market, conditions in market_cond.items():
                report += f"{market}:\n"
                for condition, metrics in conditions.items():
                    report += f"  {condition}:\n"
                    for key, value in metrics.items():
                        if key == 'win_rate':
                            report += f"    {key.replace('_', ' ').title()}: {value:.2%}\n"
                        elif isinstance(value, float):
                            report += f"    {key.replace('_', ' ').title()}: {value:.4f}\n"
                        else:
                            report += f"    {key.replace('_', ' ').title()}: {value}\n"
            report += "\n"
        
        # Parameter sensitivity
        param_sens = result.get('parameter_sensitivity', {})
        if param_sens:
            report += "PARAMETER SENSITIVITY\n"
            report += "-" * 20 + "\n"
            for param, metrics in param_sens.items():
                report += f"{param}:\n"
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report += f"  {key.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        report += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        return report