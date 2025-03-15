"""
Metrics utilities for the Autonomous Trading System.

This module provides utilities for calculating trading metrics and performance
statistics for trading strategies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger

logger = get_logger("utils.metrics.metrics_utils")


def calculate_returns(
    prices: Union[List[float], np.ndarray], is_log_returns: bool = False
) -> np.ndarray:
    """
    Calculate returns from a series of prices.
    
    Args:
        prices: List or array of prices
        is_log_returns: Whether to calculate log returns
        
    Returns:
        Array of returns
    """
    prices_array = np.array(prices)
    if is_log_returns:
        # Log returns: ln(P_t / P_{t-1})
        returns = np.diff(np.log(prices_array))
    else:
        # Simple returns: (P_t - P_{t-1}) / P_{t-1}
        returns = np.diff(prices_array) / prices_array[:-1]
    
    return returns


def calculate_cumulative_returns(
    returns: Union[List[float], np.ndarray], initial_value: float = 1.0
) -> np.ndarray:
    """
    Calculate cumulative returns from a series of returns.
    
    Args:
        returns: List or array of returns
        initial_value: Initial value
        
    Returns:
        Array of cumulative returns
    """
    returns_array = np.array(returns)
    cumulative_returns = initial_value * np.cumprod(1 + returns_array)
    return cumulative_returns


def calculate_drawdowns(
    cumulative_returns: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdowns from a series of cumulative returns.
    
    Args:
        cumulative_returns: List or array of cumulative returns
        
    Returns:
        Tuple of (drawdowns, max_drawdown, max_drawdown_duration)
    """
    cumulative_returns_array = np.array(cumulative_returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns_array)
    
    # Calculate drawdowns
    drawdowns = (cumulative_returns_array - running_max) / running_max
    
    # Calculate maximum drawdown
    max_drawdown = np.min(drawdowns)
    
    # Calculate maximum drawdown duration
    if max_drawdown < 0:
        # Find the index of the peak
        peak_idx = np.argmax(cumulative_returns_array[:np.argmin(drawdowns)])
        
        # Find the index of the recovery (or the end if no recovery)
        recovery_idx = peak_idx
        for i in range(peak_idx, len(cumulative_returns_array)):
            if cumulative_returns_array[i] >= cumulative_returns_array[peak_idx]:
                recovery_idx = i
                break
        
        max_drawdown_duration = recovery_idx - peak_idx
    else:
        max_drawdown_duration = 0
    
    return drawdowns, max_drawdown, max_drawdown_duration


def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: List or array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    returns_array = np.array(returns)
    
    # Convert risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns_array - rf_per_period
    
    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    
    if std_excess_return == 0:
        return 0.0
    
    sharpe_ratio = mean_excess_return / std_excess_return
    
    # Annualize Sharpe ratio
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(periods_per_year)
    
    return sharpe_ratio_annualized


def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: List or array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        target_return: Target return (annualized)
        
    Returns:
        Sortino ratio
    """
    returns_array = np.array(returns)
    
    # Convert risk-free rate and target return to per-period rates
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    target_per_period = (1 + target_return) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns_array - rf_per_period
    
    # Calculate downside returns
    downside_returns = np.minimum(excess_returns - target_per_period, 0)
    
    # Calculate Sortino ratio
    mean_excess_return = np.mean(excess_returns)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    sortino_ratio = mean_excess_return / downside_deviation
    
    # Annualize Sortino ratio
    sortino_ratio_annualized = sortino_ratio * np.sqrt(periods_per_year)
    
    return sortino_ratio_annualized


def calculate_calmar_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: List or array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Calmar ratio
    """
    returns_array = np.array(returns)
    
    # Convert risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns_array - rf_per_period
    
    # Calculate annualized return
    annualized_return = (1 + np.mean(excess_returns)) ** periods_per_year - 1
    
    # Calculate maximum drawdown
    cumulative_returns = calculate_cumulative_returns(excess_returns)
    _, max_drawdown, _ = calculate_drawdowns(cumulative_returns)
    
    if max_drawdown == 0:
        return 0.0
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    return calmar_ratio


def calculate_omega_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    threshold: float = 0.0,
) -> float:
    """
    Calculate the Omega ratio.
    
    Args:
        returns: List or array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        threshold: Return threshold
        
    Returns:
        Omega ratio
    """
    returns_array = np.array(returns)
    
    # Convert risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns_array - rf_per_period
    
    # Calculate returns above and below threshold
    returns_above = np.sum(np.maximum(excess_returns - threshold, 0))
    returns_below = np.sum(np.abs(np.minimum(excess_returns - threshold, 0)))
    
    if returns_below == 0:
        return float('inf')
    
    # Calculate Omega ratio
    omega_ratio = returns_above / returns_below
    
    return omega_ratio


def calculate_win_rate(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the win rate.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Win rate
    """
    returns_array = np.array(returns)
    wins = np.sum(returns_array > 0)
    total = len(returns_array)
    
    if total == 0:
        return 0.0
    
    win_rate = wins / total
    
    return win_rate


def calculate_profit_factor(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the profit factor.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Profit factor
    """
    returns_array = np.array(returns)
    gross_profit = np.sum(returns_array[returns_array > 0])
    gross_loss = np.sum(np.abs(returns_array[returns_array < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return profit_factor


def calculate_expectancy(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the expectancy.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Expectancy
    """
    returns_array = np.array(returns)
    win_rate = calculate_win_rate(returns_array)
    
    if len(returns_array) == 0:
        return 0.0
    
    avg_win = np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0
    avg_loss = np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    return expectancy


def calculate_kelly_criterion(
    win_rate: float, win_loss_ratio: float
) -> float:
    """
    Calculate the Kelly criterion.
    
    Args:
        win_rate: Win rate
        win_loss_ratio: Ratio of average win to average loss
        
    Returns:
        Kelly criterion
    """
    if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Limit Kelly to a reasonable range
    kelly = max(0.0, min(1.0, kelly))
    
    return kelly


def calculate_average_trade(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the average trade.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Average trade
    """
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        return 0.0
    
    average_trade = np.mean(returns_array)
    
    return average_trade


def calculate_average_win(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the average win.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Average win
    """
    returns_array = np.array(returns)
    wins = returns_array[returns_array > 0]
    
    if len(wins) == 0:
        return 0.0
    
    average_win = np.mean(wins)
    
    return average_win


def calculate_average_loss(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the average loss.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Average loss
    """
    returns_array = np.array(returns)
    losses = returns_array[returns_array < 0]
    
    if len(losses) == 0:
        return 0.0
    
    average_loss = np.mean(losses)
    
    return average_loss


def calculate_win_loss_ratio(returns: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the win/loss ratio.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Win/loss ratio
    """
    average_win = calculate_average_win(returns)
    average_loss = calculate_average_loss(returns)
    
    if average_loss == 0:
        return float('inf') if average_win > 0 else 0.0
    
    win_loss_ratio = average_win / abs(average_loss)
    
    return win_loss_ratio


def calculate_max_consecutive_wins(returns: Union[List[float], np.ndarray]) -> int:
    """
    Calculate the maximum number of consecutive wins.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Maximum number of consecutive wins
    """
    returns_array = np.array(returns)
    wins = returns_array > 0
    
    if not np.any(wins):
        return 0
    
    # Count consecutive wins
    consecutive_wins = []
    count = 0
    
    for win in wins:
        if win:
            count += 1
        else:
            if count > 0:
                consecutive_wins.append(count)
                count = 0
    
    # Add the last count if it's non-zero
    if count > 0:
        consecutive_wins.append(count)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    
    return max_consecutive_wins


def calculate_max_consecutive_losses(returns: Union[List[float], np.ndarray]) -> int:
    """
    Calculate the maximum number of consecutive losses.
    
    Args:
        returns: List or array of returns
        
    Returns:
        Maximum number of consecutive losses
    """
    returns_array = np.array(returns)
    losses = returns_array < 0
    
    if not np.any(losses):
        return 0
    
    # Count consecutive losses
    consecutive_losses = []
    count = 0
    
    for loss in losses:
        if loss:
            count += 1
        else:
            if count > 0:
                consecutive_losses.append(count)
                count = 0
    
    # Add the last count if it's non-zero
    if count > 0:
        consecutive_losses.append(count)
    
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    return max_consecutive_losses


def calculate_volatility(
    returns: Union[List[float], np.ndarray], periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized volatility.
    
    Args:
        returns: List or array of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility
    """
    returns_array = np.array(returns)
    
    if len(returns_array) <= 1:
        return 0.0
    
    # Calculate volatility
    volatility = np.std(returns_array, ddof=1)
    
    # Annualize volatility
    volatility_annualized = volatility * np.sqrt(periods_per_year)
    
    return volatility_annualized


def calculate_var(
    returns: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Value at Risk (VaR).
    
    Args:
        returns: List or array of returns
        confidence_level: Confidence level
        periods_per_year: Number of periods per year
        
    Returns:
        Value at Risk
    """
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        return 0.0
    
    # Calculate VaR
    var = np.percentile(returns_array, 100 * (1 - confidence_level))
    
    # Annualize VaR
    var_annualized = var * np.sqrt(periods_per_year)
    
    return var_annualized


def calculate_cvar(
    returns: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR).
    
    Args:
        returns: List or array of returns
        confidence_level: Confidence level
        periods_per_year: Number of periods per year
        
    Returns:
        Conditional Value at Risk
    """
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        return 0.0
    
    # Calculate VaR
    var = np.percentile(returns_array, 100 * (1 - confidence_level))
    
    # Calculate CVaR
    cvar = np.mean(returns_array[returns_array <= var])
    
    # Annualize CVaR
    cvar_annualized = cvar * np.sqrt(periods_per_year)
    
    return cvar_annualized


def calculate_beta(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
) -> float:
    """
    Calculate the beta.
    
    Args:
        returns: List or array of returns
        benchmark_returns: List or array of benchmark returns
        
    Returns:
        Beta
    """
    returns_array = np.array(returns)
    benchmark_returns_array = np.array(benchmark_returns)
    
    if len(returns_array) != len(benchmark_returns_array):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    if len(returns_array) <= 1:
        return 0.0
    
    # Calculate covariance and variance
    covariance = np.cov(returns_array, benchmark_returns_array)[0, 1]
    variance = np.var(benchmark_returns_array, ddof=1)
    
    if variance == 0:
        return 0.0
    
    # Calculate beta
    beta = covariance / variance
    
    return beta


def calculate_alpha(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the alpha.
    
    Args:
        returns: List or array of returns
        benchmark_returns: List or array of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Alpha
    """
    returns_array = np.array(returns)
    benchmark_returns_array = np.array(benchmark_returns)
    
    if len(returns_array) != len(benchmark_returns_array):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    if len(returns_array) <= 1:
        return 0.0
    
    # Convert risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate beta
    beta = calculate_beta(returns_array, benchmark_returns_array)
    
    # Calculate average returns
    avg_return = np.mean(returns_array)
    avg_benchmark_return = np.mean(benchmark_returns_array)
    
    # Calculate alpha
    alpha = avg_return - rf_per_period - beta * (avg_benchmark_return - rf_per_period)
    
    # Annualize alpha
    alpha_annualized = (1 + alpha) ** periods_per_year - 1
    
    return alpha_annualized


def calculate_information_ratio(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the information ratio.
    
    Args:
        returns: List or array of returns
        benchmark_returns: List or array of benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        Information ratio
    """
    returns_array = np.array(returns)
    benchmark_returns_array = np.array(benchmark_returns)
    
    if len(returns_array) != len(benchmark_returns_array):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    if len(returns_array) <= 1:
        return 0.0
    
    # Calculate tracking error
    tracking_error = np.std(returns_array - benchmark_returns_array, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    # Calculate information ratio
    information_ratio = np.mean(returns_array - benchmark_returns_array) / tracking_error
    
    # Annualize information ratio
    information_ratio_annualized = information_ratio * np.sqrt(periods_per_year)
    
    return information_ratio_annualized


def calculate_trading_metrics(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Optional[Union[List[float], np.ndarray]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate a comprehensive set of trading metrics.
    
    Args:
        returns: List or array of returns
        benchmark_returns: List or array of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary of trading metrics
    """
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        logger.warning("Empty returns array, returning default metrics")
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_trade": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_loss_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }
    
    # Calculate cumulative returns
    cumulative_returns = calculate_cumulative_returns(returns_array)
    
    # Calculate drawdowns
    _, max_drawdown, max_drawdown_duration = calculate_drawdowns(cumulative_returns)
    
    # Calculate total return
    total_return = cumulative_returns[-1] - 1.0
    
    # Calculate annualized return
    annualized_return = (1 + np.mean(returns_array)) ** periods_per_year - 1
    
    # Calculate volatility
    volatility = calculate_volatility(returns_array, periods_per_year)
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(returns_array, risk_free_rate, periods_per_year)
    
    # Calculate Sortino ratio
    sortino_ratio = calculate_sortino_ratio(returns_array, risk_free_rate, periods_per_year)
    
    # Calculate Calmar ratio
    calmar_ratio = calculate_calmar_ratio(returns_array, risk_free_rate, periods_per_year)
    
    # Calculate win rate
    win_rate = calculate_win_rate(returns_array)
    
    # Calculate profit factor
    profit_factor = calculate_profit_factor(returns_array)
    
    # Calculate expectancy
    expectancy = calculate_expectancy(returns_array)
    
    # Calculate average trade
    average_trade = calculate_average_trade(returns_array)
    
    # Calculate average win
    average_win = calculate_average_win(returns_array)
    
    # Calculate average loss
    average_loss = calculate_average_loss(returns_array)
    
    # Calculate win/loss ratio
    win_loss_ratio = calculate_win_loss_ratio(returns_array)
    
    # Calculate maximum consecutive wins
    max_consecutive_wins = calculate_max_consecutive_wins(returns_array)
    
    # Calculate maximum consecutive losses
    max_consecutive_losses = calculate_max_consecutive_losses(returns_array)
    
    # Create metrics dictionary
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_trade": average_trade,
        "average_win": average_win,
        "average_loss": average_loss,
        "win_loss_ratio": win_loss_ratio,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }
    
    # Calculate benchmark-related metrics if benchmark returns are provided
    if benchmark_returns is not None:
        benchmark_returns_array = np.array(benchmark_returns)
        
        if len(benchmark_returns_array) != len(returns_array):
            logger.warning(
                "Benchmark returns length does not match returns length, "
                "skipping benchmark-related metrics"
            )
        else:
            # Calculate beta
            beta = calculate_beta(returns_array, benchmark_returns_array)
            
            # Calculate alpha
            alpha = calculate_alpha(
                returns_array, benchmark_returns_array, risk_free_rate, periods_per_year
            )
            
            # Calculate information ratio
            information_ratio = calculate_information_ratio(
                returns_array, benchmark_returns_array, periods_per_year
            )
            
            # Add benchmark-related metrics to the dictionary
            metrics.update({
                "beta": beta,
                "alpha": alpha,
                "information_ratio": information_ratio,
            })
    
    return metrics


def calculate_trade_statistics(
    trades: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate statistics for a list of trades.
    
    Args:
        trades: List of trade dictionaries, each containing at least 'return' key
        
    Returns:
        Dictionary of trade statistics
    """
    if not trades:
        logger.warning("Empty trades list, returning default statistics")
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_trade": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_loss_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }
    
    # Extract returns from trades
    returns = [trade["return"] for trade in trades]
    
    # Calculate total trades
    total_trades = len(trades)
    
    # Calculate winning and losing trades
    winning_trades = sum(1 for r in returns if r > 0)
    losing_trades = sum(1 for r in returns if r < 0)
    
    # Calculate win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Calculate profit factor
    profit_factor = calculate_profit_factor(returns)
    
    # Calculate expectancy
    expectancy = calculate_expectancy(returns)
    
    # Calculate average trade
    average_trade = calculate_average_trade(returns)
    
    # Calculate average win
    average_win = calculate_average_win(returns)
    
    # Calculate average loss
    average_loss = calculate_average_loss(returns)
    
    # Calculate win/loss ratio
    win_loss_ratio = calculate_win_loss_ratio(returns)
    
    # Calculate maximum consecutive wins
    max_consecutive_wins = calculate_max_consecutive_wins(returns)
    
    # Calculate maximum consecutive losses
    max_consecutive_losses = calculate_max_consecutive_losses(returns)
    
    # Create statistics dictionary
    statistics = {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "average_trade": average_trade,
        "average_win": average_win,
        "average_loss": average_loss,
        "win_loss_ratio": win_loss_ratio,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }
    
    return statistics