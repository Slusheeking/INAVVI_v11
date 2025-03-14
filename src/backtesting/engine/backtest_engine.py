"""
BacktestEngine: Core component that orchestrates the backtesting process.

This module provides the main engine for running backtests of trading strategies
using historical market data. It coordinates the interaction between market simulation,
execution simulation, and strategy evaluation.
"""

import datetime
import uuid
import pandas as pd
from typing import Any, Dict, List, Optional

# Use absolute imports instead of relative imports
from src.utils.logging.logger import setup_logger
from src.backtesting.analysis.backtest_analyzer import BacktestAnalyzer
from src.backtesting.analysis.strategy_evaluator import StrategyEvaluator
from src.backtesting.engine.execution_simulator import ExecutionSimulator
from src.backtesting.engine.market_simulator import MarketSimulator


class BacktestEngine:
    """
    Main engine for running backtests of trading strategies.

    The BacktestEngine coordinates the interaction between market data, strategy signals,
    order execution, and performance analysis to provide a comprehensive backtesting
    environment.
    """

    def __init__(
        self,
        storage=None,
        feature_pipeline=None,
        model_registry=None,
        position_sizer=None,
        stop_loss_manager=None,
        profit_target_manager=None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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

        # Store external components
        self.storage = storage
        self.feature_pipeline = feature_pipeline
        self.model_registry = model_registry
        self.position_sizer = position_sizer
        self.stop_loss_manager = stop_loss_manager
        self.profit_target_manager = profit_target_manager

        # Store configuration
        if start_date and end_date:
            self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        else:
            # Default to last 90 days if dates not provided
            self.end_date = datetime.datetime.now()
            self.start_date = self.end_date - datetime.timedelta(days=90)
            
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

        self.logger.info(f"BacktestEngine initialized with ID {self.backtest_id}")
        
    def run_backtest(self, ticker: str, data: pd.DataFrame, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a backtest for a specific ticker with the given data and strategy parameters.
        
        Args:
            ticker: The ticker symbol to backtest
            data: Historical price data for the ticker
            strategy_params: Parameters for the trading strategy
            
        Returns:
            Dict containing backtest results
        """
        self.logger.info(f"Running backtest for {ticker} with {len(data)} data points")
        
        # Initialize results
        trades = []
        portfolio_history = []
        current_position = None
        
        # Set initial portfolio state
        portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "equity": self.initial_capital,
        }
        
        # Extract strategy parameters
        stop_loss_pct = strategy_params.get('stop_loss_pct', 0.02)
        profit_target_pct = strategy_params.get('profit_target_pct', 0.03)
        entry_threshold = strategy_params.get('entry_threshold', 0.6)
        exit_threshold = strategy_params.get('exit_threshold', 0.4)
        
        # Generate features if feature pipeline is available
        if self.feature_pipeline and 'open' in data.columns:
            self.logger.info(f"Generating features for {ticker}")
            try:
                data = self.feature_pipeline.generate_technical_indicators(data)
                data = self.feature_pipeline.generate_price_features(data)
                data = self.feature_pipeline.generate_volume_features(data)
                data = self.feature_pipeline.generate_volatility_features(data)
            except Exception as e:
                self.logger.error(f"Error generating features: {e}")
        
        # Iterate through data
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Skip first few rows to allow for indicator calculation
            if i < 10:
                continue
                
            # Current market data
            market_data = {
                'timestamp': timestamp,
                'open': row['open'] if 'open' in row else 0,
                'high': row['high'] if 'high' in row else 0,
                'low': row['low'] if 'low' in row else 0,
                'close': row['close'] if 'close' in row else 0,
                'volume': row['volume'] if 'volume' in row else 0,
                'price': row['close'] if 'close' in row else 0,
            }
            
            # Update portfolio value
            if current_position:
                # Update position value
                position_value = current_position['quantity'] * market_data['price']
                portfolio['positions'][ticker] = position_value
            else:
                portfolio['positions'][ticker] = 0
                
            # Calculate equity
            portfolio['equity'] = portfolio['cash'] + sum(portfolio['positions'].values())
            
            # Record portfolio history
            portfolio_history.append({
                'timestamp': timestamp,
                'cash': portfolio['cash'],
                'equity': portfolio['equity'],
                'positions': portfolio['positions'].copy()
            })
            
            # Generate trading signal
            signal = None
            if self.model_registry:
                try:
                    # Get model for this ticker
                    model = self.model_registry.get_latest_model(ticker, 'xgboost')
                    if model:
                        # Prepare features
                        features = data.iloc[i].drop(['open', 'high', 'low', 'close', 'volume', 'timestamp'],
                                                    errors='ignore').values.reshape(1, -1)
                        # Generate prediction
                        prediction = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else model.predict(features)[0]
                        
                        # Determine signal
                        if not current_position and prediction > entry_threshold:
                            signal = {'direction': 'buy', 'confidence': prediction}
                        elif current_position and prediction < exit_threshold:
                            signal = {'direction': 'sell', 'confidence': 1 - prediction}
                except Exception as e:
                    self.logger.error(f"Error generating signal: {e}")
            
            # Simple strategy if no model is available
            if not signal and i > 20:
                # Simple moving average crossover
                if 'close' in data.columns:
                    short_ma = data['close'].iloc[i-10:i].mean()
                    long_ma = data['close'].iloc[i-20:i].mean()
                    
                    if not current_position and short_ma > long_ma:
                        signal = {'direction': 'buy', 'confidence': 0.6}
                    elif current_position and short_ma < long_ma:
                        signal = {'direction': 'sell', 'confidence': 0.6}
            
            # Process signals
            if signal:
                if signal['direction'] == 'buy' and not current_position:
                    # Calculate position size
                    price = market_data['price']
                    quantity = 0
                    
                    if self.position_sizer:
                        quantity = self.position_sizer.calculate_position_size(
                            ticker=ticker,
                            price=price,
                            direction='buy'
                        )
                    else:
                        # Simple position sizing: 10% of portfolio
                        quantity = int((portfolio['equity'] * 0.1) / price)
                    
                    if quantity > 0:
                        # Create order
                        order = {
                            'symbol': ticker,
                            'qty': quantity,
                            'side': 'buy',
                            'type': 'market',
                            'time_in_force': 'day',
                            'limit_price': None
                        }
                        
                        # Execute order
                        execution = self.execution_simulator.execute_order(order, market_data)
                        
                        # Update portfolio
                        cost = execution['execution_price'] * execution['quantity'] + execution['commission']
                        portfolio['cash'] -= cost
                        
                        # Record position
                        current_position = {
                            'entry_time': timestamp,
                            'entry_price': execution['execution_price'],
                            'quantity': execution['quantity'],
                            'stop_loss': execution['execution_price'] * (1 - stop_loss_pct),
                            'profit_target': execution['execution_price'] * (1 + profit_target_pct),
                            'symbol': ticker
                        }
                        
                        self.logger.debug(f"Opened position: {current_position}")
                
                elif signal['direction'] == 'sell' and current_position:
                    # Create order
                    order = {
                        'symbol': ticker,
                        'qty': current_position['quantity'],
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'day',
                        'limit_price': None
                    }
                    
                    # Execute order
                    execution = self.execution_simulator.execute_order(order, market_data)
                    
                    # Update portfolio
                    proceeds = execution['execution_price'] * execution['quantity'] - execution['commission']
                    portfolio['cash'] += proceeds
                    
                    # Record trade
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': current_position['entry_price'],
                        'exit_price': execution['execution_price'],
                        'quantity': current_position['quantity'],
                        'symbol': ticker,
                        'pnl': (execution['execution_price'] - current_position['entry_price']) * current_position['quantity'] - execution['commission'],
                        'exit_reason': 'signal'
                    }
                    trades.append(trade)
                    
                    self.logger.debug(f"Closed position: {trade}")
                    
                    # Clear current position
                    current_position = None
            
            # Check stop loss and profit target
            if current_position:
                # Check stop loss
                if market_data['price'] <= current_position['stop_loss'] and self.stop_loss_manager:
                    # Create order
                    order = {
                        'symbol': ticker,
                        'qty': current_position['quantity'],
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'day',
                        'limit_price': None
                    }
                    
                    # Execute order
                    execution = self.execution_simulator.execute_order(order, market_data)
                    
                    # Update portfolio
                    proceeds = execution['execution_price'] * execution['quantity'] - execution['commission']
                    portfolio['cash'] += proceeds
                    
                    # Record trade
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': current_position['entry_price'],
                        'exit_price': execution['execution_price'],
                        'quantity': current_position['quantity'],
                        'symbol': ticker,
                        'pnl': (execution['execution_price'] - current_position['entry_price']) * current_position['quantity'] - execution['commission'],
                        'exit_reason': 'stop_loss'
                    }
                    trades.append(trade)
                    
                    self.logger.debug(f"Stop loss triggered: {trade}")
                    
                    # Clear current position
                    current_position = None
                
                # Check profit target
                elif market_data['price'] >= current_position['profit_target'] and self.profit_target_manager:
                    # Create order
                    order = {
                        'symbol': ticker,
                        'qty': current_position['quantity'],
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'day',
                        'limit_price': None
                    }
                    
                    # Execute order
                    execution = self.execution_simulator.execute_order(order, market_data)
                    
                    # Update portfolio
                    proceeds = execution['execution_price'] * execution['quantity'] - execution['commission']
                    portfolio['cash'] += proceeds
                    
                    # Record trade
                    trade = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': current_position['entry_price'],
                        'exit_price': execution['execution_price'],
                        'quantity': current_position['quantity'],
                        'symbol': ticker,
                        'pnl': (execution['execution_price'] - current_position['entry_price']) * current_position['quantity'] - execution['commission'],
                        'exit_reason': 'profit_target'
                    }
                    trades.append(trade)
                    
                    self.logger.debug(f"Profit target reached: {trade}")
                    
                    # Clear current position
                    current_position = None
        
        # Close any open positions at the end of the backtest
        if current_position:
            # Use the last price
            last_price = data['close'].iloc[-1] if 'close' in data.columns else 0
            
            # Record trade
            trade = {
                'entry_time': current_position['entry_time'],
                'exit_time': data.index[-1],
                'entry_price': current_position['entry_price'],
                'exit_price': last_price,
                'quantity': current_position['quantity'],
                'symbol': ticker,
                'pnl': (last_price - current_position['entry_price']) * current_position['quantity'],
                'exit_reason': 'end_of_backtest'
            }
            trades.append(trade)
            
            self.logger.debug(f"Closed position at end of backtest: {trade}")
        
        # Calculate metrics
        net_profit = sum(trade['pnl'] for trade in trades) if trades else 0
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        
        # Calculate returns
        initial_equity = self.initial_capital
        final_equity = portfolio_history[-1]['equity'] if portfolio_history else initial_equity
        total_return = (final_equity / initial_equity) - 1
        
        # Store results
        results = {
            'trades': trades,
            'portfolio_history': portfolio_history,
            'net_profit': net_profit,
            'win_rate': win_rate,
            'total_return': total_return,
            'initial_equity': initial_equity,
            'final_equity': final_equity
        }
        
        self.logger.info(f"Backtest completed for {ticker}: {len(trades)} trades, net profit: {net_profit:.2f}, win rate: {win_rate:.2f}")
        
        # Store results for later analysis
        self.results = results
        
        return results
