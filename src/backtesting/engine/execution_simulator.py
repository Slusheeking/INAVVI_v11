"""
ExecutionSimulator: Component for simulating trade execution in backtests.

This module provides functionality for simulating trade execution with various
models for commissions, slippage, and market impact.
"""
import numpy as np
from typing import Dict, Any, List, Optional

from src.utils.logging import get_logger

# Set up logger for this module
logger = get_logger("backtesting.engine.execution_simulator")

class ExecutionSimulator:
    """
    Simulator for trade execution in backtests.
    
    The ExecutionSimulator handles the simulation of trade execution with
    configurable models for execution timing, commissions, slippage, and
    market impact.
    """
    
    def __init__(self, 
                execution_model: str = "realistic",
                commission_model: Optional[Dict[str, Any]] = None,
                slippage_model: Optional[Dict[str, Any]] = None,
                market_impact_model: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution simulator.
        
        Args:
            execution_model: Type of execution model ('perfect', 'next_bar', 'realistic')
            commission_model: Configuration for commission model
            slippage_model: Configuration for slippage model
            market_impact_model: Configuration for market impact model
        """
        self.execution_model = execution_model
        
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
        
        logger.info(f"ExecutionSimulator initialized with {execution_model} execution model")
    
    def execute_order(self, 
                     order: Dict[str, Any], 
                     market_data: Dict[str, Any],
                     volume: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute an order in the simulation.
        
        Args:
            order: Order details
            market_data: Market data for the execution
            volume: Trading volume (for market impact calculation)
            
        Returns:
            Dict with execution details
        """
        # Extract order details
        symbol = order.get('symbol')
        order_type = order.get('type', 'market')
        side = order.get('side')
        quantity = order.get('quantity')
        limit_price = order.get('limit_price')
        
        # Determine execution price based on execution model
        execution_price = self._determine_execution_price(order, market_data)
        
        # Apply slippage
        execution_price = self._apply_slippage(execution_price, side)
        
        # Apply market impact if volume is provided
        if volume is not None and volume > 0:
            execution_price = self._apply_market_impact(execution_price, side, quantity, volume)
        
        # Calculate commission
        commission = self._calculate_commission(execution_price, quantity)
        
        # Create execution record
        execution = {
            'symbol': symbol,
            'order_type': order_type,
            'side': side,
            'quantity': quantity,
            'requested_price': limit_price if order_type == 'limit' else market_data.get('price'),
            'execution_price': execution_price,
            'commission': commission,
            'timestamp': market_data.get('timestamp'),
            'status': 'filled',
        }
        
        logger.debug(f"Executed order: {execution}")
        return execution
    
    def _determine_execution_price(self, 
                                  order: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> float:
        """
        Determine the execution price based on the execution model.
        
        Args:
            order: Order details
            market_data: Market data for the execution
            
        Returns:
            Execution price
        """
        order_type = order.get('type', 'market')
        side = order.get('side')
        limit_price = order.get('limit_price')
        
        # Get price from market data
        open_price = market_data.get('open')
        high_price = market_data.get('high')
        low_price = market_data.get('low')
        close_price = market_data.get('close')
        current_price = market_data.get('price', close_price)
        
        # Determine execution price based on execution model
        if self.execution_model == "perfect":
            # Perfect execution at the current price
            execution_price = current_price
        
        elif self.execution_model == "next_bar":
            # Execute at the open of the next bar
            execution_price = open_price
        
        elif self.execution_model == "realistic":
            # Realistic execution with random price within the bar
            if side == 'buy':
                # For buy orders, price is between open and high
                execution_price = np.random.uniform(open_price, high_price)
            else:
                # For sell orders, price is between open and low
                execution_price = np.random.uniform(low_price, open_price)
        
        else:
            # Default to current price
            execution_price = current_price
        
        # For limit orders, check if the price is acceptable
        if order_type == 'limit':
            if (side == 'buy' and execution_price > limit_price) or \
               (side == 'sell' and execution_price < limit_price):
                # Limit price not reached, use limit price
                execution_price = limit_price
        
        return execution_price
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to the execution price.
        
        Args:
            price: Base execution price
            side: Order side ('buy' or 'sell')
            
        Returns:
            Price with slippage applied
        """
        slippage_type = self.slippage_model.get('type', 'fixed')
        slippage_value = self.slippage_model.get('value', 0.0001)
        
        if slippage_type == 'fixed':
            # Fixed slippage in basis points
            slippage_factor = slippage_value
        
        elif slippage_type == 'variable':
            # Variable slippage with random component
            base_slippage = slippage_value
            random_component = np.random.uniform(0, slippage_value)
            slippage_factor = base_slippage + random_component
        
        elif slippage_type == 'percentage':
            # Percentage slippage
            slippage_factor = price * slippage_value
        
        else:
            # Default to no slippage
            slippage_factor = 0
        
        # Apply slippage based on order side
        if side == 'buy':
            # For buy orders, price increases
            price_with_slippage = price * (1 + slippage_factor)
        else:
            # For sell orders, price decreases
            price_with_slippage = price * (1 - slippage_factor)
        
        return price_with_slippage
    
    def _apply_market_impact(self, 
                            price: float, 
                            side: str, 
                            quantity: float, 
                            volume: float) -> float:
        """
        Apply market impact to the execution price.
        
        Args:
            price: Base execution price
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            volume: Trading volume
            
        Returns:
            Price with market impact applied
        """
        impact_type = self.market_impact_model.get('type', 'linear')
        impact_factor = self.market_impact_model.get('factor', 0.1)
        
        # Calculate order's percentage of volume
        volume_percentage = quantity / volume if volume > 0 else 0
        
        if impact_type == 'linear':
            # Linear market impact model
            impact = impact_factor * volume_percentage
        
        elif impact_type == 'square_root':
            # Square root market impact model
            impact = impact_factor * np.sqrt(volume_percentage)
        
        elif impact_type == 'quadratic':
            # Quadratic market impact model
            impact = impact_factor * (volume_percentage ** 2)
        
        else:
            # Default to no impact
            impact = 0
        
        # Apply impact based on order side
        if side == 'buy':
            # For buy orders, price increases
            price_with_impact = price * (1 + impact)
        else:
            # For sell orders, price decreases
            price_with_impact = price * (1 - impact)
        
        return price_with_impact
    
    def _calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate commission for the trade.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        commission_type = self.commission_model.get('type', 'percentage')
        commission_value = self.commission_model.get('value', 0.001)
        min_commission = self.commission_model.get('min', 0)
        max_commission = self.commission_model.get('max', float('inf'))
        
        trade_value = price * quantity
        
        if commission_type == 'percentage':
            # Percentage commission
            commission = trade_value * commission_value
        
        elif commission_type == 'fixed':
            # Fixed commission per trade
            commission = commission_value
        
        elif commission_type == 'per_share':
            # Per-share commission
            commission = quantity * commission_value
        
        elif commission_type == 'tiered':
            # Tiered commission based on trade value
            tiers = self.commission_model.get('tiers', [])
            commission = 0
            
            for tier in tiers:
                tier_min = tier.get('min', 0)
                tier_max = tier.get('max', float('inf'))
                tier_rate = tier.get('rate', 0)
                
                if tier_min <= trade_value < tier_max:
                    commission = trade_value * tier_rate
                    break
        
        else:
            # Default to no commission
            commission = 0
        
        # Apply minimum and maximum commission constraints
        commission = max(min_commission, min(commission, max_commission))
        
        return commission
    
    def simulate_fills(self, 
                      orders: List[Dict[str, Any]], 
                      market_data: Dict[str, Dict[str, Any]],
                      volumes: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Simulate fills for multiple orders.
        
        Args:
            orders: List of orders to execute
            market_data: Market data for each symbol
            volumes: Trading volumes for each symbol
            
        Returns:
            List of execution details
        """
        executions = []
        
        for order in orders:
            symbol = order.get('symbol')
            
            if symbol in market_data:
                symbol_market_data = market_data[symbol]
                symbol_volume = volumes.get(symbol) if volumes else None
                
                execution = self.execute_order(order, symbol_market_data, symbol_volume)
                executions.append(execution)
            else:
                logger.warning(f"No market data available for {symbol}, skipping order")
        
        return executions
    
    def calculate_transaction_costs(self, executions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate total transaction costs for a list of executions.
        
        Args:
            executions: List of execution details
            
        Returns:
            Dict with transaction cost breakdown
        """
        total_commission = sum(execution.get('commission', 0) for execution in executions)
        
        # Calculate slippage costs
        total_slippage = 0
        for execution in executions:
            requested_price = execution.get('requested_price', 0)
            execution_price = execution.get('execution_price', 0)
            quantity = execution.get('quantity', 0)
            side = execution.get('side')
            
            if side == 'buy':
                slippage_cost = (execution_price - requested_price) * quantity
            else:
                slippage_cost = (requested_price - execution_price) * quantity
            
            total_slippage += max(0, slippage_cost)  # Only count positive slippage as a cost
        
        total_costs = total_commission + total_slippage
        
        return {
            'total_costs': total_costs,
            'commission': total_commission,
            'slippage': total_slippage,
        }