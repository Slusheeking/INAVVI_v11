#!/usr/bin/env python3
"""
Continuous Learning Pipeline

This module provides the ContinuousLearningPipeline class which orchestrates
the process of analyzing performance, adapting strategies, and retraining models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.continuous_learning.analysis.performance_analyzer import PerformanceAnalyzer
from src.continuous_learning.adaptation.strategy_adapter import StrategyAdapter
from src.continuous_learning.retraining.model_retrainer import ModelRetrainer
from src.utils.logging import get_logger


class ContinuousLearningPipeline:
    """
    Orchestrates the continuous learning process by coordinating performance analysis,
    strategy adaptation, and model retraining.
    
    This pipeline enables the trading system to adapt to changing market conditions by:
    1. Analyzing trading performance to identify areas for improvement
    2. Adapting strategy parameters based on performance feedback
    3. Retraining models with new data to improve predictions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the continuous learning pipeline.
        
        Args:
            config: Configuration parameters for the pipeline and its components
        """
        self.config = config or {}
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer(
            config=self.config.get("performance_analyzer", {})
        )
        self.strategy_adapter = StrategyAdapter(
            strategy_params=self.config.get("strategy_params", {})
        )
        self.model_retrainer = ModelRetrainer(
            config=self.config.get("model_retrainer", {})
        )
        
        # Pipeline configuration
        self.min_trades_for_cycle = self.config.get("min_trades_for_cycle", 50)
        self.min_days_between_cycles = self.config.get("min_days_between_cycles", 7)
        self.market_regime_detection = self.config.get("market_regime_detection", True)
        
        # Pipeline state
        self.last_cycle_time = None
        self.cycle_history = []
        
        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info("Continuous learning pipeline initialized")
    
    def run_cycle(
        self,
        trades: Any,
        equity_curve: Any,
        model_registry: Any,
        features: Any = None,
        target: Any = None,
        market_data: Any = None,
        current_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete continuous learning cycle.
        
        Args:
            trades: DataFrame with trade information
            equity_curve: DataFrame with equity curve
            model_registry: Registry containing models
            features: Feature data for model retraining (optional)
            target: Target data for model retraining (optional)
            market_data: Market data for analysis (optional)
            current_regime: Current market regime (optional)
            
        Returns:
            Dictionary with cycle results
        """
        cycle_start_time = datetime.now()
        self.logger.info(f"Starting continuous learning cycle at {cycle_start_time}")
        
        # Check if we have enough data
        if len(trades) < self.min_trades_for_cycle:
            self.logger.warning(
                f"Not enough trades for continuous learning cycle: {len(trades)} < {self.min_trades_for_cycle}"
            )
            return {
                "status": "insufficient_data",
                "message": f"Not enough trades for analysis: {len(trades)} < {self.min_trades_for_cycle}",
                "timestamp": cycle_start_time.isoformat(),
            }
        
        # Check if enough time has passed since last cycle
        if self.last_cycle_time:
            days_since_last_cycle = (cycle_start_time - self.last_cycle_time).days
            if days_since_last_cycle < self.min_days_between_cycles:
                self.logger.info(
                    f"Not enough time since last cycle: {days_since_last_cycle} < {self.min_days_between_cycles} days"
                )
                return {
                    "status": "too_soon",
                    "message": f"Not enough time since last cycle: {days_since_last_cycle} < {self.min_days_between_cycles} days",
                    "timestamp": cycle_start_time.isoformat(),
                }
        
        # Detect market regime if enabled and not provided
        if self.market_regime_detection and not current_regime and market_data is not None:
            current_regime = self._detect_market_regime(market_data)
            self.logger.info(f"Detected market regime: {current_regime}")
        
        # Step 1: Analyze performance
        self.logger.info("Step 1: Analyzing performance")
        analysis_results = self.performance_analyzer.analyze_performance(
            trades, equity_curve, market_data
        )
        
        if analysis_results.get("status") != "success":
            self.logger.warning(f"Performance analysis failed: {analysis_results.get('message')}")
            return {
                "status": "analysis_failed",
                "message": analysis_results.get("message", "Performance analysis failed"),
                "timestamp": cycle_start_time.isoformat(),
            }
        
        # Step 2: Adapt strategy parameters
        self.logger.info("Step 2: Adapting strategy parameters")
        current_params = self.strategy_adapter.strategy_params
        
        # Record performance with current parameters
        self.strategy_adapter.record_performance(
            current_params, analysis_results["performance_metrics"]
        )
        
        # Suggest improved parameters
        param_ranges = self._generate_parameter_ranges(current_params)
        suggested_params = self.strategy_adapter.suggest_parameters(
            param_ranges, target_metric="sharpe_ratio"
        )
        
        # Apply regime-specific adaptation if applicable
        if current_regime:
            regime_params = self.strategy_adapter.adapt_parameters(current_regime)
            # Merge suggested and regime-specific parameters
            for key, value in regime_params.items():
                if key not in suggested_params:
                    suggested_params[key] = value
        
        # Update strategy parameters
        updated_params = self.strategy_adapter.update_strategy_params(suggested_params)
        
        # Step 3: Retrain models if features and target are provided
        retrained_model = None
        if features is not None and target is not None:
            self.logger.info("Step 3: Retraining models")
            retrained_model = self.model_retrainer.retrain_model(
                model_registry, features, target, current_regime
            )
            
            if retrained_model is None:
                self.logger.warning("Model retraining did not produce an improved model")
        else:
            self.logger.info("Skipping model retraining (no features/target provided)")
        
        # Update cycle state
        self.last_cycle_time = cycle_start_time
        
        # Record cycle results
        cycle_results = {
            "status": "success",
            "timestamp": cycle_start_time.isoformat(),
            "duration": (datetime.now() - cycle_start_time).total_seconds(),
            "analysis_results": analysis_results,
            "strategy_adaptation": {
                "previous_params": current_params,
                "updated_params": updated_params,
            },
            "model_retraining": {
                "performed": retrained_model is not None,
                "successful": retrained_model is not None,
            },
            "market_regime": current_regime,
        }
        
        self.cycle_history.append(cycle_results)
        
        self.logger.info(f"Continuous learning cycle completed in {cycle_results['duration']:.2f} seconds")
        return cycle_results
    
    def _detect_market_regime(self, market_data: Any) -> str:
        """
        Detect current market regime based on market data.
        
        Args:
            market_data: Market data
            
        Returns:
            Detected market regime
        """
        # Simple regime detection based on volatility and trend
        # This is a placeholder - in a real system, this would be more sophisticated
        try:
            # Calculate volatility (standard deviation of returns)
            if "returns" in market_data.columns:
                returns = market_data["returns"]
            else:
                # Calculate returns from close prices
                close_col = next((col for col in market_data.columns if "close" in col.lower()), None)
                if close_col:
                    returns = market_data[close_col].pct_change()
                else:
                    # Use first numeric column
                    numeric_cols = market_data.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        returns = market_data[numeric_cols[0]].pct_change()
                    else:
                        self.logger.warning("No suitable columns for regime detection")
                        return "unknown"
            
            # Calculate recent volatility (last 20 periods)
            recent_volatility = returns.tail(20).std()
            
            # Calculate trend (average return)
            recent_trend = returns.tail(20).mean()
            
            # Determine regime
            if recent_volatility > 0.015:  # High volatility threshold
                if recent_trend > 0.001:
                    return "high_volatility_bullish"
                elif recent_trend < -0.001:
                    return "high_volatility_bearish"
                else:
                    return "high_volatility_neutral"
            else:
                if recent_trend > 0.001:
                    return "low_volatility_bullish"
                elif recent_trend < -0.001:
                    return "low_volatility_bearish"
                else:
                    return "low_volatility_neutral"
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def _generate_parameter_ranges(self, current_params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Generate parameter ranges for optimization based on current parameters.
        
        Args:
            current_params: Current strategy parameters
            
        Returns:
            Dictionary with parameter ranges
        """
        param_ranges = {}
        
        for param, value in current_params.items():
            if isinstance(value, int):
                # Integer parameter: vary by ±30%
                min_val = max(1, int(value * 0.7))
                max_val = int(value * 1.3)
                param_ranges[param] = (min_val, max_val)
                
            elif isinstance(value, float):
                # Float parameter: vary by ±30%
                min_val = max(0.0001, value * 0.7)
                max_val = value * 1.3
                param_ranges[param] = (min_val, max_val)
                
            # Add more types as needed
        
        return param_ranges
    
    def get_cycle_history(self) -> List[Dict[str, Any]]:
        """
        Get history of continuous learning cycles.
        
        Returns:
            List of cycle results
        """
        return self.cycle_history
    
    def reset(self) -> None:
        """
        Reset pipeline state.
        """
        self.last_cycle_time = None
        self.cycle_history = []
        self.logger.info("Continuous learning pipeline state reset")