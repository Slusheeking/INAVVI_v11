#!/usr/bin/env python3
"""
Run ML Trading Models

This script demonstrates how to use the machine learning trading models
with data from Polygon.io and Unusual Whales APIs.

Usage:
    python run_ml_trading_models.py --symbols AAPL,MSFT,GOOGL --mode backtest
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our ML models
from ml_trading_models import (
    FeatureEngineering,
    ClassificationModels,
    TimeSeriesModels,
    HFTModels,
    ModelIntegrationFramework
)

# Import data sources
from polygon_data_source_ultra import PolygonDataSourceUltra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_ml_trading_models')

# API Keys
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf')
UNUSUAL_WHALES_API_KEY = os.environ.get('UNUSUAL_WHALES_API_KEY', '4ad71b9e-7ace-4f24-bdfc-532ace219a18')

class MLTradingRunner:
    """
    Runner for ML trading models
    """
    
    def __init__(self, polygon_api_key=POLYGON_API_KEY, unusual_whales_api_key=UNUSUAL_WHALES_API_KEY):
        """Initialize ML trading runner"""
        self.polygon_api_key = polygon_api_key
        self.unusual_whales_api_key = unusual_whales_api_key
        
        # Initialize data sources
        self.polygon_data = PolygonDataSourceUltra(api_key=polygon_api_key)
        
        # Initialize ML framework
        self.framework = ModelIntegrationFramework()
        
        logger.info("Initialized ML trading runner")
        
    def fetch_historical_data(self, symbols, days=30):
        """
        Fetch historical data for symbols
        
        Args:
            symbols: List of stock symbols
            days: Number of days of historical data
            
        Returns:
            Dictionary with historical data by symbol
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            # Fetch daily bars
            bars = self.polygon_data.get_daily_bars(
                symbol=symbol,
                days=days
            )
            
            if bars is not None and not bars.empty:
                results[symbol] = bars
                logger.info(f"Fetched {len(bars)} daily bars for {symbol}")
            else:
                logger.warning(f"No historical data for {symbol}")
                
        return results
        
    def fetch_options_data(self, symbols):
        """
        Fetch options data for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with options data by symbol
        """
        logger.info(f"Fetching options data for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            # Fetch options data
            options = self.polygon_data.get_options_data(
                underlying=symbol,
                expiration_date_gte=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                expiration_date_lte=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            )
            
            if options is not None and len(options) > 0:
                results[symbol] = options
                logger.info(f"Fetched {len(options)} options contracts for {symbol}")
            else:
                logger.warning(f"No options data for {symbol}")
                
        return results
        
    def fetch_unusual_options_flow(self, symbols):
        """
        Fetch unusual options flow data
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with unusual options flow data by symbol
        """
        logger.info(f"Fetching unusual options flow data for {len(symbols)} symbols")
        
        # For demonstration, generate mock data
        results = {}
        
        for symbol in symbols:
            # Generate mock options flow data
            flow_data = self._generate_mock_options_flow(symbol, limit=20)
            
            results[symbol] = flow_data
            logger.info(f"Generated {len(flow_data)} mock options flow entries for {symbol}")
                
        return results
        
    def fetch_dark_pool_data(self, symbols):
        """
        Fetch dark pool data for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with dark pool data by symbol
        """
        logger.info(f"Fetching dark pool data for {len(symbols)} symbols")
        
        # For demonstration, generate mock data
        results = {}
        
        for symbol in symbols:
            # Generate mock dark pool data
            dark_pool_data = self._generate_mock_dark_pool_data(symbol, limit=20)
            
            results[symbol] = dark_pool_data
            logger.info(f"Generated {len(dark_pool_data)} mock dark pool trades for {symbol}")
                
        return results
        
    def _generate_mock_options_flow(self, symbol, limit=20):
        """Generate mock options flow data for testing"""
        result = []
        
        for _ in range(limit):
            strike = round(np.random.uniform(50, 500), 2)
            premium = round(np.random.uniform(0.5, 10), 2)
            
            option = {
                "symbol": symbol,
                "strike": strike,
                "premium": premium,
                "type": np.random.choice(["call", "put"]),
                "expiration": (datetime.now() + timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d"),
                "volume": np.random.randint(100, 10000),
                "open_interest": np.random.randint(100, 50000),
                "unusual_score": round(np.random.uniform(50, 100), 2)
            }
            result.append(option)
            
        return result
        
    def _generate_mock_dark_pool_data(self, symbol, limit=20):
        """Generate mock dark pool data for testing"""
        result = []
        
        for _ in range(limit):
            trade = {
                "symbol": symbol,
                "price": round(np.random.uniform(50, 500), 2),
                "volume": np.random.randint(1000, 100000),
                "timestamp": datetime.now().isoformat(),
                "exchange": np.random.choice(["XDARK", "XADF", "XPST"]),
                "trade_id": str(np.random.randint(1000000, 9999999))
            }
            result.append(trade)
            
        return result
        
    def run_premarket_analysis(self, symbols):
        """
        Run pre-market analysis
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with pre-market analysis results
        """
        logger.info(f"Running pre-market analysis for {len(symbols)} symbols")
        
        # 1. Fetch historical data
        historical_data = self.fetch_historical_data(symbols)
        
        # 2. Fetch options data
        options_data = self.fetch_options_data(symbols)
        
        # 3. Fetch unusual options flow
        options_flow = self.fetch_unusual_options_flow(symbols)
        
        # 4. Run pre-market analysis
        results = {}
        
        for symbol in symbols:
            if symbol in historical_data and symbol in options_flow:
                # Convert historical data to DataFrame
                price_data = historical_data[symbol]
                
                # Get options flow data
                flow_data = options_flow[symbol]
                
                # Run analysis
                analysis = self.framework.premarket_analysis(price_data, flow_data)
                
                results[symbol] = analysis
                logger.info(f"Completed pre-market analysis for {symbol}")
            else:
                logger.warning(f"Insufficient data for pre-market analysis of {symbol}")
                
        return results
        
    def run_trading_hours_analysis(self, symbols):
        """
        Run trading hours analysis
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with trading hours analysis results
        """
        logger.info(f"Running trading hours analysis for {len(symbols)} symbols")
        
        # 1. Fetch historical data
        historical_data = self.fetch_historical_data(symbols)
        
        # 2. Fetch options data
        options_data = self.fetch_options_data(symbols)
        
        # 3. Fetch unusual options flow
        options_flow = self.fetch_unusual_options_flow(symbols)
        
        # 4. Fetch dark pool data
        dark_pool_data = self.fetch_dark_pool_data(symbols)
        
        # 5. Run trading hours analysis
        results = {}
        
        for symbol in symbols:
            if symbol in historical_data and symbol in options_flow and symbol in dark_pool_data:
                # Convert historical data to DataFrame
                price_data = historical_data[symbol]
                
                # Get options flow data
                flow_data = options_flow[symbol]
                
                # Get dark pool data
                dp_data = dark_pool_data[symbol]
                
                # Run analysis
                analysis = self.framework.trading_hours_analysis(
                    price_data, flow_data, dp_data, {}  # Empty real-time data for now
                )
                
                results[symbol] = analysis
                logger.info(f"Completed trading hours analysis for {symbol}")
            else:
                logger.warning(f"Insufficient data for trading hours analysis of {symbol}")
                
        return results
        
    def run_backtest(self, symbols, start_date, end_date):
        """
        Run backtest
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # 1. Fetch historical data
        historical_data = self.fetch_historical_data(symbols, days=365)  # Get a full year of data
        
        # 2. Fetch options data
        options_data = self.fetch_options_data(symbols)
        
        # 3. Fetch dark pool data
        dark_pool_data = self.fetch_dark_pool_data(symbols)
        
        # 4. Run backtest
        results = {}
        
        for symbol in symbols:
            if symbol in historical_data:
                # Convert historical data to DataFrame
                price_data = historical_data[symbol]
                
                # Get options flow data
                flow_data = options_data.get(symbol, [])
                
                # Get dark pool data
                dp_data = dark_pool_data.get(symbol, [])
                
                # Run backtest
                backtest = self.framework.backtest_strategy(
                    price_data, flow_data, dp_data, start_date, end_date
                )
                
                results[symbol] = backtest
                logger.info(f"Completed backtest for {symbol}")
            else:
                logger.warning(f"Insufficient data for backtest of {symbol}")
                
        return results
        
    def close(self):
        """Close data sources"""
        if hasattr(self, 'polygon_data'):
            self.polygon_data.close()
            
        logger.info("Closed data sources")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run ML Trading Models")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL",
                      help="Comma-separated list of symbols to analyze")
    parser.add_argument("--mode", type=str, default="backtest",
                      choices=["premarket", "trading", "backtest"],
                      help="Mode to run (premarket, trading, or backtest)")
    parser.add_argument("--start-date", type=str, default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                      help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                      help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--polygon-key", type=str, default=POLYGON_API_KEY,
                      help="Polygon API key")
    parser.add_argument("--unusual-whales-key", type=str, default=UNUSUAL_WHALES_API_KEY,
                      help="Unusual Whales API key")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    logger.info(f"Running in {args.mode} mode for symbols: {symbols}")
    
    # Initialize runner
    runner = MLTradingRunner(
        polygon_api_key=args.polygon_key,
        unusual_whales_api_key=args.unusual_whales_key
    )
    
    try:
        # Run in specified mode
        if args.mode == "premarket":
            results = runner.run_premarket_analysis(symbols)
        elif args.mode == "trading":
            results = runner.run_trading_hours_analysis(symbols)
        elif args.mode == "backtest":
            results = runner.run_backtest(symbols, args.start_date, args.end_date)
        
        # Print results summary
        print("\nResults Summary:")
        for symbol, result in results.items():
            if args.mode == "backtest" and "performance" in result:
                perf = result["performance"].get(symbol, {})
                returns = perf.get("returns", 0)
                print(f"{symbol}: Return: {returns:.2%}")
            elif args.mode in ["premarket", "trading"] and "signals" in result:
                signals = result["signals"]
                for sym, signal in signals.items():
                    print(f"{sym}: Direction: {signal.get('direction', 'UNKNOWN')}, Confidence: {signal.get('confidence', 0):.2f}")
    finally:
        # Close runner
        runner.close()
    
    logger.info("Done")

if __name__ == "__main__":
    main()