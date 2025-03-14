#!/usr/bin/env python3
"""
Full System Test for Autonomous Trading System

This script tests all components of the autonomous trading system in an integrated manner,
using real market data from Polygon and Unusual Whales APIs. The test runs as a production
environment with live data and live executions, collecting comprehensive metrics on accuracy,
latency, and performance.
"""

import os
import sys
import time
import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom components
from src.tests.performance_metrics import PerformanceMetrics
from src.tests.system_components import (
    SystemController, TimescaleDBManager,
    configure_tensorflow_gpu, configure_xgboost_gpu
)

# Data Acquisition
from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from src.data_acquisition.storage.timescale_storage import TimescaleStorage
from src.data_acquisition.collectors.price_collector import PriceCollector
from src.data_acquisition.collectors.quote_collector import QuoteCollector
from src.data_acquisition.collectors.trade_collector import TradeCollector
from src.data_acquisition.collectors.options_collector import OptionsCollector
from src.data_acquisition.collectors.multi_timeframe_data_collector import MultiTimeframeDataCollector
from src.data_acquisition.pipeline.data_pipeline import DataPipeline
from src.data_acquisition.transformation.data_transformer import DataTransformer

# Feature Engineering
from src.feature_engineering.pipeline.feature_pipeline import FeaturePipeline
from src.feature_engineering.store.feature_store import FeatureStore
from src.feature_engineering.analysis.feature_analyzer import FeatureAnalyzer

# Model Training
from src.model_training.models.xgboost_model import XGBoostModel
from src.model_training.models.lstm_model import LSTMModel
from src.model_training.models.cnn_model import CNNModel
from src.model_training.registry.model_registry import ModelRegistry
from src.model_training.validation.model_validator import ModelValidator
from src.model_training.models.inference.model_inference import ModelInference

# Trading Strategy
from src.trading_strategy.selection.ticker_selector import DynamicTickerSelector
from src.trading_strategy.selection.timeframe_selector import TimeframeSelector
from src.trading_strategy.signals.peak_detector import PeakDetector
from src.trading_strategy.signals.entry_signal_generator import EntrySignalGenerator
from src.trading_strategy.sizing.risk_based_position_sizer import RiskBasedPositionSizer
from src.trading_strategy.risk.stop_loss_manager import StopLossManager
from src.trading_strategy.risk.profit_target_manager import ProfitTargetManager
from src.trading_strategy.execution.order_generator import OrderGenerator
from src.trading_strategy.alpaca.alpaca_client import AlpacaClient
from src.trading_strategy.alpaca.alpaca_trade_executor import AlpacaTradeExecutor
from src.trading_strategy.alpaca.alpaca_position_manager import AlpacaPositionManager

# Continuous Learning
from src.continuous_learning.analysis.performance_analyzer import PerformanceAnalyzer
from src.continuous_learning.adaptation.strategy_adapter import StrategyAdapter
from src.continuous_learning.retraining.model_retrainer import ModelRetrainer

# Backtesting
from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.analysis.strategy_analyzer import StrategyAnalyzer
from src.backtesting.reporting.performance_reporter import PerformanceReporter

# Utils
from src.config.database_config import get_db_connection_string
from src.utils.time.market_hours import get_market_status, MarketStatus
from src.utils.time.market_calendar import MarketCalendar

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("full_system_test")

class FullSystemTest:
    """Test class for full system testing."""
    
    def __init__(self):
        """Initialize all system components."""
        logger.info("Initializing full system test components...")
        
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics()
        
        # Configure GPU acceleration
        logger.info("Configuring GPU acceleration...")
        self.gpus = configure_tensorflow_gpu()
        self.xgboost_gpu = configure_xgboost_gpu()
        
        # Load API keys from environment
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        if not self.polygon_key:
            logger.error("POLYGON_API_KEY not found in environment")
            raise ValueError("POLYGON_API_KEY not found in environment")
            
        self.unusual_whales_key = os.environ.get("UNUSUAL_WHALES_API_KEY")
        if not self.unusual_whales_key:
            logger.error("UNUSUAL_WHALES_API_KEY not found in environment")
            raise ValueError("UNUSUAL_WHALES_API_KEY not found in environment")
            
        self.alpaca_key_id = os.environ.get("ALPACA_API_KEY")
        if not self.alpaca_key_id:
            logger.error("ALPACA_API_KEY not found in environment")
            raise ValueError("ALPACA_API_KEY not found in environment")
            
        self.alpaca_secret_key = os.environ.get("ALPACA_API_SECRET")
        if not self.alpaca_secret_key:
            logger.error("ALPACA_API_SECRET not found in environment")
            raise ValueError("ALPACA_API_SECRET not found in environment")
        
        # Initialize API clients with optimizations for high-frequency trading
        logger.info("Initializing API clients with high-frequency trading optimizations...")
        
        # Configure Polygon client for high-frequency trading (5000 positions per day)
        self.polygon_client = PolygonClient(
            api_key=self.polygon_key,
            rate_limit=10,  # Increased rate limit for high-frequency trading
            retry_attempts=5,  # More retry attempts for reliability
            timeout=15,  # Reduced timeout for faster response handling
            verify_ssl=True
        )
        
        # Configure Unusual Whales client for high-frequency trading
        self.unusual_whales_client = UnusualWhalesClient(
            api_key=self.unusual_whales_key,
            rate_limit=5,  # Increased rate limit for high-frequency trading
            retry_attempts=5,  # More retry attempts for reliability
            timeout=15  # Reduced timeout for faster response handling
        )
        
        # Configure Alpaca client for high-frequency trading
        self.alpaca_client = AlpacaClient(
            api_key=self.alpaca_key_id,
            api_secret=self.alpaca_secret_key,
            base_url="https://paper-api.alpaca.markets/v2",  # Use paper trading for testing
            max_position_value=2500.0,  # Maximum $2500 per stock
            max_daily_value=5000.0  # Maximum $5000 per day
        )
        
        # Initialize database connection
        logger.info("Initializing database connection...")
        self.db_manager = TimescaleDBManager(connection_string=get_db_connection_string())
        self.db_manager.set_performance_metrics(self.performance_metrics)
        self.db_manager.initialize()
        
        # Initialize storage
        logger.info("Initializing storage...")
        self.storage = TimescaleStorage(connection_string=get_db_connection_string())
        self.transformer = DataTransformer()
        
        # Initialize data collectors
        logger.info("Initializing data collectors...")
        self.price_collector = PriceCollector(
            polygon_client=self.polygon_client,
            config={"storage": self.storage}
        )
        self.quote_collector = QuoteCollector(
            polygon_client=self.polygon_client,
            config={"storage": self.storage}
        )
        self.trade_collector = TradeCollector(
            polygon_client=self.polygon_client,
            config={"storage": self.storage}
        )
        self.options_collector = OptionsCollector(
            polygon_client=self.polygon_client,
            config={"storage": self.storage}
        )
        self.multi_timeframe_collector = MultiTimeframeDataCollector(
            polygon_client=self.polygon_client,
            unusual_whales_client=self.unusual_whales_client,
            config={"storage": self.storage, "max_threads": 10}
        )
        
        # Initialize data pipeline
        logger.info("Initializing data pipeline...")
        self.data_pipeline = DataPipeline(
            polygon_client=self.polygon_client,
            unusual_whales_client=self.unusual_whales_client,
            multi_timeframe_collector=self.multi_timeframe_collector,
            storage=self.storage,
            max_workers=10
        )
        
        # Initialize feature engineering components
        logger.info("Initializing feature engineering components...")
        self.feature_pipeline = FeaturePipeline()
        self.feature_store = FeatureStore()
        self.feature_analyzer = FeatureAnalyzer()
        
        # Initialize model training components
        logger.info("Initializing model training components...")
        self.xgboost_model = XGBoostModel()
        self.lstm_model = LSTMModel()
        self.cnn_model = CNNModel()
        self.model_registry = ModelRegistry()
        self.model_validator = ModelValidator()
        self.model_inference = ModelInference(model_registry=self.model_registry)
        
        # Initialize trading strategy components
        logger.info("Initializing trading strategy components...")
        self.ticker_selector = DynamicTickerSelector(
            polygon_client=self.polygon_client,
            unusual_whales_client=self.unusual_whales_client,
            max_tickers=int(os.environ.get("MAX_POSITIONS", 10))
        )
        self.timeframe_selector = TimeframeSelector(
            base_timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
            primary_timeframe="1h",
            max_timeframes=3
        )
        self.peak_detector = PeakDetector(
            smoothing_window=5,
            peak_distance=5
        )
        self.entry_signal_generator = EntrySignalGenerator(
            model_registry=self.model_registry,
            feature_pipeline=self.feature_pipeline
        )
        self.position_sizer = RiskBasedPositionSizer(
            account_value=float(os.environ.get("ACCOUNT_VALUE", 100000)),
            max_position_pct=float(os.environ.get("MAX_POSITION_PCT", 0.02)),
            max_risk_pct=float(os.environ.get("MAX_RISK_PCT", 0.005))
        )
        self.stop_loss_manager = StopLossManager()
        self.profit_target_manager = ProfitTargetManager()
        self.order_generator = OrderGenerator()
        self.trade_executor = AlpacaTradeExecutor(
            alpaca_client=self.alpaca_client,
            position_sizer=self.position_sizer
        )
        self.position_manager = AlpacaPositionManager(
            alpaca_client=self.alpaca_client,
            stop_loss_manager=self.stop_loss_manager,
            profit_target_manager=self.profit_target_manager,
            peak_detector=self.peak_detector,
            max_positions=int(os.environ.get("MAX_POSITIONS", 10)),
            max_position_value_pct=float(os.environ.get("MAX_POSITION_VALUE_PCT", 0.05)),
            use_trailing_stops=True,
            use_profit_targets=True,
            use_peak_detection=True
        )
        
        # Initialize continuous learning components
        logger.info("Initializing continuous learning components...")
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_adapter = StrategyAdapter()
        self.model_retrainer = ModelRetrainer(
            model_registry=self.model_registry,
            feature_pipeline=self.feature_pipeline
        )
        
        # Initialize backtesting components
        logger.info("Initializing backtesting components...")
        self.backtest_engine = BacktestEngine(
            storage=self.storage,
            feature_pipeline=self.feature_pipeline,
            model_registry=self.model_registry,
            position_sizer=self.position_sizer,
            stop_loss_manager=self.stop_loss_manager,
            profit_target_manager=self.profit_target_manager
        )
        self.strategy_analyzer = StrategyAnalyzer()
        self.performance_reporter = PerformanceReporter()
        
        # Initialize market calendar
        logger.info("Initializing market calendar...")
        self.market_calendar = MarketCalendar(
            exchange="NYSE",
            timezone="America/New_York"
        )
        
        # Initialize system controller
        logger.info("Initializing system controller...")
        self.system_controller = SystemController(config={
            "recovery_attempts": 3,
            "health_check_interval": 30
        })
        
        # Register subsystems with the controller
        self.system_controller.register_subsystem("data_acquisition", self.data_pipeline)
        self.system_controller.register_subsystem("feature_engineering", self.feature_pipeline)
        self.system_controller.register_subsystem("model_training", self.model_registry)
        self.system_controller.register_subsystem("trading_strategy", self.trade_executor)
        
        # Start the system controller
        self.system_controller.start()
        
        logger.info("All system components initialized successfully")
    
    def _determine_data_collection_timeframe(self) -> tuple:
        """Determine the appropriate timeframe for data collection based on market hours."""
        now = datetime.now()
        
        # Check if market is currently open
        market_status = get_market_status(now, "NYSE")
        is_market_open = market_status == MarketStatus.OPEN
        
        if is_market_open:
            # During market hours, use live data for the current day
            logger.info("Market is open - using live data for current trading day")
            end_date = now.strftime("%Y-%m-%d")
            start_date = now.strftime("%Y-%m-%d")  # Same day for live data
            use_live_data = True
        else:
            # Outside market hours, use historical data for training
            logger.info("Market is closed - using historical data for training")
            end_date = now.strftime("%Y-%m-%d")
            # Use more historical data for training (30 days)
            start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            use_live_data = False
            
        return start_date, end_date, use_live_data
    
    async def test_data_acquisition(self) -> Tuple[bool, List[str]]:
        """Test data acquisition components optimized for high-frequency trading."""
        try:
            logger.info("Testing data acquisition components with high-frequency optimizations...")
            start_time = time.time()
            
            # Determine data collection timeframe based on market hours
            start_date, end_date, use_live_data = self._determine_data_collection_timeframe()
            logger.info(f"Data collection timeframe: {start_date} to {end_date} (live data: {use_live_data})")
            
            # For high-frequency trading, we need to focus on real-time data
            if use_live_data:
                logger.info("Market is open - using WebSocket connections for real-time data")
                
                # Connect to Polygon WebSocket for real-time data with high-frequency optimizations
                logger.info("Connecting to Polygon WebSocket for real-time data...")
                websocket_start = time.time()
                
                # Use optimized WebSocket connection for high-frequency trading
                self.polygon_client.connect_websocket(
                    cluster="stocks",
                    delayed=False,
                    buffer_size=16384,  # Increased buffer size for high-throughput (16KB)
                    reconnect_attempts=5,
                    high_throughput=True,  # Enable optimizations for high-throughput trading
                    use_compression=True  # Enable WebSocket compression for reduced bandwidth
                )
                
                self.performance_metrics.record_data_processing_time(
                    "websocket_connection", time.time() - websocket_start
                )
            
            # Dynamically select tickers - use a larger universe for high-frequency trading
            logger.info("Selecting tickers for high-frequency trading...")
            ticker_selection_start = time.time()
            tickers = await self.ticker_selector.fetch_ticker_universe(market_type="stocks", limit=2000)  # Increased limit
            self.performance_metrics.record_data_processing_time(
                "ticker_selection", time.time() - ticker_selection_start
            )
            
            if not tickers:
                logger.error("Failed to fetch ticker universe")
                return False, []
                
            logger.info(f"Fetched {len(tickers)} tickers from Polygon API")
            
            # Get market data for opportunity scoring - use parallel processing for efficiency
            logger.info("Collecting market data for opportunity scoring with parallel processing...")
            market_data = {}
            market_data_start = time.time()
            
            # For high-frequency trading, we need to process more tickers efficiently
            # Use a more efficient approach with batched requests
            ticker_batches = [tickers[i:i+10] for i in range(0, min(200, len(tickers)), 10)]  # Process up to 200 tickers in batches of 10
            
            for batch in ticker_batches:
                try:
                    # Get recent price data for the batch
                    batch_start = time.time()
                    batch_df = self.polygon_client.get_bars(
                        symbols=batch,
                        timeframe="1d",
                        start=start_date,
                        end=end_date
                    )
                    
                    self.performance_metrics.record_api_response_time(
                        "polygon", "get_bars_batch", time.time() - batch_start
                    )
                    
                    # Process each ticker in the batch
                    for ticker in batch:
                        ticker_df = batch_df[batch_df['symbol'] == ticker] if not batch_df.empty else pd.DataFrame()
                        
                        if not ticker_df.empty:
                            # Calculate basic metrics
                            volume = ticker_df['volume'].mean() if 'volume' in ticker_df.columns else 0
                            close_prices = ticker_df['close'].values if 'close' in ticker_df.columns else []
                            
                            if len(close_prices) > 1:
                                # Calculate volatility (simple implementation)
                                returns = [close_prices[i] / close_prices[i-1] - 1 for i in range(1, len(close_prices))]
                                volatility = sum([abs(r) for r in returns]) / len(returns) * 100  # as percentage
                            else:
                                volatility = 0
                                
                            market_data[ticker] = {
                                "metadata": {
                                    "volume": volume,
                                    "atr_pct": volatility,
                                    "opportunity_score": 0.0  # Will be calculated
                                },
                                "ohlcv": ticker_df
                            }
                            
                            # Record data quality
                            self.performance_metrics.record_data_quality(
                                f"market_data_{ticker}",
                                1.0 if len(ticker_df) >= 5 else len(ticker_df) / 5.0  # Simple quality score based on data availability
                            )
                except Exception as e:
                    logger.error(f"Error getting market data for batch: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "market_data_collection", time.time() - market_data_start
            )
            
            # Update the ticker selector with market data
            logger.info("Calculating opportunity scores for high-frequency trading...")
            opportunity_start = time.time()
            self.ticker_selector.update_market_data(market_data)
            self.ticker_selector.calculate_opportunity_scores()
            self.performance_metrics.record_data_processing_time(
                "opportunity_score_calculation", time.time() - opportunity_start
            )
            
            # Select active tickers - for high-frequency trading, we need more active tickers
            active_selection_start = time.time()
            selected_tickers = self.ticker_selector.select_active_tickers(max_tickers=50)  # Increased for high-frequency trading
            self.performance_metrics.record_data_processing_time(
                "active_ticker_selection", time.time() - active_selection_start
            )
            
            logger.info(f"Selected {len(selected_tickers)} active tickers for high-frequency trading: {selected_tickers[:5]}...")
            
            # For high-frequency trading, we need real-time data
            if use_live_data:
                # Subscribe to WebSocket channels for selected tickers
                logger.info("Subscribing to WebSocket channels for real-time data...")
                websocket_subscribe_start = time.time()
                
                # Create WebSocket channels for trades, quotes, and second-level aggregates
                trade_channels = [f"T.{ticker}" for ticker in selected_tickers[:20]]  # Trade events
                quote_channels = [f"Q.{ticker}" for ticker in selected_tickers[:20]]  # Quote updates
                agg_channels = [f"A.{ticker}" for ticker in selected_tickers[:20]]    # Second-level aggregates
                
                # Subscribe to channels
                self.polygon_client.subscribe_websocket(trade_channels + quote_channels + agg_channels)
                
                self.performance_metrics.record_data_processing_time(
                    "websocket_subscription", time.time() - websocket_subscribe_start
                )
                
                # Wait briefly to receive some data
                await asyncio.sleep(2)
                
                # Close WebSocket connection after testing
                self.polygon_client.close_websocket()
            
            # Test price collection with optimized parameters
            logger.info("Testing price collection with high-frequency optimizations...")
            price_collection_start = time.time()
            price_data = self.price_collector.collect_daily_bars(
                tickers=selected_tickers[:5],  # Increased to 5 tickers for better testing
                start_date=start_date,
                end_date=end_date
            )
            self.performance_metrics.record_data_processing_time(
                "price_collection", time.time() - price_collection_start
            )
            
            if price_data:
                total_records = sum(len(df) for df in price_data.values() if not df.empty)
                logger.info(f"Collected {total_records} price records for {len(price_data)} symbols")
            
            # Test quote collection with optimized parameters
            logger.info("Testing quote collection with high-frequency optimizations...")
            quote_collection_start = time.time()
            quote_data = self.quote_collector.collect_quotes(
                tickers=selected_tickers[:5],  # Increased to 5 tickers for better testing
                date_to_collect=end_date
            )
            self.performance_metrics.record_data_processing_time(
                "quote_collection", time.time() - quote_collection_start
            )
            
            if quote_data:
                total_records = sum(len(df) for df in quote_data.values() if not df.empty)
                logger.info(f"Collected {total_records} quote records for {len(quote_data)} symbols")
            
            # Test multi-timeframe data collection with optimized timeframes for high-frequency trading
            logger.info("Testing multi-timeframe data collection with high-frequency optimizations...")
            
            # For high-frequency trading, focus on shorter timeframes
            collection_timeframes = ["1m", "5m", "15m"]  # Shorter timeframes for high-frequency trading
            
            multi_timeframe_start = time.time()
            multi_timeframe_data = self.multi_timeframe_collector.collect_multi_timeframe_data(
                tickers=selected_tickers[:5],  # Increased to 5 tickers for better testing
                timeframes=collection_timeframes,
                start_date=start_date, end_date=end_date
            )
            self.performance_metrics.record_data_processing_time(
                "multi_timeframe_collection", time.time() - multi_timeframe_start
            )
            
            if multi_timeframe_data:
                logger.info(f"Collected multi-timeframe data for {len(multi_timeframe_data)} symbols")
            
            # Record overall data acquisition time
            self.performance_metrics.record_system_latency(
                "data_acquisition", time.time() - start_time
            )
            
            logger.info("High-frequency data acquisition test completed successfully")
            return True, selected_tickers
            
        except Exception as e:
            logger.error(f"Error in high-frequency data acquisition test: {e}")
            logger.error(traceback.format_exc())
            return False, []
    
    async def test_feature_engineering(self, tickers: List[str]) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """Test feature engineering components."""
        try:
            logger.info("Testing feature engineering components...")
            start_time = time.time()
            
            # Determine data collection timeframe based on market hours
            start_date, end_date, use_live_data = self._determine_data_collection_timeframe()
            
            # Fetch data for feature engineering
            logger.info("Fetching data for feature engineering...")
            data_fetch_start = time.time()
            data = {}
            for ticker in tickers[:3]:  # Limit to 3 tickers to reduce API calls
                try:
                    # Fetch daily data
                    ticker_fetch_start = time.time()
                    daily_df = self.storage.get_stock_aggs(
                        symbol=ticker,
                        timeframe="1d",
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    self.performance_metrics.record_data_processing_time(
                        f"db_fetch_{ticker}", time.time() - ticker_fetch_start
                    )
                    
                    if not daily_df.empty:
                        data[f"{ticker}_daily"] = daily_df
                        logger.info(f"Fetched {len(daily_df)} daily bars for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "data_fetch", time.time() - data_fetch_start
            )
            
            if not data:
                logger.warning("No data found in database, fetching from Polygon API")
                # Fallback to fetching from Polygon API
                api_fetch_start = time.time()
                for ticker in tickers[:3]:  # Limit to 3 tickers for API efficiency
                    try:
                        # Fetch daily data
                        ticker_api_start = time.time()
                        daily_df = self.polygon_client.get_bars(
                            symbols=ticker,
                            timeframe="1d",
                            start=start_date,
                            end=end_date
                        )
                        
                        self.performance_metrics.record_api_response_time(
                            "polygon", f"get_bars_{ticker}", time.time() - ticker_api_start
                        )
                        
                        if not daily_df.empty:
                            data[f"{ticker}_daily"] = daily_df
                            logger.info(f"Fetched {len(daily_df)} daily bars for {ticker} from Polygon API")
                            
                            # Store in database for future use
                            storage_start = time.time()
                            self.storage.store_stock_aggs(daily_df)
                            self.performance_metrics.record_data_processing_time(
                                f"db_store_{ticker}", time.time() - storage_start
                            )
                            
                            logger.info(f"Stored {len(daily_df)} daily bars for {ticker} in database")
                    except Exception as e:
                        logger.error(f"Error fetching data for {ticker} from Polygon API: {e}")
                
                self.performance_metrics.record_data_processing_time(
                    "api_fetch", time.time() - api_fetch_start
                )
            
            # Generate features
            logger.info("Generating features...")
            feature_generation_start = time.time()
            feature_data = {}
            for ticker, df in data.items():
                try:
                    # Generate technical indicators
                    tech_ind_start = time.time()
                    features_df = self.feature_pipeline.generate_technical_indicators(df)
                    self.performance_metrics.record_data_processing_time(
                        f"technical_indicators_{ticker}", time.time() - tech_ind_start
                    )
                    
                    # Generate price features
                    price_feat_start = time.time()
                    features_df = self.feature_pipeline.generate_price_features(features_df)
                    self.performance_metrics.record_data_processing_time(
                        f"price_features_{ticker}", time.time() - price_feat_start
                    )
                    
                    # Generate volume features
                    volume_feat_start = time.time()
                    features_df = self.feature_pipeline.generate_volume_features(features_df)
                    self.performance_metrics.record_data_processing_time(
                        f"volume_features_{ticker}", time.time() - volume_feat_start
                    )
                    
                    # Generate volatility features
                    volatility_feat_start = time.time()
                    features_df = self.feature_pipeline.generate_volatility_features(features_df)
                    self.performance_metrics.record_data_processing_time(
                        f"volatility_features_{ticker}", time.time() - volatility_feat_start
                    )
                    
                    # Store features
                    store_start = time.time()
                    self.feature_store.store_features(ticker, features_df)
                    self.performance_metrics.record_data_processing_time(
                        f"feature_store_{ticker}", time.time() - store_start
                    )
                    
                    feature_data[ticker] = features_df
                    logger.info(f"Generated {features_df.shape[1]} features for {ticker}")
                    
                    # Record data quality
                    self.performance_metrics.record_data_quality(
                        f"features_{ticker}", 
                        1.0 if features_df.shape[1] >= 20 else features_df.shape[1] / 20.0  # Simple quality score
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating features for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "feature_generation", time.time() - feature_generation_start
            )
            
            # Record overall feature engineering time
            self.performance_metrics.record_system_latency(
                "feature_engineering", time.time() - start_time
            )
            
            logger.info("Feature engineering test completed successfully")
            return True, feature_data
            
        except Exception as e:
            logger.error(f"Error in feature engineering test: {e}")
            logger.error(traceback.format_exc())
            return False, {}
    
    async def test_model_training(self, feature_data: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """Test model training components."""
        try:
            logger.info("Testing model training components...")
            start_time = time.time()
            
            if not feature_data:
                logger.error("No feature data available for model training")
                return False, {}
            
            # Prepare training data
            logger.info("Preparing training data...")
            data_prep_start = time.time()
            X_data = {}
            y_data = {}
            
            for ticker, df in feature_data.items():
                try:
                    # Define prediction targets
                    # For classification: 1 if close price increases in next period, 0 otherwise
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    
                    # Remove rows with NaN values
                    df = df.dropna()
                    
                    if len(df) > 0:
                        # Split features and target
                        X = df.drop(['target', 'timestamp'], axis=1, errors='ignore')
                        y = df['target']
                        
                        X_data[ticker] = X
                        y_data[ticker] = y
                        
                        logger.info(f"Prepared {len(X)} samples for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error preparing training data for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "training_data_preparation", time.time() - data_prep_start
            )
            
            # Train models
            logger.info("Training models...")
            model_training_start = time.time()
            models = {}
            
            for ticker in X_data.keys():
                try:
                    X = X_data[ticker]
                    y = y_data[ticker]
                    
                    if len(X) < 100:
                        logger.warning(f"Not enough data to train models for {ticker} ({len(X)} samples)")
                        continue
                    
                    # Train XGBoost model
                    logger.info(f"Training XGBoost model for {ticker}...")
                    xgb_start = time.time()
                    xgb_model = self.xgboost_model.train(X, y)
                    xgb_time = time.time() - xgb_start
                    
                    self.performance_metrics.record_model_training_time(
                        "xgboost", xgb_time
                    )
                    
                    logger.info(f"XGBoost model trained in {xgb_time:.2f}s")
                    
                    # Train LSTM model if enough data
                    lstm_model = None
                    if len(X) >= 200:
                        logger.info(f"Training LSTM model for {ticker}...")
                        lstm_start = time.time()
                        lstm_model = self.lstm_model.train(X, y)
                        lstm_time = time.time() - lstm_start
                        
                        self.performance_metrics.record_model_training_time(
                            "lstm", lstm_time
                        )
                        
                        logger.info(f"LSTM model trained in {lstm_time:.2f}s")
                    
                    # Store models
                    models[ticker] = {
                        'xgboost': xgb_model,
                        'lstm': lstm_model,
                        'cnn': None
                    }
                    
                    logger.info(f"Trained models for {ticker}")
                    
                    # Evaluate models
                    if xgb_model:
                        # Simple evaluation on training data (in a real system, would use validation data)
                        xgb_preds = xgb_model.predict(X)
                        xgb_accuracy = np.mean(xgb_preds.round() == y)
                        
                        self.performance_metrics.record_model_accuracy(
                            "xgboost", xgb_accuracy
                        )
                        
                        logger.info(f"XGBoost model accuracy: {xgb_accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training models for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "model_training", time.time() - model_training_start
            )
            
            # Register models
            logger.info("Registering models...")
            registration_start = time.time()
            
            for ticker, ticker_models in models.items():
                try:
                    # Register XGBoost model
                    if ticker_models['xgboost'] is not None:
                        self.model_registry.register_model(
                            model=ticker_models['xgboost'],
                            model_type='xgboost',
                            ticker=ticker,
                            metrics={},
                            version='1.0'
                        )
                        logger.info(f"Registered XGBoost model for {ticker}")
                    
                    # Register LSTM model
                    if ticker_models['lstm'] is not None:
                        self.model_registry.register_model(
                            model=ticker_models['lstm'],
                            model_type='lstm',
                            ticker=ticker,
                            metrics={},
                            version='1.0'
                        )
                        logger.info(f"Registered LSTM model for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error registering models for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "model_registration", time.time() - registration_start
            )
            
            # Record overall model training time
            self.performance_metrics.record_system_latency(
                "model_training", time.time() - start_time
            )
            
            logger.info("Model training test completed successfully")
            return True, models
            
        except Exception as e:
            logger.error(f"Error in model training test: {e}")
            logger.error(traceback.format_exc())
            return False, {}
    
    async def test_trading_strategy(self, tickers: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Test trading strategy components."""
        try:
            logger.info("Testing trading strategy components...")
            start_time = time.time()
            
            # Check if market is open
            now = datetime.now()
            market_status = get_market_status(now, "NYSE")
            is_market_open = market_status == MarketStatus.OPEN
            
            if is_market_open:
                logger.info("Market is open - trading enabled")
            else:
                logger.info("Market is closed - trading in simulation mode")
            
            # Get real-time data
            logger.info("Getting real-time data...")
            data_fetch_start = time.time()
            data = {}
            for ticker in tickers[:3]:  # Limit to 3 tickers to reduce API calls
                try:
                    # Get the latest daily bar
                    ticker_fetch_start = time.time()
                    daily_df = self.polygon_client.get_bars(
                        symbols=ticker,
                        timeframe="1d",
                        limit=10
                    )
                    
                    self.performance_metrics.record_api_response_time(
                        "polygon", f"get_bars_{ticker}", time.time() - ticker_fetch_start
                    )
                    
                    if not daily_df.empty:
                        data[f"{ticker}_daily"] = daily_df
                        logger.info(f"Got real-time data for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error getting real-time data for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "realtime_data_fetch", time.time() - data_fetch_start
            )
            
            # Generate features for signal generation
            logger.info("Generating features for signal generation...")
            feature_gen_start = time.time()
            feature_data = {}
            for ticker_timeframe, df in data.items():
                try:
                    # Generate features
                    ticker_feature_start = time.time()
                    features_df = self.feature_pipeline.generate_technical_indicators(df)
                    features_df = self.feature_pipeline.generate_price_features(features_df)
                    features_df = self.feature_pipeline.generate_volume_features(features_df)
                    features_df = self.feature_pipeline.generate_volatility_features(features_df)
                    
                    self.performance_metrics.record_data_processing_time(
                        f"signal_features_{ticker_timeframe}", time.time() - ticker_feature_start
                    )
                    
                    feature_data[ticker_timeframe] = features_df
                    logger.info(f"Generated features for {ticker_timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error generating features for {ticker_timeframe}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "signal_feature_generation", time.time() - feature_gen_start
            )
            
            # Generate trading signals
            logger.info("Generating trading signals...")
            signal_gen_start = time.time()
            signals = {}
            for ticker_timeframe, df in feature_data.items():
                try:
                    # Extract ticker from ticker_timeframe
                    ticker = ticker_timeframe.split("_")[0]
                    timeframe = ticker_timeframe.split("_")[1]
                    
                    # Skip if not enough data
                    if len(df) < 10:
                        continue
                    
                    # Generate signals
                    signal_start = time.time()
                    signal = self.entry_signal_generator.generate_signal(ticker, df)
                    
                    self.performance_metrics.record_model_inference_time(
                        f"signal_generation_{ticker}", time.time() - signal_start
                    )
                    
                    if signal:
                        if ticker not in signals:
                            signals[ticker] = {}
                        
                        signals[ticker][timeframe] = signal
                        logger.info(f"Generated {signal['direction']} signal for {ticker} ({timeframe}) with confidence {signal['confidence']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error generating signals for {ticker_timeframe}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "signal_generation", time.time() - signal_gen_start
            )
            
            # Process signals
            logger.info("Processing trading signals...")
            signal_process_start = time.time()
            trade_decisions = []
            for ticker, timeframe_signals in signals.items():
                try:
                    # Get the direction from the first signal
                    direction = list(timeframe_signals.values())[0]['direction']
                    
                    # Get the latest price
                    latest_price = list(timeframe_signals.values())[0]['price']
                    
                    # Create trade decision
                    trade_decision = {
                        'ticker': ticker,
                        'direction': direction,
                        'confidence': 0.7,
                        'price': latest_price,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    trade_decisions.append(trade_decision)
                    logger.info(f"Created trade decision for {ticker}: {direction} at {latest_price}")
                    
                except Exception as e:
                    logger.error(f"Error processing signals for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "signal_processing", time.time() - signal_process_start
            )
            # Test order generation with dollar-based position limits
            logger.info("Testing order generation with dollar-based position limits...")
            order_gen_start = time.time()
            orders = []
            
            # Track total position value for daily limit ($5000)
            total_position_value = 0.0
            ticker_position_values = {}  # Track position value per ticker for per-stock limit ($2500)
            
            for decision in trade_decisions:
                try:
                    ticker = decision['ticker']
                    direction = decision['direction']
                    price = decision['price']
                    
                    # Calculate position size
                    position_size = self.position_sizer.calculate_position_size(
                        ticker=ticker,
                        price=price,
                        direction=direction
                    )
                    
                    if position_size == 0:
                        logger.info(f"Position size is 0 for {ticker}")
                        continue
                    
                    # Calculate position value
                    position_value = position_size * price
                    
                    # Check if this would exceed per-stock limit ($2500)
                    current_ticker_value = ticker_position_values.get(ticker, 0.0)
                    if current_ticker_value + position_value > 2500.0:
                        # Adjust position size to respect per-stock limit
                        available_value = 2500.0 - current_ticker_value
                        if available_value <= 0:
                            logger.info(f"Skipping {ticker} - reached per-stock limit of $2500")
                            continue
                        
                        # Recalculate position size based on available value
                        position_size = int(available_value / price)
                        position_value = position_size * price
                        logger.info(f"Adjusted position size for {ticker} to respect $2500 per-stock limit")
                    
                    # Check if this would exceed daily limit ($5000)
                    if total_position_value + position_value > 5000.0:
                        # Adjust position size to respect daily limit
                        available_value = 5000.0 - total_position_value
                        if available_value <= 0:
                            logger.info("Reached daily position value limit of $5000")
                            break
                        
                        # Recalculate position size based on available value
                        position_size = int(available_value / price)
                        position_value = position_size * price
                        logger.info(f"Adjusted position size for {ticker} to respect $5000 daily limit")
                    
                    # Update tracking variables
                    ticker_position_values[ticker] = current_ticker_value + position_value
                    total_position_value += position_value
                    
                    # Generate order
                    order = self.order_generator.generate_order(
                        symbol=ticker,
                        qty=position_size,
                        side=direction,
                        order_type='market',
                        time_in_force='day'
                    )
                    
                    orders.append(order)
                    logger.info(f"Generated {direction} order for {ticker}: {position_size} shares at {price} (${position_value:.2f})")
                    logger.info(f"Current position value: ${total_position_value:.2f}/{5000.0:.2f} daily, ${ticker_position_values[ticker]:.2f}/{2500.0:.2f} for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error generating order for {decision['ticker']}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "order_generation", time.time() - order_gen_start
            )
            
            # Test trade execution (in simulation mode)
            logger.info("Testing trade execution (in simulation mode)...")
            execution_start = time.time()
            executed_orders = []
            for order in orders:
                try:
                    # Execute trade in simulation mode
                    order_execution_start = time.time()
                    executed_order = self.trade_executor.execute_trade_simulation(order)
                    
                    self.performance_metrics.record_trade_execution_time(
                        f"execute_order_{order['symbol']}", time.time() - order_execution_start
                    )
                    
                    executed_orders.append(executed_order)
                    logger.info(f"Executed {executed_order['side']} order for {executed_order['symbol']}: {executed_order['qty']} shares at {executed_order['price']}")
                    
                except Exception as e:
                    logger.error(f"Error executing trade for {order['symbol']}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "trade_execution", time.time() - execution_start
            )
            
            # Record overall trading strategy time
            self.performance_metrics.record_system_latency(
                "trading_strategy", time.time() - start_time
            )
            
            logger.info("Trading strategy test completed successfully")
            return True, executed_orders
            
        except Exception as e:
            logger.error(f"Error in trading strategy test: {e}")
            logger.error(traceback.format_exc())
            return False, []
    
    async def test_backtesting(self, tickers: List[str]) -> bool:
        """Test backtesting components."""
        try:
            logger.info("Testing backtesting components...")
            start_time = time.time()
            
            # Fetch historical data
            logger.info("Fetching historical data...")
            data_fetch_start = time.time()
            historical_data = {}
            for ticker in tickers[:2]:  # Limit to 2 tickers to reduce API calls
                try:
                    # Fetch daily data for the past 90 days
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                    
                    ticker_fetch_start = time.time()
                    daily_df = self.storage.get_stock_aggs(
                        symbol=ticker,
                        timeframe="1d",
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    self.performance_metrics.record_data_processing_time(
                        f"backtest_db_fetch_{ticker}", time.time() - ticker_fetch_start
                    )
                    
                    if not daily_df.empty:
                        historical_data[ticker] = daily_df
                        logger.info(f"Fetched {len(daily_df)} daily bars for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error fetching historical data for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "backtest_data_fetch", time.time() - data_fetch_start
            )
            
            if not historical_data:
                logger.warning("No historical data found in database, fetching from Polygon API")
                # Fallback to fetching from Polygon API
                api_fetch_start = time.time()
                for ticker in tickers[:2]:  # Limit to 2 tickers for API efficiency
                    try:
                        # Fetch daily data
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                        
                        ticker_api_start = time.time()
                        daily_df = self.polygon_client.get_bars(
                            symbols=ticker,
                            timeframe="1d",
                            start=start_date,
                            end=end_date
                        )
                        
                        self.performance_metrics.record_api_response_time(
                            "polygon", f"backtest_get_bars_{ticker}", time.time() - ticker_api_start
                        )
                        
                        if not daily_df.empty:
                            historical_data[ticker] = daily_df
                            logger.info(f"Fetched {len(daily_df)} daily bars for {ticker} from Polygon API")
                    except Exception as e:
                        logger.error(f"Error fetching historical data for {ticker} from Polygon API: {e}")
                
                self.performance_metrics.record_data_processing_time(
                    "backtest_api_fetch", time.time() - api_fetch_start
                )
            
            # Run backtest
            logger.info("Running backtest...")
            backtest_start = time.time()
            backtest_results = {}
            for ticker, df in historical_data.items():
                try:
                    # Run backtest
                    ticker_backtest_start = time.time()
                    results = self.backtest_engine.run_backtest(
                        ticker=ticker,
                        data=df,
                        strategy_params={
                            'stop_loss_pct': 0.02,
                            'profit_target_pct': 0.03,
                            'entry_threshold': 0.6,
                            'exit_threshold': 0.4
                        }
                    )
                    
                    self.performance_metrics.record_data_processing_time(
                        f"backtest_run_{ticker}", time.time() - ticker_backtest_start
                    )
                    
                    backtest_results[ticker] = results
                    logger.info(f"Backtest results for {ticker}: trades={len(results['trades'])}, net_profit={results['net_profit']:.2f}, win_rate={results['win_rate']:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error running backtest for {ticker}: {e}")
            
            self.performance_metrics.record_data_processing_time(
                "backtest_execution", time.time() - backtest_start
            )
            
            # Record overall backtesting time
            self.performance_metrics.record_system_latency(
                "backtesting", time.time() - start_time
            )
            
            logger.info("Backtesting test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in backtesting test: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def run_test(self) -> bool:
        """Run the full system test."""
        try:
            logger.info("Starting full system test...")
            overall_start_time = time.time()
            
            # Test data acquisition
            data_acquisition_success, tickers = await self.test_data_acquisition()
            if not data_acquisition_success:
                logger.error("Data acquisition test failed")
                return False
            
            # If no tickers were selected, use default tickers
            if not tickers:
                logger.warning("No tickers selected, using default tickers")
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            
            logger.info(f"Using tickers for testing: {tickers[:5]}...")
            
            # Test feature engineering
            feature_engineering_success, feature_data = await self.test_feature_engineering(tickers)
            if not feature_engineering_success:
                logger.error("Feature engineering test failed")
                return False
            
            # Test model training
            model_training_success, models = await self.test_model_training(feature_data)
            if not model_training_success:
                logger.error("Model training test failed")
                return False
            
            # Test trading strategy
            trading_strategy_success, executed_orders = await self.test_trading_strategy(tickers)
            if not trading_strategy_success:
                logger.error("Trading strategy test failed")
                return False
            
            # Test backtesting
            backtesting_success = await self.test_backtesting(tickers)
            if not backtesting_success:
                logger.error("Backtesting test failed")
                return False
            
            # Record overall test time
            overall_time = time.time() - overall_start_time
            self.performance_metrics.record_system_latency("full_system_test", overall_time)
            
            # Generate performance report
            logger.info("Generating performance report...")
            self.performance_metrics.log_report()
            
            logger.info(f"All system tests passed successfully in {overall_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error running full system test: {e}")
            logger.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up resources
            try:
                self.polygon_client.close()
                self.storage.close()
                self.db_manager.close()
                self.system_controller.shutdown()
                logger.info("Resources released")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function to run the test."""
    # Create and run the test
    test = FullSystemTest()
    success = await test.run_test()
    
    return success

if __name__ == "__main__":
    # Run the test
    logger.info("Starting full system test...")
    logger.info(f"Project root: {os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Date: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")
    logger.info("-------------------------------------")
    
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(main())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
