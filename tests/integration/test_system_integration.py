#!/usr/bin/env python3
"""
Production Integration Test for Trading System

This test verifies that the core components of the trading system work together
correctly with real API endpoints (no mocks). It tests:

1. API Clients (Polygon.io REST, Polygon.io WebSocket, Unusual Whales)
2. Stock Selection Engine
3. Data Pipeline

Prerequisites:
- Valid API keys in environment variables
- Redis server running
- Internet connection for API access

This test is designed to run in a production-like environment.
"""

from stock_selection_engine import GPUStockSelectionSystem
from data_pipeline import DataPipeline
from api_clients import PolygonRESTClient, PolygonWebSocketClient, UnusualWhalesClient
import asyncio
import datetime
import logging
import os
import sys
import time
from datetime import timedelta

import pandas as pd
import pytest
import redis

# Add project root to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Import components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("integration_test")


class TestSystemIntegration:
    """Integration tests for the trading system with real API endpoints"""

    @pytest.fixture(scope="class")
    async def setup_components(self, api_keys, redis_client):
        """Set up all system components with real connections"""
        logger.info("Setting up system components for integration testing")

        # Skip test if API keys are not provided
        if not api_keys["polygon"]:
            pytest.skip("POLYGON_API_KEY not set in environment variables")

        # Initialize Polygon REST client
        polygon_rest = PolygonRESTClient(
            api_key=api_keys["polygon"],
            redis_client=redis_client,
            use_gpu=True,
        )

        # Initialize Polygon WebSocket client
        polygon_ws = PolygonWebSocketClient(
            api_key=api_keys["polygon"],
            redis_client=redis_client,
            use_gpu=True,
        )

        # Initialize Unusual Whales client if API key is available
        unusual_whales = None
        if api_keys["unusual_whales"]:
            unusual_whales = UnusualWhalesClient(
                api_key=api_keys["unusual_whales"],
                redis_client=redis_client,
                use_gpu=True,
            )

        # Initialize Stock Selection Engine
        stock_selection = GPUStockSelectionSystem(
            redis_client=redis_client,
            polygon_api_client=polygon_rest,
            polygon_websocket_client=polygon_ws,
            unusual_whales_client=unusual_whales,
        )

        # Initialize Data Pipeline
        data_pipeline = DataPipeline(
            polygon_client=polygon_rest,
            polygon_ws=polygon_ws,
            unusual_whales_client=unusual_whales,
            redis_client=redis_client,
            use_gpu=True,
        )

        # Start WebSocket client
        polygon_ws.start()

        # Wait for WebSocket connection to establish
        await asyncio.sleep(5)

        # Yield components for testing
        yield {
            "polygon_rest": polygon_rest,
            "polygon_ws": polygon_ws,
            "unusual_whales": unusual_whales,
            "stock_selection": stock_selection,
            "data_pipeline": data_pipeline,
        }

        # Cleanup
        logger.info("Cleaning up system components")
        await polygon_rest.close()
        polygon_ws.stop()
        if unusual_whales:
            await unusual_whales.close()
        await stock_selection.stop()

    @pytest.mark.asyncio
    async def test_polygon_rest_api(self, setup_components):
        """Test Polygon REST API client with real endpoints"""
        logger.info("Testing Polygon REST API client")
        polygon_rest = setup_components["polygon_rest"]

        # Test market status
        market_status = await polygon_rest.get_market_status()
        assert market_status is not None
        assert isinstance(market_status, dict)
        assert "market" in market_status
        logger.info(f"Market status: {market_status['market']}")

        # Test ticker details
        ticker = "AAPL"
        details = await polygon_rest.get_ticker_details(ticker)
        assert details is not None
        assert "ticker" in details
        assert details["ticker"] == ticker
        assert "name" in details
        logger.info(f"Ticker details for {ticker}: {details['name']}")

        # Test aggregates
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=5)
        aggs = await polygon_rest.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )
        assert isinstance(aggs, pd.DataFrame)
        assert not aggs.empty
        assert "open" in aggs.columns
        assert "close" in aggs.columns
        assert "high" in aggs.columns
        assert "low" in aggs.columns
        assert "volume" in aggs.columns
        logger.info(f"Retrieved {len(aggs)} aggregates for {ticker}")

        # Test batch aggregates
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        batch_aggs = await polygon_rest.get_aggregates_batch(
            tickers=test_tickers,
            multiplier=1,
            timespan="day",
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )
        assert isinstance(batch_aggs, dict)
        assert len(batch_aggs) > 0
        for ticker, df in batch_aggs.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
        logger.info(
            f"Retrieved batch aggregates for {len(batch_aggs)} tickers")

        # Test GPU processing
        processed_data = await polygon_rest.process_data_with_gpu(batch_aggs)
        assert isinstance(processed_data, dict)
        assert len(processed_data) > 0
        logger.info(
            f"Processed data with GPU for {len(processed_data)} tickers")

    @pytest.mark.asyncio
    async def test_polygon_websocket(self, setup_components):
        """Test Polygon WebSocket client with real endpoints"""
        logger.info("Testing Polygon WebSocket client")
        polygon_ws = setup_components["polygon_ws"]

        # Verify connection
        assert polygon_ws.is_connected()
        logger.info("WebSocket connection established")

        # Test message handling
        received_messages = []

        def message_handler(message):
            received_messages.append(message)

        # Add message handler
        polygon_ws.add_message_handler("T", message_handler)

        # Subscribe to trades for a test ticker
        polygon_ws.subscribe(["T.AAPL"])

        # Wait for some messages (market might be closed, so this could be 0)
        await asyncio.sleep(10)

        # Unsubscribe
        polygon_ws.unsubscribe(["T.AAPL"])

        # Remove message handler
        polygon_ws.remove_message_handler("T", message_handler)

        # Log results
        logger.info(f"Received {len(received_messages)} WebSocket messages")

    @pytest.mark.asyncio
    async def test_unusual_whales_api(self, setup_components):
        """Test Unusual Whales API client with real endpoints"""
        logger.info("Testing Unusual Whales API client")
        unusual_whales = setup_components["unusual_whales"]

        # Skip test if Unusual Whales client is not available
        if unusual_whales is None:
            pytest.skip(
                "UNUSUAL_WHALES_API_KEY not set in environment variables")

        # Test API health
        health = await unusual_whales.check_api_health()
        assert health is True
        logger.info("Unusual Whales API health check passed")

        # Test flow alerts
        ticker = "AAPL"
        alerts = await unusual_whales.get_flow_alerts(ticker=ticker, limit=10)
        assert isinstance(alerts, dict)
        logger.info(f"Retrieved flow alerts for {ticker}")

        # Test alerts endpoint
        alerts_data = await unusual_whales.get_alerts(limit=10)
        assert isinstance(alerts_data, dict)
        logger.info("Retrieved alerts data")

        # Test alert configurations
        configs = await unusual_whales.get_alert_configurations()
        assert isinstance(configs, dict)
        logger.info("Retrieved alert configurations")

    @pytest.mark.asyncio
    async def test_stock_selection_engine(self, setup_components):
        """Test Stock Selection Engine with real data"""
        logger.info("Testing Stock Selection Engine")
        stock_selection = setup_components["stock_selection"]

        # Start the stock selection engine
        await stock_selection.start()
        logger.info("Stock selection engine started")

        # Wait for initial universe to build
        await asyncio.sleep(5)

        # Verify universe is built
        assert len(stock_selection.full_universe) > 0
        logger.info(f"Universe size: {len(stock_selection.full_universe)}")

        # Verify watchlist is created
        assert len(stock_selection.active_watchlist) > 0
        logger.info(f"Watchlist size: {len(stock_selection.active_watchlist)}")

        # Verify focused list is created
        assert len(stock_selection.focused_list) > 0
        logger.info(f"Focused list size: {len(stock_selection.focused_list)}")

        # Test GPU acceleration
        if stock_selection.gpu_available:
            mem_info = stock_selection.gh200_accelerator.get_memory_info()
            assert "error" not in mem_info
            logger.info(f"GPU memory usage: {mem_info['used_gb']:.2f}GB")

        # Stop the stock selection engine
        await stock_selection.stop()
        logger.info("Stock selection engine stopped")

    @pytest.mark.asyncio
    async def test_data_pipeline(self, setup_components):
        """Test Data Pipeline with real data"""
        logger.info("Testing Data Pipeline")
        data_pipeline = setup_components["data_pipeline"]

        # Test price data loading
        test_tickers = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=5)

        price_data = data_pipeline.load_price_data(
            tickers=test_tickers,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
        )
        assert isinstance(price_data, dict)
        assert len(price_data) > 0
        for ticker, df in price_data.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
        logger.info(f"Loaded price data for {len(price_data)} tickers")

        # Test market data loading
        market_data = data_pipeline.load_market_data(
            start_date=start_date,
            end_date=end_date,
            symbols=["SPY", "QQQ"],
        )
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        logger.info(f"Loaded market data with {len(market_data)} rows")

        # Test technical indicators calculation
        for ticker, df in price_data.items():
            indicators_df = data_pipeline.calculate_technical_indicators(df)
            assert "sma5" in indicators_df.columns
            assert "rsi" in indicators_df.columns
            assert "macd" in indicators_df.columns
            logger.info(f"Calculated technical indicators for {ticker}")
            break  # Test just one ticker

        # Test target generation
        for ticker, df in price_data.items():
            targets_df = data_pipeline.generate_targets(df)
            assert "signal_target" in targets_df.columns
            assert "future_return_10min" in targets_df.columns
            logger.info(f"Generated targets for {ticker}")
            break  # Test just one ticker

        # Test training data preparation
        training_data = data_pipeline.prepare_training_data(
            price_data=price_data,
            market_data=market_data,
        )
        assert isinstance(training_data, pd.DataFrame)
        assert not training_data.empty
        logger.info(f"Prepared training data with {len(training_data)} rows")

        # Test ML data preparation
        X, y = data_pipeline.prepare_signal_detection_data(training_data)
        assert isinstance(X, pd.DataFrame)
        assert not X.empty
        assert len(X) == len(y)
        logger.info(f"Prepared signal detection data with {len(X)} samples")

    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, setup_components):
        """Test end-to-end integration of all components"""
        logger.info("Testing end-to-end integration")
        polygon_rest = setup_components["polygon_rest"]
        stock_selection = setup_components["stock_selection"]
        data_pipeline = setup_components["data_pipeline"]

        # Start the stock selection engine
        await stock_selection.start()
        logger.info("Stock selection engine started")

        # Wait for initial universe to build
        await asyncio.sleep(5)

        # Get focused list from stock selection engine
        focused_tickers = list(stock_selection.focused_list)
        assert len(focused_tickers) > 0
        logger.info(f"Focused tickers: {focused_tickers[:5]}")

        # Use data pipeline to load data for focused tickers
        end_date = datetime.datetime.now()
        start_date = end_date - timedelta(days=5)

        price_data = data_pipeline.load_price_data(
            tickers=focused_tickers[:5],  # Use first 5 tickers for testing
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
        )
        assert isinstance(price_data, dict)
        assert len(price_data) > 0
        logger.info(f"Loaded price data for {len(price_data)} focused tickers")

        # Calculate technical indicators
        for ticker, df in price_data.items():
            indicators_df = data_pipeline.calculate_technical_indicators(df)
            assert "sma5" in indicators_df.columns
            assert "rsi" in indicators_df.columns
            logger.info(f"Calculated technical indicators for {ticker}")

            # Use Polygon REST client to get current price
            current_price = await polygon_rest._get_current_price(ticker)
            assert current_price >= 0
            logger.info(f"Current price for {ticker}: {current_price}")

            break  # Test just one ticker

        # Stop the stock selection engine
        await stock_selection.stop()
        logger.info("Stock selection engine stopped")

        logger.info("End-to-end integration test completed successfully")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
