"""
Data Processing Service

This module combines both data acquisition and feature engineering functionality,
eliminating the network hop between these closely related components.
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logging import configure_logger
from src.utils.config import load_config
from src.utils.api.service_registry_extension import ServiceRegistryExtension
from src.data_acquisition.pipeline.data_pipeline import DataPipeline
from src.feature_engineering.pipeline.feature_pipeline import FeaturePipeline
from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from src.data_acquisition.storage.timescale_storage import TimescaleStorage
from src.feature_engineering.store.feature_store import FeatureStore

# Configure logging
logger = configure_logger("data_processing")

# Create FastAPI app
app = FastAPI(title="Data Processing Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
data_pipeline = None
feature_pipeline = None
storage = None
feature_store = None
running = True
collection_thread = None
feature_generation_thread = None


@app.on_event("startup")
async def startup_event():
    """Initialize the data processing service on startup."""
    global data_pipeline, feature_pipeline, config, storage, feature_store

    try:
        # Load configuration
        config = load_config(os.path.join("config", "system_config.yaml"))
        if not config:
            logger.error("Failed to load configuration")
            return

        # Initialize clients
        polygon_client = PolygonClient(
            api_key=os.environ.get("POLYGON_API_KEY"))
        unusual_whales_client = UnusualWhalesClient(
            api_key=os.environ.get("UNUSUAL_WHALES_API_KEY"))

        # Initialize storage
        storage = TimescaleStorage()

        # Initialize feature store
        feature_store = FeatureStore()

        # Initialize data pipeline
        data_pipeline = DataPipeline(
            polygon_client=polygon_client,
            unusual_whales_client=unusual_whales_client,
            storage=storage,
            max_workers=config.get("data_acquisition", {}
                                   ).get("max_workers", 10)
        )

        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline(
            storage=storage,
            feature_store=feature_store,
            max_workers=config.get("feature_engineering",
                                   {}).get("max_workers", 10)
        )

        logger.info("Data processing service initialized successfully")

        # Register with service registry
        registry_extension = ServiceRegistryExtension(
            app=app,
            service_name="data-processing",
            port=8001
        )

        # Start data collection and feature generation threads
        start_collection_thread()
        start_feature_generation_thread()

    except Exception as e:
        logger.error(f"Error initializing data processing service: {e}")
        raise


def start_collection_thread():
    """Start the data collection thread."""
    global collection_thread

    if collection_thread is not None and collection_thread.is_alive():
        logger.info("Collection thread is already running")
        return

    collection_thread = threading.Thread(target=collection_loop)
    collection_thread.daemon = True
    collection_thread.start()

    logger.info("Started data collection thread")


def start_feature_generation_thread():
    """Start the feature generation thread."""
    global feature_generation_thread

    if feature_generation_thread is not None and feature_generation_thread.is_alive():
        logger.info("Feature generation thread is already running")
        return

    feature_generation_thread = threading.Thread(
        target=feature_generation_loop)
    feature_generation_thread.daemon = True
    feature_generation_thread.start()

    logger.info("Started feature generation thread")


def collection_loop():
    """Main data collection loop."""
    global running, data_pipeline, config

    logger.info("Starting data collection loop")

    while running:
        try:
            # Get symbols to collect
            symbols = config.get("data_acquisition", {}).get(
                "symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])

            # Get collection parameters
            timeframes = config.get("data_acquisition", {}).get(
                "timeframes", ["1m", "5m", "15m", "1h", "1d"])
            include_quotes = config.get(
                "data_acquisition", {}).get("include_quotes", True)
            include_trades = config.get(
                "data_acquisition", {}).get("include_trades", False)
            include_options_flow = config.get(
                "data_acquisition", {}).get("include_options_flow", True)
            days_back = config.get("data_acquisition", {}).get("days_back", 1)

            # Run daily collection
            logger.info(f"Running daily collection for {len(symbols)} symbols")
            stats = data_pipeline.run_daily_collection(
                symbols=symbols,
                timeframes=timeframes,
                include_quotes=include_quotes,
                include_trades=include_trades,
                include_options_flow=include_options_flow,
                days_back=days_back
            )

            logger.info(f"Daily collection completed: {stats}")

            # Sleep until next collection
            collection_interval = config.get("data_acquisition", {}).get(
                "collection_interval_seconds", 3600)
            logger.info(
                f"Sleeping for {collection_interval} seconds until next collection")

            # Sleep in small increments to allow for graceful shutdown
            sleep_increment = 10
            for _ in range(collection_interval // sleep_increment):
                if not running:
                    break
                time.sleep(sleep_increment)

            # Sleep any remaining time
            remaining_time = collection_interval % sleep_increment
            if remaining_time > 0 and running:
                time.sleep(remaining_time)

        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
            # Sleep for a short time before retrying
            time.sleep(60)


def feature_generation_loop():
    """Main feature generation loop."""
    global running, feature_pipeline, config

    logger.info("Starting feature generation loop")

    while running:
        try:
            # Get symbols for feature generation
            symbols = config.get("feature_engineering", {}).get(
                "symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])

            # Get feature generation parameters
            timeframes = config.get("feature_engineering", {}).get(
                "timeframes", ["1m", "5m", "15m", "1h", "1d"])
            lookback_periods = config.get("feature_engineering", {}).get(
                "lookback_periods", [5, 10, 20, 60])
            feature_sets = config.get("feature_engineering", {}).get(
                "feature_sets", ["technical", "statistical", "volume", "volatility"])

            # Run feature generation
            logger.info(
                f"Running feature generation for {len(symbols)} symbols")
            stats = feature_pipeline.generate_features(
                symbols=symbols,
                timeframes=timeframes,
                lookback_periods=lookback_periods,
                feature_sets=feature_sets
            )

            logger.info(f"Feature generation completed: {stats}")

            # Sleep until next feature generation
            generation_interval = config.get("feature_engineering", {}).get(
                "generation_interval_seconds", 900)
            logger.info(
                f"Sleeping for {generation_interval} seconds until next feature generation")

            # Sleep in small increments to allow for graceful shutdown
            sleep_increment = 10
            for _ in range(generation_interval // sleep_increment):
                if not running:
                    break
                time.sleep(sleep_increment)

            # Sleep any remaining time
            remaining_time = generation_interval % sleep_increment
            if remaining_time > 0 and running:
                time.sleep(remaining_time)

        except Exception as e:
            logger.error(f"Error in feature generation loop: {e}")
            # Sleep for a short time before retrying
            time.sleep(60)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if data_pipeline is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_pipeline": "healthy",
            "feature_pipeline": "healthy"
        }
    }


@app.get("/status")
async def get_status():
    """Get the current status of the data processing service."""
    if data_pipeline is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if running else "stopped",
        "collection_thread_active": collection_thread is not None and collection_thread.is_alive(),
        "feature_generation_thread_active": feature_generation_thread is not None and feature_generation_thread.is_alive(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/collect")
async def trigger_collection(symbols: Optional[List[str]] = None, background_tasks: BackgroundTasks = None):
    """Trigger a data collection run."""
    if data_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Data pipeline not initialized")

    try:
        # Use provided symbols or default from config
        if symbols is None:
            symbols = config.get("data_acquisition", {}).get(
                "symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])

        # Get collection parameters from config
        timeframes = config.get("data_acquisition", {}).get(
            "timeframes", ["1m", "5m", "15m", "1h", "1d"])
        include_quotes = config.get(
            "data_acquisition", {}).get("include_quotes", True)
        include_trades = config.get(
            "data_acquisition", {}).get("include_trades", False)
        include_options_flow = config.get(
            "data_acquisition", {}).get("include_options_flow", True)
        days_back = config.get("data_acquisition", {}).get("days_back", 1)

        # Run collection in a background task
        def run_collection():
            try:
                stats = data_pipeline.run_daily_collection(
                    symbols=symbols,
                    timeframes=timeframes,
                    include_quotes=include_quotes,
                    include_trades=include_trades,
                    include_options_flow=include_options_flow,
                    days_back=days_back
                )
                logger.info(f"Manual collection completed: {stats}")
            except Exception as e:
                logger.error(f"Error in manual collection: {e}")

        background_tasks.add_task(run_collection)

        return {"status": "collection_started", "symbols": symbols, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error triggering collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_features")
async def trigger_feature_generation(symbols: Optional[List[str]] = None, background_tasks: BackgroundTasks = None):
    """Trigger a feature generation run."""
    if feature_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Feature pipeline not initialized")

    try:
        # Use provided symbols or default from config
        if symbols is None:
            symbols = config.get("feature_engineering", {}).get(
                "symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])

        # Get feature generation parameters from config
        timeframes = config.get("feature_engineering", {}).get(
            "timeframes", ["1m", "5m", "15m", "1h", "1d"])
        lookback_periods = config.get("feature_engineering", {}).get(
            "lookback_periods", [5, 10, 20, 60])
        feature_sets = config.get("feature_engineering", {}).get(
            "feature_sets", ["technical", "statistical", "volume", "volatility"])

        # Run feature generation in a background task
        def run_feature_generation():
            try:
                stats = feature_pipeline.generate_features(
                    symbols=symbols,
                    timeframes=timeframes,
                    lookback_periods=lookback_periods,
                    feature_sets=feature_sets
                )
                logger.info(f"Manual feature generation completed: {stats}")
            except Exception as e:
                logger.error(f"Error in manual feature generation: {e}")

        background_tasks.add_task(run_feature_generation)

        return {"status": "feature_generation_started", "symbols": symbols, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error triggering feature generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{symbol}")
async def get_features(symbol: str, timeframe: str = "1d", limit: int = 100):
    """Get features for a symbol."""
    if feature_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Feature pipeline not initialized")

    try:
        features = feature_pipeline.get_features(symbol, timeframe, limit)
        return features
    except Exception as e:
        logger.error(f"Error retrieving features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global running, data_pipeline, feature_pipeline

    logger.info("Shutting down data processing service")

    # Stop background threads
    running = False

    # Shutdown pipelines
    if data_pipeline is not None:
        data_pipeline.shutdown()

    if feature_pipeline is not None:
        feature_pipeline.shutdown()

    logger.info("Data processing service shutdown complete")


def main():
    """Main entry point for the data processing service."""
    try:
        # Start the FastAPI server
        uvicorn.run(
            "src.data_processing.main:app",
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting data processing service: {e}")
        raise


if __name__ == "__main__":
    main()
