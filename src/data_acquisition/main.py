"""
Data Acquisition Service

This module provides the main entry point for the data acquisition service.
It initializes the data pipeline and exposes a health check endpoint.
"""

import os
import time
import signal
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logging import configure_logger
from src.utils.config import load_config
from src.data_acquisition.pipeline.data_pipeline import DataPipeline
from src.data_acquisition.api.polygon_client import PolygonClient
from src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from src.data_acquisition.storage.timescale_storage import TimescaleStorage

# Configure logging
logger = configure_logger("data_acquisition")

# Create FastAPI app
app = FastAPI(title="Data Acquisition Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
data_pipeline = None
config = None
running = True
collection_thread = None

@app.on_event("startup")
async def startup_event():
    """Initialize the data acquisition service on startup."""
    global data_pipeline, config
    
    try:
        # Load configuration
        config = load_config(os.path.join("config", "system_config.yaml"))
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Initialize clients
        polygon_client = PolygonClient(api_key=os.environ.get("POLYGON_API_KEY"))
        unusual_whales_client = UnusualWhalesClient(api_key=os.environ.get("UNUSUAL_WHALES_API_KEY"))
        
        # Initialize storage
        storage = TimescaleStorage()
        
        # Initialize data pipeline
        data_pipeline = DataPipeline(
            polygon_client=polygon_client,
            unusual_whales_client=unusual_whales_client,
            storage=storage,
            max_workers=config.get("data_acquisition", {}).get("max_workers", 10)
        )
        
        logger.info("Data acquisition service initialized successfully")
        
        # Start data collection thread
        start_collection_thread()
        
    except Exception as e:
        logger.error(f"Error initializing data acquisition service: {e}")
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

def collection_loop():
    """Main data collection loop."""
    global running, data_pipeline, config
    
    logger.info("Starting data collection loop")
    
    while running:
        try:
            # Get symbols to collect
            symbols = config.get("data_acquisition", {}).get("symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
            
            # Get collection parameters
            timeframes = config.get("data_acquisition", {}).get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
            include_quotes = config.get("data_acquisition", {}).get("include_quotes", True)
            include_trades = config.get("data_acquisition", {}).get("include_trades", False)
            include_options_flow = config.get("data_acquisition", {}).get("include_options_flow", True)
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
            collection_interval = config.get("data_acquisition", {}).get("collection_interval_seconds", 3600)
            logger.info(f"Sleeping for {collection_interval} seconds until next collection")
            
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if data_pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def get_status():
    """Get the current status of the data acquisition service."""
    if data_pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    return {
        "status": "running" if running else "stopped",
        "collection_thread_active": collection_thread is not None and collection_thread.is_alive(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/collect")
async def trigger_collection(symbols: List[str] = None):
    """Trigger a data collection run."""
    if data_pipeline is None:
        raise HTTPException(status_code=503, detail="Data pipeline not initialized")
    
    try:
        # Use provided symbols or default from config
        if symbols is None:
            symbols = config.get("data_acquisition", {}).get("symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
        
        # Get collection parameters from config
        timeframes = config.get("data_acquisition", {}).get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
        include_quotes = config.get("data_acquisition", {}).get("include_quotes", True)
        include_trades = config.get("data_acquisition", {}).get("include_trades", False)
        include_options_flow = config.get("data_acquisition", {}).get("include_options_flow", True)
        days_back = config.get("data_acquisition", {}).get("days_back", 1)
        
        # Run collection in a separate thread
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
        
        collection_thread = threading.Thread(target=run_collection)
        collection_thread.daemon = True
        collection_thread.start()
        
        return {"status": "collection_started", "symbols": symbols, "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error triggering collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global running, data_pipeline
    
    logger.info("Shutting down data acquisition service")
    
    # Stop collection thread
    running = False
    
    # Shutdown data pipeline
    if data_pipeline is not None:
        data_pipeline.shutdown()
    
    logger.info("Data acquisition service shutdown complete")

def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    global running
    
    logger.info("Received SIGTERM signal, shutting down")
    running = False

# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    """Main entry point for the data acquisition service."""
    try:
        # Start the FastAPI server
        uvicorn.run(
            "src.data_acquisition.main:app",
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting data acquisition service: {e}")
        raise

if __name__ == "__main__":
    main()