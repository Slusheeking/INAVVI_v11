"""
Feature Engineering Service

This module provides the main entry point for the feature engineering service.
It initializes the feature pipeline and exposes a health check endpoint.
"""

import os
import time
import signal
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from src.utils.logging import configure_logger
from src.utils.config import load_config
from src.feature_engineering.pipeline.feature_pipeline import FeaturePipeline
from src.feature_engineering.store.feature_store import feature_store
from src.utils.database import get_db_manager

# Configure logging
logger = configure_logger("feature_engineering")

# Create FastAPI app
app = FastAPI(title="Feature Engineering Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
feature_pipeline = None
config = None
running = True
processing_thread = None
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the feature engineering service on startup."""
    global feature_pipeline, config, db_manager
    
    try:
        # Load configuration
        config = load_config(os.path.join("config", "feature_engineering_config.yaml"))
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Initialize database manager
        db_manager = get_db_manager()
        
        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline(
            db_manager=db_manager,
            use_redis_cache=config.get("use_redis_cache", False),
            redis_cache_ttl=config.get("redis_cache_ttl", 3600),
            max_workers=config.get("max_workers", 4)
        )
        
        # Initialize the pipeline
        feature_pipeline.initialize()
        
        logger.info("Feature engineering service initialized successfully")
        
        # Start feature processing thread
        start_processing_thread()
        
    except Exception as e:
        logger.error(f"Error initializing feature engineering service: {e}")
        raise

def start_processing_thread():
    """Start the feature processing thread."""
    global processing_thread
    
    if processing_thread is not None and processing_thread.is_alive():
        logger.info("Processing thread is already running")
        return
    
    processing_thread = threading.Thread(target=processing_loop)
    processing_thread.daemon = True
    processing_thread.start()
    
    logger.info("Started feature processing thread")

def processing_loop():
    """Main feature processing loop."""
    global running, feature_pipeline, config
    
    logger.info("Starting feature processing loop")
    
    while running:
        try:
            # Get symbols to process
            symbols = config.get("symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
            
            # Get processing parameters
            timeframes = config.get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
            lookback_days = config.get("lookback_days", 30)
            
            # Process each symbol
            for symbol in symbols:
                try:
                    logger.info(f"Processing features for {symbol}")
                    
                    # Generate multi-timeframe features
                    results = feature_pipeline.generate_multi_timeframe_features(
                        symbol=symbol,
                        timeframes=timeframes,
                        lookback_days=lookback_days,
                        parallel=True,
                        store_features=True,
                        include_target=True
                    )
                    
                    # Log results
                    for timeframe, (features, targets) in results.items():
                        if features is not None:
                            logger.info(f"Generated {features.shape[1]} features for {symbol} {timeframe}")
                        else:
                            logger.warning(f"No features generated for {symbol} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error processing features for {symbol}: {e}")
            
            # Sleep until next processing
            processing_interval = config.get("processing_interval_seconds", 3600)
            logger.info(f"Sleeping for {processing_interval} seconds until next processing")
            
            # Sleep in small increments to allow for graceful shutdown
            sleep_increment = 10
            for _ in range(processing_interval // sleep_increment):
                if not running:
                    break
                time.sleep(sleep_increment)
            
            # Sleep any remaining time
            remaining_time = processing_interval % sleep_increment
            if remaining_time > 0 and running:
                time.sleep(remaining_time)
                
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            # Sleep for a short time before retrying
            time.sleep(60)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Feature pipeline not initialized")
    
    health_status = feature_pipeline.get_health()
    health_status["timestamp"] = datetime.now().isoformat()
    
    return health_status

@app.get("/status")
async def get_status():
    """Get the current status of the feature engineering service."""
    if feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Feature pipeline not initialized")
    
    return {
        "status": feature_pipeline.get_status(),
        "processing_thread_active": processing_thread is not None and processing_thread.is_alive(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_features(
    symbol: str,
    timeframes: Optional[List[str]] = None,
    lookback_days: int = 30,
    background_tasks: BackgroundTasks = None
):
    """Generate features for a symbol."""
    if feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Feature pipeline not initialized")
    
    try:
        # Use provided timeframes or default from config
        if timeframes is None:
            timeframes = config.get("timeframes", ["1m", "5m", "15m", "1h", "1d"])
        
        # Function to run in background
        def generate():
            try:
                results = feature_pipeline.generate_multi_timeframe_features(
                    symbol=symbol,
                    timeframes=timeframes,
                    lookback_days=lookback_days,
                    parallel=True,
                    store_features=True,
                    include_target=True
                )
                
                logger.info(f"Generated features for {symbol} in timeframes {timeframes}")
                
                # Return summary of results
                summary = {}
                for tf, (features, targets) in results.items():
                    if features is not None:
                        summary[tf] = {
                            "features_shape": features.shape,
                            "targets_shape": targets.shape if targets is not None else None
                        }
                    else:
                        summary[tf] = None
                
                return summary
            
            except Exception as e:
                logger.error(f"Error generating features for {symbol}: {e}")
                return {"error": str(e)}
        
        # Run in background if background_tasks is provided
        if background_tasks:
            background_tasks.add_task(generate)
            return {
                "status": "processing_started",
                "symbol": symbol,
                "timeframes": timeframes,
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Run synchronously
            return generate()
    
    except Exception as e:
        logger.error(f"Error triggering feature generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{symbol}")
async def get_features(symbol: str, timeframe: str = "1d", limit: int = 100):
    """Get features for a symbol."""
    if feature_pipeline is None or db_manager is None:
        raise HTTPException(status_code=503, detail="Feature pipeline not initialized")
    
    try:
        # Query features from the database
        query = """
        SELECT timestamp, feature_name, feature_value
        FROM features
        WHERE symbol = :symbol AND timeframe = :timeframe
        ORDER BY timestamp DESC
        LIMIT :limit
        """
        
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        
        df = db_manager.execute_query(query, params)
        
        if df.empty:
            return {"symbol": symbol, "timeframe": timeframe, "features": {}}
        
        # Pivot the data to get features as columns
        pivot_df = df.pivot(index="timestamp", columns="feature_name", values="feature_value")
        
        # Convert to dictionary
        features_dict = {}
        for timestamp, row in pivot_df.iterrows():
            features_dict[timestamp.isoformat()] = row.dropna().to_dict()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "features": features_dict
        }
    
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global running, feature_pipeline
    
    logger.info("Shutting down feature engineering service")
    
    # Stop processing thread
    running = False
    
    # Shutdown feature pipeline
    if feature_pipeline is not None:
        feature_pipeline.shutdown()
    
    logger.info("Feature engineering service shutdown complete")

def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    global running
    
    logger.info("Received SIGTERM signal, shutting down")
    running = False

# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def main():
    """Main entry point for the feature engineering service."""
    try:
        # Start the FastAPI server
        uvicorn.run(
            "src.feature_engineering.main:app",
            host="0.0.0.0",
            port=8004,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting feature engineering service: {e}")
        raise

if __name__ == "__main__":
    main()