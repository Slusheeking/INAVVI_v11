"""
Model Services

This module combines both model training and continuous learning functionality,
providing integrated model lifecycle management.
"""

import os
import logging
import threading
import time
import signal
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.utils.logging import configure_logger
from src.utils.config import load_config
from src.utils.api.service_registry_extension import ServiceRegistryExtension
from src.model_training.pipeline import ModelPipeline
from src.continuous_learning.pipeline.continuous_learning_pipeline import ContinuousLearningPipeline
from src.model_training.models import ModelFactory
from src.continuous_learning.adaptation.strategy_adapter import StrategyAdapter
from src.continuous_learning.retraining.model_retrainer import ModelRetrainer
from src.utils.api.resilient_client import ResilientServiceClient

# Configure logging
logger = configure_logger("model_services")

# Create FastAPI app
app = FastAPI(title="Model Services", version="1.0.0")

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
model_pipeline = None
continuous_learning_pipeline = None
model_factory = None
model_retrainer = None
strategy_adapter = None
data_client = None
running = True
training_thread = None
retraining_thread = None

# Track running processes
training_processes = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the model services on startup."""
    global model_pipeline, continuous_learning_pipeline, config, model_factory
    global model_retrainer, strategy_adapter, data_client

    try:
        # Load configuration
        config = load_config(os.path.join("config", "system_config.yaml"))
        if not config:
            logger.error("Failed to load configuration")
            return

        # Initialize client for data processing service
        data_client = ResilientServiceClient(
            service_name="data-processing",
            base_url="http://ats-data-processing:8001",
            use_service_registry=True
        )

        # Initialize model factory
        model_factory = ModelFactory()

        # Initialize model pipeline
        model_pipeline = ModelPipeline(
            config=config,
            model_factory=model_factory
        )

        # Initialize model retrainer
        model_retrainer = ModelRetrainer(
            model_pipeline=model_pipeline,
            config=config
        )

        # Initialize strategy adapter
        strategy_adapter = StrategyAdapter(
            config=config
        )

        # Initialize continuous learning pipeline
        continuous_learning_pipeline = ContinuousLearningPipeline(
            model_retrainer=model_retrainer,
            strategy_adapter=strategy_adapter,
            config=config
        )

        logger.info("Model services initialized successfully")

        # Register with service registry
        registry_extension = ServiceRegistryExtension(
            app=app,
            service_name="model-services",
            port=8003
        )

        # Start training and retraining threads
        start_training_thread()
        start_retraining_thread()

    except Exception as e:
        logger.error(f"Error initializing model services: {e}")
        raise


def start_training_thread():
    """Start the model training thread."""
    global training_thread

    if training_thread is not None and training_thread.is_alive():
        logger.info("Training thread is already running")
        return

    training_thread = threading.Thread(target=training_loop)
    training_thread.daemon = True
    training_thread.start()

    logger.info("Started model training thread")


def start_retraining_thread():
    """Start the model retraining thread."""
    global retraining_thread

    if retraining_thread is not None and retraining_thread.is_alive():
        logger.info("Retraining thread is already running")
        return

    retraining_thread = threading.Thread(target=retraining_loop)
    retraining_thread.daemon = True
    retraining_thread.start()

    logger.info("Started model retraining thread")


def training_loop():
    """Main model training loop."""
    global running, model_pipeline, config

    logger.info("Starting model training loop")

    while running:
        try:
            # Get training parameters
            symbols = config.get("model_training", {}).get(
                "symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
            model_type = config.get("model_training", {}).get(
                "model_type", "lstm")
            timeframes = config.get("model_training", {}).get(
                "timeframes", ["1d"])
            training_days = config.get(
                "model_training", {}).get("training_days", 365)

            # Run model training
            logger.info(
                f"Running model training for {len(symbols)} symbols with {model_type} model")
            training_id = f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store in processes dict to track progress
            training_processes[training_id] = {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "symbols": symbols,
                "model_type": model_type,
                "progress": 0.0
            }

            try:
                result = model_pipeline.train_model(
                    symbols=symbols,
                    model_type=model_type,
                    timeframes=timeframes,
                    training_days=training_days,
                    validation_split=0.2,
                    progress_callback=lambda p: update_training_progress(
                        training_id, p)
                )

                training_processes[training_id].update({
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "model_id": result.get("model_id"),
                    "performance": result.get("performance_metrics"),
                    "progress": 1.0
                })

                logger.info(f"Model training completed: {result}")

            except Exception as e:
                training_processes[training_id].update({
                    "status": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": str(e)
                })
                logger.error(f"Error in model training: {e}")

            # Sleep until next training
            training_interval = config.get("model_training", {}).get(
                "training_interval_seconds", 86400)  # Default: daily
            logger.info(
                f"Sleeping for {training_interval} seconds until next training")

            # Sleep in small increments to allow for graceful shutdown
            sleep_increment = 30
            for _ in range(training_interval // sleep_increment):
                if not running:
                    break
                time.sleep(sleep_increment)

            # Sleep any remaining time
            remaining_time = training_interval % sleep_increment
            if remaining_time > 0 and running:
                time.sleep(remaining_time)

        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            # Sleep for a short time before retrying
            time.sleep(60)


def update_training_progress(training_id: str, progress: float):
    """Update the progress of a training job."""
    if training_id in training_processes:
        training_processes[training_id]["progress"] = progress


def retraining_loop():
    """Main model retraining loop."""
    global running, continuous_learning_pipeline, config

    logger.info("Starting model retraining loop")

    while running:
        try:
            # Check if retraining is needed
            retraining_needed = continuous_learning_pipeline.check_retraining_needed()

            if retraining_needed:
                logger.info("Retraining needed, triggering retraining process")
                result = continuous_learning_pipeline.run_retraining_cycle()
                logger.info(f"Retraining completed: {result}")

            # Sleep until next check
            check_interval = config.get("continuous_learning", {}).get(
                "check_interval_seconds", 3600)  # Default: hourly
            logger.info(
                f"Sleeping for {check_interval} seconds until next retraining check")

            # Sleep in small increments to allow for graceful shutdown
            sleep_increment = 30
            for _ in range(check_interval // sleep_increment):
                if not running:
                    break
                time.sleep(sleep_increment)

            # Sleep any remaining time
            remaining_time = check_interval % sleep_increment
            if remaining_time > 0 and running:
                time.sleep(remaining_time)

        except Exception as e:
            logger.error(f"Error in retraining loop: {e}")
            # Sleep for a short time before retrying
            time.sleep(60)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_pipeline is None or continuous_learning_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model_pipeline": "healthy",
            "continuous_learning": "healthy"
        }
    }


@app.get("/status")
async def get_status():
    """Get the current status of the model services."""
    if model_pipeline is None or continuous_learning_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if running else "stopped",
        "training_thread_active": training_thread is not None and training_thread.is_alive(),
        "retraining_thread_active": retraining_thread is not None and retraining_thread.is_alive(),
        "active_training_processes": len([p for p in training_processes.values() if p.get("status") == "running"]),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/train")
async def trigger_training(
    symbols: List[str],
    model_type: str,
    timeframes: Optional[List[str]] = None,
    training_days: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """Trigger a model training job."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline not initialized")

    try:
        # Use provided parameters or defaults from config
        timeframes = timeframes or config.get(
            "model_training", {}).get("timeframes", ["1d"])
        training_days = training_days or config.get(
            "model_training", {}).get("training_days", 365)

        # Generate unique training ID
        training_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store in processes dict to track progress
        training_processes[training_id] = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "symbols": symbols,
            "model_type": model_type,
            "progress": 0.0
        }

        # Define training function for background task
        def run_training():
            try:
                result = model_pipeline.train_model(
                    symbols=symbols,
                    model_type=model_type,
                    timeframes=timeframes,
                    training_days=training_days,
                    validation_split=0.2,
                    progress_callback=lambda p: update_training_progress(
                        training_id, p)
                )

                training_processes[training_id].update({
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "model_id": result.get("model_id"),
                    "performance": result.get("performance_metrics"),
                    "progress": 1.0
                })

                logger.info(f"Manual training completed: {result}")
            except Exception as e:
                training_processes[training_id].update({
                    "status": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": str(e)
                })
                logger.error(f"Error in manual training: {e}")

        # Run training in background
        background_tasks.add_task(run_training)

        return {
            "status": "training_started",
            "training_id": training_id,
            "symbols": symbols,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/{training_id}")
async def get_training_status(training_id: str):
    """Get the status of a training job."""
    if training_id not in training_processes:
        raise HTTPException(
            status_code=404, detail=f"Training job {training_id} not found")

    return training_processes[training_id]


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks = None):
    """Trigger a model retraining cycle."""
    if continuous_learning_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Continuous learning pipeline not initialized")

    try:
        # Define retraining function for background task
        def run_retraining():
            try:
                result = continuous_learning_pipeline.run_retraining_cycle()
                logger.info(f"Manual retraining completed: {result}")
                return result
            except Exception as e:
                logger.error(f"Error in manual retraining: {e}")
                raise

        # Run retraining in background
        background_tasks.add_task(run_retraining)

        return {
            "status": "retraining_started",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all available models."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline not initialized")

    try:
        models = model_pipeline.list_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline not initialized")

    try:
        model_info = model_pipeline.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not found")
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a specific model."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline not initialized")

    try:
        success = model_pipeline.delete_model(model_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not found or could not be deleted")
        return {"status": "deleted", "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global running, model_pipeline, continuous_learning_pipeline

    logger.info("Shutting down model services")

    # Stop background threads
    running = False

    # Shutdown processes
    # Kill TensorFlow Serving if running
    try:
        tensorflow_serving_pid = os.environ.get("TENSORFLOW_SERVING_PID")
        if tensorflow_serving_pid:
            os.kill(int(tensorflow_serving_pid), signal.SIGTERM)
            logger.info(
                f"Terminated TensorFlow Serving (PID {tensorflow_serving_pid})")
    except Exception as e:
        logger.error(f"Error terminating TensorFlow Serving: {e}")

    # Shutdown pipelines
    if model_pipeline is not None:
        model_pipeline.shutdown()

    if continuous_learning_pipeline is not None:
        continuous_learning_pipeline.shutdown()

    logger.info("Model services shutdown complete")


def main():
    """Main entry point for the model services."""
    try:
        # Start the FastAPI server
        uvicorn.run(
            "src.model_services.main:app",
            host="0.0.0.0",
            port=8003,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting model services: {e}")
        raise


if __name__ == "__main__":
    main()
