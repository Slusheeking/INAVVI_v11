#!/usr/bin/env python3
"""
TensorFlow Serving REST API for arm64 architecture
This script provides a REST API compatible with TensorFlow Serving's REST API
but implemented in pure Python for arm64 compatibility.
"""

import os
import json
import logging
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import base64
import asyncio
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tf_serving_rest_api')

# Model registry path
MODEL_BASE_PATH = os.environ.get('MODEL_REGISTRY_PATH', '/app/models/registry')
MODEL_NAME = 'trading_model'

# Create FastAPI app
app = FastAPI(title="TensorFlow Serving REST API",
              description="REST API for TensorFlow models on arm64 architecture",
              version="1.0.0")

# Global model cache
model_cache = {}


def load_model(model_name: str, version: Optional[str] = None):
    """Load a model from the registry"""
    try:
        # Find all model versions in the registry
        if not os.path.exists(MODEL_BASE_PATH):
            logger.warning(f'Model base path {MODEL_BASE_PATH} does not exist')
            return None

        versions = [d for d in os.listdir(MODEL_BASE_PATH) if os.path.isdir(
            os.path.join(MODEL_BASE_PATH, d))]
        if not versions:
            logger.warning(f'No model versions found in {MODEL_BASE_PATH}')
            return None

        # Determine which version to load
        if version is None or version == "latest":
            # Load the latest version
            latest_version = max([int(v)
                                 for v in versions if v.isdigit()], default=0)
            if latest_version == 0:
                logger.warning('No valid model versions found')
                return None
            version_to_load = str(latest_version)
        else:
            # Load the specified version
            if version not in versions:
                logger.warning(f'Version {version} not found')
                return None
            version_to_load = version

        model_path = os.path.join(MODEL_BASE_PATH, version_to_load)
        logger.info(f'Loading model from {model_path}')

        # Load the model
        model = tf.saved_model.load(model_path)

        # Cache the model
        model_cache[f"{model_name}/{version_to_load}"] = {
            'version': version_to_load,
            'model': model,
            'signatures': model.signatures,
            'loaded_at': time.time()
        }

        logger.info(
            f'Successfully loaded model {model_name} version {version_to_load}')
        return model_cache[f"{model_name}/{version_to_load}"]
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        return None


@app.get("/v1/models/{model_name}")
async def get_model_status(model_name: str):
    """Get model status"""
    # Check if model exists in cache
    model_versions = [
        k.split('/')[1] for k in model_cache.keys() if k.startswith(f"{model_name}/")]

    if not model_versions:
        # Try to load the latest version
        model_info = load_model(model_name)
        if model_info is None:
            raise HTTPException(
                status_code=404, detail=f"Model {model_name} not found")
        model_versions = [model_info['version']]

    return {
        "model_version_status": [
            {
                "version": version,
                "state": "AVAILABLE",
                "status": {"error_code": "OK", "error_message": ""}
            }
            for version in model_versions
        ]
    }


@app.get("/v1/models/{model_name}/versions/{version}")
async def get_model_version_status(model_name: str, version: str):
    """Get model version status"""
    # Check if model exists in cache
    cache_key = f"{model_name}/{version}"
    if cache_key not in model_cache:
        # Try to load the specified version
        model_info = load_model(model_name, version)
        if model_info is None:
            raise HTTPException(
                status_code=404, detail=f"Model {model_name} version {version} not found")

    return {
        "model_version_status": [
            {
                "version": version,
                "state": "AVAILABLE",
                "status": {"error_code": "OK", "error_message": ""}
            }
        ]
    }


@app.post("/v1/models/{model_name}:predict")
async def predict(model_name: str, request: Request):
    """Make a prediction using the latest model version"""
    return await predict_version(model_name, "latest", request)


@app.post("/v1/models/{model_name}/versions/{version}:predict")
async def predict_version(model_name: str, version: str, request: Request):
    """Make a prediction using a specific model version"""
    try:
        # Parse request body
        body = await request.json()

        # Get model from cache or load it
        cache_key = f"{model_name}/{version}"
        if cache_key not in model_cache:
            model_info = load_model(model_name, version)
            if model_info is None:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_name} version {version} not found")
        else:
            model_info = model_cache[cache_key]

        model = model_info['model']
        signatures = model_info['signatures']

        # Get the serving signature
        serving_signature = signatures.get(
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        if serving_signature is None:
            raise HTTPException(
                status_code=500, detail="Model does not have a serving signature")

        # Convert inputs from JSON to Tensors
        inputs = {}
        for key, value in body.get("inputs", {}).items():
            # Handle different input formats (lists, nested lists, base64 encoded, etc.)
            if isinstance(value, dict) and "b64" in value:
                # Handle base64 encoded data
                decoded = base64.b64decode(value["b64"])
                if "dtype" in value and value["dtype"] == "DT_FLOAT":
                    # Convert to float32 tensor
                    tensor = tf.convert_to_tensor(
                        np.frombuffer(decoded, dtype=np.float32))
                else:
                    # Default to float32
                    tensor = tf.convert_to_tensor(
                        np.frombuffer(decoded, dtype=np.float32))

                # Reshape if shape is provided
                if "shape" in value:
                    tensor = tf.reshape(tensor, value["shape"])
            else:
                # Handle regular lists/arrays
                tensor = tf.convert_to_tensor(value)

            inputs[key] = tensor

        # Run prediction
        outputs = serving_signature(**inputs)

        # Convert outputs to JSON-serializable format
        response = {"outputs": {}}
        for key, tensor in outputs.items():
            # Convert tensor to numpy and then to list
            value = tensor.numpy()
            if isinstance(value, np.ndarray):
                response["outputs"][key] = value.tolist()
            else:
                response["outputs"][key] = value

        # Add model metadata
        response["model_spec"] = {
            "name": model_name,
            "version": model_info['version'],
            "signature_name": "serving_default"
        }

        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    # Try to load the default model
    load_model(MODEL_NAME)


def start_server():
    """Start the server"""
    uvicorn.run(app, host="0.0.0.0", port=8501)


if __name__ == "__main__":
    start_server()
