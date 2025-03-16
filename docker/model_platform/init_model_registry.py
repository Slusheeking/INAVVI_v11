#!/usr/bin/env python3
"""
Initialize the model registry with a placeholder model
"""
import os
import tensorflow as tf
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('init_model_registry')

# Model registry path
MODEL_REGISTRY_PATH = os.environ.get(
    'MODEL_REGISTRY_PATH', '/app/models/registry')
MODEL_VERSION = "1"
MODEL_NAME = "trading_model"


def create_simple_model():
    """Create a simple TensorFlow model for testing"""
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="mse")

        # Generate some random data and train for one step
        x = np.random.random((10, 5))
        y = np.random.random((10, 1))
        model.fit(x, y, epochs=1, verbose=0)

        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return None


def save_model():
    """Save a simple model to the registry"""
    try:
        # Create model directory
        model_path = os.path.join(MODEL_REGISTRY_PATH, MODEL_VERSION)
        os.makedirs(model_path, exist_ok=True)

        # Create and save the model
        model = create_simple_model()
        if model is None:
            logger.error("Failed to create model")
            return False

        # Save the model
        tf.saved_model.save(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Create a metadata file
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write(
                '{"name": "trading_model", "version": "1", "type": "test_model"}')

        logger.info(f"Model metadata saved to {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


if __name__ == "__main__":
    # Set matplotlib config dir to avoid permission issues
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

    # Check if model registry exists
    if not os.path.exists(MODEL_REGISTRY_PATH):
        logger.info(
            f"Creating model registry directory: {MODEL_REGISTRY_PATH}")
        os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)

    # Check if model already exists
    model_version_path = os.path.join(MODEL_REGISTRY_PATH, MODEL_VERSION)
    if os.path.exists(model_version_path):
        logger.info(
            f"Model version {MODEL_VERSION} already exists at {model_version_path}")
        sys.exit(0)

    # Save the model
    if save_model():
        logger.info("Model registry initialized successfully")
        sys.exit(0)
    else:
        logger.error("Failed to initialize model registry")
        sys.exit(1)
