import os
from src.utils.logging import configure_logger
from src.utils.config import load_config
from src.model_training.registry import ModelRegistry
from src.model_training.pipeline import ModelTrainingPipeline

def main():
    """Main entry point for model training service."""
    # Configure logging
    logger = configure_logger("model_training")

    # Load configuration
    config = load_config(os.path.join("config", "model_training_config.yaml"))
    if not config:
        logger.error("Failed to load configuration")
        return

    # Initialize model registry
    try:
        registry = ModelRegistry(
            registry_dir=config["model_registry"]["registry_dir"],
            create_if_missing=config["model_registry"]["create_if_missing"],
            max_models_per_type=config["model_registry"]["max_models_per_type"],
            backup_dir=config["model_registry"].get("backup_dir"),
            use_port_manager=config["model_registry"].get("use_port_manager", True),
        )
        logger.info("Initialized model registry")
    except Exception as e:
        logger.error(f"Failed to initialize model registry: {e}")
        return

    # Initialize training pipeline
    try:
        pipeline = ModelTrainingPipeline(config=config, registry=registry)
        logger.info("Initialized model training pipeline")
    except Exception as e:
        logger.error(f"Failed to initialize training pipeline: {e}")
        return

    # Start training
    try:
        pipeline.run()
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()