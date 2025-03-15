#!/usr/bin/env python3
"""
Model Registry

This module provides a registry for storing and retrieving machine learning models.
It includes versioning, metadata tracking, and model lifecycle management.
"""

import glob
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any
import pandas as pd

from src.utils.logging import get_logger
from src.utils.serialization import (
    save_json,
    load_json,
    serialize_model,
    deserialize_model,
)

# Define dummy functions for port management
def allocate_port(service_name):
    """Dummy function to allocate a port."""
    return 8000

def release_port(port):
    """Dummy function to release a port."""
    pass

# Define a simple model metadata class
class ModelMetadata:
    """Simple model metadata class."""
    def __init__(self, model_id=None, model_type=None, model_version=None, description=None):
        """Initialize model metadata."""
        self.model_id = model_id or f"model_{uuid.uuid4().hex[:8]}"
        self.model_type = model_type or "unknown"
        self.model_version = model_version or "1.0.0"
        self.metadata = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "description": description or "",
            "created_at": datetime.now().isoformat(),
        }
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        self.metadata["parameters"] = parameters
    
    def set_performance(self, performance):
        """Set model performance metrics."""
        self.metadata["performance"] = performance
    
    def get_metadata(self):
        """Get model metadata."""
        return self.metadata
    
    def save(self, directory):
        """Save metadata to file."""
        os.makedirs(directory, exist_ok=True)
        metadata_path = os.path.join(directory, f"{self.model_id}_metadata.json")
        save_json(self.metadata, metadata_path, indent=2)
        return metadata_path
    
    def load(self, metadata_path):
        """Load metadata from file."""
        self.metadata = load_json(metadata_path)
        self.model_id = self.metadata.get("model_id", self.model_id)
        self.model_type = self.metadata.get("model_type", self.model_type)
        self.model_version = self.metadata.get("model_version", self.model_version)

# Configure logging
logger = get_logger("model_registry")


class ModelRegistry:
    """
    Registry for storing and retrieving machine learning models.

    This class provides utilities for storing, retrieving, and managing machine
    learning models, including versioning, metadata tracking, and lifecycle management.
    """

    def __init__(
        self,
        registry_dir: str,
        create_if_missing: bool = True,
        max_models_per_type: int = 10,
        backup_dir: str | None = None,
        port: int = None,
        use_port_manager: bool = True,
    ):
        """
        Initialize the model registry.

        Args:
            registry_dir: Directory for the model registry
            create_if_missing: Whether to create the registry directory if it doesn't exist
            max_models_per_type: Maximum number of models to keep per type
            backup_dir: Directory for model backups
            port: Port to use for the registry service (if None, will be allocated)
            use_port_manager: Whether to use port_manager for port allocation
        """
        self.registry_dir = registry_dir
        self.max_models_per_type = max_models_per_type
        self.backup_dir = backup_dir
        self.use_port_manager = use_port_manager
        self.port = port
        self.allocated_port = None
        
        # No need for a serializer instance as we'll use the utility functions directly

        # Create registry directory if it doesn't exist
        if create_if_missing and not os.path.exists(registry_dir):
            os.makedirs(registry_dir, exist_ok=True)
            logger.info(f"Created model registry directory: {registry_dir}")

        # Create backup directory if specified and it doesn't exist
        if self.backup_dir and not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir, exist_ok=True)
            logger.info(f"Created model backup directory: {self.backup_dir}")

        # Initialize model index
        self.model_index = self._load_model_index()

        # Allocate port if needed
        if self.use_port_manager and self.port is None:
            self.allocated_port = allocate_port("model_registry")
            self.port = self.allocated_port
            logger.info(f"Allocated port {self.port} for model registry")

        logger.info("Initialized model registry at {} on port {}".format(registry_dir, self.port))

    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_files: dict[str, str] | None = None,
        copy_files: bool = True,
        stage: str = "development",
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Model to register
            metadata: Model metadata
            model_files: Dictionary mapping file types to file paths
            copy_files: Whether to copy model files to the registry
            stage: Model lifecycle stage ('development', 'staging', 'production', 'archived')

        Returns:
            Model ID
        """
        model_id = metadata.model_id
        model_type = metadata.model_type
        model_version = metadata.model_version

        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_type, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, f"{model_id}.pkl")
        serialize_model(model, model_path)

        # Save metadata
        metadata_path = metadata.save(model_dir)

        # Copy additional model files if provided
        if model_files and copy_files:
            for file_type, file_path in model_files.items():
                if os.path.exists(file_path):
                    dest_path = os.path.join(model_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    logger.info(
                        f"Copied {file_type} file from {file_path} to {dest_path}"
                    )

        # Update model index
        model_entry = {
            "model_id": model_id,
            "model_type": model_type,
            "model_version": model_version,
            "created_at": metadata.metadata.get(
                "created_at", datetime.now().isoformat()
            ),
            "updated_at": metadata.metadata.get(
                "updated_at", datetime.now().isoformat()
            ),
            "stage": stage,
            "path": model_dir,
            "metadata_path": metadata_path,
            "model_path": model_path,
            "files": model_files or {},
        }

        if model_type not in self.model_index:
            self.model_index[model_type] = []

        # Check if model already exists in index
        existing_model_idx = None
        for i, entry in enumerate(self.model_index[model_type]):
            if entry["model_id"] == model_id:
                existing_model_idx = i
                break

        if existing_model_idx is not None:
            # Update existing entry
            self.model_index[model_type][existing_model_idx] = model_entry
        else:
            # Add new entry
            self.model_index[model_type].append(model_entry)

        # Enforce maximum models per type
        self._enforce_max_models(model_type)

        # Save model index
        self._save_model_index()

        logger.info(
            f"Registered model {model_id} of type {model_type} in {stage} stage"
        )

        return model_id

    def load_model(
        self,
        model_id: str,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
    ) -> tuple[Any, ModelMetadata]:
        """
        Load a model from the registry.

        Args:
            model_id: ID of the model to load
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, loads latest if not specified)
            stage: Stage of the model (optional, loads from any stage if not specified)

        Returns:
            Tuple of (model, metadata)
        """
        # Find model entry
        model_entry = self._find_model_entry(model_id, model_type, version, stage)

        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Load model
        model_path = model_entry["model_path"]
        model = deserialize_model(model_path)

        # Load metadata
        metadata = ModelMetadata()
        metadata.load(model_entry["metadata_path"])

        logger.info(f"Loaded model {model_id} from {model_path}")

        return model, metadata

    def get_metadata(
        self,
        model_id: str,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
    ) -> ModelMetadata:
        """
        Get metadata for a model.

        Args:
            model_id: ID of the model
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, loads latest if not specified)
            stage: Stage of the model (optional, loads from any stage if not specified)

        Returns:
            Model metadata
        """
        # Find model entry
        model_entry = self._find_model_entry(model_id, model_type, version, stage)

        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Load metadata
        metadata = ModelMetadata()
        metadata.load(model_entry["metadata_path"])

        return metadata

    def update_stage(
        self,
        model_id: str,
        stage: str,
        model_type: str | None = None,
        version: str | None = None,
    ) -> None:
        """
        Update the stage of a model.

        Args:
            model_id: ID of the model
            stage: New stage ('development', 'staging', 'production', 'archived')
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, updates latest if not specified)
        """
        # Find model entry
        model_entry = self._find_model_entry(model_id, model_type, version)

        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Update stage
        model_type = model_entry["model_type"]
        for i, entry in enumerate(self.model_index[model_type]):
            if entry["model_id"] == model_id:
                if version is None or entry["model_version"] == version:
                    # If moving to production, archive current production model
                    if stage == "production":
                        self._archive_production_models(model_type, model_id)

                    # Update stage
                    self.model_index[model_type][i]["stage"] = stage
                    self.model_index[model_type][i][
                        "updated_at"
                    ] = datetime.now().isoformat()

                    logger.info(f"Updated model {model_id} to {stage} stage")

        # Save model index
        self._save_model_index()

    def delete_model(
        self,
        model_id: str,
        model_type: str | None = None,
        version: str | None = None,
        delete_files: bool = True,
    ) -> None:
        """
        Delete a model from the registry.

        Args:
            model_id: ID of the model
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, deletes latest if not specified)
            delete_files: Whether to delete model files
        """
        # Find model entry
        model_entry = self._find_model_entry(model_id, model_type, version)

        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Backup model if backup directory is specified
        if self.backup_dir:
            self._backup_model(model_entry)

        # Delete model files
        if delete_files:
            model_dir = model_entry["path"]
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model files at {model_dir}")

        # Remove from model index
        model_type = model_entry["model_type"]
        self.model_index[model_type] = [
            entry
            for entry in self.model_index[model_type]
            if not (
                entry["model_id"] == model_id
                and (version is None or entry["model_version"] == version)
            )
        ]

        # Save model index
        self._save_model_index()

        logger.info(f"Deleted model {model_id} from registry")

    def list_models(
        self,
        model_type: str | None = None,
        stage: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List models in the registry.

        Args:
            model_type: Type of models to list (optional, lists all types if not specified)
            stage: Stage of models to list (optional, lists all stages if not specified)
            limit: Maximum number of models to return
            offset: Offset for pagination

        Returns:
            List of model entries
        """
        models = []

        # Get models of specified type or all types
        if model_type:
            model_types = [model_type] if model_type in self.model_index else []
        else:
            model_types = list(self.model_index.keys())

        # Collect models
        for mt in model_types:
            for entry in self.model_index[mt]:
                if stage is None or entry["stage"] == stage:
                    models.append(entry)

        # Sort by updated_at (newest first)
        models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        # Apply pagination
        models = models[offset : offset + limit]

        return models

    def get_latest_model(
        self, model_type: str, stage: str = "production"
    ) -> tuple[Any, ModelMetadata] | None:
        """
        Get the latest model of a specific type and stage.

        Args:
            model_type: Type of the model
            stage: Stage of the model

        Returns:
            Tuple of (model, metadata) or None if not found
        """
        if model_type not in self.model_index:
            return None

        # Filter models by type and stage
        models = [
            entry for entry in self.model_index[model_type] if entry["stage"] == stage
        ]

        if not models:
            return None

        # Sort by updated_at (newest first)
        models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        # Get latest model
        latest_model_entry = models[0]

        # Load model and metadata
        try:
            model, metadata = self.load_model(
                latest_model_entry["model_id"],
                model_type,
                latest_model_entry["model_version"],
                stage,
            )
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return None

    def compare_models(
        self, model_ids: list[str], metrics: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compare multiple models based on their metadata.

        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare (optional, compares all metrics if not specified)

        Returns:
            DataFrame with model comparison
        """
        # Load metadata for each model
        model_metadata = []
        for model_id in model_ids:
            try:
                metadata = self.get_metadata(model_id)
                model_metadata.append(metadata.get_metadata())
            except Exception as e:
                logger.error(f"Error loading metadata for model {model_id}: {e}")

        if not model_metadata:
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison = {
            "model_id": [m["model_id"] for m in model_metadata],
            "model_type": [m["model_type"] for m in model_metadata],
            "model_version": [m["model_version"] for m in model_metadata],
            "created_at": [m["created_at"] for m in model_metadata],
        }

        # Add performance metrics
        if metrics:
            for metric in metrics:
                comparison[metric] = [
                    m.get("performance", {}).get("validation", {}).get(metric, None)
                    for m in model_metadata
                ]
        else:
            # Add all validation metrics
            all_metrics = set()
            for m in model_metadata:
                all_metrics.update(
                    m.get("performance", {}).get("validation", {}).keys()
                )

            for metric in all_metrics:
                comparison[metric] = [
                    m.get("performance", {}).get("validation", {}).get(metric, None)
                    for m in model_metadata
                ]

        # Create DataFrame
        df = pd.DataFrame(comparison)

        return df

    def export_model(
        self,
        model_id: str,
        export_dir: str,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Export a model to a directory.

        Args:
            model_id: ID of the model
            export_dir: Directory to export the model to
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, exports latest if not specified)
            stage: Stage of the model (optional, exports from any stage if not specified)
            include_metadata: Whether to include metadata in the export

        Returns:
            Path to the exported model
        """
        # Find model entry
        model_entry = self._find_model_entry(model_id, model_type, version, stage)

        if not model_entry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Create export directory
        os.makedirs(export_dir, exist_ok=True)

        # Copy model file
        model_path = model_entry["model_path"]
        export_model_path = os.path.join(export_dir, os.path.basename(model_path))
        shutil.copy2(model_path, export_model_path)

        # Copy metadata if requested
        if include_metadata:
            metadata_path = model_entry["metadata_path"]
            export_metadata_path = os.path.join(
                export_dir, os.path.basename(metadata_path)
            )
            shutil.copy2(metadata_path, export_metadata_path)

        # Copy additional files
        for file_type, file_path in model_entry.get("files", {}).items():
            if os.path.exists(file_path):
                export_file_path = os.path.join(export_dir, os.path.basename(file_path))
                shutil.copy2(file_path, export_file_path)

        logger.info(f"Exported model {model_id} to {export_dir}")

        return export_model_path

    def import_model(
        self,
        model_path: str,
        metadata_path: str | None = None,
        stage: str = "development",
    ) -> str:
        """
        Import a model from a file.

        Args:
            model_path: Path to the model file
            metadata_path: Path to the metadata file (optional)
            stage: Stage for the imported model

        Returns:
            ID of the imported model
        """
        # Load model
        model = deserialize_model(model_path)

        # Load or create metadata
        if metadata_path and os.path.exists(metadata_path):
            metadata = ModelMetadata()
            metadata.load(metadata_path)
        else:
            # Create metadata from model
            model_id = f"imported_{uuid.uuid4().hex[:8]}"
            model_type = type(model).__name__.lower()

            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                description=f"Imported {model_type} model",
            )

        # Register model
        model_id = self.register_model(model=model, metadata=metadata, stage=stage)

        logger.info(f"Imported model {model_id} from {model_path}")

        return model_id

    def _load_model_index(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load the model index from disk.

        Returns:
            Dictionary mapping model types to lists of model entries
        """
        index_path = os.path.join(self.registry_dir, "model_index.json")

        if os.path.exists(index_path):
            try:
                return load_json(index_path)
            except Exception as e:
                logger.error(f"Error loading model index: {e}")
                return {}
        else:
            # Create index from existing models
            index = {}

            # Scan registry directory for model types
            for model_type_dir in glob.glob(os.path.join(self.registry_dir, "*")):
                if os.path.isdir(model_type_dir):
                    model_type = os.path.basename(model_type_dir)

                    # Skip special directories
                    if model_type in ["__pycache__", ".git"]:
                        continue

                    index[model_type] = []

                    # Scan model type directory for models
                    for model_dir in glob.glob(os.path.join(model_type_dir, "*")):
                        if os.path.isdir(model_dir):
                            model_id = os.path.basename(model_dir)

                            # Find metadata file
                            metadata_files = glob.glob(
                                os.path.join(model_dir, "*_metadata.json")
                            )
                            if metadata_files:
                                metadata_path = metadata_files[0]

                                # Load metadata
                                try:
                                    with open(metadata_path) as f:
                                        metadata = json.load(f)

                                    # Find model file
                                    model_files = glob.glob(
                                        os.path.join(model_dir, "*.pkl")
                                    )
                                    model_path = model_files[0] if model_files else None

                                    # Create model entry
                                    model_entry = {
                                        "model_id": model_id,
                                        "model_type": model_type,
                                        "model_version": metadata.get(
                                            "model_version", ""
                                        ),
                                        "created_at": metadata.get("created_at", ""),
                                        "updated_at": metadata.get("updated_at", ""),
                                        "stage": "development",  # Default stage
                                        "path": model_dir,
                                        "metadata_path": metadata_path,
                                        "model_path": model_path,
                                        "files": {},
                                    }

                                    index[model_type].append(model_entry)
                                except Exception as e:
                                    logger.error(
                                        f"Error loading metadata for model {model_id}: {e}"
                                    )

            return index

    def _save_model_index(self) -> None:
        """Save the model index to disk."""
        index_path = os.path.join(self.registry_dir, "model_index.json")

        try:
            save_json(self.model_index, index_path, indent=2)
            logger.debug(f"Saved model index to {index_path}")
        except Exception as e:
            logger.error(f"Error saving model index: {e}")

    def _find_model_entry(
        self,
        model_id: str,
        model_type: str | None = None,
        version: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Find a model entry in the index.

        Args:
            model_id: ID of the model
            model_type: Type of the model (optional if model_id is unique)
            version: Version of the model (optional, finds latest if not specified)
            stage: Stage of the model (optional, finds from any stage if not specified)

        Returns:
            Model entry or None if not found
        """
        # If model_type is specified, search only in that type
        if model_type and model_type in self.model_index:
            model_types = [model_type]
        else:
            # Search in all model types
            model_types = list(self.model_index.keys())

        # Collect matching models
        matching_models = []
        for mt in model_types:
            for entry in self.model_index[mt]:
                if entry["model_id"] == model_id:
                    if version is None or entry["model_version"] == version:
                        if stage is None or entry["stage"] == stage:
                            matching_models.append(entry)

        if not matching_models:
            return None

        # If multiple matches, return the latest one
        if len(matching_models) > 1:
            # Sort by updated_at (newest first)
            matching_models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return matching_models[0]

    def _enforce_max_models(self, model_type: str) -> None:
        """
        Enforce maximum number of models per type.

        Args:
            model_type: Type of models to enforce limit on
        """
        if model_type not in self.model_index:
            return

        # Get models of this type
        models = self.model_index[model_type]

        # If number of models exceeds limit, remove oldest ones
        if len(models) > self.max_models_per_type:
            # Sort by updated_at (oldest first)
            models.sort(key=lambda x: x.get("updated_at", ""))

            # Keep only production and staging models, and newest development models
            production_models = [m for m in models if m["stage"] == "production"]
            staging_models = [m for m in models if m["stage"] == "staging"]
            development_models = [m for m in models if m["stage"] == "development"]
            archived_models = [m for m in models if m["stage"] == "archived"]

            # Calculate how many development models to keep
            keep_dev_count = max(
                0,
                self.max_models_per_type - len(production_models) - len(staging_models),
            )

            # Sort development models by updated_at (newest first)
            development_models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

            # Keep only the newest development models
            keep_dev_models = development_models[:keep_dev_count]

            # Remove excess development models
            remove_dev_models = development_models[keep_dev_count:]

            # Remove excess archived models (keep only a few)
            keep_archived_count = max(
                0,
                self.max_models_per_type
                - len(production_models)
                - len(staging_models)
                - len(keep_dev_models),
            )
            archived_models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            keep_archived_models = archived_models[:keep_archived_count]
            remove_archived_models = archived_models[keep_archived_count:]

            # Update model index
            self.model_index[model_type] = (
                production_models
                + staging_models
                + keep_dev_models
                + keep_archived_models
            )

            # Delete removed models
            for model_entry in remove_dev_models + remove_archived_models:
                # Backup model if backup directory is specified
                if self.backup_dir:
                    self._backup_model(model_entry)

                # Delete model files
                model_dir = model_entry["path"]
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    logger.info(
                        f"Deleted excess model {model_entry['model_id']} files at {model_dir}"
                    )

    def _archive_production_models(
        self, model_type: str, exclude_model_id: str
    ) -> None:
        """
        Archive current production models.

        Args:
            model_type: Type of models to archive
            exclude_model_id: ID of model to exclude from archiving
        """
        if model_type not in self.model_index:
            return

        # Update stage of current production models
        for i, entry in enumerate(self.model_index[model_type]):
            if entry["stage"] == "production" and entry["model_id"] != exclude_model_id:
                self.model_index[model_type][i]["stage"] = "archived"
                self.model_index[model_type][i][
                    "updated_at"
                ] = datetime.now().isoformat()

                logger.info(f"Archived production model {entry['model_id']}")

    def _backup_model(self, model_entry: dict[str, Any]) -> None:
        """
        Backup a model to the backup directory.

        Args:
            model_entry: Model entry to backup
        """
        if not self.backup_dir:
            return

        model_id = model_entry["model_id"]
        model_type = model_entry["model_type"]
        model_dir = model_entry["path"]

        # Create backup directory
        backup_dir = os.path.join(self.backup_dir, model_type, model_id)
        os.makedirs(backup_dir, exist_ok=True)

        # Copy model files
        if os.path.exists(model_dir):
            for file_name in os.listdir(model_dir):
                src_path = os.path.join(model_dir, file_name)
                dst_path = os.path.join(backup_dir, file_name)

                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

            logger.info(f"Backed up model {model_id} to {backup_dir}")

    def shutdown(self) -> None:
        """Shutdown the model registry and release resources."""
        logger.info("Shutting down model registry")
        
        # Release allocated port if using port_manager
        if self.use_port_manager and self.allocated_port:
            release_port(self.allocated_port)
            logger.info(f"Released port {self.allocated_port}")

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    # Create model registry
    registry = ModelRegistry(
        registry_dir="/tmp/model_registry",
        create_if_missing=True,
        max_models_per_type=5,
        backup_dir="/tmp/model_backups",
        use_port_manager=True,
    )

    # Create a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model.fit(X, y)

    # Create metadata
    metadata = ModelMetadata(
        model_type="random_forest", description="Random Forest model for testing"
    )

    # Set parameters
    metadata.set_parameters({"n_estimators": 100, "random_state": 42})

    # Set performance metrics
    metadata.set_performance(
        {"train": {"mse": 0.1, "r2": 0.8}, "validation": {"mse": 0.12, "r2": 0.75}}
    )

    # Register model
    model_id = registry.register_model(
        model=model, metadata=metadata, stage="development"
    )

    print(f"Registered model with ID: {model_id}")

    # List models
    models = registry.list_models()
    print(f"Models in registry: {models}")

    # Load model
    loaded_model, loaded_metadata = registry.load_model(model_id)
    print(f"Loaded model: {loaded_model}")
    print(f"Loaded metadata: {loaded_metadata.get_metadata()}")

    # Update stage
    registry.update_stage(model_id, "production")
    print("Updated model stage to production")

    # Get latest production model
    latest_model, latest_metadata = registry.get_latest_model(
        "random_forest", "production"
    )
    print(f"Latest production model: {latest_model}")
    print(f"Latest production metadata: {latest_metadata.get_metadata()}")

    # Export model
    export_path = registry.export_model(model_id, "/tmp/exported_models")
    print(f"Exported model to {export_path}")

    # Delete model
    registry.delete_model(model_id)
    print(f"Deleted model {model_id}")
    
    # Shutdown registry
    registry.shutdown()
