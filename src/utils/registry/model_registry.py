"""
Redis-based model registry for managing model metadata.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Union, Any
import redis
from datetime import datetime, timedelta

logger = logging.getLogger("model_registry")


class ModelRegistry:
    """
    Redis-based model registry for tracking model metadata.
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        namespace: str = "model"
    ):
        """
        Initialize the model registry.

        Args:
            redis_host: Redis host (default: from environment)
            redis_port: Redis port (default: from environment)
            redis_password: Redis password (default: from environment)
            namespace: Namespace for Redis keys
        """
        self.namespace = namespace

        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=redis_host or os.environ.get("REDIS_HOST", "redis"),
                port=redis_port or int(os.environ.get("REDIS_PORT", "6379")),
                password=redis_password or os.environ.get(
                    "REDIS_PASSWORD", ""),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            self.registry_available = True
            logger.info(
                f"Connected to model registry (Redis) at {redis_host or os.environ.get('REDIS_HOST', 'redis')}")
        except Exception as e:
            self.redis = None
            self.registry_available = False
            logger.warning(f"Could not connect to model registry: {e}")
            logger.warning("Model registry will operate in local-only mode")

    def _model_key(self, model_id: str) -> str:
        """Generate a Redis key for a model."""
        return f"{self.namespace}:model:{model_id}"

    def _active_model_key(self, model_type: str, symbols: List[str]) -> str:
        """Generate a Redis key for tracking the active model of a type."""
        symbols_key = ",".join(sorted(symbols))
        return f"{self.namespace}:active:{model_type}:{symbols_key}"

    def register_model(
        self,
        model_id: str,
        model_type: str,
        symbols: List[str],
        metadata: Dict[str, Any],
        is_active: bool = False
    ) -> bool:
        """
        Register a model in the registry.

        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "lstm", "xgboost")
            symbols: List of symbols this model is trained for
            metadata: Additional metadata about the model
            is_active: Whether this should be marked as the active model

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_available or not self.redis:
            logger.warning(
                f"Model registry not available, model {model_id} registration skipped")
            return False

        try:
            # Create model data
            model_data = {
                "model_id": model_id,
                "model_type": model_type,
                "symbols": json.dumps(sorted(symbols)),
                "created_at": datetime.utcnow().isoformat(),
                "metadata": json.dumps(metadata),
                "is_active": "1" if is_active else "0"
            }

            # Store model metadata
            key = self._model_key(model_id)
            self.redis.hmset(key, model_data)

            # If active, update active model key
            if is_active:
                active_key = self._active_model_key(model_type, symbols)
                self.redis.set(active_key, model_id)

            logger.info(f"Registered model {model_id} in registry")
            return True
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata from the registry.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Optional[Dict[str, Any]]: Model metadata or None if not found
        """
        if not self.registry_available or not self.redis:
            logger.warning(
                f"Model registry not available, cannot retrieve model {model_id}")
            return None

        try:
            # Get model metadata
            key = self._model_key(model_id)
            data = self.redis.hgetall(key)

            if not data:
                return None

            # Parse JSON fields
            if "symbols" in data:
                data["symbols"] = json.loads(data["symbols"])

            if "metadata" in data:
                data["metadata"] = json.loads(data["metadata"])

            # Convert string to boolean
            if "is_active" in data:
                data["is_active"] = data["is_active"] == "1"

            return data
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None

    def get_active_model(self, model_type: str, symbols: List[str]) -> Optional[str]:
        """
        Get the active model ID for a model type and symbols.

        Args:
            model_type: Type of model (e.g., "lstm", "xgboost")
            symbols: List of symbols

        Returns:
            Optional[str]: Model ID of the active model or None if not found
        """
        if not self.registry_available or not self.redis:
            logger.warning(
                f"Model registry not available, cannot get active model for {model_type}")
            return None

        try:
            # Get active model ID
            active_key = self._active_model_key(model_type, symbols)
            model_id = self.redis.get(active_key)

            return model_id
        except Exception as e:
            logger.error(f"Error getting active model for {model_type}: {e}")
            return None

    def set_active_model(self, model_id: str, model_type: str, symbols: List[str]) -> bool:
        """
        Set a model as the active model for a type and symbols.

        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "lstm", "xgboost")
            symbols: List of symbols this model is trained for

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_available or not self.redis:
            logger.warning(
                f"Model registry not available, cannot set {model_id} as active")
            return False

        try:
            # Update the model's active status
            model_key = self._model_key(model_id)
            self.redis.hset(model_key, "is_active", "1")

            # Set as active model
            active_key = self._active_model_key(model_type, symbols)
            self.redis.set(active_key, model_id)

            logger.info(
                f"Set model {model_id} as active for {model_type} and symbols {symbols}")
            return True
        except Exception as e:
            logger.error(f"Error setting model {model_id} as active: {e}")
            return False

    def list_models(
        self,
        model_type: Optional[str] = None,
        symbol: Optional[str] = None,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry with optional filtering.

        Args:
            model_type: Filter by model type
            symbol: Filter by symbol
            active_only: Only return active models

        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        if not self.registry_available or not self.redis:
            logger.warning("Model registry not available, cannot list models")
            return []

        try:
            # Scan for model keys
            pattern = f"{self.namespace}:model:*"
            keys = []
            cursor = 0

            while True:
                cursor, new_keys = self.redis.scan(
                    cursor, match=pattern, count=100)
                keys.extend(new_keys)

                if cursor == 0:
                    break

            # Get model data for each key
            models = []
            for key in keys:
                data = self.redis.hgetall(key)

                if not data:
                    continue

                # Parse JSON fields
                if "symbols" in data:
                    data["symbols"] = json.loads(data["symbols"])

                if "metadata" in data:
                    data["metadata"] = json.loads(data["metadata"])

                # Convert string to boolean
                if "is_active" in data:
                    data["is_active"] = data["is_active"] == "1"

                # Apply filters
                if model_type and data.get("model_type") != model_type:
                    continue

                if symbol and symbol not in data.get("symbols", []):
                    continue

                if active_only and not data.get("is_active", False):
                    continue

                models.append(data)

            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
