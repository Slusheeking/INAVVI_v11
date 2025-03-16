"""
Redis-based feature cache for efficient feature sharing between components.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Union, Any
import redis
from datetime import datetime, timedelta

logger = logging.getLogger("feature_cache")


class FeatureCache:
    """
    Redis-based feature cache for efficient feature access.
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour default TTL
        namespace: str = "feature"
    ):
        """
        Initialize the feature cache.

        Args:
            redis_host: Redis host (default: from environment)
            redis_port: Redis port (default: from environment)
            redis_password: Redis password (default: from environment)
            default_ttl: Default time-to-live in seconds for cached features
            namespace: Namespace for Redis keys
        """
        self.default_ttl = default_ttl
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
            self.cache_available = True
            logger.info(
                f"Connected to feature cache (Redis) at {redis_host or os.environ.get('REDIS_HOST', 'redis')}")
        except Exception as e:
            self.redis = None
            self.cache_available = False
            logger.warning(f"Could not connect to feature cache: {e}")
            logger.warning("Feature cache will be disabled")

    def _generate_key(self, symbol: str, timeframe: str, feature_names: Optional[List[str]] = None, **params) -> str:
        """
        Generate a cache key based on parameters.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "1d", "1h")
            feature_names: List of feature names to include in the key
            **params: Additional parameters to include in the key

        Returns:
            str: Cache key
        """
        # Start with base key components
        key_parts = [self.namespace, symbol, timeframe]

        # Add feature names if provided
        if feature_names:
            key_parts.append(",".join(sorted(feature_names)))

        # Add sorted parameters
        if params:
            # Convert params to a stable string representation
            params_str = json.dumps(params, sort_keys=True)
            # Use a hash for potentially long parameter strings
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key_parts.append(params_hash)

        # Join with colons
        return ":".join(key_parts)

    def set(
        self,
        symbol: str,
        timeframe: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
        **params
    ) -> bool:
        """
        Store features in the cache.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "1d", "1h")
            features: Dictionary of features to cache
            ttl: Time-to-live in seconds (default: use instance default)
            **params: Additional parameters that identify this feature set

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cache_available or not self.redis:
            return False

        try:
            # Generate feature names from the features dict
            feature_names = list(features.keys())

            # Generate the cache key
            key = self._generate_key(
                symbol, timeframe, feature_names, **params)

            # Store as JSON string
            value = json.dumps(features)

            # Set in Redis with TTL
            ttl = ttl if ttl is not None else self.default_ttl
            self.redis.setex(key, ttl, value)

            logger.debug(f"Cached features at key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error caching features: {e}")
            return False

    def get(
        self,
        symbol: str,
        timeframe: str,
        feature_names: Optional[List[str]] = None,
        **params
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from the cache.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "1d", "1h")
            feature_names: List of feature names to retrieve
            **params: Additional parameters that identify this feature set

        Returns:
            Optional[Dict[str, Any]]: Dictionary of features or None if not found
        """
        if not self.cache_available or not self.redis:
            return None

        try:
            # Generate the cache key
            key = self._generate_key(
                symbol, timeframe, feature_names, **params)

            # Get from Redis
            value = self.redis.get(key)
            if not value:
                return None

            # Parse JSON
            features = json.loads(value)

            # Filter by feature_names if provided
            if feature_names:
                return {k: v for k, v in features.items() if k in feature_names}

            return features
        except Exception as e:
            logger.error(f"Error retrieving features from cache: {e}")
            return None

    def invalidate(
        self,
        symbol: str,
        timeframe: str,
        feature_names: Optional[List[str]] = None,
        **params
    ) -> bool:
        """
        Invalidate cached features.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "1d", "1h")
            feature_names: List of feature names to invalidate
            **params: Additional parameters that identify this feature set

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.cache_available or not self.redis:
            return False

        try:
            # Generate the cache key
            key = self._generate_key(
                symbol, timeframe, feature_names, **params)

            # Delete from Redis
            deleted = self.redis.delete(key)

            return deleted > 0
        except Exception as e:
            logger.error(f"Error invalidating features in cache: {e}")
            return False

    def clear_all(self, namespace: Optional[str] = None) -> int:
        """
        Clear all cached features.

        Args:
            namespace: Optional namespace override (default: use instance namespace)

        Returns:
            int: Number of keys deleted
        """
        if not self.cache_available or not self.redis:
            return 0

        try:
            # Use provided namespace or instance namespace
            ns = namespace or self.namespace

            # Scan for keys with pattern
            pattern = f"{ns}:*"
            keys = []
            cursor = 0

            while True:
                cursor, new_keys = self.redis.scan(
                    cursor, match=pattern, count=100)
                keys.extend(new_keys)

                if cursor == 0:
                    break

            # Delete all found keys
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} keys from feature cache")
                return deleted

            return 0
        except Exception as e:
            logger.error(f"Error clearing feature cache: {e}")
            return 0
