#!/usr/bin/env python3
"""
Comprehensive Trading System Test Suite
Tests all aspects of the trading system container and validates production readiness.
"""

from stock_selection_system.gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
from stock_selection_system.gpu_optimized_polygon_websocket_client import GPUPolygonWebSocketClient
from stock_selection_system.gpu_optimized_polygon_api_client import GPUPolygonAPIClient
from stock_selection_system.gpu_stock_selection_core import GPUStockSelectionSystem
from gpu_system.gh200_accelerator import GH200Accelerator
import os
import sys
import json
import redis
import pytest
import logging
import numpy as np
import tensorflow as tf
import cupy as cp
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import datetime
import pytz

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTradingSystem:
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.required_env_vars = [
            'POLYGON_API_KEY',
            'UNUSUAL_WHALES_API_KEY',
            'REDIS_HOST',
            'REDIS_PORT',
            'REDIS_PASSWORD'
        ]

        cls.required_directories = [
            'data',
            'models',
            'logs'
        ]

        cls.system_files = [
            'gpu_system/docker-compose.unified.yml',
            'gpu_system/Dockerfile.unified',
            'gpu_system/prometheus/prometheus.yml',
            'gpu_system/redis/redis.conf'
        ]

    def test_environment_variables(self):
        """Test all required environment variables are set"""
        for var in self.required_env_vars:
            assert os.getenv(
                var) is not None, f"Missing environment variable: {var}"
            assert os.getenv(var) != "", f"Empty environment variable: {var}"

            # Check for placeholder values
            value = os.getenv(var)
            assert "your_" not in value.lower(
            ), f"Placeholder value found in {var}: {value}"
            assert "example" not in value.lower(
            ), f"Example value found in {var}: {value}"

    def test_directory_structure(self):
        """Test required directories exist"""
        for directory in self.required_directories:
            path = Path(directory)
            assert path.exists(), f"Missing directory: {directory}"
            assert path.is_dir(), f"Not a directory: {directory}"

    def test_system_files(self):
        """Test all required system files exist and are valid"""
        for file_path in self.system_files:
            path = Path(file_path)
            assert path.exists(), f"Missing file: {file_path}"
            assert path.is_file(), f"Not a file: {file_path}"
            assert path.stat().st_size > 0, f"Empty file: {file_path}"

    def test_gpu_configuration(self):
        """Test GPU setup and acceleration"""
        # Test GH200 Accelerator
        accelerator = GH200Accelerator()
        assert accelerator.has_tensorflow_gpu or accelerator.has_cupy_gpu, "No GPU acceleration available"

        # Test TensorFlow GPU
        if accelerator.has_tensorflow_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            assert len(gpus) > 0, "TensorFlow cannot see GPU"

            # Test basic GPU operation
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                assert c.device.endswith('GPU:0'), "TensorFlow not using GPU"

        # Test CuPy GPU
        if accelerator.has_cupy_gpu:
            assert cp.cuda.runtime.getDeviceCount() > 0, "CuPy cannot see GPU"

            # Test basic GPU operation
            a = cp.random.normal(0, 1, (1000, 1000))
            b = cp.random.normal(0, 1, (1000, 1000))
            c = cp.matmul(a, b)
            assert isinstance(c, cp.ndarray), "CuPy not using GPU"

    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """Test Redis connection and operations"""
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )

        # Test connection
        assert redis_client.ping(), "Redis connection failed"

        # Test basic operations
        test_key = "test:connection"
        test_value = "test_value"
        redis_client.set(test_key, test_value)
        assert redis_client.get(test_key) == test_value, "Redis get/set failed"

        # Clean up
        redis_client.delete(test_key)
        redis_client.close()

    @pytest.mark.asyncio
    async def test_polygon_connection(self):
        """Test Polygon.io API connection"""
        polygon_client = GPUPolygonAPIClient()

        # Test market status
        status = await polygon_client.get_market_status()
        assert isinstance(status, dict), "Failed to get market status"

        # Test ticker details (using a major index)
        ticker_details = await polygon_client.get_ticker_details("SPY")
        assert isinstance(ticker_details, dict), "Failed to get ticker details"

    @pytest.mark.asyncio
    async def test_unusual_whales_connection(self):
        """Test Unusual Whales API connection"""
        whales_client = GPUUnusualWhalesClient()

        # Test API health
        health = await whales_client.check_api_health()
        assert health, "Unusual Whales API health check failed"

    def test_file_content_validation(self):
        """Test source files for hard-coded values and example data"""
        def check_file_content(file_path: str) -> List[str]:
            issues = []
            with open(file_path, 'r') as f:
                content = f.read()

                # Check for example tickers
                example_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                for ticker in example_tickers:
                    if ticker in content:
                        issues.append(
                            f"Found example ticker {ticker} in {file_path}")

                # Check for example values
                if 'example' in content.lower():
                    issues.append(f"Found 'example' in {file_path}")

                # Check for TODO comments
                if 'TODO' in content:
                    issues.append(f"Found TODO comment in {file_path}")

                # Check for hard-coded numerical values without constants
                if 'price = 100' in content or 'amount = 1000' in content:
                    issues.append(
                        f"Found potential hard-coded values in {file_path}")

            return issues

        # Get all Python files
        python_files = []
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        # Check each file
        all_issues = []
        for file_path in python_files:
            issues = check_file_content(file_path)
            all_issues.extend(issues)

        assert len(all_issues) == 0, f"Found issues in files:\n" + \
            "\n".join(all_issues)

    @pytest.mark.asyncio
    async def test_trading_system_initialization(self):
        """Test trading system initialization and basic operations"""
        # Initialize components
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=int(os.getenv('REDIS_PORT')),
            password=os.getenv('REDIS_PASSWORD')
        )

        polygon_client = GPUPolygonAPIClient()
        polygon_ws = GPUPolygonWebSocketClient()
        whales_client = GPUUnusualWhalesClient()

        # Initialize trading system
        trading_system = GPUStockSelectionSystem(
            redis_client=redis_client,
            polygon_api_client=polygon_client,
            polygon_websocket_client=polygon_ws,
            unusual_whales_client=whales_client
        )

        # Test system initialization
        assert trading_system.gpu_available, "GPU not available for trading system"
        assert trading_system.config is not None, "Trading system config not initialized"

        # Test basic operations
        await trading_system.start()
        assert trading_system.running, "Trading system not running"

        # Test universe building
        await trading_system.build_initial_universe()
        assert len(
            trading_system.full_universe) > 0, "Failed to build initial universe"

        # Clean up
        await trading_system.stop()
        assert not trading_system.running, "Trading system not stopped"

    def test_prometheus_config(self):
        """Test Prometheus configuration"""
        with open('gpu_system/prometheus/prometheus.yml', 'r') as f:
            config = f.read()

            # Check for required components
            assert 'scrape_configs:' in config, "Missing scrape configs in Prometheus config"
            assert 'Redis Exporter' in config, "Missing Redis exporter in Prometheus config"

    def test_redis_config(self):
        """Test Redis configuration"""
        with open('gpu_system/redis/redis.conf', 'r') as f:
            config = f.read()

            # Check for required settings
            assert 'maxmemory' in config, "Missing maxmemory setting in Redis config"
            assert 'maxmemory-policy' in config, "Missing maxmemory-policy in Redis config"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
