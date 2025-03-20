#!/usr/bin/env python3
"""
GPU-Optimized Stock Selection System - Main Entry Point
This module provides a unified interface to the GPU-accelerated stock selection system.
"""

from trading_system.execution_system import ExecutionSystem
from stock_selection_system.websocket_core import WebSocketCore
from stock_selection_system.stock_selection_core import StockSelectionCore
from stock_selection_system.gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
from stock_selection_system.gpu_optimized_polygon_websocket_client import GPUPolygonWebSocketClient
from stock_selection_system.gpu_optimized_polygon_api_client import GPUPolygonAPIClient
from stock_selection_system.gpu_stock_selection_core import GPUStockSelectionSystem
from dotenv import dotenv_values
import logging
import asyncio
import os
import redis
import alpaca_trade_api as tradeapi
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the core system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_stock_selection_main')


async def main():
    """Main entry point for the GPU-optimized stock selection system"""
    try:
        # Load environment variables from .env file
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        env_values = {}
        if os.path.exists(env_path):
            env_values = dotenv_values(env_path)
        else:
            # Try to load from the main .env file
            env_values = dotenv_values(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), '.env'))

        # Get Redis connection details and API keys
        redis_host = env_values.get('REDIS_HOST')
        redis_port = int(env_values.get('REDIS_PORT', 6380))
        redis_db = int(env_values.get('REDIS_DB', '0'))
        redis_password = env_values.get('REDIS_PASSWORD')
        polygon_api_key = env_values.get('POLYGON_API_KEY')
        alpaca_api_key = env_values.get('ALPACA_API_KEY')
        alpaca_api_secret = env_values.get('ALPACA_API_SECRET')
        alpaca_api_url = env_values.get(
            'ALPACA_API_URL', 'https://paper-api.alpaca.markets')
        unusual_whales_api_key = env_values.get('UNUSUAL_WHALES_API_KEY')

        # Validate required environment variables
        missing_vars = []
        if not redis_host:
            missing_vars.append('REDIS_HOST')
        if not redis_port:
            missing_vars.append('REDIS_PORT')
        if not polygon_api_key:
            missing_vars.append('POLYGON_API_KEY')
        if not alpaca_api_key:
            missing_vars.append('ALPACA_API_KEY')
        if not alpaca_api_secret:
            missing_vars.append('ALPACA_API_SECRET')
        if not unusual_whales_api_key:
            missing_vars.append('UNUSUAL_WHALES_API_KEY')

        if missing_vars:
            logger.error(
                f"Missing required variables in .env file: {', '.join(missing_vars)}")
            logger.error(
                "Please ensure all required variables are set in the .env file")
            return

        # Connect to Redis with full configuration
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=env_values.get('REDIS_USERNAME', 'default'),
            password=redis_password,
            db=redis_db,
            ssl=env_values.get('REDIS_SSL', 'false').lower() == 'true',
            socket_timeout=int(env_values.get('REDIS_TIMEOUT', 5))
        )

        # Test Redis connection
        try:
            logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
            redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return

        # Create Alpaca client
        alpaca_client = tradeapi.REST(
            key_id=alpaca_api_key,
            secret_key=alpaca_api_secret,
            base_url=alpaca_api_url
        )

        # Initialize execution system
        execution_system = ExecutionSystem(redis_client, alpaca_client)

        # Initialize Polygon API client
        polygon_api_client = GPUPolygonAPIClient(
            api_key=polygon_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Initialize Polygon WebSocket client
        polygon_websocket_client = GPUPolygonWebSocketClient(
            api_key=polygon_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Initialize Unusual Whales client
        unusual_whales_client = GPUUnusualWhalesClient(
            api_key=unusual_whales_api_key,
            redis_client=redis_client,
            use_gpu=True
        )

        # Create stock selection system
        system = GPUStockSelectionSystem(
            redis_client=redis_client,
            polygon_api_client=polygon_api_client,
            polygon_websocket_client=polygon_websocket_client,
            unusual_whales_client=unusual_whales_client
        )

        # Verify GPU is being used
        if not system.gpu_available:
            logger.error(
                "GPU is not available. This system requires either TensorFlow or CuPy GPU acceleration.")
            logger.error(f"TensorFlow GPU: {system.gh200_accelerator.has_tensorflow_gpu}, "
                         f"CuPy GPU: {system.gh200_accelerator.has_cupy_gpu}")
            logger.error(
                "Please ensure you are running in the GPU container with NVIDIA GH200 support.")
            return

        # Start execution system
        execution_system.start()

        # Start the stock selection system
        await system.start()

        # Run until interrupted
        try:
            logger.info("System running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping system...")
        finally:
            # Stop the system
            await system.stop()

            # Stop execution system
            execution_system.stop()

            # Close clients
            await polygon_api_client.close()
            await polygon_websocket_client.close()
            if unusual_whales_client:
                await unusual_whales_client.close()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
