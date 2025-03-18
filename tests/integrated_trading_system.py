#!/usr/bin/env python3
"""
Integrated Trading System

This module provides a complete integrated trading system that combines:
1. Data ingestion from multiple sources (Polygon.io, Unusual Whales)
2. Real-time data processing with GPU acceleration
3. ML model integration for signal generation
4. Risk management and position sizing
5. Execution and monitoring

The system is designed for high-performance trading with robust error handling,
failover mechanisms, and comprehensive monitoring capabilities.
"""

import os
import sys
import time
import json
import logging
import asyncio
import argparse
import signal
import redis
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Import system components
from tests.data_pipeline_integration import DataPipelineIntegration
from tests.model_integration_system import ModelIntegrationSystem
from tests.gpu_optimized_polygon_api_client import GPUPolygonAPIClient
from tests.gpu_optimized_polygon_websocket_client import GPUPolygonWebSocketClient
from tests.gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient
from tests.execution_system import ExecutionSystem
from tests.monitoring_system import MonitoringSystem
from tests.reporting_system import ReportingSystem
from tests.ml_model_trainer import MLModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_system.log")
    ]
)
logger = logging.getLogger('integrated_trading_system')

# Import continual learning system if available
try:
    from tests.continual_learning_system import ContinualLearningSystem
    CONTINUAL_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUAL_LEARNING_AVAILABLE = False
    logger.warning("Continual learning system not available")

# Environment variables
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
UNUSUAL_WHALES_API_KEY = os.environ.get('UNUSUAL_WHALES_API_KEY', '')
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET', '')
ALPACA_API_URL = os.environ.get('ALPACA_API_URL', 'https://paper-api.alpaca.markets')
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'


class IntegratedTradingSystem:
    """
    Integrated Trading System that combines data ingestion, processing,
    model inference, and execution into a cohesive trading platform.
    """

    def __init__(self, redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB,
                 polygon_api_key=POLYGON_API_KEY, unusual_whales_api_key=UNUSUAL_WHALES_API_KEY,
                 alpaca_api_key=ALPACA_API_KEY, alpaca_api_secret=ALPACA_API_SECRET, 
                 alpaca_api_url=ALPACA_API_URL, use_gpu=USE_GPU, config_path=None):
        """
        Initialize the Integrated Trading System
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            polygon_api_key: Polygon.io API key
            unusual_whales_api_key: Unusual Whales API key
            alpaca_api_key: Alpaca API key
            alpaca_api_secret: Alpaca API secret
            alpaca_api_url: Alpaca API URL
            use_gpu: Whether to use GPU acceleration
            config_path: Path to configuration file
        """
        self.use_gpu = use_gpu
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize Redis client
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except (redis.RedisError, ConnectionError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Initialize system components
        self.data_pipeline = DataPipelineIntegration(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            polygon_api_key=polygon_api_key,
            unusual_whales_api_key=unusual_whales_api_key,
            use_gpu=use_gpu
        )
        
        self.model_system = ModelIntegrationSystem(self.redis)
        
        # Initialize Alpaca client
        try:
            import alpaca_trade_api as tradeapi
            self.alpaca = tradeapi.REST(
                key_id=alpaca_api_key,
                secret_key=alpaca_api_secret,
                base_url=alpaca_api_url
            )
            logger.info(f"Connected to Alpaca API at {alpaca_api_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            self.alpaca = None
            
        # Initialize execution system
        self.execution_system = ExecutionSystem(self.redis, self.alpaca) if self.alpaca else None
        
        # Initialize monitoring system
        self.monitoring_system = MonitoringSystem(self.redis)
        
        # Initialize reporting system
        self.reporting_system = ReportingSystem(self.redis)
        
        # Initialize ML model trainer
        self.model_trainer = MLModelTrainer(self.redis, self.data_pipeline)
        
        # Initialize continual learning system if available
        self.continual_learning = None
        if CONTINUAL_LEARNING_AVAILABLE:
            try:
                self.continual_learning = ContinualLearningSystem(
                    self.redis, 
                    self.data_pipeline, 
                    self.model_trainer
                )
                logger.info("Continual learning system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize continual learning system: {e}")
        
        # Control flags
        self.running = False
        self.event_loop = None
        self.main_task = None
        self.background_thread = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Integrated Trading System initialized")

    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        asyncio.create_task(self.stop())

    def _load_config(self):
        """Load configuration from file or use defaults"""
        config = {
            'system': {
                'name': 'GPU-Accelerated Trading System',
                'version': '1.0.0',
                'description': 'High-performance trading system with GPU acceleration',
                'log_level': 'INFO',
                'health_check_interval': 60,  # seconds
                'max_memory_usage': 0.8,  # 80% of available memory
                'gpu_memory_limit': 0.9  # 90% of GPU memory
            },
            'data': {
                'sources': ['polygon', 'unusual_whales'],
                'cache_ttl': 300,  # 5 minutes
                'batch_size': 100,
                'update_interval': {
                    'market_status': 60,
                    'ticker_details': 86400,
                    'aggregates': 60,
                    'quotes': 5,
                    'trades': 5
                }
            },
            'models': {
                'signal_threshold': 0.7,
                'signal_expiry': 300,
                'batch_size': 32,
                'update_interval': 86400
            },
            'trading': {
                'max_positions': 5,
                'risk_per_trade': 0.005,  # 0.5% of account
                'max_drawdown': 0.05,  # 5% max drawdown
                'trading_hours': {
                    'start': '09:30',
                    'end': '16:00',
                    'timezone': 'America/New_York'
                },
                'position_sizing': {
                    'method': 'risk_based',  # risk_based, equal, kelly
                    'max_position_size': 0.1  # 10% of account
                }
            },
            'watchlist': {
                'default': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'sectors': {
                    'technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
                    'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
                    'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS']
                }
            }
        }
        
        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    
                # Merge configurations
                self._deep_update(config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {e}")
                logger.info("Using default configuration")
        
        return config

    def _deep_update(self, d, u):
        """Recursively update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    async def start(self):
        """Start the integrated trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return
            
        self.running = True
        logger.info("Starting integrated trading system")
        
        # Store the event loop
        self.event_loop = asyncio.get_event_loop()
        
        # Initialize system state in Redis
        self.redis.set("system:status", "starting")
        self.redis.set("system:start_time", datetime.now().isoformat())
        
        # Start data pipeline
        await self.data_pipeline.start()
        self.monitoring_system.component_status["data_ingestion"] = "running"
        
        # Start model system
        self.model_system.start()
        self.monitoring_system.component_status["model_integration"] = "running"
        
        # Start execution system if available
        if self.execution_system:
            self.execution_system.start()
            self.monitoring_system.component_status["execution"] = "running"
        else:
            logger.warning("Execution system not available, running in simulation mode")
            self.monitoring_system.component_status["execution"] = "unavailable"
            
        # Start monitoring system
        self.monitoring_system.start()
        
        # Start reporting system
        self.reporting_system.start()
        self.monitoring_system.component_status["reporting"] = "running"
            
        # Start ML model trainer and continual learning if available
        if self.continual_learning:
            try:
                self.continual_learning.start()
                logger.info("Continual learning system started")
            except Exception as e:
                logger.error(f"Failed to start continual learning system: {e}")
            
        # Start system monitoring
        self.main_task = self.event_loop.create_task(self._system_monitor())
        
        # Initialize watchlist
        await self._initialize_watchlist()
        
        # Update system status
        self.redis.set("system:status", "running")
        
        logger.info("Integrated trading system started")

    async def stop(self):
        """Stop the integrated trading system"""
        if not self.running:
            logger.warning("Trading system is not running")
            return
            
        logger.info("Stopping integrated trading system")
        self.running = False
        
        # Update system status
        self.redis.set("system:status", "stopping")
        
        # Cancel main task
        if self.main_task:
            self.main_task.cancel()
            
        # Stop components
        await self.data_pipeline.stop()
        self.model_system.stop()
        if self.execution_system:
            self.execution_system.stop()
        # Stop continual learning if available
        if self.continual_learning:
            try:
                self.continual_learning.stop()
                logger.info("Continual learning system stopped")
            except Exception as e:
                logger.error(f"Error stopping continual learning system: {e}")
        self.monitoring_system.stop()
        
        # Stop reporting system
        try:
            self.reporting_system.stop()
            logger.info("Reporting system stopped")
        except Exception as e:
            logger.error(f"Error stopping reporting system: {e}")
        
        
        # Update system status
        self.redis.set("system:status", "stopped")
        self.redis.set("system:stop_time", datetime.now().isoformat())
        
        logger.info("Integrated trading system stopped")

    async def _system_monitor(self):
        """Monitor system health and performance"""
        logger.info("Starting system monitor")
        
        while self.running:
            try:
                # Check system health
                health_status = await self._check_health()
                
                # Store health status in Redis
                self.redis.set("system:health", json.dumps(health_status))
                
                # Log any issues
                if health_status['status'] != 'healthy':
                    logger.warning(f"System health issues detected: {health_status['issues']}")
                    
                    # Take corrective action if needed
                    if health_status['critical_issues']:
                        await self._handle_critical_issues(health_status['critical_issues'])
                
                # Wait before next check
                await asyncio.sleep(self.config['system']['health_check_interval'])
                
            except asyncio.CancelledError:
                logger.info("System monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(10)  # Shorter interval on error

    async def _check_health(self):
        """Check system health"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'issues': [],
            'critical_issues': [],
            'components': {}
        }
        
        # Check Redis connection
        try:
            self.redis.ping()
            health['components']['redis'] = 'healthy'
        except redis.RedisError:
            health['components']['redis'] = 'error'
            health['issues'].append('Redis connection error')
            health['critical_issues'].append('redis_connection')
            health['status'] = 'critical'
        
        # Check data pipeline
        data_pipeline_status = self.redis.get("system:health")
        if data_pipeline_status:
            try:
                dp_health = json.loads(data_pipeline_status)
                health['components']['data_pipeline'] = dp_health.get('status', 'unknown')
                
                if health['components']['data_pipeline'] != 'OK':
                    health['issues'].append('Data pipeline health issues')
                    if health['status'] == 'healthy':
                        health['status'] = 'warning'
            except (json.JSONDecodeError, AttributeError):
                health['components']['data_pipeline'] = 'unknown'
                health['issues'].append('Invalid data pipeline health data')
                if health['status'] == 'healthy':
                    health['status'] = 'warning'
        else:
            health['components']['data_pipeline'] = 'unknown'
            health['issues'].append('No data pipeline health data')
            if health['status'] == 'healthy':
                health['status'] = 'warning'
        
        # Check model system
        # This would require the model system to expose health metrics
        health['components']['model_system'] = 'unknown'
        
        # Check memory usage
        
        # Check execution system
        if self.execution_system:
            health['components']['execution_system'] = 'healthy'
            self.monitoring_system.component_status["execution"] = "running"
            
            # Check active positions
            active_positions = len(self.execution_system.active_positions)
            health['active_positions'] = active_positions
            
            # Check daily stats
            daily_stats = self.execution_system.daily_stats
            health['daily_stats'] = {
                'trades_executed': daily_stats.get('trades_executed', 0),
                'profitable_trades': daily_stats.get('profitable_trades', 0),
                'total_pnl': daily_stats.get('total_pnl', 0.0),
                'current_exposure': daily_stats.get('current_exposure', 0.0)
            }
        else:
            health['components']['execution_system'] = 'unavailable'
            health['issues'].append('Execution system not available')
            if health['status'] == 'healthy':
                health['status'] = 'warning'
            self.monitoring_system.component_status["execution"] = "unavailable"

        # Check reporting system
        try:
            if hasattr(self.reporting_system, 'running') and self.reporting_system.running:
                health['components']['reporting_system'] = 'healthy'
                self.monitoring_system.component_status["reporting"] = "running"
            else:
                health['components']['reporting_system'] = 'stopped'
                health['issues'].append('Reporting system not running')
                self.monitoring_system.component_status["reporting"] = "stopped"
        except Exception as e:
            health['components']['reporting_system'] = 'error'
            health['issues'].append(f'Reporting system error: {str(e)}')

        # Check continual learning system
        if self.continual_learning:
            health['components']['continual_learning'] = 'healthy'
            # Add any specific metrics from continual learning
            if hasattr(self.continual_learning, 'model_versions'):
                health['model_versions'] = {
                    model: len(versions) for model, versions in 
                    self.continual_learning.model_versions.items()
                }
        else:
            health['components']['continual_learning'] = 'unavailable'
            if CONTINUAL_LEARNING_AVAILABLE:
                health['issues'].append('Continual learning system not available')
                if health['status'] == 'healthy':
                    health['status'] = 'warning'
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            health['memory_usage'] = memory_usage
            
            if memory_usage > self.config['system']['max_memory_usage']:
                health['issues'].append(f'High memory usage: {memory_usage:.1%}')
                if health['status'] == 'healthy':
                    health['status'] = 'warning'
        except ImportError:
            health['memory_usage'] = 'unknown'
        
        # Check GPU memory if using GPU
        if self.use_gpu:
            try:
                import cupy as cp
                device = cp.cuda.Device(0)
                gpu_memory = device.mem_info
                gpu_usage = 1.0 - (gpu_memory[0] / gpu_memory[1])
                health['gpu_memory_usage'] = gpu_usage
                
                if gpu_usage > self.config['system']['gpu_memory_limit']:
                    health['issues'].append(f'High GPU memory usage: {gpu_usage:.1%}')
                    if health['status'] == 'healthy':
                        health['status'] = 'warning'
            except ImportError:
                health['gpu_memory_usage'] = 'unknown'
            except Exception as e:
                health['gpu_memory_usage'] = 'error'
                health['issues'].append(f'GPU memory check error: {str(e)}')
                if health['status'] == 'healthy':
                    health['status'] = 'warning'
        
        return health

    async def _handle_critical_issues(self, issues):
        """Handle critical system issues"""
        logger.critical(f"Handling critical issues: {issues}")
        
        # Handle Redis connection issues
        if 'redis_connection' in issues:
            logger.critical("Redis connection lost, attempting to reconnect")
            try:
                # Attempt to reconnect
                self.redis = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.redis.ping()
                logger.info("Redis reconnection successful")
            except (redis.RedisError, ConnectionError) as e:
                logger.critical(f"Redis reconnection failed: {e}")
                # If reconnection fails, initiate system shutdown
                await self.stop()
        
        # Handle other critical issues as needed
        # ...

    async def _initialize_watchlist(self):
        """Initialize system watchlist"""
        logger.info("Initializing watchlist")
        
        # Get default watchlist from config
        default_watchlist = self.config['watchlist']['default']
        
        # Check if watchlist already exists in Redis
        existing_watchlist = self.redis.smembers("watchlist:active")
        if existing_watchlist:
            logger.info(f"Watchlist already exists with {len(existing_watchlist)} tickers")
            return
        
        # Initialize watchlist with default tickers
        pipeline = self.redis.pipeline()
        pipeline.delete("watchlist:active")
        pipeline.sadd("watchlist:active", *default_watchlist)
        
        # Initialize focused list with top 5 tickers
        pipeline.delete("watchlist:focused")
        pipeline.sadd("watchlist:focused", *default_watchlist[:5])
        
        # Execute pipeline
        pipeline.execute()
        
        logger.info(f"Initialized watchlist with {len(default_watchlist)} tickers")
        logger.info(f"Initialized focused list with {len(default_watchlist[:5])} tickers")

    def run_in_background(self):
        """Run the trading system in a background thread"""
        def run_event_loop():
            """Run the event loop in a background thread"""
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Start the trading system
                loop.run_until_complete(self.start())
                
                # Run the event loop
                loop.run_forever()
                
            except Exception as e:
                logger.error(f"Error in background thread: {e}")
                
            finally:
                # Clean up
                if loop and not loop.is_closed():
                    loop.close()
                logger.info("Background thread stopped")
        
        # Start background thread
        self.background_thread = threading.Thread(target=run_event_loop)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        logger.info("Trading system started in background thread")

    def stop_background(self):
        """Stop the trading system running in background"""
        if not self.background_thread or not self.background_thread.is_alive():
            logger.warning("Trading system is not running in background")
            return
            
        logger.info("Stopping trading system in background")
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Stop the trading system
        loop.run_until_complete(self.stop())
        
        # Wait for background thread to stop
        self.background_thread.join(timeout=10)
        
        # Clean up
        if loop and not loop.is_closed():
            loop.close()
            
        logger.info("Trading system stopped in background")


async def main():
    """Main entry point for the integrated trading system"""
    parser = argparse.ArgumentParser(description='Integrated Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--redis-host', type=str, default=REDIS_HOST, help='Redis host')
    parser.add_argument('--redis-port', type=int, default=REDIS_PORT, help='Redis port')
    parser.add_argument('--redis-db', type=int, default=REDIS_DB, help='Redis database')
    parser.add_argument('--polygon-key', type=str, default=POLYGON_API_KEY, help='Polygon.io API key')
    parser.add_argument('--unusual-whales-key', type=str, default=UNUSUAL_WHALES_API_KEY, help='Unusual Whales API key')
    parser.add_argument('--alpaca-key', type=str, default=ALPACA_API_KEY, help='Alpaca API key')
    parser.add_argument('--alpaca-secret', type=str, default=ALPACA_API_SECRET, help='Alpaca API secret')
    parser.add_argument('--alpaca-url', type=str, default=ALPACA_API_URL, help='Alpaca API URL')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    args = parser.parse_args()
    
    # Create trading system
    trading_system = IntegratedTradingSystem(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        polygon_api_key=args.polygon_key,
        unusual_whales_api_key=args.unusual_whales_key,
        alpaca_api_key=args.alpaca_key,
        alpaca_api_secret=args.alpaca_secret,
        alpaca_api_url=args.alpaca_url,
        use_gpu=not args.no_gpu,
        config_path=args.config
    )
    
    # Start trading system
    await trading_system.start()
    
    try:
        # Keep the main thread running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop trading system
        await trading_system.stop()


if __name__ == "__main__":
    asyncio.run(main())