#!/usr/bin/env python3
"""
Trading System

Main entry point for the integrated trading system that coordinates all components.
Features include:
1. Component lifecycle management
2. Configuration handling
3. API client initialization
4. Signal handling for graceful shutdown
5. System status reporting
"""

import os
import time
import json
import logging
import argparse
import threading
import redis
import signal
from datetime import datetime

# Import custom modules
from data_pipeline_integration import DataPipelineIntegration
from model_integration_system import ModelIntegrationSystem
from execution_system import ExecutionSystem
from monitoring_system import MonitoringSystem
from data_loader import DataLoader
from ml_model_trainer import MLModelTrainer
from learning_system.continual_learning_system import ContinualLearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log')
    ]
)
logger = logging.getLogger('main')


class TradingSystem:
    """
    Main trading system application that integrates all components
    """

    def __init__(self, config_path=None):
        """Initialize the trading system"""
        # Load configuration
        self.config = self._load_config(config_path)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Initialize Redis connection
        self.redis = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            password=self.config['redis']['password'] if 'password' in self.config['redis'] else None
        )

        # Initialize API clients
        self.polygon_client = self._initialize_polygon_client()
        self.unusual_whales_client = self._initialize_unusual_whales_client()
        self.alpaca_client = self._initialize_alpaca_client()

        # Initialize components
        self.components = {}
        self.running = False

        logger.info("Trading system initialized")

    def _load_config(self, config_path):
        """Load configuration from file"""
        default_config = {
            'redis': {
                'host': 'localhost',
                'port': 6380,
                'db': 0
            },
            'api_keys': {
                'polygon': os.environ.get('POLYGON_API_KEY', ''),
                'unusual_whales': os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
                'alpaca': {
                    'api_key': os.environ.get('ALPACA_API_KEY', ''),
                    'api_secret': os.environ.get('ALPACA_API_SECRET', ''),
                    'base_url': os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                }
            },
            'system': {
                'max_exposure': 5000.0,
                'market_hours_only': True,
                'data_dir': '/data',
                'models_dir': '/models',
                'use_gpu': os.environ.get('USE_GPU', 'true').lower() == 'true',
                'continual_learning': {
                    'enabled': True,
                    'daily_update_time': '23:30',
                    'full_retrain_time': '00:30'
                }
            }
        }

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)

            # Merge configs
            self._merge_configs(default_config, file_config)

        # Ensure directories exist
        os.makedirs(default_config['system']['data_dir'], exist_ok=True)
        os.makedirs(default_config['system']['models_dir'], exist_ok=True)

        logger.info("Configuration loaded")
        return default_config

    def _merge_configs(self, base, override):
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _initialize_polygon_client(self):
        """Initialize Polygon client"""
        try:
            from gpu_optimized_polygon_api_client import GPUPolygonAPIClient

            client = GPUPolygonAPIClient(
                api_key=self.config['api_keys']['polygon'],
                use_gpu=self.config['system']['use_gpu']
            )

            logger.info(
                f"Initialized Polygon client with GPU acceleration: {self.config['system']['use_gpu']}")
            return client

        except Exception as e:
            logger.error(
                f"Error initializing Polygon client: {str(e)}", exc_info=True)
            return None

    def _initialize_unusual_whales_client(self):
        """Initialize Unusual Whales client"""
        try:
            from gpu_optimized_unusual_whales_client import GPUUnusualWhalesClient

            client = GPUUnusualWhalesClient(
                api_key=self.config['api_keys']['unusual_whales'],
                use_gpu=self.config['system']['use_gpu']
            )

            logger.info(
                f"Initialized Unusual Whales client with GPU acceleration: {self.config['system']['use_gpu']}")
            return client

        except Exception as e:
            logger.error(
                f"Error initializing Unusual Whales client: {str(e)}", exc_info=True)
            return None

    def _initialize_alpaca_client(self):
        """Initialize Alpaca client"""
        try:
            import alpaca_trade_api as tradeapi

            alpaca = tradeapi.REST(
                key_id=self.config['api_keys']['alpaca']['api_key'],
                secret_key=self.config['api_keys']['alpaca']['api_secret'],
                base_url=self.config['api_keys']['alpaca']['base_url']
            )

            # Test connection
            account = alpaca.get_account()
            logger.info(
                f"Connected to Alpaca - Account ID: {account.id}, Status: {account.status}")

            return alpaca

        except Exception as e:
            logger.error(
                f"Error initializing Alpaca client: {str(e)}", exc_info=True)
            return None

    def _handle_signal(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return

        self.running = True
        logger.info("Starting trading system...")

        try:
            # Initialize and start each component
            logger.info("Initializing data loader...")
            data_loader = DataLoader(
                polygon_client=self.polygon_client,
                unusual_whales_client=self.unusual_whales_client,
                redis_client=self.redis
            )

            logger.info("Initializing ML model trainer...")
            model_trainer = MLModelTrainer(
                redis_client=self.redis,
                data_loader=data_loader
            )

            # Initialize continual learning system if enabled
            if self.config['system']['continual_learning']['enabled']:
                logger.info("Initializing continual learning system...")
                continual_learning = ContinualLearningSystem(
                    redis_client=self.redis,
                    data_loader=data_loader,
                    model_trainer=model_trainer
                )
                self.components['continual_learning'] = continual_learning

            logger.info("Initializing data pipeline integration...")
            data_pipeline = DataPipelineIntegration(
                redis_host=self.config['redis']['host'],
                redis_port=self.config['redis']['port'],
                redis_db=self.config['redis']['db'],
                polygon_api_key=self.config['api_keys']['polygon'],
                unusual_whales_api_key=self.config['api_keys']['unusual_whales'],
                use_gpu=self.config['system']['use_gpu']
            )
            self.components['data_pipeline'] = data_pipeline

            logger.info("Initializing model integration system...")
            model_integration = ModelIntegrationSystem(
                redis_client=self.redis,
                data_pipeline=data_pipeline
            )
            self.components['model_integration'] = model_integration

            logger.info("Initializing execution system...")
            execution_system = ExecutionSystem(
                redis_client=self.redis,
                alpaca_client=self.alpaca_client
            )
            self.components['execution'] = execution_system

            logger.info("Initializing monitoring system...")
            monitoring_system = MonitoringSystem(
                redis_client=self.redis
            )
            self.components['monitoring'] = monitoring_system

            # Start ML training in background if models don't exist
            models_exist = os.path.exists(os.path.join(
                self.config['system']['models_dir'], 'signal_detection_model.xgb'))
            if not models_exist:
                logger.info(
                    "No existing models found, initiating model training...")
                threading.Thread(
                    target=model_trainer.train_all_models, daemon=True).start()

            # Start each component
            if 'continual_learning' in self.components:
                logger.info("Starting continual learning system...")
                self.components['continual_learning'].start()

            logger.info("Starting data pipeline...")
            self.components['data_pipeline'].start()

            logger.info("Starting model integration system...")
            self.components['model_integration'].start()

            logger.info("Starting execution system...")
            self.components['execution'].start()

            logger.info("Starting monitoring system...")
            self.components['monitoring'].start()

            logger.info("Trading system started successfully")

        except Exception as e:
            logger.error(
                f"Error starting trading system: {str(e)}", exc_info=True)
            self.running = False

    def stop(self):
        """Stop the trading system"""
        if not self.running:
            logger.warning("Trading system is not running")
            return

        logger.info("Stopping trading system...")

        # Stop components in reverse order
        stop_order = ['monitoring', 'execution',
                      'model_integration', 'data_pipeline', 'continual_learning']

        for component_name in stop_order:
            if component_name in self.components:
                logger.info(f"Stopping {component_name} system...")
                try:
                    self.components[component_name].stop()
                except Exception as e:
                    logger.error(f"Error stopping {component_name}: {str(e)}")

        # Close API clients
        try:
            logger.info("Closing API clients...")
            if hasattr(self.polygon_client, 'close'):
                self.polygon_client.close()
            if hasattr(self.unusual_whales_client, 'close'):
                self.unusual_whales_client.close()
        except Exception as e:
            logger.error(f"Error closing API clients: {str(e)}")

        self.running = False
        logger.info("Trading system stopped")

    def status(self):
        """Get the status of all components"""
        status = {
            'system': 'running' if self.running else 'stopped',
            'components': {}
        }

        for name, component in self.components.items():
            status['components'][name] = 'running' if hasattr(
                component, 'running') and component.running else 'stopped'

        # Get positions and orders
        if self.alpaca_client:
            try:
                positions = self.alpaca_client.list_positions()
                status['positions'] = [
                    {
                        'symbol': pos.symbol,
                        'qty': int(pos.qty),
                        'side': 'long' if int(pos.qty) > 0 else 'short',
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc)
                    }
                    for pos in positions
                ]

                orders = self.alpaca_client.list_orders(status='open')
                status['orders'] = [
                    {
                        'id': order.id,
                        'symbol': order.symbol,
                        'qty': int(order.qty),
                        'side': order.side,
                        'type': order.type,
                        'submitted_at': order.submitted_at.isoformat() if hasattr(order.submitted_at, 'isoformat') else order.submitted_at
                    }
                    for order in orders
                ]

            except Exception as e:
                logger.error(f"Error getting positions and orders: {str(e)}")
                status['positions'] = []
                status['orders'] = []

        return status


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--action', type=str, default='start',
                        choices=['start', 'stop', 'status'], help='Action to perform')
    parser.add_argument('--enable-continual-learning',
                        action='store_true', help='Enable continual learning system')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    args = parser.parse_args()

    # Create trading system
    trading_system = TradingSystem(config_path=args.config)

    # Override config with command line arguments
    if args.enable_continual_learning:
        trading_system.config['system']['continual_learning']['enabled'] = True

    if args.no_gpu:
        trading_system.config['system']['use_gpu'] = False

    # Perform requested action
    if args.action == 'start':
        trading_system.start()

        # Keep running until interrupted
        try:
            while trading_system.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
            trading_system.stop()

    elif args.action == 'stop':
        trading_system.stop()

    elif args.action == 'status':
        status = trading_system.status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
