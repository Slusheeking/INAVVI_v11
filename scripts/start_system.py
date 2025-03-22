#!/usr/bin/env python3
"""
Unified Trading System Starter

This script initializes and starts the entire trading system with all components:
1. Data Pipeline
2. ML Engine
3. Trading Engine
4. Stock Selection
5. Monitoring System
6. Continuous Learning

The script includes comprehensive health checks and sends status updates to Slack.

Usage:
    python start_system.py [--config CONFIG] [--debug] [--no-gpu]
"""

import os
import sys
import time
import logging
import argparse
import signal
import subprocess
import json
import psutil
import platform
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
# Use local logs directory when running outside of a container
if os.path.exists('/app'):
    logs_dir = os.environ.get('LOGS_DIR', '/app/logs')
    os.makedirs(logs_dir, exist_ok=True)  # Ensure logs directory exists
else:
    logs_dir = './logs'
    os.makedirs(logs_dir, exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'system.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('system_starter')

# Global variables
system = None
running = False
monitoring_system = None
slack_notifier = None
components = {}


def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    logger.info(f"Received signal {sig}, shutting down...")
    running = False
    stop_all_components()


def load_environment_variables():
    """Load environment variables from .env file if it exists"""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value
        logger.info("Environment variables loaded")
    else:
        logger.warning(f"Environment file {env_file} not found")


def ensure_directories():
    """Ensure all required directories exist"""
    # Only create local directories, not /app directories
    if os.environ.get('LOGS_DIR', './logs').startswith('/app'):
        logger.info("Skipping directory creation for container paths")
        return

    dirs = [
        './logs',
        './data',
        './models',
        os.path.join('./data', 'market_data'),
        os.path.join('./data', 'processed'),
        os.path.join('./data', 'signals'),
        os.path.join('./models', 'signal_detection'),
        os.path.join('./models', 'price_prediction'),
        os.path.join('./models', 'risk_assessment'),
        os.path.join('./logs', 'trading'),
        os.path.join('./logs', 'ml'),
        os.path.join('./logs', 'monitoring')
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

    logger.info("All required directories created")


def initialize_slack_notifier():
    """Initialize Slack notifier for system notifications"""
    global slack_notifier
    try:
        from monitoring_system.monitoring_system import SlackNotifier
        slack_notifier = SlackNotifier()

        # Test Slack connection
        if slack_notifier.slack_available:
            success = slack_notifier.send_notification(
                "Trading system startup initiated",
                level='info',
                channel=slack_notifier.notification_channel
            )
            if success:
                logger.info(
                    "Slack notifier initialized and connected successfully")
            else:
                logger.warning(
                    "Slack notifier initialized but test message failed")
        else:
            logger.warning(
                "Slack integration not available - notifications will be logged only")

        return slack_notifier
    except Exception as e:
        logger.error(f"Failed to initialize Slack notifier: {str(e)}")
        return None


def check_redis():
    """Check if Redis is running and start it if needed"""
    try:
        import redis
        redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6380)),
            db=int(os.environ.get('REDIS_DB', 0)),
            password=os.environ.get('REDIS_PASSWORD', ''),
            username=os.environ.get('REDIS_USERNAME', 'default'),
            socket_timeout=1
        )
        redis_client.ping()
        logger.info("Redis is already running")

        if slack_notifier:
            slack_notifier.send_success("Redis connection verified")

        return redis_client
    except:
        logger.warning("Redis is not running, attempting to start...")
        try:
            # Try to start Redis using the configuration file
            redis_conf = os.path.join(os.path.dirname(
                os.path.dirname(__file__)), 'redis', 'redis.conf')
            if os.path.exists(redis_conf):
                subprocess.Popen(['redis-server', redis_conf],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                subprocess.Popen(
                    ['redis-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for Redis to start
            time.sleep(2)

            # Check if Redis is now running
            redis_client = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'localhost'),
                port=int(os.environ.get('REDIS_PORT', 6380)),
                db=int(os.environ.get('REDIS_DB', 0)),
                password=os.environ.get(
                    'REDIS_PASSWORD', 'trading_system_2025'),
                username=os.environ.get('REDIS_USERNAME', 'default'),
                socket_timeout=1
            )
            redis_client.ping()
            logger.info("Redis started successfully")

            if slack_notifier:
                slack_notifier.send_success("Redis started successfully")

            return redis_client
        except Exception as e:
            error_msg = f"Failed to start Redis: {str(e)}"
            logger.error(error_msg)

            if slack_notifier:
                slack_notifier.send_error("Redis startup failed", error_msg)

            return None


def check_prometheus():
    """Check if Prometheus is running and start it if needed"""
    try:
        import requests
        response = requests.get('http://localhost:9090/-/healthy', timeout=1)
        if response.status_code == 200:
            logger.info("Prometheus is already running")

            if slack_notifier:
                slack_notifier.send_success("Prometheus connection verified")

            return True
    except:
        logger.warning("Prometheus is not running, attempting to start...")
        try:
            # Try to start Prometheus using the configuration file
            prometheus_conf = os.path.join(os.path.dirname(
                os.path.dirname(__file__)), 'prometheus', 'prometheus.yml')
            if os.path.exists(prometheus_conf):
                subprocess.Popen(['prometheus', '--config.file=' + prometheus_conf],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                logger.warning(
                    "Prometheus configuration file not found, skipping Prometheus start")

                if slack_notifier:
                    slack_notifier.send_warning(
                        "Prometheus configuration file not found, metrics collection will be limited")

                return False

            # Wait for Prometheus to start
            time.sleep(2)

            # Check if Prometheus is now running
            response = requests.get(
                'http://localhost:9090/-/healthy', timeout=1)
            if response.status_code == 200:
                logger.info("Prometheus started successfully")

                if slack_notifier:
                    slack_notifier.send_success(
                        "Prometheus started successfully")

                return True
            else:
                logger.warning("Prometheus failed to start")

                if slack_notifier:
                    slack_notifier.send_warning(
                        "Prometheus failed to start, metrics collection will be limited")

                return False
        except Exception as e:
            error_msg = f"Failed to start Prometheus: {str(e)}"
            logger.error(error_msg)

            if slack_notifier:
                slack_notifier.send_warning(
                    f"Prometheus startup failed: {error_msg}")

            return False

    return False


def start_monitoring_system(redis_client=None):
    """Start the monitoring system"""
    global monitoring_system
    try:
        from monitoring_system.monitoring_system import MonitoringSystem

        logger.info("Starting monitoring system...")
        monitoring_system = MonitoringSystem(redis_client=redis_client)
        monitoring_system.start()
        logger.info("Monitoring system started successfully")

        if slack_notifier:
            slack_notifier.send_success(
                "Monitoring system started successfully")

        return monitoring_system
    except Exception as e:
        error_msg = f"Failed to start monitoring system: {str(e)}"
        logger.error(error_msg)

        if slack_notifier:
            slack_notifier.send_error(
                "Monitoring system startup failed", error_msg)

        return None


def start_individual_components(redis_client=None, use_gpu=True):
    """Start individual components of the trading system"""
    global components

    try:
        # Start data pipeline
        logger.info("Starting data pipeline...")
        from data_pipeline import DataPipeline
        data_pipeline = DataPipeline(
            redis_client=redis_client, use_gpu=use_gpu)
        if hasattr(data_pipeline, 'start'):
            data_pipeline.start()
        components['data_pipeline'] = data_pipeline
        logger.info("Data pipeline started successfully")

        if slack_notifier:
            slack_notifier.send_success("Data pipeline started successfully")

        # Start ML engine
        logger.info("Starting ML engine...")
        from ml_engine import MLModelTrainer
        ml_engine = MLModelTrainer(
            redis_client=redis_client, data_loader=data_pipeline)
        if hasattr(ml_engine, 'start'):
            ml_engine.start()
        components['ml_engine'] = ml_engine
        logger.info("ML engine started successfully")

        if slack_notifier:
            slack_notifier.send_success("ML engine started successfully")

        # Start stock selection engine
        logger.info("Starting stock selection engine...")
        from stock_selection_engine import GPUStockSelectionSystem
        stock_selection = GPUStockSelectionSystem(
            redis_client=redis_client, use_gpu=use_gpu)
        if hasattr(stock_selection, 'start'):
            stock_selection.start()
        components['stock_selection'] = stock_selection
        logger.info("Stock selection engine started successfully")

        if slack_notifier:
            slack_notifier.send_success(
                "Stock selection engine started successfully")

        # Start trading engine
        logger.info("Starting trading engine...")
        from trading_engine import TradingEngine
        trading_engine = TradingEngine(
            redis_client=redis_client, data_pipeline=data_pipeline)
        if hasattr(trading_engine, 'start'):
            trading_engine.start()
        components['trading_engine'] = trading_engine
        logger.info("Trading engine started successfully")

        if slack_notifier:
            slack_notifier.send_success("Trading engine started successfully")

        # Start continuous learning system
        logger.info("Starting continuous learning system...")
        from learning_engine import ContinualLearningSystem
        learning_engine = ContinualLearningSystem(
            redis_client=redis_client,
            data_loader=data_pipeline,
            model_trainer=ml_engine
        )
        if hasattr(learning_engine, 'start'):
            learning_engine.start()
        components['learning_engine'] = learning_engine
        logger.info("Continuous learning system started successfully")

        if slack_notifier:
            slack_notifier.send_success(
                "Continuous learning system started successfully")

        return True
    except Exception as e:
        error_msg = f"Failed to start individual components: {str(e)}"
        logger.error(error_msg)

        if slack_notifier:
            slack_notifier.send_error("Component startup failed", error_msg)

        return False


def stop_all_components():
    """Stop all system components"""
    global components, monitoring_system, system

    logger.info("Stopping all system components...")

    # Stop individual components in reverse order
    component_names = list(components.keys())
    component_names.reverse()

    for name in component_names:
        component = components.get(name)
        if component and hasattr(component, 'stop'):
            try:
                logger.info(f"Stopping {name}...")
                component.stop()
                logger.info(f"{name} stopped successfully")

                if slack_notifier:
                    slack_notifier.send_info(f"{name} stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping {name}: {str(e)}")

                if slack_notifier:
                    slack_notifier.send_warning(
                        f"Error stopping {name}: {str(e)}")

    # Stop monitoring system
    if monitoring_system and hasattr(monitoring_system, 'stop'):
        try:
            logger.info("Stopping monitoring system...")
            monitoring_system.stop()
            logger.info("Monitoring system stopped successfully")

            if slack_notifier:
                slack_notifier.send_info(
                    "Monitoring system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {str(e)}")

            if slack_notifier:
                slack_notifier.send_warning(
                    f"Error stopping monitoring system: {str(e)}")

    # Stop unified system if it was started
    if system and hasattr(system, 'stop'):
        try:
            logger.info("Stopping unified system...")
            system.stop()
            logger.info("Unified system stopped successfully")

            if slack_notifier:
                slack_notifier.send_info("Unified system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping unified system: {str(e)}")

            if slack_notifier:
                slack_notifier.send_warning(
                    f"Error stopping unified system: {str(e)}")

    # Final notification
    if slack_notifier:
        slack_notifier.send_info("Trading system shutdown complete")


def run_health_check(redis_client=None):
    """Run a comprehensive health check of the system"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        },
        "components": {},
        "services": {
            "redis": False,
            "prometheus": False
        },
        "gpu": {}
    }

    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["services"]["redis"] = True
    except:
        health_status["services"]["redis"] = False

    # Check Prometheus
    try:
        import requests
        response = requests.get('http://localhost:9090/-/healthy', timeout=1)
        health_status["services"]["prometheus"] = response.status_code == 200
    except:
        health_status["services"]["prometheus"] = False

    # Check GPU if enabled
    if os.environ.get('USE_GPU', 'true').lower() == 'true':
        try:
            from gpu_utils import get_gpu_memory_usage
            mem_info = get_gpu_memory_usage()
            if mem_info:
                used_mb, total_mb = mem_info
                health_status["gpu"] = {
                    "available": True,
                    "memory_used_mb": used_mb,
                    "memory_total_mb": total_mb,
                    "memory_usage_percent": (used_mb / total_mb) * 100 if total_mb > 0 else 0
                }
            else:
                health_status["gpu"] = {"available": False}
        except Exception as e:
            health_status["gpu"] = {"available": False, "error": str(e)}
    else:
        health_status["gpu"] = {"available": False,
                                "reason": "GPU disabled by configuration"}

    # Check individual components
    for name, component in components.items():
        if component:
            health_status["components"][name] = {
                "running": hasattr(component, 'running') and component.running,
                "status": "healthy" if hasattr(component, 'running') and component.running else "not running"
            }

    # Check monitoring system
    if monitoring_system:
        health_status["components"]["monitoring_system"] = {
            "running": hasattr(monitoring_system, 'running') and monitoring_system.running,
            "status": "healthy" if hasattr(monitoring_system, 'running') and monitoring_system.running else "not running"
        }

    # Check unified system
    if system:
        health_status["components"]["unified_system"] = {
            "running": hasattr(system, 'running') and system.running,
            "status": "healthy" if hasattr(system, 'running') and system.running else "not running"
        }

    # Log health status
    logger.info(f"Health check results: {json.dumps(health_status, indent=2)}")

    # Send to Slack if available
    if slack_notifier:
        # Determine overall health
        components_healthy = all(c.get("running", False)
                                 for c in health_status["components"].values())
        services_healthy = all(health_status["services"].values())
        system_healthy = health_status["system"]["cpu_usage_percent"] < 90 and health_status["system"]["memory_usage_percent"] < 90

        if components_healthy and services_healthy and system_healthy:
            slack_notifier.send_success(
                "System health check: All systems operational")
        else:
            # Create a detailed report
            slack_notifier.send_report(
                "System Health Check Report", health_status)

    return health_status


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Trading System Starter')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--no-redis', action='store_true',
                        help='Skip Redis check/start')
    parser.add_argument('--no-prometheus', action='store_true',
                        help='Skip Prometheus check/start')
    parser.add_argument('--individual', action='store_true',
                        help='Start individual components instead of unified system')
    parser.add_argument('--health-check-interval', type=int,
                        default=300, help='Health check interval in seconds')
    return parser.parse_args()


def main():
    """Main entry point"""
    global system, running, monitoring_system, slack_notifier, components

    # Parse arguments
    args = parse_arguments()

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load environment variables
    load_environment_variables()

    # Ensure directories exist
    ensure_directories()

    # Initialize Slack notifier
    slack_notifier = initialize_slack_notifier()

    # Check and start Redis if needed
    redis_client = None
    if not args.no_redis:
        redis_client = check_redis()
        if not redis_client:
            logger.error("Redis is required but could not be started")
            if slack_notifier:
                slack_notifier.send_error(
                    "System startup failed", "Redis is required but could not be started")
            sys.exit(1)

    # Check and start Prometheus if needed
    if not args.no_prometheus:
        check_prometheus()  # Continue even if Prometheus fails

    # Start monitoring system
    monitoring_system = start_monitoring_system(redis_client=redis_client)

    # Start the trading system
    try:
        if args.individual:
            # Start individual components
            logger.info("Starting individual components...")
            if start_individual_components(redis_client=redis_client, use_gpu=not args.no_gpu):
                logger.info("All individual components started successfully")
                if slack_notifier:
                    slack_notifier.send_success(
                        "All trading system components started successfully")
            else:
                logger.error("Failed to start individual components")
                if slack_notifier:
                    slack_notifier.send_error(
                        "System startup failed", "Failed to start individual components")
                sys.exit(1)
        else:
            # Create and start the unified trading system
            logger.info("Initializing unified trading system...")
            from unified_system import UnifiedTradingSystem
            system = UnifiedTradingSystem(
                config_path=args.config,
                debug=args.debug,
                use_gpu=not args.no_gpu
            )

            logger.info("Starting unified trading system...")
            if system.start():
                logger.info("Unified trading system started successfully")
                if slack_notifier:
                    slack_notifier.send_success(
                        "Unified trading system started successfully")
            else:
                logger.error("Failed to start the unified trading system")
                if slack_notifier:
                    slack_notifier.send_error(
                        "System startup failed", "Failed to start the unified trading system")
                sys.exit(1)

        # Run initial health check
        health_status = run_health_check(redis_client)

        # Set running flag
        running = True

        # Main loop with periodic health checks
        last_health_check = time.time()
        while running:
            time.sleep(1)

            # Run health check at specified interval
            current_time = time.time()
            if current_time - last_health_check > args.health_check_interval:
                health_status = run_health_check(redis_client)
                last_health_check = current_time

    except Exception as e:
        error_msg = f"Error starting trading system: {str(e)}"
        logger.error(error_msg)
        if slack_notifier:
            slack_notifier.send_error("System startup failed", error_msg)
        sys.exit(1)
    finally:
        # Stop all components
        stop_all_components()
        logger.info("Trading system shutdown complete")


if __name__ == "__main__":
    main()
