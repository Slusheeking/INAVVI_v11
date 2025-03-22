#!/usr/bin/env python3
"""
Monitoring System Module

This module provides a unified monitoring system for the trading platform:
1. Prometheus metrics collection and exposure
2. System health monitoring (CPU, memory, GPU)
3. Trading metrics collection
4. ML model performance tracking
5. Slack notifications for alerts and status updates
6. Redis integration for real-time metrics

The monitoring system is designed to work with the unified trading system
and provides comprehensive observability.
"""

import os
import time
import logging
import threading
import json
import datetime
import socket
import requests
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

# Import Prometheus client
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.getLogger('monitoring_system').warning(
        "Prometheus client not available. Metrics collection will be limited.")

# Import GPU monitoring tools if available
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.getLogger('monitoring_system').warning(
        "pynvml not available. GPU monitoring will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring_system')


class SlackNotifier:
    """
    Slack notification system for alerts and status updates
    """

    def __init__(self,
                 webhook_url: Optional[str] = None,
                 bot_token: Optional[str] = None,
                 notification_channel: str = '#notifications',
                 reports_channel: str = '#reports',
                 portfolio_channel: str = '#portfolio',
                 positions_channel: str = '#positions',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Slack notifier

        Args:
            webhook_url: Slack webhook URL for notifications
            bot_token: Slack bot token for API access
            notification_channel: Channel for general notifications
            reports_channel: Channel for detailed reports
            portfolio_channel: Channel for portfolio updates
            positions_channel: Channel for position updates
            config: Configuration dictionary from config module
        """
        # Try to get configuration from config module
        try:
            from config import config as system_config
            slack_config = system_config.get('slack', {})
        except ImportError:
            slack_config = {}

        # Override with provided config if available
        if config and 'slack' in config:
            slack_config = config['slack']

        # Get webhook URLs from config or environment variables
        self.notification_webhook = webhook_url or slack_config.get('webhooks', {}).get(
            'notifications') or os.environ.get('SLACK_WEBHOOK_NOTIFICATIONS')
        self.reports_webhook = slack_config.get('webhooks', {}).get(
            'reports') or os.environ.get('SLACK_WEBHOOK_REPORTS')
        self.portfolio_webhook = slack_config.get('webhooks', {}).get(
            'portfolio') or os.environ.get('SLACK_WEBHOOK_PORTFOLIO')
        self.positions_webhook = slack_config.get('webhooks', {}).get(
            'positions') or os.environ.get('SLACK_WEBHOOK_POSITIONS')

        # Get bot token from config or environment variable
        self.bot_token = bot_token or slack_config.get(
            'bot_token') or os.environ.get('SLACK_BOT_TOKEN')

        # Set channels
        self.notification_channel = notification_channel
        self.reports_channel = reports_channel
        self.portfolio_channel = portfolio_channel
        self.positions_channel = positions_channel

        # Track notification counts to prevent flooding
        self.notification_count = 0
        self.last_reset_time = time.time()
        self.max_notifications_per_hour = slack_config.get(
            'max_notifications_per_hour', 20)

        # Check if Slack integration is available
        if not self.notification_webhook and not self.bot_token:
            logger.warning(
                "No Slack webhook URL or bot token provided. Notifications will be logged only.")
            self.slack_available = False
        else:
            self.slack_available = True
            logger.info("Slack notifier initialized")

    def send_notification(self,
                          message: str,
                          level: str = 'info',
                          channel: Optional[str] = None,
                          attachments: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Send a notification to Slack

        Args:
            message: Message text
            level: Notification level (info, warning, error, success)
            channel: Override default channel
            attachments: Optional Slack message attachments

        Returns:
            bool: True if notification was sent successfully
        """
        # Always log the message
        if level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

        # Check if we should throttle notifications
        current_time = time.time()
        if current_time - self.last_reset_time > 3600:  # 1 hour
            self.notification_count = 0
            self.last_reset_time = current_time

        if self.notification_count >= self.max_notifications_per_hour:
            logger.warning(f"Notification throttled: {message}")
            return False

        if not self.slack_available:
            return False

        try:
            # Format message with appropriate emoji based on level
            if level == 'error':
                formatted_message = f"🚨 *ERROR:* {message}"
            elif level == 'warning':
                formatted_message = f"⚠️ *WARNING:* {message}"
            elif level == 'success':
                formatted_message = f"✅ *SUCCESS:* {message}"
            else:  # info
                formatted_message = f"ℹ️ *INFO:* {message}"

            # Determine which webhook to use based on channel
            webhook_url = self.notification_webhook
            if channel == self.reports_channel and self.reports_webhook:
                webhook_url = self.reports_webhook
            elif channel == self.portfolio_channel and self.portfolio_webhook:
                webhook_url = self.portfolio_webhook
            elif channel == self.positions_channel and self.positions_webhook:
                webhook_url = self.positions_webhook

            # Prepare payload
            payload = {
                "text": formatted_message,
                "channel": channel or self.notification_channel
            }

            if attachments:
                payload["attachments"] = attachments

            # Send notification
            if webhook_url:
                response = requests.post(webhook_url, json=payload)
                success = response.status_code == 200
            elif self.bot_token:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=headers,
                    json=payload
                )
                success = response.status_code == 200 and response.json().get("ok", False)
            else:
                success = False

            if success:
                self.notification_count += 1

            return success

        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False

    def send_error(self, message: str, details: Optional[str] = None) -> bool:
        """
        Send an error notification

        Args:
            message: Error message
            details: Optional error details

        Returns:
            bool: True if notification was sent successfully
        """
        full_message = message
        if details:
            full_message += f"\n```{details}```"

        return self.send_notification(full_message, level='error')

    def send_warning(self, message: str) -> bool:
        """
        Send a warning notification

        Args:
            message: Warning message

        Returns:
            bool: True if notification was sent successfully
        """
        return self.send_notification(message, level='warning')

    def send_success(self, message: str) -> bool:
        """
        Send a success notification

        Args:
            message: Success message

        Returns:
            bool: True if notification was sent successfully
        """
        return self.send_notification(message, level='success')

    def send_report(self, title: str, data: Dict[str, Any]) -> bool:
        """
        Send a detailed report

        Args:
            title: Report title
            data: Report data

        Returns:
            bool: True if report was sent successfully
        """
        # Format data as a code block
        formatted_data = json.dumps(data, indent=2)
        message = f"{title}\n```{formatted_data}```"

        return self.send_notification(message, channel=self.reports_channel)

    def send_portfolio_update(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Send a portfolio update

        Args:
            portfolio_data: Portfolio data

        Returns:
            bool: True if update was sent successfully
        """
        # Extract key metrics
        total_value = portfolio_data.get('total_value', 0)
        cash = portfolio_data.get('cash', 0)
        positions_value = portfolio_data.get('positions_value', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        daily_pnl_pct = portfolio_data.get('daily_pnl_pct', 0)

        # Format message
        message = f"*Portfolio Update*\n"
        message += f"Total Value: ${total_value:,.2f}\n"
        message += f"Cash: ${cash:,.2f}\n"
        message += f"Positions Value: ${positions_value:,.2f}\n"

        # Add PnL with emoji based on performance
        if daily_pnl > 0:
            message += f"Daily P&L: 📈 +${daily_pnl:,.2f} (+{daily_pnl_pct:.2f}%)\n"
        else:
            message += f"Daily P&L: 📉 ${daily_pnl:,.2f} ({daily_pnl_pct:.2f}%)\n"

        return self.send_notification(message, channel=self.portfolio_channel)

    def send_position_update(self, position_data: Dict[str, Any]) -> bool:
        """
        Send a position update

        Args:
            position_data: Position data

        Returns:
            bool: True if update was sent successfully
        """
        # Extract key information
        symbol = position_data.get('symbol', 'UNKNOWN')
        action = position_data.get('action', 'UNKNOWN')
        quantity = position_data.get('quantity', 0)
        price = position_data.get('price', 0)
        value = position_data.get('value', 0)
        reason = position_data.get('reason', '')

        # Format message based on action
        if action.lower() == 'buy':
            message = f"🟢 *BOUGHT {symbol}*\n"
        elif action.lower() == 'sell':
            message = f"🔴 *SOLD {symbol}*\n"
        else:
            message = f"⚪ *{action.upper()} {symbol}*\n"

        message += f"Quantity: {quantity}\n"
        message += f"Price: ${price:,.2f}\n"
        message += f"Value: ${value:,.2f}\n"

        if reason:
            message += f"Reason: {reason}\n"

        return self.send_notification(message, channel=self.positions_channel)


class MonitoringSystem:
    """
    Unified monitoring system for the trading platform
    """

    def __init__(self, redis_client=None, config=None):
        """
        Initialize the monitoring system

        Args:
            redis_client: Redis client for storing metrics
            config: Configuration parameters
        """
        self.redis = redis_client

        # Default configuration
        self.default_config = {
            'metrics_port': int(os.environ.get('METRICS_PORT', 8000)),
            'collection_interval': int(os.environ.get('METRICS_COLLECTION_INTERVAL', 60)),
            'system_metrics': {
                'cpu_usage_threshold': float(os.environ.get('CPU_USAGE_THRESHOLD', 80.0)),
                'memory_usage_threshold': float(os.environ.get('MEMORY_USAGE_THRESHOLD', 80.0)),
                'gpu_memory_threshold': float(os.environ.get('GPU_MEMORY_THRESHOLD', 90.0)),
                'disk_usage_threshold': float(os.environ.get('DISK_USAGE_THRESHOLD', 85.0)),
                'latency_threshold_ms': float(os.environ.get('LATENCY_THRESHOLD_MS', 100.0)),
            },
            'api_metrics': {
                'api_success_rate_threshold': float(os.environ.get('API_SUCCESS_RATE_THRESHOLD', 95.0)),
                'api_response_time_threshold': float(os.environ.get('API_RESPONSE_TIME_THRESHOLD', 500.0)),
                'websocket_reconnect_threshold': int(os.environ.get('WEBSOCKET_RECONNECT_THRESHOLD', 3)),
                'rate_limit_threshold_pct': float(os.environ.get('RATE_LIMIT_THRESHOLD_PCT', 80.0)),
            },
            'trading_metrics': {
                'max_drawdown_alert_threshold': float(os.environ.get('MAX_DRAWDOWN_ALERT_THRESHOLD', 0.05)),
                'win_rate_threshold': float(os.environ.get('WIN_RATE_THRESHOLD', 0.4)),
                'risk_reward_ratio_threshold': float(os.environ.get('RISK_REWARD_RATIO_THRESHOLD', 1.5)),
                'profit_factor_threshold': float(os.environ.get('PROFIT_FACTOR_THRESHOLD', 1.2)),
                'sharpe_ratio_threshold': float(os.environ.get('SHARPE_RATIO_THRESHOLD', 0.5)),
            },
            'alerts': {
                'enabled': os.environ.get('ALERTS_ENABLED', 'true').lower() == 'true',
                'slack_alerts': os.environ.get('SLACK_ALERTS_ENABLED', 'true').lower() == 'true',
                'console_alerts': os.environ.get('CONSOLE_ALERTS_ENABLED', 'true').lower() == 'true',
                'max_alerts_per_hour': int(os.environ.get('MAX_ALERTS_PER_HOUR', 20)),
            }
        }

        # Update with provided config
        self.config = self.default_config.copy()
        if config:
            self._update_config_recursive(self.config, config)

        # Initialize Slack notifier with config
        self.slack = SlackNotifier(config=self.config)

        # Initialize Prometheus metrics if available
        self.metrics = {}
        self.initialize_prometheus_metrics()

        # Initialize GPU monitoring if available
        self.gpu_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_initialized = True
                logger.info("GPU monitoring initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize GPU monitoring: {str(e)}")

        # Start monitoring thread
        self.running = False
        self.monitor_thread = None

        logger.info("Monitoring system initialized")

    def _update_config_recursive(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary with values from another dictionary

        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    # If both values are dictionaries, update recursively
                    self._update_config_recursive(target[key], value)
                else:
                    # Otherwise, update the value directly
                    target[key] = value
            else:
                # If the key doesn't exist in the target, add it
                target[key] = value

    def initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Prometheus client not available, skipping metrics initialization")
            return

        try:
            # Start Prometheus HTTP server
            prom.start_http_server(self.config['metrics_port'])
            logger.info(
                f"Prometheus metrics server started on port {self.config['metrics_port']}")

            # System metrics
            self.metrics['system'] = {
                'cpu_usage': Gauge('system_cpu_usage_percent', 'CPU usage percentage'),
                'memory_usage': Gauge('system_memory_usage_percent', 'Memory usage percentage'),
                'disk_usage': Gauge('system_disk_usage_percent', 'Disk usage percentage'),
                'uptime': Gauge('system_uptime_seconds', 'System uptime in seconds'),
            }

            # GPU metrics
            self.metrics['gpu'] = {
                'memory_usage': Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage', ['device']),
                'gpu_utilization': Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['device']),
                'temperature': Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius', ['device']),
                'power_usage': Gauge('gpu_power_usage_watts', 'GPU power usage in watts', ['device']),
            }

            # API metrics
            self.metrics['api'] = {
                'request_count': Counter('api_request_count', 'API request count', ['endpoint', 'method']),
                'request_latency': Histogram('api_request_latency_seconds', 'API request latency in seconds', ['endpoint', 'method']),
                'error_count': Counter('api_error_count', 'API error count', ['endpoint', 'method', 'error_type']),
                'rate_limit_remaining': Gauge('api_rate_limit_remaining', 'API rate limit remaining', ['endpoint']),
                'websocket_reconnects': Counter('websocket_reconnects', 'WebSocket reconnection count', ['endpoint']),
            }

            # Trading metrics
            self.metrics['trading'] = {
                'order_count': Counter('trading_order_count', 'Order count', ['symbol', 'side', 'type']),
                'order_value': Counter('trading_order_value_dollars', 'Order value in dollars', ['symbol', 'side']),
                'position_count': Gauge('trading_position_count', 'Number of open positions'),
                'position_value': Gauge('trading_position_value_dollars', 'Total value of open positions'),
                'cash_balance': Gauge('trading_cash_balance_dollars', 'Cash balance in dollars'),
                'total_equity': Gauge('trading_total_equity_dollars', 'Total equity in dollars'),
                'daily_pnl': Gauge('trading_daily_pnl_dollars', 'Daily profit and loss in dollars'),
                'daily_pnl_percent': Gauge('trading_daily_pnl_percent', 'Daily profit and loss percentage'),
            }

            # ML metrics
            self.metrics['ml'] = {
                'prediction_count': Counter('ml_prediction_count', 'Prediction count', ['model', 'symbol']),
                'prediction_latency': Histogram('ml_prediction_latency_seconds', 'Prediction latency in seconds', ['model']),
                'model_accuracy': Gauge('ml_model_accuracy', 'Model accuracy', ['model']),
                'model_f1_score': Gauge('ml_model_f1_score', 'Model F1 score', ['model']),
                'training_duration': Gauge('ml_training_duration_seconds', 'Model training duration in seconds', ['model']),
                'feature_importance': Gauge('ml_feature_importance', 'Feature importance', ['model', 'feature']),
            }

            # Data pipeline metrics
            self.metrics['data'] = {
                'data_freshness': Gauge('data_freshness_seconds', 'Data freshness in seconds', ['data_type']),
                'data_processing_time': Histogram('data_processing_time_seconds', 'Data processing time in seconds', ['operation']),
                'data_size': Gauge('data_size_bytes', 'Data size in bytes', ['data_type']),
                'cache_hit_ratio': Gauge('data_cache_hit_ratio', 'Cache hit ratio'),
                'api_data_points': Counter('data_api_points_count', 'Number of data points from API', ['source', 'data_type']),
            }

            logger.info("Prometheus metrics initialized")

        except Exception as e:
            logger.error(f"Error initializing Prometheus metrics: {str(e)}")

    def start(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring system is already running")
            return

        self.running = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Send startup notification
        if self.config['alerts']['slack_alerts']:
            hostname = socket.gethostname()
            self.slack.send_success(f"Trading system started on {hostname}")

        logger.info("Monitoring system started")

    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            logger.warning("Monitoring system is not running")
            return

        self.running = False

        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # Send shutdown notification
        if self.config['alerts']['slack_alerts']:
            hostname = socket.gethostname()
            self.slack.send_notification(
                f"Trading system stopped on {hostname}")

        logger.info("Monitoring system stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect GPU metrics
                if self.gpu_initialized:
                    self._collect_gpu_metrics()

                # Sleep until next collection
                time.sleep(self.config['collection_interval'])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Sleep briefly before retrying

    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Uptime
            uptime = time.time() - psutil.boot_time()

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and 'system' in self.metrics:
                self.metrics['system']['cpu_usage'].set(cpu_percent)
                self.metrics['system']['memory_usage'].set(memory_percent)
                self.metrics['system']['disk_usage'].set(disk_percent)
                self.metrics['system']['uptime'].set(uptime)

            # Store in Redis if available
            if self.redis:
                self.redis.hset('system:metrics', 'cpu_percent', cpu_percent)
                self.redis.hset('system:metrics',
                                'memory_percent', memory_percent)
                self.redis.hset('system:metrics', 'disk_percent', disk_percent)
                self.redis.hset('system:metrics', 'uptime', uptime)

            # Check thresholds and send alerts if needed
            if cpu_percent > self.config['system_metrics']['cpu_usage_threshold']:
                message = f"High CPU usage: {cpu_percent:.1f}%"
                logger.warning(message)
                if self.config['alerts']['slack_alerts']:
                    self.slack.send_warning(message)

            if memory_percent > self.config['system_metrics']['memory_usage_threshold']:
                message = f"High memory usage: {memory_percent:.1f}%"
                logger.warning(message)
                if self.config['alerts']['slack_alerts']:
                    self.slack.send_warning(message)

            if disk_percent > self.config['system_metrics']['disk_usage_threshold']:
                message = f"High disk usage: {disk_percent:.1f}%"
                logger.warning(message)
                if self.config['alerts']['slack_alerts']:
                    self.slack.send_warning(message)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")

    def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        if not self.gpu_initialized:
            return

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # Memory usage
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_percent = (memory.used / memory.total) * 100

                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization.gpu

                # Temperature
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)

                # Power usage
                power_usage = pynvml.nvmlDeviceGetPowerUsage(
                    handle) / 1000.0  # Convert from mW to W

                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and 'gpu' in self.metrics:
                    self.metrics['gpu']['memory_usage'].labels(
                        device=name).set(memory_percent)
                    self.metrics['gpu']['gpu_utilization'].labels(
                        device=name).set(gpu_utilization)
                    self.metrics['gpu']['temperature'].labels(
                        device=name).set(temperature)
                    self.metrics['gpu']['power_usage'].labels(
                        device=name).set(power_usage)

                # Store in Redis if available
                if self.redis:
                    self.redis.hset(f'gpu:{i}:metrics', 'name', name)
                    self.redis.hset(f'gpu:{i}:metrics',
                                    'memory_percent', memory_percent)
                    self.redis.hset(f'gpu:{i}:metrics',
                                    'gpu_utilization', gpu_utilization)
                    self.redis.hset(f'gpu:{i}:metrics',
                                    'temperature', temperature)
                    self.redis.hset(f'gpu:{i}:metrics',
                                    'power_usage', power_usage)

                # Check thresholds and send alerts if needed
                if memory_percent > self.config['system_metrics']['gpu_memory_threshold']:
                    message = f"High GPU memory usage on {name}: {memory_percent:.1f}%"
                    logger.warning(message)
                    if self.config['alerts']['slack_alerts']:
                        self.slack.send_warning(message)

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {str(e)}")

    def record_api_request(self, endpoint, method, latency, success, error_type=None):
        """
        Record API request metrics

        Args:
            endpoint: API endpoint
            method: HTTP method
            latency: Request latency in seconds
            success: Whether the request was successful
            error_type: Error type if the request failed
        """
        if PROMETHEUS_AVAILABLE and 'api' in self.metrics:
            self.metrics['api']['request_count'].labels(
                endpoint=endpoint, method=method).inc()
            self.metrics['api']['request_latency'].labels(
                endpoint=endpoint, method=method).observe(latency)

            if not success and error_type:
                self.metrics['api']['error_count'].labels(
                    endpoint=endpoint, method=method, error_type=error_type).inc()

        # Store in Redis if available
        if self.redis:
            # Update request count
            self.redis.hincrby(
                f'api:{endpoint}:metrics', f'{method}_requests', 1)

            # Update latency stats
            self.redis.lpush(f'api:{endpoint}:latency', latency)
            # Keep last 1000 latency values
            self.redis.ltrim(f'api:{endpoint}:latency', 0, 999)

            # Update error count if applicable
            if not success:
                self.redis.hincrby(
                    f'api:{endpoint}:metrics', f'{method}_errors', 1)
                if error_type:
                    self.redis.hincrby(
                        f'api:{endpoint}:metrics', f'error_{error_type}', 1)

        # Check if latency exceeds threshold and alert if needed
        # Convert ms to seconds
        if latency > (self.config['api_metrics']['api_response_time_threshold'] / 1000.0):
            message = f"High API latency for {endpoint} ({method}): {latency*1000:.1f}ms"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

    def record_rate_limit(self, endpoint, remaining, limit):
        """
        Record API rate limit metrics

        Args:
            endpoint: API endpoint
            remaining: Remaining requests
            limit: Total limit
        """
        if not remaining or not limit:
            return

        # Calculate percentage remaining
        percent_remaining = (remaining / limit) * 100

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'api' in self.metrics:
            self.metrics['api']['rate_limit_remaining'].labels(
                endpoint=endpoint).set(remaining)

        # Store in Redis if available
        if self.redis:
            self.redis.hset(f'api:{endpoint}:rate_limit',
                            'remaining', remaining)
            self.redis.hset(f'api:{endpoint}:rate_limit', 'limit', limit)
            self.redis.hset(f'api:{endpoint}:rate_limit',
                            'percent_remaining', percent_remaining)

        # Alert if rate limit is getting low
        threshold = self.config['api_metrics']['rate_limit_threshold_pct']
        if percent_remaining < threshold:
            message = f"API rate limit for {endpoint} is low: {percent_remaining:.1f}% remaining ({remaining}/{limit})"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

    def record_websocket_reconnect(self, endpoint):
        """
        Record WebSocket reconnection

        Args:
            endpoint: WebSocket endpoint
        """
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'api' in self.metrics:
            self.metrics['api']['websocket_reconnects'].labels(
                endpoint=endpoint).inc()

        # Store in Redis if available
        if self.redis:
            # Increment reconnect count
            reconnect_count = self.redis.hincrby(
                f'websocket:{endpoint}:metrics', 'reconnect_count', 1)

            # Store timestamp of last reconnect
            self.redis.hset(
                f'websocket:{endpoint}:metrics', 'last_reconnect', time.time())

            # Alert if reconnect count exceeds threshold
            threshold = self.config['api_metrics']['websocket_reconnect_threshold']
            if reconnect_count >= threshold:
                message = f"WebSocket {endpoint} has reconnected {reconnect_count} times"
                logger.warning(message)
                if self.config['alerts']['slack_alerts']:
                    self.slack.send_warning(message)

    def record_order(self, symbol, side, order_type, quantity, price, value):
        """
        Record order metrics

        Args:
            symbol: Stock symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit/etc)
            quantity: Order quantity
            price: Order price
            value: Order value (quantity * price)
        """
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'trading' in self.metrics:
            self.metrics['trading']['order_count'].labels(
                symbol=symbol, side=side, type=order_type).inc()
            self.metrics['trading']['order_value'].labels(
                symbol=symbol, side=side).inc(value)

        # Store in Redis if available
        if self.redis:
            # Store order details
            order_id = f"{symbol}_{side}_{int(time.time())}"
            self.redis.hset(f'order:{order_id}', 'symbol', symbol)
            self.redis.hset(f'order:{order_id}', 'side', side)
            self.redis.hset(f'order:{order_id}', 'type', order_type)
            self.redis.hset(f'order:{order_id}', 'quantity', quantity)
            self.redis.hset(f'order:{order_id}', 'price', price)
            self.redis.hset(f'order:{order_id}', 'value', value)
            self.redis.hset(f'order:{order_id}', 'timestamp', time.time())

            # Add to recent orders list
            self.redis.lpush('recent_orders', order_id)
            self.redis.ltrim('recent_orders', 0, 99)  # Keep last 100 orders

            # Update symbol-specific metrics
            self.redis.hincrby(f'symbol:{symbol}:metrics', f'{side}_orders', 1)
            self.redis.hincrby(
                f'symbol:{symbol}:metrics', f'{side}_quantity', quantity)
            self.redis.hincrbyfloat(
                f'symbol:{symbol}:metrics', f'{side}_value', value)

        # Send position update notification
        if self.config['alerts']['slack_alerts']:
            position_data = {
                'symbol': symbol,
                'action': side,
                'quantity': quantity,
                'price': price,
                'value': value
            }
            self.slack.send_position_update(position_data)

    def update_portfolio_metrics(self, portfolio_data):
        """
        Update portfolio metrics

        Args:
            portfolio_data: Dictionary with portfolio metrics
        """
        # Extract key metrics
        total_equity = portfolio_data.get('total_value', 0)
        cash_balance = portfolio_data.get('cash', 0)
        position_value = portfolio_data.get('positions_value', 0)
        position_count = portfolio_data.get('position_count', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        daily_pnl_pct = portfolio_data.get('daily_pnl_pct', 0)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'trading' in self.metrics:
            self.metrics['trading']['total_equity'].set(total_equity)
            self.metrics['trading']['cash_balance'].set(cash_balance)
            self.metrics['trading']['position_value'].set(position_value)
            self.metrics['trading']['position_count'].set(position_count)
            self.metrics['trading']['daily_pnl'].set(daily_pnl)
            self.metrics['trading']['daily_pnl_percent'].set(daily_pnl_pct)

        # Store in Redis if available
        if self.redis:
            self.redis.hset('portfolio:metrics', 'total_equity', total_equity)
            self.redis.hset('portfolio:metrics', 'cash_balance', cash_balance)
            self.redis.hset('portfolio:metrics',
                            'position_value', position_value)
            self.redis.hset('portfolio:metrics',
                            'position_count', position_count)
            self.redis.hset('portfolio:metrics', 'daily_pnl', daily_pnl)
            self.redis.hset('portfolio:metrics',
                            'daily_pnl_pct', daily_pnl_pct)
            self.redis.hset('portfolio:metrics', 'last_update', time.time())

            # Store historical data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d')
            self.redis.hset(
                f'portfolio:history:{timestamp}', 'total_equity', total_equity)
            self.redis.hset(
                f'portfolio:history:{timestamp}', 'daily_pnl', daily_pnl)
            self.redis.hset(
                f'portfolio:history:{timestamp}', 'daily_pnl_pct', daily_pnl_pct)

        # Check for significant drawdown and alert if needed
        if daily_pnl_pct < -self.config['trading_metrics']['max_drawdown_alert_threshold'] * 100:
            message = f"Significant drawdown detected: {daily_pnl_pct:.2f}% (${daily_pnl:.2f})"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

        # Send portfolio update notification
        if self.config['alerts']['slack_alerts']:
            self.slack.send_portfolio_update(portfolio_data)

    def record_ml_prediction(self, model_name, symbol, prediction_time, prediction_value=None):
        """
        Record ML prediction metrics

        Args:
            model_name: Name of the ML model
            symbol: Stock symbol
            prediction_time: Time taken for prediction in seconds
            prediction_value: Optional prediction value
        """
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'ml' in self.metrics:
            self.metrics['ml']['prediction_count'].labels(
                model=model_name, symbol=symbol).inc()
            self.metrics['ml']['prediction_latency'].labels(
                model=model_name).observe(prediction_time)

        # Store in Redis if available
        if self.redis:
            # Update prediction count
            self.redis.hincrby(
                f'ml:{model_name}:metrics', 'prediction_count', 1)
            self.redis.hincrby(f'ml:{model_name}:metrics',
                               f'prediction_count_{symbol}', 1)

            # Store prediction latency
            self.redis.lpush(f'ml:{model_name}:latency', prediction_time)
            # Keep last 1000 latency values
            self.redis.ltrim(f'ml:{model_name}:latency', 0, 999)

            # Calculate average latency
            latency_values = self.redis.lrange(
                f'ml:{model_name}:latency', 0, -1)
            if latency_values:
                avg_latency = sum(float(v)
                                  for v in latency_values) / len(latency_values)
                self.redis.hset(f'ml:{model_name}:metrics',
                                'avg_latency', avg_latency)

            # Store prediction value if provided
            if prediction_value is not None:
                self.redis.hset(
                    f'ml:{model_name}:predictions:{symbol}', 'value', prediction_value)
                self.redis.hset(
                    f'ml:{model_name}:predictions:{symbol}', 'timestamp', time.time())

        # Check if prediction latency is too high and alert if needed
        if prediction_time > self.config['system_metrics']['latency_threshold_ms'] / 1000.0:
            message = f"High prediction latency for {model_name}: {prediction_time*1000:.1f}ms"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

    def update_ml_model_metrics(self, model_name, metrics):
        """
        Update ML model metrics

        Args:
            model_name: Name of the ML model
            metrics: Dictionary with model metrics
        """
        # Extract key metrics
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1', 0)
        training_duration = metrics.get('training_duration', 0)
        feature_importance = metrics.get('feature_importance', {})

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'ml' in self.metrics:
            self.metrics['ml']['model_accuracy'].labels(
                model=model_name).set(accuracy)
            self.metrics['ml']['model_f1_score'].labels(
                model=model_name).set(f1_score)
            self.metrics['ml']['training_duration'].labels(
                model=model_name).set(training_duration)

            # Update feature importance metrics
            for feature, importance in feature_importance.items():
                self.metrics['ml']['feature_importance'].labels(
                    model=model_name, feature=feature).set(importance)

        # Store in Redis if available
        if self.redis:
            # Store model metrics
            self.redis.hset(f'ml:{model_name}:metrics', 'accuracy', accuracy)
            self.redis.hset(f'ml:{model_name}:metrics', 'f1_score', f1_score)
            self.redis.hset(f'ml:{model_name}:metrics',
                            'training_duration', training_duration)
            self.redis.hset(f'ml:{model_name}:metrics',
                            'last_update', time.time())

            # Store feature importance
            for feature, importance in feature_importance.items():
                self.redis.hset(
                    f'ml:{model_name}:feature_importance', feature, importance)

        # Send model training report
        if self.config['alerts']['slack_alerts']:
            report_data = {
                'model': model_name,
                'accuracy': f"{accuracy:.4f}",
                'f1_score': f"{f1_score:.4f}",
                'training_duration': f"{training_duration:.2f}s",
                'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            self.slack.send_report(
                f"Model Training: {model_name}", report_data)

    def record_data_processing(self, operation, processing_time, data_size=None, data_type=None):
        """
        Record data processing metrics

        Args:
            operation: Data processing operation
            processing_time: Time taken for processing in seconds
            data_size: Optional size of data in bytes
            data_type: Optional type of data
        """
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'data' in self.metrics:
            self.metrics['data']['data_processing_time'].labels(
                operation=operation).observe(processing_time)

            if data_size is not None and data_type is not None:
                self.metrics['data']['data_size'].labels(
                    data_type=data_type).set(data_size)

        # Store in Redis if available
        if self.redis:
            # Store processing time
            self.redis.lpush(
                f'data:{operation}:processing_time', processing_time)
            # Keep last 1000 values
            self.redis.ltrim(f'data:{operation}:processing_time', 0, 999)

            # Calculate average processing time
            times = self.redis.lrange(
                f'data:{operation}:processing_time', 0, -1)
            if times:
                avg_time = sum(float(t) for t in times) / len(times)
                self.redis.hset(
                    'data:metrics', f'{operation}_avg_time', avg_time)

            # Store data size if provided
            if data_size is not None and data_type is not None:
                self.redis.hset('data:metrics', f'{data_type}_size', data_size)

        # Check if processing time is too high and alert if needed
        if processing_time > self.config['system_metrics']['latency_threshold_ms'] / 1000.0:
            message = f"High data processing time for {operation}: {processing_time*1000:.1f}ms"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

    def update_data_freshness(self, data_type, timestamp):
        """
        Update data freshness metrics

        Args:
            data_type: Type of data
            timestamp: Timestamp of the data
        """
        # Calculate freshness in seconds
        current_time = time.time()
        if isinstance(timestamp, str):
            try:
                # Try to parse ISO format
                dt = datetime.datetime.fromisoformat(
                    timestamp.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
            except ValueError:
                # Try to parse Unix timestamp
                try:
                    timestamp = float(timestamp)
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {timestamp}")
                    return

        freshness = current_time - timestamp

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'data' in self.metrics:
            self.metrics['data']['data_freshness'].labels(
                data_type=data_type).set(freshness)

        # Store in Redis if available
        if self.redis:
            self.redis.hset('data:freshness', data_type, freshness)
            self.redis.hset('data:freshness',
                            f'{data_type}_timestamp', timestamp)

        # Check if data is too old and alert if needed
        threshold = self.config['system_metrics']['data_freshness_threshold_sec']
        if threshold and freshness > threshold:
            message = f"Data for {data_type} is stale: {freshness:.1f}s old"
            logger.warning(message)
            if self.config['alerts']['slack_alerts']:
                self.slack.send_warning(message)

    def get_system_status(self):
        """
        Get overall system status

        Returns:
            Dictionary with system status information
        """
        status = {
            'timestamp': time.time(),
            'hostname': socket.gethostname(),
            'uptime': 0,
            'system': {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0
            },
            'gpu': [],
            'trading': {
                'total_equity': 0,
                'cash_balance': 0,
                'position_count': 0,
                'daily_pnl_pct': 0
            },
            'api': {
                'request_count': 0,
                'error_count': 0,
                'avg_latency': 0
            },
            'ml': {
                'prediction_count': 0,
                'avg_latency': 0
            }
        }

        # Get system metrics from Redis if available
        if self.redis:
            # System metrics
            system_metrics = self.redis.hgetall('system:metrics')
            if system_metrics:
                status['uptime'] = float(system_metrics.get('uptime', 0))
                status['system']['cpu_percent'] = float(
                    system_metrics.get('cpu_percent', 0))
                status['system']['memory_percent'] = float(
                    system_metrics.get('memory_percent', 0))
                status['system']['disk_percent'] = float(
                    system_metrics.get('disk_percent', 0))

            # GPU metrics
            gpu_keys = self.redis.keys('gpu:*:metrics')
            for key in gpu_keys:
                gpu_id = key.split(':')[1]
                gpu_metrics = self.redis.hgetall(key)
                if gpu_metrics:
                    status['gpu'].append({
                        'id': gpu_id,
                        'name': gpu_metrics.get('name', 'Unknown'),
                        'memory_percent': float(gpu_metrics.get('memory_percent', 0)),
                        'gpu_utilization': float(gpu_metrics.get('gpu_utilization', 0)),
                        'temperature': float(gpu_metrics.get('temperature', 0))
                    })

            # Trading metrics
            portfolio_metrics = self.redis.hgetall('portfolio:metrics')
            if portfolio_metrics:
                status['trading']['total_equity'] = float(
                    portfolio_metrics.get('total_equity', 0))
                status['trading']['cash_balance'] = float(
                    portfolio_metrics.get('cash_balance', 0))
                status['trading']['position_count'] = int(
                    portfolio_metrics.get('position_count', 0))
                status['trading']['daily_pnl_pct'] = float(
                    portfolio_metrics.get('daily_pnl_pct', 0))

        return status
