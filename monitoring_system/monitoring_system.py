#!/usr/bin/env python3
"""
Monitoring System

This module provides a production-ready monitoring and observability system that tracks:
1. System health metrics (CPU, memory, disk usage)
2. Data quality and freshness metrics
3. Trading performance metrics
4. Queue and latency metrics
5. Alert generation and management

The system uses Prometheus for metrics collection and provides comprehensive
observability for the entire trading platform.
"""

import logging
import time
import json
import threading
import queue
import datetime
import pytz
import psutil
import os
import numpy as np
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter, Summary
import socket

# Environment variables
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6380))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
METRICS_PORT = int(os.environ.get('METRICS_PORT', 8000))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitoring_system')


class MonitoringSystem:
    """Production-ready monitoring and observability system"""

    def __init__(self, redis_client):
        self.redis = redis_client

        # Configuration
        self.config = {
            'metrics_port': METRICS_PORT,
            'log_level': logging.INFO,
            'alert_thresholds': {
                'api_error_rate': 0.05,        # 5% error rate
                'data_freshness': 30,          # 30 seconds
                'signal_to_execution': 2.0,    # 2 seconds
                'memory_usage': 85.0,          # 85% usage
                'cpu_usage': 80.0,             # 80% usage
                'active_signals': 20,          # 20 active signals
                'queue_depth': 50              # 50 items in any queue
            },
            'update_interval': 10,             # 10 seconds
            'retention_period': 86400          # 1 day
        }

        # Initialize Prometheus metrics
        self._setup_metrics()

        # State tracking
        self.running = False
        self.threads = []
        self.system_status = "initializing"
        self.component_status = {
            "data_ingestion": "unknown",
            "stock_selection": "unknown",
            "model_integration": "unknown",
            "execution": "unknown"
        }
        self.active_alerts = {}

        logger.info("Monitoring System initialized")

    def _find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        port = start_port
        attempts = 0

        while attempts < max_attempts:
            try:
                # Try to open a socket on the port
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('0.0.0.0', port))
                s.close()
                # If we get here, the port is available
                return port
            except OSError:
                port += 1
                attempts += 1
        return None  # No available ports found within range

    def start(self):
        """Start the monitoring system"""
        if self.running:
            logger.warning("Monitoring system already running")
            return

        self.running = True
        logger.info("Starting monitoring system")

        # Start Prometheus metrics server
        port = self._find_available_port(self.config['metrics_port'])
        if port is None:
            error_msg = f"Could not find an available port after trying {self.config['metrics_port']} to {self.config['metrics_port'] + 9}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Update the config with the port we're actually using
        self.config['metrics_port'] = port
        start_http_server(port=port, addr='0.0.0.0')
        logger.info(f"Started Prometheus metrics server on port {port}")

        # Start worker threads
        self.threads.append(threading.Thread(
            target=self._system_metrics_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._data_metrics_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._trading_metrics_worker, daemon=True))
        self.threads.append(threading.Thread(
            target=self._alert_worker, daemon=True))

        for thread in self.threads:
            thread.start()

        # Set system status to running
        self.system_status = "running"
        self.metrics['system_status'].set(1)

        logger.info(
            f"Monitoring system started on port {self.config['metrics_port']}")

    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            logger.warning("Monitoring system not running")
            return

        logger.info("Stopping monitoring system")
        self.running = False

        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)

        # Set system status to stopped
        self.system_status = "stopped"
        self.metrics['system_status'].set(0)

        logger.info("Monitoring system stopped")

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            # System metrics
            'system_status': Gauge('trading_system_status', 'Trading system status (1=running, 0=stopped)'),
            'cpu_usage': Gauge('cpu_usage', 'CPU usage percentage'),
            'memory_usage': Gauge('memory_usage', 'Memory usage percentage'),
            'disk_usage': Gauge('disk_usage', 'Disk usage percentage'),
            'process_uptime': Gauge('process_uptime', 'Process uptime in seconds'),

            # Data metrics
            'api_requests': Counter('api_requests_total', 'Total API requests', ['api', 'endpoint']),
            'api_errors': Counter('api_errors_total', 'Total API errors', ['api', 'endpoint']),
            'data_freshness': Gauge('data_freshness', 'Data freshness in seconds', ['data_type']),
            'data_completeness': Gauge('data_completeness', 'Data completeness percentage', ['data_type']),
            'redis_keys': Gauge('redis_keys', 'Number of keys in Redis', ['prefix']),
            'redis_memory': Gauge('redis_memory', 'Redis memory usage in MB'),

            # Trading metrics
            'active_positions': Gauge('trading_active_positions', 'Number of active positions'),
            'pending_orders': Gauge('trading_pending_orders', 'Number of pending orders'),
            'signals_generated': Counter('trading_signals_generated_total', 'Total signals generated', ['source']),
            'trades_executed': Counter('trading_trades_executed_total', 'Total trades executed', ['direction']),
            'pnl_total': Gauge('trading_pnl_total', 'Total P&L in dollars'),
            'pnl_daily': Gauge('trading_pnl_daily', 'Daily P&L in dollars'),
            'win_rate': Gauge('trading_win_rate', 'Win rate percentage'),
            'average_profit': Gauge('trading_average_profit', 'Average profit per trade in dollars'),
            'average_loss': Gauge('trading_average_loss', 'Average loss per trade in dollars'),
            'max_drawdown': Gauge('trading_max_drawdown', 'Maximum drawdown percentage'),
            'exposure': Gauge('exposure', 'Current exposure in dollars'),

            # Queue metrics
            'queue_depth': Gauge('queue_depth', 'Queue depth', ['queue_name']),
            'queue_latency': Summary('queue_latency', 'Queue processing latency in ms', ['queue_name']),

            # Model metrics
            'model_prediction_count': Counter('trading_model_prediction_count', 'Number of model predictions', ['model_name']),
            'model_inference_time': Summary('trading_model_inference_time', 'Model inference time in ms', ['model_name']),
            'signal_quality': Gauge('trading_signal_quality', 'Signal quality score', ['model_name'])
        }

    def _system_metrics_worker(self):
        """Worker thread for system metrics collection"""
        logger.info("Starting system metrics worker")
        start_time = time.time()

        while self.running:
            try:
                # Collect CPU metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].set(cpu_usage)

                # Collect memory metrics
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].set(memory.percent)

                # Collect disk metrics
                disk = psutil.disk_usage('/')
                self.metrics['disk_usage'].set(disk.percent)

                # Process uptime
                uptime = time.time() - start_time
                self.metrics['process_uptime'].set(uptime)

                # Check system health
                self._check_system_health()

                # Sleep until next update
                time.sleep(self.config['update_interval'])

            except Exception as e:
                logger.error(f"Error in system metrics worker: {str(e)}")
                time.sleep(10.0)

    def _data_metrics_worker(self):
        """Worker thread for data metrics collection"""
        logger.info("Starting data metrics worker")

        while self.running:
            try:
                # Collect Redis metrics
                self._collect_redis_metrics()

                # Check data freshness
                self._check_data_freshness()

                # Track API metrics
                self._track_api_metrics()

                # Sleep until next update
                time.sleep(self.config['update_interval'])

            except Exception as e:
                logger.error(f"Error in data metrics worker: {str(e)}")
                time.sleep(10.0)

    def _trading_metrics_worker(self):
        """Worker thread for trading metrics collection"""
        logger.info("Starting trading metrics worker")

        while self.running:
            try:
                # Collect position metrics
                self._collect_position_metrics()

                # Collect order metrics
                self._collect_order_metrics()

                # Collect performance metrics
                self._collect_performance_metrics()

                # Collect queue metrics
                self._collect_queue_metrics()

                # Sleep until next update
                time.sleep(self.config['update_interval'])

            except Exception as e:
                logger.error(f"Error in trading metrics worker: {str(e)}")
                time.sleep(10.0)

    def _alert_worker(self):
        """Worker thread for alert generation and handling"""
        logger.info("Starting alert worker")

        while self.running:
            try:
                # Check all alerting conditions
                self._check_alerting_conditions()

                # Sleep until next check
                time.sleep(10.0)

            except Exception as e:
                logger.error(f"Error in alert worker: {str(e)}")
                time.sleep(10.0)

    def _collect_redis_metrics(self):
        """Collect Redis metrics"""
        try:
            # Get Redis info
            info = self.redis.info()

            # Memory usage
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            self.metrics['redis_memory'].set(used_memory_mb)

            # Count keys by prefix
            prefixes = ['stock:', 'options:', 'darkpool:',
                        'watchlist:', 'positions:', 'orders:', 'signals:']

            for prefix in prefixes:
                count = len(self.redis.keys(f"{prefix}*"))
                self.metrics['redis_keys'].labels(prefix=prefix).set(count)

        except Exception as e:
            logger.error(f"Error collecting Redis metrics: {str(e)}")

    def _check_data_freshness(self):
        """Check freshness of critical data"""
        try:
            # Check stock data freshness
            for ticker in self._get_active_tickers():
                last_update = self.redis.hget(
                    f"stock:{ticker}:last_trade", "timestamp")

                if last_update:
                    last_update = float(last_update) / \
                        1000.0  # Convert ms to seconds
                    freshness = time.time() - last_update
                    self.metrics['data_freshness'].labels(
                        data_type=f"stock:{ticker}").set(freshness)

                    # Alert if data is stale
                    if freshness > self.config['alert_thresholds']['data_freshness']:
                        self._trigger_alert(
                            'data_freshness',
                            f"Data for {ticker} is {freshness:.1f}s old",
                            {
                                'ticker': ticker,
                                'freshness': freshness,
                                'threshold': self.config['alert_thresholds']['data_freshness']
                            }
                        )

            # Check market data freshness
            market_data_types = ['SPY', 'VIX']
            for data_type in market_data_types:
                market_data = self.redis.get(f"market:{data_type}:data")

                if market_data:
                    data = json.loads(market_data)
                    last_update = data.get('timestamp', 0)
                    freshness = time.time() - last_update
                    self.metrics['data_freshness'].labels(
                        data_type=f"market:{data_type}").set(freshness)

                    # Alert if market data is stale
                    # More lenient for market data
                    if freshness > self.config['alert_thresholds']['data_freshness'] * 2:
                        self._trigger_alert(
                            'market_data_freshness',
                            f"Market data for {data_type} is {freshness:.1f}s old",
                            {
                                'data_type': data_type,
                                'freshness': freshness,
                                'threshold': self.config['alert_thresholds']['data_freshness'] * 2
                            }
                        )

        except Exception as e:
            logger.error(f"Error checking data freshness: {str(e)}")

    def _track_api_metrics(self):
        """Track API metrics from Redis"""
        try:
            # Get API metrics from Redis
            api_metrics = self.redis.hgetall("metrics:api")

            for key, value in api_metrics.items():
                if b':' in key:
                    api, endpoint = key.decode('utf-8').split(':')
                    metrics = json.loads(value)

                    # Update Prometheus metrics
                    self.metrics['api_requests'].labels(
                        api=api, endpoint=endpoint).inc(metrics.get('requests', 0))
                    self.metrics['api_errors'].labels(
                        api=api, endpoint=endpoint).inc(metrics.get('errors', 0))

                    # Calculate error rate
                    if metrics.get('requests', 0) > 0:
                        error_rate = metrics.get(
                            'errors', 0) / metrics.get('requests', 0)

                        # Alert on high error rate
                        if error_rate > self.config['alert_thresholds']['api_error_rate']:
                            self._trigger_alert(
                                'api_error_rate',
                                f"High error rate for {api}:{endpoint}: {error_rate:.1%}",
                                {
                                    'api': api,
                                    'endpoint': endpoint,
                                    'error_rate': error_rate,
                                    'threshold': self.config['alert_thresholds']['api_error_rate'],
                                    'errors': metrics.get('errors', 0),
                                    'requests': metrics.get('requests', 0)
                                }
                            )

            # Reset counters in Redis
            self.redis.delete("metrics:api")

        except Exception as e:
            logger.error(f"Error tracking API metrics: {str(e)}")

    def _collect_position_metrics(self):
        """Collect position metrics"""
        try:
            # Get active positions
            positions = self.redis.hgetall("positions:active")

            # Count active positions
            self.metrics['active_positions'].set(len(positions))

            # Calculate total exposure
            total_exposure = 0.0

            for _, pos_data in positions.items():
                position = json.loads(pos_data)
                total_exposure += position.get('current_value', 0)

            self.metrics['exposure'].set(total_exposure)

        except Exception as e:
            logger.error(f"Error collecting position metrics: {str(e)}")

    def _collect_order_metrics(self):
        """Collect order metrics"""
        try:
            # Get pending orders
            orders = self.redis.hgetall("orders:pending")

            # Count pending orders
            self.metrics['pending_orders'].set(len(orders))

        except Exception as e:
            logger.error(f"Error collecting order metrics: {str(e)}")

    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            # Get daily stats
            daily_stats = self.redis.hgetall("execution:daily_stats")

            if daily_stats:
                # Convert bytes to string for keys
                stats = {k.decode('utf-8'): v for k, v in daily_stats.items()}

                # Update metrics
                if 'total_pnl' in stats:
                    self.metrics['pnl_daily'].set(float(stats['total_pnl']))

                if 'trades_executed' in stats:
                    trades = int(stats['trades_executed'])
                    profitable = int(stats.get('profitable_trades', 0))

                    if trades > 0:
                        win_rate = (profitable / trades) * 100
                        self.metrics['win_rate'].set(win_rate)

                if 'max_drawdown' in stats:
                    self.metrics['max_drawdown'].set(
                        float(stats['max_drawdown']) * 100)

                if 'current_exposure' in stats:
                    self.metrics['exposure'].set(
                        float(stats['current_exposure']))

            # Get historical trades
            trades = self.redis.hgetall("trades:history")

            if trades:
                # Calculate total P&L
                total_pnl = 0.0
                profitable_trades = []
                losing_trades = []

                for _, trade_data in trades.items():
                    trade = json.loads(trade_data)
                    pnl = trade.get('realized_pnl', 0)
                    total_pnl += pnl

                    if pnl > 0:
                        profitable_trades.append(pnl)
                    elif pnl < 0:
                        losing_trades.append(pnl)

                # Update metrics
                self.metrics['pnl_total'].set(total_pnl)

                # Update trade direction counts
                long_trades = sum(1 for _, t in trades.items()
                                  if json.loads(t).get('direction') == 'long')
                short_trades = sum(1 for _, t in trades.items(
                ) if json.loads(t).get('direction') == 'short')

                self.metrics['trades_executed'].labels(
                    direction='long').inc(long_trades)
                self.metrics['trades_executed'].labels(
                    direction='short').inc(short_trades)

                # Calculate average profit and loss
                if profitable_trades:
                    avg_profit = sum(profitable_trades) / \
                        len(profitable_trades)
                    self.metrics['average_profit'].set(avg_profit)

                if losing_trades:
                    avg_loss = sum(losing_trades) / len(losing_trades)
                    self.metrics['average_loss'].set(abs(avg_loss))

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")

    def _collect_queue_metrics(self):
        """Collect queue metrics"""
        try:
            # Get queue metrics from Redis
            queue_metrics = self.redis.hgetall("metrics:queues")

            if queue_metrics:
                for queue_name, metrics_data in queue_metrics.items():
                    queue_name = queue_name.decode('utf-8')
                    metrics = json.loads(metrics_data)

                    # Update Prometheus metrics
                    self.metrics['queue_depth'].labels(
                        queue_name=queue_name).set(metrics.get('depth', 0))

                    # Record latency observations
                    latencies = metrics.get('latencies', [])
                    for latency in latencies:
                        self.metrics['queue_latency'].labels(
                            queue_name=queue_name).observe(latency)

                    # Alert on deep queues
                    if metrics.get('depth', 0) > self.config['alert_thresholds']['queue_depth']:
                        self._trigger_alert(
                            'queue_depth',
                            f"Queue {queue_name} has high depth: {metrics.get('depth')}",
                            {
                                'queue_name': queue_name,
                                'depth': metrics.get('depth', 0),
                                'threshold': self.config['alert_thresholds']['queue_depth']
                            }
                        )

        except Exception as e:
            logger.error(f"Error collecting queue metrics: {str(e)}")

    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check CPU usage
            if self.metrics['cpu_usage']._value.get() > self.config['alert_thresholds']['cpu_usage']:
                self._trigger_alert(
                    'high_cpu',
                    f"High CPU usage: {self.metrics['cpu_usage']._value.get():.1f}%",
                    {
                        'cpu_usage': self.metrics['cpu_usage']._value.get(),
                        'threshold': self.config['alert_thresholds']['cpu_usage']
                    }
                )

            # Check memory usage
            if self.metrics['memory_usage']._value.get() > self.config['alert_thresholds']['memory_usage']:
                self._trigger_alert(
                    'high_memory',
                    f"High memory usage: {self.metrics['memory_usage']._value.get():.1f}%",
                    {
                        'memory_usage': self.metrics['memory_usage']._value.get(),
                        'threshold': self.config['alert_thresholds']['memory_usage']
                    }
                )

            # Check component status
            for component, status in self.component_status.items():
                if status != "running":
                    self._trigger_alert(
                        f"{component}_status",
                        f"Component {component} is not running: {status}",
                        {
                            'component': component,
                            'status': status
                        }
                    )

        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")

    def _check_alerting_conditions(self):
        """Check all alerting conditions"""
        try:
            # Most alerting is done in the specific metric collection methods
            # This method can handle any cross-cutting concerns

            # Check for active alerts that have been resolved
            resolved_alerts = []

            for alert_id, alert in self.active_alerts.items():
                if self._is_alert_resolved(alert):
                    # Handle alert resolution
                    self._resolve_alert(alert_id)
                    resolved_alerts.append(alert_id)

            # Remove resolved alerts
            for alert_id in resolved_alerts:
                self.active_alerts.pop(alert_id, None)

        except Exception as e:
            logger.error(f"Error checking alerting conditions: {str(e)}")

    def _trigger_alert(self, alert_type, message, data):
        """Trigger an alert"""
        try:
            # Create alert ID
            alert_id = f"{alert_type}:{int(time.time())}"

            # Skip if this type of alert is already active
            for active_id, active_alert in self.active_alerts.items():
                if active_alert['type'] == alert_type:
                    # Update existing alert with new data
                    active_alert['last_triggered'] = time.time()
                    active_alert['count'] += 1
                    active_alert['data'] = data
                    return

            # Create new alert
            alert = {
                'id': alert_id,
                'type': alert_type,
                'message': message,
                'data': data,
                'first_triggered': time.time(),
                'last_triggered': time.time(),
                'status': 'active',
                'count': 1
            }

            # Store alert
            self.active_alerts[alert_id] = alert

            # Log alert
            logger.warning(f"ALERT: {message}")

            # Store in Redis
            self.redis.hset("alerts:active", alert_id, json.dumps(alert))

        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")

    def _is_alert_resolved(self, alert):
        """Check if an alert has been resolved"""
        try:
            alert_type = alert['type']
            data = alert['data']

            # Check based on alert type
            if alert_type == 'high_cpu':
                return self.metrics['cpu_usage']._value.get() <= self.config['alert_thresholds']['cpu_usage']

            elif alert_type == 'high_memory':
                return self.metrics['memory_usage']._value.get() <= self.config['alert_thresholds']['memory_usage']

            elif alert_type == 'data_freshness':
                ticker = data.get('ticker')
                last_update = self.redis.hget(
                    f"stock:{ticker}:last_trade", "timestamp")

                if last_update:
                    last_update = float(last_update) / 1000.0
                    freshness = time.time() - last_update
                    return freshness <= self.config['alert_thresholds']['data_freshness']

                return False

            elif alert_type == 'api_error_rate':
                # Assume resolved after some time since we reset the counters
                return (time.time() - alert['last_triggered']) > 60

            elif alert_type == 'queue_depth':
                queue_name = data.get('queue_name')
                queue_metrics = self.redis.hget("metrics:queues", queue_name)

                if queue_metrics:
                    metrics = json.loads(queue_metrics)
                    return metrics.get('depth', 0) <= self.config['alert_thresholds']['queue_depth']

                return True  # Assume resolved if metrics are gone

            # Default: consider resolved after 5 minutes
            return (time.time() - alert['last_triggered']) > 300

        except Exception as e:
            logger.error(f"Error checking if alert is resolved: {str(e)}")
            return False

    def _resolve_alert(self, alert_id):
        """Resolve an alert"""
        try:
            # Get alert
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return

            # Update status
            alert['status'] = 'resolved'
            alert['resolved_at'] = time.time()

            # Log resolution
            logger.info(f"RESOLVED: {alert['message']}")

            # Move to resolved alerts
            self.redis.hdel("alerts:active", alert_id)
            self.redis.hset("alerts:resolved", alert_id, json.dumps(alert))

        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")

    def _get_active_tickers(self):
        """Get list of active tickers (watchlist + positions)"""
        try:
            # Get focused watchlist
            watchlist = self.redis.zrange("watchlist:focused", 0, -1)

            # Convert to strings
            watchlist = [ticker.decode(
                'utf-8') if isinstance(ticker, bytes) else ticker for ticker in watchlist]

            # Get active positions
            positions = self.redis.hgetall("positions:active")
            position_tickers = []

            for key in positions:
                ticker = key.decode('utf-8').split(':')[0]
                position_tickers.append(ticker)

            # Combine and deduplicate
            active_tickers = list(set(watchlist + position_tickers))

            return active_tickers

        except Exception as e:
            logger.error(f"Error getting active tickers: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    import redis

    # Create Redis client
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )

    # Create monitoring system
    monitoring_system = MonitoringSystem(redis_client)

    # Start system
    monitoring_system.start()

    try:
        # Run for a while
        print("Monitoring system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        monitoring_system.stop()
