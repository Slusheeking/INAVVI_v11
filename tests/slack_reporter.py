#!/usr/bin/env python3
"""
Slack Reporter Module

This module provides Slack reporting capabilities for the ML model trainer.
It includes classes for sending notifications to Slack and tracking GPU statistics.
"""

import os
import time
import json
import logging
import threading
import subprocess
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('slack_reporter')

class GPUStatsTracker:
    """Track GPU statistics during model training"""
    
    def __init__(self, polling_interval=5.0):
        """Initialize GPU stats tracker
        
        Args:
            polling_interval (float): How often to poll GPU stats in seconds
        """
        self.polling_interval = polling_interval
        self.running = False
        self.thread = None
        self.stats = {
            'max_memory_used': 0,
            'max_utilization': 0,
            'avg_utilization': 0,
            'max_temperature': 0,
            'samples': []
        }
    
    def _poll_gpu_stats(self):
        """Poll GPU statistics in a background thread"""
        total_util = 0
        sample_count = 0
        
        while self.running:
            try:
                # Run nvidia-smi to get GPU stats
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # Parse the output
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            utilization = float(parts[0].strip())
                            memory_used = float(parts[1].strip())
                            temperature = float(parts[2].strip())
                            
                            # Update stats
                            self.stats['max_utilization'] = max(self.stats['max_utilization'], utilization)
                            self.stats['max_memory_used'] = max(self.stats['max_memory_used'], memory_used)
                            self.stats['max_temperature'] = max(self.stats['max_temperature'], temperature)
                            
                            # Track for average
                            total_util += utilization
                            sample_count += 1
                            
                            # Store sample
                            self.stats['samples'].append({
                                'timestamp': time.time(),
                                'utilization': utilization,
                                'memory_used': memory_used,
                                'temperature': temperature
                            })
            except Exception as e:
                logger.warning(f"Error polling GPU stats: {str(e)}")
            
            # Sleep for the polling interval
            time.sleep(self.polling_interval)
        
        # Calculate average utilization
        if sample_count > 0:
            self.stats['avg_utilization'] = total_util / sample_count
    
    def start(self):
        """Start tracking GPU statistics"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._poll_gpu_stats)
        self.thread.daemon = True
        self.thread.start()
        logger.info("GPU stats tracking started")
    
    def stop(self):
        """Stop tracking GPU statistics and return the stats"""
        if not self.running:
            return self.stats
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        logger.info(f"GPU stats tracking stopped. Max memory: {self.stats['max_memory_used']} MB, " +
                   f"Avg utilization: {self.stats['avg_utilization']:.1f}%, " +
                   f"Max temperature: {self.stats['max_temperature']}°C")
        
        return self.stats

class SlackReporter:
    """Send reports and notifications to Slack"""
    
    def __init__(self, webhook_url=None, bot_token=None, channel='#system-notifications'):
        """Initialize Slack reporter
        
        Args:
            webhook_url (str): Slack webhook URL for sending messages
            bot_token (str): Slack bot token for API access
            channel (str): Default channel to send messages to
        """
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel = channel
        
        if not webhook_url and not bot_token:
            logger.warning("No Slack webhook URL or bot token provided. Slack reporting will be disabled.")
    
    def send_message(self, message, blocks=None):
        """Send a message to Slack
        
        Args:
            message (str): Message text
            blocks (list): Optional blocks for rich formatting
        """
        if not self.webhook_url and not self.bot_token:
            logger.debug(f"Slack message (not sent): {message}")
            return False
        
        try:
            if self.webhook_url:
                # Use webhook
                payload = {
                    'text': message
                }
                
                if blocks:
                    payload['blocks'] = blocks
                
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to send Slack message via webhook: {response.status_code} {response.text}")
                    return False
                
                return True
            
            elif self.bot_token:
                # Use bot token
                payload = {
                    'channel': self.channel,
                    'text': message
                }
                
                if blocks:
                    payload['blocks'] = blocks
                
                response = requests.post(
                    'https://slack.com/api/chat.postMessage',
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.bot_token}'
                    }
                )
                
                if not response.json().get('ok', False):
                    logger.warning(f"Failed to send Slack message via API: {response.json()}")
                    return False
                
                return True
        
        except Exception as e:
            logger.warning(f"Error sending Slack message: {str(e)}")
            return False
    
    def report_error(self, error_message, traceback_text=None, phase="unknown"):
        """Report an error to Slack
        
        Args:
            error_message (str): Error message
            traceback_text (str): Optional traceback text
            phase (str): Phase where the error occurred
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "❌ ML Trainer Error",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Phase:*\n{phase}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{error_message}```"
                }
            }
        ]
        
        if traceback_text:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Traceback:*\n```{traceback_text[:1000]}{'...' if len(traceback_text) > 1000 else ''}```"
                }
            })
        
        return self.send_message(f"❌ ML Trainer Error: {error_message}", blocks)
    
    def report_training_start(self, config, gpu_info=None):
        """Report training start to Slack
        
        Args:
            config (dict): Training configuration
            gpu_info (dict): GPU information
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚀 ML Training Started",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Lookback:*\n{config.get('lookback_days', 'N/A')} days"
                    }
                ]
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Tickers:*\n{config.get('ticker_count', 'N/A')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Models:*\n{', '.join(config.get('models', ['N/A']))}"
                    }
                ]
            }
        ]
        
        if gpu_info:
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*GPU:*\n{gpu_info.get('name', 'N/A')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Memory:*\n{gpu_info.get('memory', 'N/A')}"
                    }
                ]
            })
        
        return self.send_message("🚀 ML Training Started", blocks)
    
    def report_training_complete(self, duration, models_summary, gpu_stats=None):
        """Report training completion to Slack
        
        Args:
            duration (float): Training duration in seconds
            models_summary (dict): Summary of model training results
            gpu_stats (dict): GPU statistics
        """
        # Format duration
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Count successful models
        success_count = sum(1 for model in models_summary.values() if model.get('success', False))
        total_count = len(models_summary)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ ML Training Complete",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Duration:*\n{duration_str}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Models:*\n{success_count}/{total_count} successful"
                    }
                ]
            }
        ]
        
        if gpu_stats:
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*GPU Utilization:*\nAvg: {gpu_stats.get('avg_utilization', 0):.1f}%\nMax: {gpu_stats.get('max_utilization', 0):.1f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*GPU Memory:*\nMax: {gpu_stats.get('max_memory_used', 0)} MB\n*Temp:* {gpu_stats.get('max_temperature', 0)}°C"
                    }
                ]
            })
        
        return self.send_message(f"✅ ML Training Complete in {duration_str}", blocks)
    
    def report_model_metrics(self, model_name, metrics, training_time):
        """Report model metrics to Slack
        
        Args:
            model_name (str): Name of the model
            metrics (dict): Model metrics
            training_time (float): Training time in seconds
        """
        # Format metrics based on model type
        metrics_text = ""
        
        if model_name == "signal_detection":
            metrics_text = (
                f"*Accuracy:* {metrics.get('accuracy', 0):.4f}\n"
                f"*Precision:* {metrics.get('precision', 0):.4f}\n"
                f"*Recall:* {metrics.get('recall', 0):.4f}\n"
                f"*F1:* {metrics.get('f1', 0):.4f}\n"
                f"*AUC:* {metrics.get('auc', 0):.4f}"
            )
        elif model_name == "price_prediction":
            metrics_text = (
                f"*MSE:* {metrics.get('mse', 0):.6f}\n"
                f"*Direction Accuracy:* {metrics.get('direction_accuracy', 0):.4f}"
            )
        elif model_name == "risk_assessment":
            metrics_text = (
                f"*MSE:* {metrics.get('mse', 0):.6f}\n"
                f"*R²:* {metrics.get('r2', 0):.4f}"
            )
        elif model_name == "exit_strategy":
            metrics_text = (
                f"*RMSE:* {metrics.get('rmse', 0):.6f}\n"
                f"*MSE:* {metrics.get('mse', 0):.6f}"
            )
        elif model_name == "market_regime":
            metrics_text = (
                f"*Inertia:* {metrics.get('inertia', 0):.2f}\n"
                f"*Clusters:* {len(metrics.get('cluster_counts', []))}"
            )
        
        # Format training time
        minutes, seconds = divmod(training_time, 60)
        training_time_str = f"{int(minutes)}m {int(seconds)}s"
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{model_name.replace('_', ' ').title()} Model Metrics*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": metrics_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Training Time:*\n{training_time_str}"
                    }
                ]
            }
        ]
        
        # Add feature importance if available
        if 'feature_importance' in metrics and metrics['feature_importance']:
            # Get top 5 features
            top_features = sorted(
                metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            feature_text = "*Top Features:*\n" + "\n".join(
                f"{i+1}. {feature}: {importance:.4f}"
                for i, (feature, importance) in enumerate(top_features)
            )
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": feature_text
                }
            })
        
        return self.send_message(f"{model_name.replace('_', ' ').title()} Model Trained", blocks)