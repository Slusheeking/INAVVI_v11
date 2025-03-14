"""
Logging Utility for Autonomous Trading System

This module provides a centralized logging configuration for the autonomous trading system.
It sets up loggers with appropriate handlers for console and file output, with configurable
log levels and formatting.
"""

import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Create logs directory if it doesn't exist
# Get the project root directory (3 levels up from the current file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Log levels
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_logger(name, log_level='INFO', log_to_console=True, log_to_file=True,
                log_format=DEFAULT_LOG_FORMAT, max_file_size=10*1024*1024, backup_count=5,
                force_file_creation=True):
    """
    Set up a logger with the specified configuration.
    
    Args:
        name (str): Name of the logger
        log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console (bool): Whether to log to console
        log_to_file (bool): Whether to log to file
        log_format (str): Log format
        max_file_size (int): Maximum log file size in bytes
        backup_count (int): Number of backup files to keep
        force_file_creation (bool): Whether to force creation of log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    
    # Set log level
    logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        log_file = os.path.join(LOG_DIR, f"{name}.log")
        
        # Check if file exists before creating it
        file_exists = os.path.exists(log_file)
        
        # Create the rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count,
            # Use append mode to preserve existing logs
            mode='a'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
        logger.addHandler(file_handler)
        
        # Also add a daily rotating handler for archival purposes
        daily_log_file = os.path.join(LOG_DIR, f"{name}_daily.log")
        daily_handler = TimedRotatingFileHandler(
            daily_log_file, when='midnight', interval=1, backupCount=30
        )
        daily_handler.setFormatter(formatter)
        daily_handler.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
        logger.addHandler(daily_handler)
        
        # Force file creation by writing an initialization message if the file doesn't exist
        if force_file_creation and not file_exists:
            logger.info(f"Log file initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Make sure the logger propagates to the root logger
    logger.propagate = True
    
    return logger

def get_logger(name, **kwargs):
    """
    Get a logger with the specified configuration.
    If the logger already exists, return it, otherwise create a new one.
    
    Args:
        name (str): Name of the logger
        **kwargs: Additional arguments to pass to setup_logger
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it
    if logger.handlers:
        return logger
    
    # Otherwise, set up a new logger
    return setup_logger(name, **kwargs)

# Set up root logger
root_logger = setup_logger('root', log_level='INFO')

# Module-specific loggers
# Feature Engineering loggers
feature_analyzer_logger = setup_logger('feature_analyzer', log_level='INFO')
feature_pipeline_logger = setup_logger('feature_pipeline', log_level='INFO')
feature_store_logger = setup_logger('feature_store', log_level='INFO')

# Model Training loggers
model_registry_logger = setup_logger('model_registry', log_level='INFO')
model_validator_logger = setup_logger('model_validator', log_level='INFO')
model_inference_logger = setup_logger('model_inference', log_level='INFO')
xgboost_model_logger = setup_logger('xgboost_model', log_level='INFO')
lstm_model_logger = setup_logger('lstm_model', log_level='INFO')
cnn_model_logger = setup_logger('cnn_model', log_level='INFO')

# Trading Strategy loggers
entry_signal_generator_logger = setup_logger('entry_signal_generator', log_level='INFO')

# Data Acquisition loggers
polygon_client_logger = setup_logger('polygon_client', log_level='INFO')
unusual_whales_client_logger = setup_logger('unusual_whales_client', log_level='INFO')
multi_timeframe_data_collector_logger = setup_logger('multi_timeframe_data_collector', log_level='INFO')
options_collector_logger = setup_logger('options_collector', log_level='INFO')
price_collector_logger = setup_logger('price_collector', log_level='INFO')
quote_collector_logger = setup_logger('quote_collector', log_level='INFO')
trade_collector_logger = setup_logger('trade_collector', log_level='INFO')
data_pipeline_logger = setup_logger('data_pipeline', log_level='INFO')
timescale_storage_logger = setup_logger('timescale_storage', log_level='INFO')
data_transformer_logger = setup_logger('data_transformer', log_level='INFO')

# Test and System loggers
system_components_logger = setup_logger('system_components', log_level='INFO')
performance_metrics_logger = setup_logger('performance_metrics', log_level='INFO')
full_system_test_logger = setup_logger('full_system_test', log_level='INFO')

# Dictionary of loggers for easy access
LOGGERS = {
    # Feature Engineering loggers
    'feature_analyzer': feature_analyzer_logger,
    'feature_pipeline': feature_pipeline_logger,
    'feature_store': feature_store_logger,
    
    # Model Training loggers
    'model_registry': model_registry_logger,
    'model_validator': model_validator_logger,
    'model_inference': model_inference_logger,
    'xgboost_model': xgboost_model_logger,
    'lstm_model': lstm_model_logger,
    'cnn_model': cnn_model_logger,
    
    # Trading Strategy loggers
    'entry_signal_generator': entry_signal_generator_logger,
    
    # Data Acquisition loggers
    'polygon_client': polygon_client_logger,
    'unusual_whales_client': unusual_whales_client_logger,
    'multi_timeframe_data_collector': multi_timeframe_data_collector_logger,
    'options_collector': options_collector_logger,
    'price_collector': price_collector_logger,
    'quote_collector': quote_collector_logger,
    'trade_collector': trade_collector_logger,
    'data_pipeline': data_pipeline_logger,
    'timescale_storage': timescale_storage_logger,
    'data_transformer': data_transformer_logger,
    
    # Test and System loggers
    'system_components': system_components_logger,
    'performance_metrics': performance_metrics_logger,
    'full_system_test': full_system_test_logger,
}

def log_to_file(name, message, level='INFO'):
    """
    Log a message to a file using the configured logger.
    
    Args:
        name (str): Name of the logger/file
        message (str): Message to log
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get the logger by name
    logger = get_logger(name)
    
    # Convert level string to logging level
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Log the message with the appropriate level
    if log_level == logging.DEBUG:
        logger.debug(message)
    elif log_level == logging.INFO:
        logger.info(message)
    elif log_level == logging.WARNING:
        logger.warning(message)
    elif log_level == logging.ERROR:
        logger.error(message)
    elif log_level == logging.CRITICAL:
        logger.critical(message)
    
    # Force flush handlers to ensure message is written immediately
    for handler in logger.handlers:
        handler.flush()

def get_all_loggers():
    """
    Get all configured loggers.
    
    Returns:
        dict: Dictionary of loggers
    """
    return LOGGERS