"""
Logging utilities for the trading system.

This module provides utilities for configuring and using logging.
"""

import os
import sys
import logging
import logging.config
import yaml
from datetime import datetime
from typing import Optional, Dict, Any

# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

def configure_logger(name: str, config_path: Optional[str] = None) -> logging.Logger:
    """
    Configure and get a logger.
    
    Args:
        name: Logger name
        config_path: Path to logging configuration file
        
    Returns:
        Configured logger
    """
    # Try to load configuration from file
    config = None
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading logging configuration from {config_path}: {e}")
    
    # If no config file or error loading, try environment variable
    if not config:
        config_path = os.environ.get("LOGGING_CONFIG_PATH")
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading logging configuration from {config_path}: {e}")
    
    # If still no config, try default path
    if not config:
        default_path = os.path.join("config", "logging.yaml")
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading logging configuration from {default_path}: {e}")
    
    # If still no config, use default configuration
    if not config:
        config = DEFAULT_CONFIG
    
    # Apply log level from environment variable if set
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        # Update root logger level
        if "" in config.get("loggers", {}):
            config["loggers"][""]["level"] = log_level
        
        # Update console handler level
        if "console" in config.get("handlers", {}):
            config["handlers"]["console"]["level"] = log_level
    
    # Ensure logs directory exists
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Update log file path with service name
    if "file" in config.get("handlers", {}):
        config["handlers"]["file"]["filename"] = os.path.join(logs_dir, f"{name}.log")
    
    # Configure logging
    try:
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error configuring logging: {e}")
        # Fall back to basic configuration
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(logs_dir, f"{name}.log"))
            ]
        )
    
    # Get logger
    logger = logging.getLogger(name)
    logger.info(f"Logger {name} configured with level {log_level}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger. If the logger doesn't exist, it will be created with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, configure it
    if not logger.handlers and not logger.parent.handlers:
        return configure_logger(name)
    
    return logger

def log_exception(logger: logging.Logger, e: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an exception with context.
    
    Args:
        logger: Logger to use
        e: Exception to log
        context: Additional context information
    """
    if context:
        logger.error(f"Exception: {str(e)}, Context: {context}", exc_info=True)
    else:
        logger.error(f"Exception: {str(e)}", exc_info=True)

def log_startup(logger: logging.Logger, service_name: str):
    """
    Log service startup information.
    
    Args:
        logger: Logger to use
        service_name: Name of the service
    """
    logger.info(f"Starting {service_name} service")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    logger.info(f"Log level: {os.environ.get('LOG_LEVEL', 'INFO')}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

def log_shutdown(logger: logging.Logger, service_name: str):
    """
    Log service shutdown information.
    
    Args:
        logger: Logger to use
        service_name: Name of the service
    """
    logger.info(f"Shutting down {service_name} service")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")