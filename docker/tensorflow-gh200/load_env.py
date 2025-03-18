#!/usr/bin/env python3
"""
Environment Variable Loader

This module loads environment variables from a .env file and provides
utility functions for accessing them.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('load_env')

def load_env_file(env_file=None):
    """
    Load environment variables from a .env file
    
    Args:
        env_file: Path to the .env file (default: .env in the current directory)
        
    Returns:
        True if successful, False otherwise
    """
    if env_file is None:
        # Try to find .env file in the current directory
        script_dir = Path(__file__).resolve().parent
        env_file = script_dir / '.env'
    
    if not os.path.exists(env_file):
        logger.warning(f"Environment file not found: {env_file}")
        return False
    
    try:
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Don't override existing environment variables
                if key not in os.environ:
                    os.environ[key] = value
                    
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False

def get_env(key, default=None):
    """
    Get an environment variable
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)

def get_env_int(key, default=0):
    """
    Get an environment variable as an integer
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value as an integer or default
    """
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def get_env_float(key, default=0.0):
    """
    Get an environment variable as a float
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value as a float or default
    """
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def get_env_bool(key, default=False):
    """
    Get an environment variable as a boolean
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value as a boolean or default
    """
    value = os.environ.get(key, str(default)).lower()
    return value in ('true', 'yes', 'y', '1', 'on')

def get_env_list(key, default=None, separator=','):
    """
    Get an environment variable as a list
    
    Args:
        key: Environment variable name
        default: Default value if not found
        separator: List item separator
        
    Returns:
        Environment variable value as a list or default
    """
    if default is None:
        default = []
        
    value = os.environ.get(key)
    if not value:
        return default
        
    return [item.strip() for item in value.split(separator)]

# Load environment variables when imported
load_env_file()

# Export API keys and other important variables
POLYGON_API_KEY = get_env('POLYGON_API_KEY', '')
UNUSUAL_WHALES_API_KEY = get_env('UNUSUAL_WHALES_API_KEY', '')
REDIS_HOST = get_env('REDIS_HOST', 'localhost')
REDIS_PORT = get_env_int('REDIS_PORT', 6379)
REDIS_DB = get_env_int('REDIS_DB', 0)
REDIS_TTL = get_env_int('REDIS_TTL', 3600)
MAX_CONNECTIONS = get_env_int('MAX_CONNECTIONS', 50)
MAX_POOL_SIZE = get_env_int('MAX_POOL_SIZE', 30)
CONNECTION_TIMEOUT = get_env_int('CONNECTION_TIMEOUT', 15)
MAX_RETRIES = get_env_int('MAX_RETRIES', 5)
RETRY_BACKOFF_FACTOR = get_env_float('RETRY_BACKOFF_FACTOR', 0.5)
NUM_WORKERS = get_env_int('NUM_WORKERS', 4)
QUEUE_SIZE = get_env_int('QUEUE_SIZE', 10000)
BATCH_SIZE = get_env_int('BATCH_SIZE', 1024)
MAX_DATA_POINTS = get_env_int('MAX_DATA_POINTS', 50000)
DEFAULT_WATCHLIST = get_env_list('DEFAULT_WATCHLIST', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

if __name__ == "__main__":
    # Print environment variables if run directly
    print(f"POLYGON_API_KEY: {POLYGON_API_KEY[:4]}****{POLYGON_API_KEY[-4:] if len(POLYGON_API_KEY) > 8 else ''}")
    print(f"UNUSUAL_WHALES_API_KEY: {UNUSUAL_WHALES_API_KEY[:4]}****{UNUSUAL_WHALES_API_KEY[-4:] if len(UNUSUAL_WHALES_API_KEY) > 8 else ''}")
    print(f"REDIS_HOST: {REDIS_HOST}")
    print(f"REDIS_PORT: {REDIS_PORT}")
    print(f"REDIS_DB: {REDIS_DB}")
    print(f"DEFAULT_WATCHLIST: {DEFAULT_WATCHLIST}")