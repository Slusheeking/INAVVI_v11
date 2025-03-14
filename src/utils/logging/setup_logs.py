"""
Setup Logs Script

This script initializes the logging system for the autonomous trading system.
It creates the logs directory if it doesn't exist and sets up the logging configuration.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.logging.logger import get_all_loggers, log_to_file

def setup_logs():
    """
    Set up the logging system.
    
    This function creates the logs directory if it doesn't exist and initializes
    all loggers for the autonomous trading system.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamp for the log initialization
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize all loggers
    loggers = get_all_loggers()
    
    # Initialize log files using the log_to_file function
    for name in loggers.keys():
        # Use the log_to_file function to write initialization messages
        log_to_file(name, f"Log file initialized at {timestamp}", "INFO")
        log_to_file(name, f"Logging system ready for {name}", "INFO")
    
    # Log initialization message to each logger
    for name, logger in loggers.items():
        logger.info(f"Logger initialized at {timestamp}")
        logger.info(f"Logging system ready for {name}")
        
    # Create a log initialization file
    with open(os.path.join(logs_dir, 'log_initialization.txt'), 'a') as f:
        f.write(f"Logs initialized at {timestamp}\n")
        f.write(f"Initialized loggers: {', '.join(loggers.keys())}\n\n")
    
    print(f"Logging system initialized at {timestamp}")
    print(f"Log files are stored in: {logs_dir}")
    print(f"Initialized loggers: {', '.join(loggers.keys())}")

if __name__ == "__main__":
    setup_logs()