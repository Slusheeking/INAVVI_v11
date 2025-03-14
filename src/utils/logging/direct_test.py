"""
Direct Test Script for Logging System

This script directly tests the logging system by creating a new logger
and writing log messages to it.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.logging.logger import setup_logger, log_to_file, LOG_DIR

def direct_test():
    """
    Directly test the logging system by creating a new logger and writing log messages to it.
    """
    # Create a timestamp for the log test
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Starting direct logging test at {timestamp}")
    
    # Ensure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create a test log file directly in the root logs directory
    test_log_file = os.path.join(LOG_DIR, 'direct_test.log')
    print(f"LOG_DIR is: {LOG_DIR}")
    with open(test_log_file, 'w') as f:
        f.write(f"Direct test log file created at {timestamp}\n")
    
    print(f"Created test log file: {test_log_file}")
    
    # Append to the test log file
    with open(test_log_file, 'a') as f:
        f.write(f"{timestamp} - DEBUG - Direct debug message\n")
        f.write(f"{timestamp} - INFO - Direct info message\n")
        f.write(f"{timestamp} - WARNING - Direct warning message\n")
        f.write(f"{timestamp} - ERROR - Direct error message\n")
        f.write(f"{timestamp} - CRITICAL - Direct critical message\n")
    
    # Create a test logger
    test_logger = setup_logger('test_direct', log_level='DEBUG', force_file_creation=True)
    
    # Log messages using the logger
    test_logger.debug(f"DEBUG direct message at {timestamp}")
    test_logger.info(f"INFO direct message at {timestamp}")
    test_logger.warning(f"WARNING direct message at {timestamp}")
    test_logger.error(f"ERROR direct message at {timestamp}")
    test_logger.critical(f"CRITICAL direct message at {timestamp}")
    
    # Also test the log_to_file function
    log_to_file('test_direct', f"Direct file log message at {timestamp}", "INFO")
    
    # Force flush all handlers
    for handler in test_logger.handlers:
        handler.flush()
    
    print(f"Direct logging test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check the logs/direct_test.log and logs/test_direct.log files for the test messages.")

if __name__ == "__main__":
    direct_test()