"""
Test Logging Script

This script tests the logging system by writing log messages to each logger.
It demonstrates both direct logger usage and the log_to_file utility function.
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.logging.logger import get_all_loggers, log_to_file

def test_logging():
    """
    Test the logging system by writing log messages to each logger.
    Tests both direct logger usage and the log_to_file utility function.
    """
    # Get all loggers
    loggers = get_all_loggers()
    
    # Create a timestamp for the log test
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Testing logging system at {timestamp}")
    print(f"Writing test messages to {len(loggers)} loggers...")
    
    # Write log messages to each logger
    for name, logger in loggers.items():
        # Log using the logger directly
        logger.debug(f"DEBUG message for {name} at {timestamp}")
        logger.info(f"INFO message for {name} at {timestamp}")
        logger.warning(f"WARNING message for {name} at {timestamp}")
        logger.error(f"ERROR message for {name} at {timestamp}")
        
        # Test the log_to_file utility function with different log levels
        log_to_file(name, f"DEBUG log_to_file message for {name} at {timestamp}", "DEBUG")
        log_to_file(name, f"INFO log_to_file message for {name} at {timestamp}", "INFO")
        log_to_file(name, f"WARNING log_to_file message for {name} at {timestamp}", "WARNING")
        log_to_file(name, f"ERROR log_to_file message for {name} at {timestamp}", "ERROR")
        log_to_file(name, f"CRITICAL log_to_file message for {name} at {timestamp}", "CRITICAL")
        
        # Add a small delay to ensure messages are written in order
        time.sleep(0.1)
    
    # Test specific logger - feature_analyzer
    log_to_file("feature_analyzer", f"Special test message for feature_analyzer at {timestamp}", "INFO")
    
    print(f"Logging test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check the log files in the logs directory for the test messages.")

if __name__ == "__main__":
    test_logging()