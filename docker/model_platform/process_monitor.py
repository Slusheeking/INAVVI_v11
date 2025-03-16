#!/usr/bin/env python3
"""
Process monitor for supervisord processes
"""
import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('process_monitor')

# Process status file
STATUS_FILE = '/app/process_status.json'


def get_process_status():
    """Get the status of all supervisord processes"""
    try:
        result = subprocess.run(['supervisorctl', 'status'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)

        processes = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                process_name = parts[0]
                status = parts[1]

                processes[process_name] = {
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }

                # Add additional info if available
                if len(parts) > 2:
                    processes[process_name]['info'] = ' '.join(parts[2:])

        return processes
    except Exception as e:
        logger.error(f"Error getting process status: {e}")
        return {}


def check_process_logs(process_name):
    """Check the logs for a specific process to find error messages"""
    try:
        # Get the last 50 lines of stderr log
        log_file = f"/app/logs/{process_name}_stderr.log"
        if not os.path.exists(log_file):
            return "Log file not found"

        result = subprocess.run(['tail', '-n', '50', log_file],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        log_content = result.stdout.strip()

        # Look for common error patterns
        error_lines = []
        for line in log_content.split('\n'):
            if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
                error_lines.append(line)

        if error_lines:
            return '\n'.join(error_lines[-5:])  # Return the last 5 error lines
        else:
            return "No specific errors found in logs"
    except Exception as e:
        logger.error(f"Error checking logs for {process_name}: {e}")
        return f"Error checking logs: {e}"


def save_status(processes):
    """Save the process status to a file"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(processes, f, indent=2)
        logger.info(f"Process status saved to {STATUS_FILE}")
    except Exception as e:
        logger.error(f"Error saving process status: {e}")


def main():
    """Main function"""
    logger.info("Process monitor started")

    while True:
        try:
            # Get process status
            processes = get_process_status()

            # Check for FATAL or BACKOFF status
            for process_name, info in processes.items():
                if info['status'] in ['FATAL', 'BACKOFF']:
                    # Check logs for errors
                    error_info = check_process_logs(process_name)
                    info['error'] = error_info
                    logger.warning(
                        f"Process {process_name} is in {info['status']} state: {error_info}")

            # Save status
            save_status(processes)

            # Sleep for a while
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Process monitor stopped")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Sleep longer on error


if __name__ == "__main__":
    main()
