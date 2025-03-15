#!/usr/bin/env python3
"""
CLI command to check the status of the autonomous trading system.
"""

import argparse
import json
import subprocess
import sys
from tabulate import tabulate

from src.utils.logging import get_logger

# Set up logger for this module
logger = get_logger("cli.status")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check the status of the autonomous trading system")
    parser.add_argument(
        "--log-level", "-l",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--format", "-f",
        help="Output format",
        choices=["table", "json"],
        default="table"
    )
    parser.add_argument(
        "--services", "-s",
        help="Comma-separated list of services to check (default: all)",
        default="all"
    )
    return parser.parse_args()

def get_docker_status():
    """Get the status of Docker containers."""
    try:
        # Get container information in JSON format
        result = subprocess.run(
            ["docker-compose", "ps", "--format", "json"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Parse JSON output
        containers = json.loads(result.stdout)
        
        # Format container information
        status_info = []
        for container in containers:
            status_info.append({
                "name": container.get("Name", "Unknown"),
                "service": container.get("Service", "Unknown"),
                "state": container.get("State", "Unknown"),
                "health": container.get("Health", "N/A"),
                "ports": container.get("Ports", ""),
                "created": container.get("CreatedAt", ""),
                "running_for": container.get("RunningFor", "")
            })
        
        return status_info
    except subprocess.CalledProcessError:
        logger.error("Failed to get Docker container status")
        return []
    except json.JSONDecodeError:
        logger.error("Failed to parse Docker container status")
        return []

def display_status(status_info, format_type="table"):
    """Display the status information."""
    if not status_info:
        logger.info("No containers found")
        return
    
    if format_type == "json":
        # Display as JSON
        print(json.dumps(status_info, indent=2))
    else:
        # Display as table
        headers = ["Name", "Service", "State", "Health", "Ports", "Created", "Running For"]
        table_data = [
            [
                info["name"],
                info["service"],
                info["state"],
                info["health"],
                info["ports"],
                info["created"],
                info["running_for"]
            ]
            for info in status_info
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    # Note: get_logger already sets up the logger, but we can update the level if needed
    if args.log_level:
        logger.setLevel(args.log_level)
    
    # Get system status
    logger.info("Checking system status...")
    status_info = get_docker_status()
    
    # Filter services if specified
    if args.services != "all":
        services = args.services.split(",")
        status_info = [info for info in status_info if info["service"] in services]
    
    # Display status
    display_status(status_info, args.format)
    
    # Check if all services are running
    all_running = all(info["state"] == "running" for info in status_info)
    if all_running:
        logger.info("All services are running")
    else:
        logger.warning("Some services are not running")
    
    # Return exit code based on status
    return 0 if all_running else 1

if __name__ == "__main__":
    sys.exit(main())