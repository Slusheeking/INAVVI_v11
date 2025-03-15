#!/usr/bin/env python3
"""
CLI command to start the autonomous trading system.
"""

import argparse
import os
import subprocess
import sys

from src.utils.logging import get_logger

# Set up logger for this module
logger = get_logger("cli.start")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the autonomous trading system")
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default=os.environ.get("INAVVI_CONFIG", ".env")
    )
    parser.add_argument(
        "--log-level", "-l",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("LOG_LEVEL", "INFO")
    )
    parser.add_argument(
        "--no-docker", "-n",
        help="Run without Docker",
        action="store_true"
    )
    parser.add_argument(
        "--services", "-s",
        help="Comma-separated list of services to start (default: all)",
        default="all"
    )
    return parser.parse_args()

def check_prerequisites():
    """Check if all prerequisites are met."""
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not in PATH")
        return False
    
    # Check if Docker Compose is installed
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker Compose is not installed or not in PATH")
        return False
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        logger.warning(".env file not found. Using default configuration.")
    
    return True

def start_system(args):
    """Start the autonomous trading system."""
    if not args.no_docker:
        # Start with Docker
        logger.info("Starting the system with Docker...")
        
        # Build the services
        logger.info("Building Docker images...")
        build_cmd = ["docker-compose", "build"]
        if args.services != "all":
            build_cmd.extend(args.services.split(","))
        
        try:
            subprocess.run(build_cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to build Docker images")
            return False
        
        # Start the services
        logger.info("Starting Docker containers...")
        up_cmd = ["docker-compose", "up", "-d"]
        if args.services != "all":
            up_cmd.extend(args.services.split(","))
        
        try:
            subprocess.run(up_cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to start Docker containers")
            return False
        
        logger.info("System started successfully!")
        logger.info("To view logs, run: docker-compose logs -f")
    else:
        # Start without Docker
        logger.info("Starting the system without Docker...")
        logger.error("Running without Docker is not yet implemented")
        return False
    
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    # Note: get_logger already sets up the logger, but we can update the level if needed
    if args.log_level:
        logger.setLevel(args.log_level)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        sys.exit(1)
    
    # Start the system
    if not start_system(args):
        logger.error("Failed to start the system")
        sys.exit(1)
    
    logger.info("System startup completed successfully")

if __name__ == "__main__":
    main()