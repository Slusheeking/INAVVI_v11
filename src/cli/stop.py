#!/usr/bin/env python3
"""
CLI command to stop the autonomous trading system.
"""

import argparse
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("inavvi-stop")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stop the autonomous trading system")
    parser.add_argument(
        "--log-level", "-l",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--no-docker", "-n",
        help="Run without Docker",
        action="store_true"
    )
    parser.add_argument(
        "--services", "-s",
        help="Comma-separated list of services to stop (default: all)",
        default="all"
    )
    parser.add_argument(
        "--remove", "-r",
        help="Remove containers after stopping",
        action="store_true"
    )
    return parser.parse_args()

def stop_system(args):
    """Stop the autonomous trading system."""
    if not args.no_docker:
        # Stop with Docker
        logger.info("Stopping the system with Docker...")
        
        # Stop the services
        if args.remove:
            logger.info("Stopping and removing Docker containers...")
            down_cmd = ["docker-compose", "down"]
            if args.services != "all":
                logger.warning("--services option is ignored when using --remove")
        else:
            logger.info("Stopping Docker containers...")
            down_cmd = ["docker-compose", "stop"]
            if args.services != "all":
                down_cmd.extend(args.services.split(","))
        
        try:
            subprocess.run(down_cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to stop Docker containers")
            return False
        
        logger.info("System stopped successfully!")
    else:
        # Stop without Docker
        logger.info("Stopping the system without Docker...")
        logger.error("Running without Docker is not yet implemented")
        return False
    
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Stop the system
    if not stop_system(args):
        logger.error("Failed to stop the system")
        sys.exit(1)
    
    logger.info("System shutdown completed successfully")

if __name__ == "__main__":
    main()