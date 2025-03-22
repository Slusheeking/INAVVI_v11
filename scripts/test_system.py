#!/usr/bin/env python3
"""
Trading System Test Script

This script tests the startup and shutdown of the trading system.
It performs a basic health check and verifies that all components
can be started and stopped correctly.

Usage:
    python test_system.py [--no-gpu] [--debug]
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('system_tester')


def run_command(command, timeout=None):
    """Run a command and return its output"""
    try:
        logger.info(f"Running command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False, "Command timed out"
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False, str(e)


def test_redis():
    """Test Redis connection"""
    logger.info("Testing Redis connection...")
    success, output = run_command("redis-cli ping", timeout=5)
    if success and "PONG" in output:
        logger.info("Redis connection successful")
        return True
    else:
        logger.error("Redis connection failed")
        return False


def test_system_startup(args):
    """Test system startup"""
    logger.info("Testing system startup...")

    # Build command with arguments
    cmd = ["python3", "scripts/start_system.py", "--health-check-interval=10"]
    if args.no_gpu:
        cmd.append("--no-gpu")
    if args.debug:
        cmd.append("--debug")

    # Start the system in a separate process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for system to initialize (30 seconds)
    logger.info("Waiting for system to initialize (30 seconds)...")
    time.sleep(30)

    # Check if process is still running
    if process.poll() is not None:
        # Process has terminated
        stdout, stderr = process.communicate()
        logger.error("System startup failed")
        logger.error(f"Error output: {stderr}")
        return False, process

    logger.info("System startup successful")
    return True, process


def test_system_health():
    """Test system health by checking Redis keys"""
    logger.info("Testing system health...")
    try:
        import redis
        redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6380)),
            db=int(os.environ.get('REDIS_DB', 0)),
            password=os.environ.get('REDIS_PASSWORD', ''),
            username=os.environ.get('REDIS_USERNAME', 'default'),
            socket_timeout=1
        )

        # Check if system status key exists
        if redis_client.exists("system:status"):
            status = redis_client.hgetall("system:status")
            logger.info(f"System status: {status}")

            # Check if system is running
            if b'running' in status and status[b'running'] == b'true':
                logger.info("System is running")
                return True

        logger.error("System health check failed - system not running")
        return False
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        return False


def test_system_shutdown(process):
    """Test system shutdown"""
    logger.info("Testing system shutdown...")

    # Run the stop script
    success, output = run_command("python3 scripts/stop_system.py", timeout=60)

    # Check if process has terminated
    if process.poll() is None:
        # Process is still running, try to terminate it
        logger.warning(
            "Process still running after stop script, terminating...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error("Process did not terminate, killing...")
            process.kill()

    if success:
        logger.info("System shutdown successful")
        return True
    else:
        logger.error("System shutdown failed")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System Test Script')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Starting system test...")

    # Test Redis
    if not test_redis():
        logger.error("Redis test failed, aborting")
        return 1

    # Test system startup
    startup_success, process = test_system_startup(args)
    if not startup_success:
        logger.error("System startup test failed, aborting")
        return 1

    # Test system health
    health_success = test_system_health()

    # Test system shutdown
    shutdown_success = test_system_shutdown(process)

    # Report results
    logger.info("Test results:")
    logger.info(f"- Redis test: {'PASSED' if test_redis() else 'FAILED'}")
    logger.info(
        f"- System startup test: {'PASSED' if startup_success else 'FAILED'}")
    logger.info(
        f"- System health test: {'PASSED' if health_success else 'FAILED'}")
    logger.info(
        f"- System shutdown test: {'PASSED' if shutdown_success else 'FAILED'}")

    # Overall result
    if startup_success and health_success and shutdown_success:
        logger.info("All tests PASSED")
        return 0
    else:
        logger.error("Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
