#!/usr/bin/env python3
"""
Unified Trading System Stopper

This script gracefully stops all components of the trading system:
1. Data Pipeline
2. ML Engine
3. Trading Engine
4. Stock Selection
5. Monitoring System
6. Continuous Learning
7. Redis and Prometheus services

The script sends status updates to Slack and ensures all components are properly shut down.

Usage:
    python stop_system.py [--force] [--debug]
"""

import argparse
import logging
import os
import sys
import time

import psutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.environ.get("LOGS_DIR", "./logs"), "system_stop.log"),
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("system_stopper")

# Global variables
slack_notifier = None


def load_environment_variables() -> None:
    """Load environment variables from .env file if it exists"""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
        logger.info("Environment variables loaded")
    else:
        logger.warning(f"Environment file {env_file} not found")


def initialize_slack_notifier():
    """Initialize Slack notifier for system notifications"""
    global slack_notifier
    try:
        from monitoring_system.monitoring_system import SlackNotifier

        slack_notifier = SlackNotifier()

        # Test Slack connection
        if slack_notifier.slack_available:
            success = slack_notifier.send_notification(
                "Trading system shutdown initiated",
                level="info",
                channel=slack_notifier.notification_channel,
            )
            if success:
                logger.info("Slack notifier initialized and connected successfully")
            else:
                logger.warning("Slack notifier initialized but test message failed")
        else:
            logger.warning(
                "Slack integration not available - notifications will be logged only",
            )

        return slack_notifier
    except Exception as e:
        logger.exception(f"Failed to initialize Slack notifier: {e!s}")
        return None


def find_system_processes():
    """Find all running processes related to the trading system"""
    system_processes = []

    # Define process names to look for
    process_names = [
        "python3 unified_system.py",
        "python3 scripts/run_data_pipeline.py",
        "python3 scripts/run_model_training.py",
        "python3 scripts/run_stock_selection.py",
        "python3 scripts/run_peak_monitor.py",
        "python3 scripts/start_system.py",
    ]

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
            for process_name in process_names:
                if process_name in cmdline:
                    system_processes.append(
                        {"pid": proc.pid, "name": proc.info["name"], "cmdline": cmdline},
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return system_processes


def stop_process(pid, force=False) -> bool | None:
    """Stop a process by PID"""
    try:
        process = psutil.Process(pid)
        process_name = process.name()

        logger.info(f"Stopping process {pid} ({process_name})...")

        if force:
            process.kill()
        else:
            process.terminate()

            # Wait for process to terminate
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                logger.warning(
                    f"Process {pid} did not terminate within timeout, killing...",
                )
                process.kill()

        logger.info(f"Process {pid} ({process_name}) stopped")
        return True
    except psutil.NoSuchProcess:
        logger.warning(f"Process {pid} not found")
        return False
    except Exception as e:
        logger.exception(f"Error stopping process {pid}: {e!s}")
        return False


def stop_redis() -> bool | None:
    """Stop Redis server if running"""
    try:
        # Check if Redis is running
        import redis

        redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6380)),
            db=int(os.environ.get("REDIS_DB", 0)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            username=os.environ.get("REDIS_USERNAME", "default"),
            socket_timeout=1,
        )
        redis_client.ping()

        # Redis is running, try to shut it down
        logger.info("Stopping Redis server...")
        try:
            redis_client.shutdown()
            logger.info("Redis server stopped gracefully")

            if slack_notifier:
                slack_notifier.send_info("Redis server stopped gracefully")

            return True
        except BaseException:
            # If shutdown command fails, try to find and kill the process
            for proc in psutil.process_iter(["pid", "name"]):
                if proc.info["name"] == "redis-server":
                    stop_process(proc.info["pid"], force=True)
                    logger.info("Redis server stopped forcefully")

                    if slack_notifier:
                        slack_notifier.send_warning("Redis server stopped forcefully")

                    return True

            logger.warning("Could not stop Redis server")

            if slack_notifier:
                slack_notifier.send_warning("Could not stop Redis server")

            return False
    except BaseException:
        logger.info("Redis server is not running")
        return True


def stop_prometheus() -> bool:
    """Stop Prometheus if running"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
            if "prometheus" in cmdline and "--config.file" in cmdline:
                stop_process(proc.info["pid"])
                logger.info("Prometheus stopped")

                if slack_notifier:
                    slack_notifier.send_info("Prometheus stopped")

                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    logger.info("Prometheus is not running")
    return True


def stop_unified_system(force=False) -> bool | None:
    """Stop the unified trading system using Python API if possible"""
    try:
        # Try to import and use the UnifiedTradingSystem class

        # Try to connect to Redis to get system status
        import redis

        redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6380)),
            db=int(os.environ.get("REDIS_DB", 0)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            username=os.environ.get("REDIS_USERNAME", "default"),
            socket_timeout=1,
        )

        # Check if system is running by checking Redis keys
        system_running = (
            redis_client.exists("system:status")
            and redis_client.hget("system:status", "running") == "true"
        )

        if system_running:
            logger.info("Unified system is running, attempting graceful shutdown...")

            # Set shutdown flag in Redis
            redis_client.hset("system:status", "shutdown_requested", "true")

            # Wait for system to acknowledge shutdown
            timeout = 30  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                if (
                    not redis_client.exists("system:status")
                    or redis_client.hget("system:status", "running") != "true"
                ):
                    logger.info("Unified system shutdown completed gracefully")

                    if slack_notifier:
                        slack_notifier.send_success(
                            "Unified system shutdown completed gracefully",
                        )

                    return True
                time.sleep(1)

            logger.warning("Unified system did not shut down within timeout")

            if slack_notifier:
                slack_notifier.send_warning(
                    "Unified system did not shut down within timeout, proceeding with process termination",
                )

        # If we get here, either the system wasn't running or didn't shut down gracefully
        # Find and stop all system processes
        system_processes = find_system_processes()

        if system_processes:
            logger.info(f"Found {len(system_processes)} system processes to stop")

            for process in system_processes:
                stop_process(process["pid"], force=force)

            logger.info("All system processes stopped")

            if slack_notifier:
                slack_notifier.send_info("All system processes stopped")

            return True
        logger.info("No running system processes found")
        return True

    except Exception as e:
        logger.exception(f"Error stopping unified system: {e!s}")

        if slack_notifier:
            slack_notifier.send_error("Error during system shutdown", str(e))

        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Trading System Stopper")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force kill processes instead of graceful termination",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-redis", action="store_true", help="Skip Redis shutdown")
    parser.add_argument(
        "--no-prometheus", action="store_true", help="Skip Prometheus shutdown",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()

    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load environment variables
    load_environment_variables()

    # Initialize Slack notifier
    initialize_slack_notifier()

    if slack_notifier:
        slack_notifier.send_info("Trading system shutdown initiated")

    # Stop the unified system
    logger.info("Stopping unified trading system...")
    stop_unified_system(force=args.force)

    # Stop Redis if requested
    if not args.no_redis:
        stop_redis()

    # Stop Prometheus if requested
    if not args.no_prometheus:
        stop_prometheus()

    # Final check to make sure everything is stopped
    system_processes = find_system_processes()
    if system_processes:
        logger.warning(
            f"Found {len(system_processes)} system processes still running after shutdown",
        )

        if args.force:
            logger.info("Force killing remaining processes...")
            for process in system_processes:
                stop_process(process["pid"], force=True)

            if slack_notifier:
                slack_notifier.send_warning(
                    f"Force killed {len(system_processes)} remaining system processes",
                )
        else:
            process_list = "\n".join(
                [f"PID: {p['pid']}, CMD: {p['cmdline']}" for p in system_processes],
            )
            logger.warning(f"Remaining processes:\n{process_list}")

            if slack_notifier:
                slack_notifier.send_warning(
                    f"System shutdown incomplete. {len(system_processes)} processes still running. Use --force to kill them.",
                )
    else:
        logger.info("All system processes successfully stopped")

        if slack_notifier:
            slack_notifier.send_success(
                "Trading system shutdown completed successfully",
            )

    logger.info("System shutdown procedure completed")


if __name__ == "__main__":
    main()
