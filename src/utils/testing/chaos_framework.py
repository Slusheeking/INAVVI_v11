"""
Chaos testing framework for validating system resilience.
"""

import os
import random
import time
import logging
import threading
import subprocess
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger("chaos_framework")


class ChaosTest:
    """Base class for chaos tests."""

    def __init__(self, name: str, description: str):
        """
        Initialize a chaos test.

        Args:
            name: Name of the test
            description: Description of the test
        """
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.success = None
        self.error = None

    def setup(self) -> bool:
        """
        Set up the test environment.

        Returns:
            bool: True if setup successful, False otherwise
        """
        logger.info(f"Setting up test: {self.name}")
        return True

    def execute(self) -> bool:
        """
        Execute the test.

        Returns:
            bool: True if test passed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement execute")

    def cleanup(self) -> None:
        """Clean up after the test."""
        logger.info(f"Cleaning up after test: {self.name}")

    def run(self) -> bool:
        """
        Run the full test lifecycle: setup, execute, cleanup.

        Returns:
            bool: True if test passed, False otherwise
        """
        self.start_time = datetime.now()

        try:
            if not self.setup():
                logger.error(f"Test setup failed: {self.name}")
                self.success = False
                self.error = "Setup failed"
                return False

            success = self.execute()
            self.success = success

            if not success:
                logger.error(f"Test failed: {self.name}")
                self.error = "Test failed"
            else:
                logger.info(f"Test passed: {self.name}")

            return success
        except Exception as e:
            logger.exception(f"Error running test {self.name}: {e}")
            self.success = False
            self.error = str(e)
            return False
        finally:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error during test cleanup: {e}")

            self.end_time = datetime.now()

    def get_result(self) -> Dict[str, Any]:
        """
        Get the test result.

        Returns:
            Dict[str, Any]: Test result information
        """
        return {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "success": self.success,
            "error": self.error
        }


class ContainerRestartTest(ChaosTest):
    """Test system resilience by restarting containers."""

    def __init__(self, container_name: str, check_endpoint: Optional[str] = None):
        """
        Initialize a container restart test.

        Args:
            container_name: Name of the container to restart
            check_endpoint: Optional endpoint to check after restart
        """
        super().__init__(
            name=f"restart_{container_name}",
            description=f"Test system resilience when {container_name} is restarted"
        )
        self.container_name = container_name
        self.check_endpoint = check_endpoint

    def execute(self) -> bool:
        """Execute the container restart test."""
        try:
            logger.info(f"Restarting container: {self.container_name}")

            # Restart the container
            result = subprocess.run(
                ["docker", "restart", self.container_name],
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Container restart output: {result.stdout.strip()}")

            # Wait for container to be healthy
            max_wait = 60  # seconds
            wait_interval = 5  # seconds

            for _ in range(max_wait // wait_interval):
                result = subprocess.run(
                    ["docker", "inspect", "--format",
                        "{{.State.Health.Status}}", self.container_name],
                    capture_output=True,
                    text=True
                )

                status = result.stdout.strip()
                logger.info(f"Container health status: {status}")

                if status == "healthy":
                    break

                time.sleep(wait_interval)
            else:
                logger.error(
                    f"Container did not become healthy within {max_wait} seconds")
                return False

            # Check endpoint if provided
            if self.check_endpoint:
                time.sleep(5)  # Additional wait for service to stabilize

                result = subprocess.run(
                    ["curl", "-s", "-f", self.check_endpoint],
                    capture_output=True
                )

                if result.returncode != 0:
                    logger.error(
                        f"Failed to access endpoint {self.check_endpoint} after restart")
                    return False

                logger.info(
                    f"Successfully accessed endpoint {self.check_endpoint} after restart")

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error restarting container: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during container restart test: {e}")
            return False


class NetworkPartitionTest(ChaosTest):
    """Test system resilience with network partitioning."""

    def __init__(self, container_name: str, duration: int = 30):
        """
        Initialize a network partition test.

        Args:
            container_name: Name of the container to partition
            duration: Duration of the partition in seconds
        """
        super().__init__(
            name=f"network_partition_{container_name}",
            description=f"Test system resilience when {container_name} is network partitioned for {duration}s"
        )
        self.container_name = container_name
        self.duration = duration

    def execute(self) -> bool:
        """Execute the network partition test."""
        try:
            logger.info(
                f"Creating network partition for: {self.container_name}")

            # Disconnect container from network
            result = subprocess.run(
                ["docker", "network", "disconnect",
                    "trading-network", self.container_name],
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Network disconnect output: {result.stdout.strip()}")

            # Wait for the specified duration
            logger.info(f"Waiting for {self.duration} seconds...")
            time.sleep(self.duration)

            # Reconnect container to network
            result = subprocess.run(
                ["docker", "network", "connect",
                    "trading-network", self.container_name],
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Network reconnect output: {result.stdout.strip()}")

            # Wait for container to recover
            time.sleep(10)

            # Check container health
            result = subprocess.run(
                ["docker", "inspect", "--format",
                    "{{.State.Health.Status}}", self.container_name],
                capture_output=True,
                text=True
            )

            status = result.stdout.strip()
            logger.info(
                f"Container health status after network partition: {status}")

            return status == "healthy"
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during network partition test: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during network partition test: {e}")
            return False


class ResourceConstraintTest(ChaosTest):
    """Test system resilience under resource constraints."""

    def __init__(self, container_name: str, cpu_limit: str, memory_limit: str, duration: int = 60):
        """
        Initialize a resource constraint test.

        Args:
            container_name: Name of the container to constrain
            cpu_limit: CPU limit (e.g., "0.5")
            memory_limit: Memory limit (e.g., "256m")
            duration: Duration of the constraint in seconds
        """
        super().__init__(
            name=f"resource_constraint_{container_name}",
            description=f"Test system resilience when {container_name} has limited resources (CPU: {cpu_limit}, Memory: {memory_limit})"
        )
        self.container_name = container_name
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.duration = duration
        self.original_cpu = None
        self.original_memory = None

    def setup(self) -> bool:
        """Set up the resource constraint test."""
        try:
            # Get original resource limits
            result = subprocess.run(
                ["docker", "inspect", "--format",
                    "{{.HostConfig.NanoCpus}}", self.container_name],
                capture_output=True,
                text=True
            )
            self.original_cpu = result.stdout.strip()

            result = subprocess.run(
                ["docker", "inspect", "--format",
                    "{{.HostConfig.Memory}}", self.container_name],
                capture_output=True,
                text=True
            )
            self.original_memory = result.stdout.strip()

            logger.info(
                f"Original resource limits for {self.container_name}: CPU={self.original_cpu}, Memory={self.original_memory}")

            return True
        except Exception as e:
            logger.error(f"Error getting original resource limits: {e}")
            return False

    def execute(self) -> bool:
        """Execute the resource constraint test."""
        try:
            logger.info(
                f"Applying resource constraints to: {self.container_name}")

            # Apply resource constraints
            result = subprocess.run(
                ["docker", "update", "--cpus", self.cpu_limit,
                    "--memory", self.memory_limit, self.container_name],
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(
                f"Resource constraint update output: {result.stdout.strip()}")

            # Wait for the specified duration
            logger.info(f"Waiting for {self.duration} seconds...")
            time.sleep(self.duration)

            # Check container health
            result = subprocess.run(
                ["docker", "inspect", "--format",
                    "{{.State.Health.Status}}", self.container_name],
                capture_output=True,
                text=True
            )

            status = result.stdout.strip()
            logger.info(
                f"Container health status under resource constraints: {status}")

            return status == "healthy"
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during resource constraint test: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during resource constraint test: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up after the resource constraint test."""
        try:
            # Restore original resource limits
            if self.original_cpu and self.original_memory:
                logger.info(
                    f"Restoring original resource limits for {self.container_name}")

                nano_cpus = int(float(self.original_cpu) *
                                1_000_000_000) if self.original_cpu != "0" else 0
                memory = int(
                    self.original_memory) if self.original_memory != "0" else 0

                update_cmd = ["docker", "update"]

                if nano_cpus > 0:
                    update_cmd.extend(
                        ["--cpus", str(nano_cpus / 1_000_000_000)])
                else:
                    update_cmd.extend(["--cpus", "0"])

                if memory > 0:
                    update_cmd.extend(["--memory", str(memory)])
                else:
                    update_cmd.extend(["--memory", "0"])

                update_cmd.append(self.container_name)

                result = subprocess.run(
                    update_cmd,
                    capture_output=True,
                    text=True
                )

                logger.info(
                    f"Resource limit restore output: {result.stdout.strip()}")
        except Exception as e:
            logger.error(f"Error restoring original resource limits: {e}")


class ChaosTestSuite:
    """Suite of chaos tests to run."""

    def __init__(self, name: str, description: str):
        """
        Initialize a chaos test suite.

        Args:
            name: Name of the test suite
            description: Description of the test suite
        """
        self.name = name
        self.description = description
        self.tests: List[ChaosTest] = []
        self.results: List[Dict[str, Any]] = []

    def add_test(self, test: ChaosTest) -> None:
        """
        Add a test to the suite.

        Args:
            test: Test to add
        """
        self.tests.append(test)

    def run(self) -> bool:
        """
        Run all tests in the suite.

        Returns:
            bool: True if all tests passed, False otherwise
        """
        logger.info(f"Running test suite: {self.name}")
        logger.info(f"Description: {self.description}")
        logger.info(f"Number of tests: {len(self.tests)}")

        success = True
        self.results = []

        for i, test in enumerate(self.tests, 1):
            logger.info(f"Running test {i}/{len(self.tests)}: {test.name}")

            test_success = test.run()
            success = success and test_success

            self.results.append(test.get_result())

            logger.info(
                f"Test {i}/{len(self.tests)} completed: {'PASS' if test_success else 'FAIL'}")

            # Add a short break between tests
            if i < len(self.tests):
                time.sleep(10)

        logger.info(f"Test suite completed: {'PASS' if success else 'FAIL'}")
        return success

    def get_results(self) -> Dict[str, Any]:
        """
        Get the test suite results.

        Returns:
            Dict[str, Any]: Test suite results
        """
        passed = sum(1 for result in self.results if result["success"])
        failed = len(self.results) - passed

        return {
            "name": self.name,
            "description": self.description,
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.results) if self.results else 0,
            "tests": self.results
        }


def create_standard_test_suite() -> ChaosTestSuite:
    """
    Create a standard test suite with common chaos tests.

    Returns:
        ChaosTestSuite: Standard test suite
    """
    suite = ChaosTestSuite(
        name="Standard Resilience Test Suite",
        description="Standard suite of tests to validate system resilience"
    )

    # Add container restart tests
    for container, endpoint in [
        ("ats-data-processing", "http://localhost:8001/health"),
        ("ats-model-services", "http://localhost:8003/health"),
        ("ats-trading-strategy", "http://localhost:8002/health"),
        ("ats-system-controller", "http://localhost:8000/health")
    ]:
        suite.add_test(ContainerRestartTest(container, endpoint))

    # Add network partition tests
    for container in [
        "ats-data-processing",
        "ats-model-services",
        "ats-trading-strategy"
    ]:
        suite.add_test(NetworkPartitionTest(container, duration=20))

    # Add resource constraint tests
    for container, cpu, memory in [
        ("ats-data-processing", "0.3", "512m"),
        ("ats-model-services", "0.5", "1g"),
        ("ats-trading-strategy", "0.3", "512m")
    ]:
        suite.add_test(ResourceConstraintTest(
            container, cpu, memory, duration=30))

    return suite
