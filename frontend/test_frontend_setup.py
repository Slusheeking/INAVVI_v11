#!/usr/bin/env python3
"""
Frontend Setup Test Script

This script verifies that the frontend environment is properly set up
by checking for the presence of required packages and dependencies.
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

# Define required packages
REQUIRED_PACKAGES = [
    "flask",
    "flask_cors",
    "flask_socketio",
    "redis",
    "eventlet",
    "gevent",
    "werkzeug"
]


def check_venv():
    """Check if running in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("WARNING: Not running in a virtual environment!")
        print("It's recommended to run this script within the virtual environment.")
        print("Activate the virtual environment with: source venv/bin/activate")
        return False
    else:
        print("✓ Running in virtual environment")
        return True


def check_packages():
    """Check if required packages are installed"""
    missing_packages = []
    installed_packages = []

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            installed_packages.append(f"{package} (v{version})")
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(
            f"ERROR: Missing required packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages are installed:")
        for pkg in installed_packages:
            print(f"  - {pkg}")
        return True


def check_directories():
    """Check if required directories exist"""
    frontend_dir = Path(__file__).parent
    required_dirs = [
        frontend_dir / "templates",
        frontend_dir / "sessions",
        frontend_dir.parent / "logs" / "frontend",
        frontend_dir.parent / "logs" / "events"
    ]

    missing_dirs = []
    for directory in required_dirs:
        if not directory.exists():
            missing_dirs.append(str(directory))

    if missing_dirs:
        print(f"WARNING: Missing directories: {', '.join(missing_dirs)}")
        print("These directories will be created when running start_frontend.sh")
        return False
    else:
        print("✓ All required directories exist")
        return True


def check_redis():
    """Check if Redis is accessible"""
    try:
        import redis

        # Get Redis connection parameters from environment or use defaults
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", 6380))
        db = int(os.environ.get("REDIS_DB", 0))
        password = os.environ.get("REDIS_PASSWORD", "trading_system_2025")
        username = os.environ.get("REDIS_USERNAME", "default")

        # Try to connect to Redis
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            username=username,
            socket_timeout=2
        )

        # Test connection with ping
        if r.ping():
            print(f"✓ Redis connection successful at {host}:{port}")
            return True
        else:
            print(f"ERROR: Redis connection failed at {host}:{port}")
            return False
    except Exception as e:
        print(f"ERROR: Redis connection failed: {str(e)}")
        return False


def check_flask_app():
    """Check if Flask app can be imported"""
    try:
        sys.path.append(str(Path(__file__).parent))
        from app import app
        print("✓ Flask app imported successfully")
        print("  - Flask app configured to run on port 5000")
        return True
    except Exception as e:
        print(f"ERROR: Failed to import Flask app: {str(e)}")
        return False


def main():
    """Main function to run all checks"""
    print("\n=== Frontend Setup Test ===\n")

    # Run all checks
    venv_ok = check_venv()
    packages_ok = check_packages()
    dirs_ok = check_directories()
    redis_ok = check_redis()
    flask_ok = check_flask_app()

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Virtual Environment: {'✓' if venv_ok else '✗'}")
    print(f"Required Packages: {'✓' if packages_ok else '✗'}")
    print(f"Required Directories: {'✓' if dirs_ok else '✗'}")
    print(f"Redis Connection: {'✓' if redis_ok else '✗'}")
    print(f"Flask App: {'✓' if flask_ok else '✗'}")

    # Overall status
    if all([packages_ok, flask_ok]):
        print("\n✅ Frontend setup is READY")
        return 0
    else:
        print("\n❌ Frontend setup has ISSUES that need to be fixed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
