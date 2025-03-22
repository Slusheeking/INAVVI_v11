#!/usr/bin/env python3
"""
Memory Bank Activation Script

This script automatically activates the memory bank by checking if it exists
and loading all memory bank files. It's designed to be run at the start of each session.
"""

import os
import sys
import time
from datetime import datetime


def check_memory_bank_exists():
    """Check if the memory bank directory exists."""
    return os.path.isdir("memory-bank")


def read_memory_bank_file(file_path):
    """Read a memory bank file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def activate_memory_bank():
    """Activate the memory bank by reading all memory bank files."""
    if not check_memory_bank_exists():
        print("[MEMORY BANK: INACTIVE]")
        print("No Memory Bank was found. Please create one to maintain project context.")
        return False

    # List of mandatory memory bank files
    memory_bank_files = [
        "memory-bank/productContext.md",
        "memory-bank/activeContext.md",
        "memory-bank/systemPatterns.md",
        "memory-bank/decisionLog.md",
        "memory-bank/progress.md"
    ]

    # Read all memory bank files
    for file_path in memory_bank_files:
        content = read_memory_bank_file(file_path)
        if content is None:
            print(f"[MEMORY BANK: INACTIVE] Failed to read {file_path}")
            return False

        # Print a summary of the file
        file_name = os.path.basename(file_path)
        print(f"Loaded {file_name} ({len(content.splitlines())} lines)")

    print("[MEMORY BANK: ACTIVE]")
    print("Memory Bank has been successfully activated.")
    return True


def main():
    """Main function to activate the memory bank."""
    print(
        f"Memory Bank Activation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    success = activate_memory_bank()

    print("-" * 50)
    if success:
        print("Memory Bank is now active and ready for use.")
    else:
        print("Memory Bank activation failed. Please check the error messages above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
