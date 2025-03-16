#!/usr/bin/env python3
"""
Fix matplotlib permissions issue by setting the MPLCONFIGDIR environment variable
"""
import os
import sys

# Set the MPLCONFIGDIR environment variable to a writable directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Create the directory if it doesn't exist
if not os.path.exists('/tmp/matplotlib'):
    try:
        os.makedirs('/tmp/matplotlib', exist_ok=True)
        print("Created matplotlib config directory at /tmp/matplotlib")
    except Exception as e:
        print(
            f"Error creating matplotlib config directory: {e}", file=sys.stderr)
        sys.exit(1)

print("MPLCONFIGDIR set to /tmp/matplotlib")
