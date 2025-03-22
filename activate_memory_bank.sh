#!/bin/bash
# Memory Bank Activation Script Wrapper
# This script runs the Python script to activate the memory bank

# Make the script executable if it's not already
chmod +x scripts/activate_memory_bank.py

# Run the Python script
python3 scripts/activate_memory_bank.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo "Memory Bank activation completed successfully."
else
    echo "Memory Bank activation failed. Please check the error messages above."
    exit 1
fi

# Return success
exit 0