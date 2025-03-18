#!/bin/bash
# Run the reporting system independently

# Set environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
if [ -z "$SLACK_BOT_TOKEN" ]; then
    echo "Error: SLACK_BOT_TOKEN environment variable is not set"
    echo "Please set it in the .env file or export it before running this script"
    exit 1
fi

if [ -z "$SLACK_WEBHOOK_SYSTEM_NOTIFICATIONS" ] || [ -z "$SLACK_WEBHOOK_REPORTS" ] || [ -z "$SLACK_WEBHOOK_PORTFOLIO" ] || [ -z "$SLACK_WEBHOOK_CURRENT_POSITIONS" ]; then
    echo "Warning: One or more Slack webhook URLs are not set"
    echo "Some reporting features may not work correctly"
fi

# Run the reporting system
echo "Starting reporting system..."
python -m tests.reporting_system

# Exit with the same code as the Python script
exit $?