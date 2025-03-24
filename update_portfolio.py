#!/usr/bin/env python3
"""
Update Portfolio Script

This script connects to Alpaca API, fetches account and position data,
and updates Redis with the latest portfolio information.

It runs as a service, updating the portfolio data at regular intervals.
"""

import os
import json
import time
import redis
import logging
import signal
import sys
from datetime import datetime
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_portfolio')

# Alpaca API configuration
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET', '')
ALPACA_BASE_URL = os.environ.get(
    'ALPACA_API_URL', 'https://paper-api.alpaca.markets')

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6380'))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', 'trading_system_2025')

# Update interval in seconds
# Default: update every minute
UPDATE_INTERVAL = int(os.environ.get('PORTFOLIO_UPDATE_INTERVAL', '60'))

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    logger.info(f"Received signal {sig}, shutting down...")
    running = False


# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def connect_to_alpaca():
    """Connect to Alpaca API"""
    try:
        alpaca = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            base_url=ALPACA_BASE_URL
        )

        # Test connection
        account = alpaca.get_account()
        logger.info(
            f"Connected to Alpaca - Account ID: {account.id}, Status: {account.status}")
        return alpaca
    except Exception as e:
        logger.error(f"Error connecting to Alpaca: {str(e)}")
        return None


def connect_to_redis():
    """Connect to Redis"""
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Test connection
        redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return redis_client
    except Exception as e:
        logger.error(f"Error connecting to Redis: {str(e)}")
        return None


def get_account_info(alpaca_client):
    """Get account information from Alpaca"""
    try:
        account = alpaca_client.get_account()

        # Convert to dictionary
        account_info = {
            'id': account.id,
            'status': account.status,
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'long_market_value': float(account.long_market_value),
            'short_market_value': float(account.short_market_value),
            'initial_margin': float(account.initial_margin),
            'maintenance_margin': float(account.maintenance_margin),
            'last_equity': float(account.last_equity),
            'last_maintenance_margin': float(account.last_maintenance_margin),
            'multiplier': float(account.multiplier),
            'daytrade_count': int(account.daytrade_count),
            'last_updated': datetime.now().isoformat()
        }

        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        return None


def get_positions(alpaca_client):
    """Get positions from Alpaca"""
    try:
        positions = alpaca_client.list_positions()

        # Convert to list of dictionaries
        positions_list = []
        for position in positions:
            position_data = {
                'ticker': position.symbol,
                'quantity': float(position.qty),
                'entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'current_value': float(position.market_value),
                'unrealized_pnl': float(position.unrealized_pl),
                'unrealized_pnl_pct': float(position.unrealized_plpc) * 100,
                'last_updated': datetime.now().isoformat()
            }
            positions_list.append(position_data)

        return positions_list
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return []


def update_redis_portfolio(redis_client, account_info, positions):
    """Update Redis with portfolio information"""
    try:
        # Update portfolio summary
        portfolio_data = {
            'total_equity': account_info['equity'],
            'cash': account_info['cash'],
            'buying_power': account_info['buying_power'],
            'total_pnl': sum(pos['unrealized_pnl'] for pos in positions),
            'current_exposure': account_info['long_market_value'] - account_info['short_market_value'],
            'timestamp': time.time()
        }

        # Store portfolio summary
        redis_client.set('portfolio:summary', json.dumps(portfolio_data))

        # Store individual portfolio values
        for key, value in portfolio_data.items():
            redis_client.set(f'portfolio:{key}', value)

        # Update positions
        # First, clear existing active positions
        redis_client.delete('active_positions')

        # Add new positions
        for position in positions:
            ticker = position['ticker']

            # Add to active positions set
            redis_client.sadd('active_positions', ticker)

            # Store position details
            redis_client.hmset(f'position:{ticker}', {
                'ticker': position['ticker'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': position['current_price'],
                'unrealized_pnl': position['unrealized_pnl'],
                'unrealized_pnl_pct': position['unrealized_pnl_pct']
            })

        # Update frontend data
        # Store for frontend access
        redis_client.set('frontend:portfolio:latest',
                         json.dumps(portfolio_data))

        # Add to portfolio history
        redis_client.lpush('frontend:portfolio:history',
                           json.dumps(portfolio_data))
        redis_client.ltrim('frontend:portfolio:history',
                           0, 99)  # Keep last 100 entries

        # Add equity point to equity curve
        equity_point = {
            'timestamp': time.time(),
            'value': account_info['equity']
        }
        redis_client.lpush('frontend:portfolio:equity_curve',
                           json.dumps(equity_point))
        redis_client.ltrim('frontend:portfolio:equity_curve',
                           0, 999)  # Keep last 1000 entries

        # Publish events for real-time frontend updates
        redis_client.publish('frontend:events', json.dumps({
            'type': 'portfolio_update',
            'data': portfolio_data
        }))

        redis_client.publish('frontend:events', json.dumps({
            'type': 'equity_update',
            'data': equity_point
        }))

        # Send notification
        notification = {
            'timestamp': time.time(),
            'message': 'Portfolio data updated from Alpaca',
            'level': 'info',
            'category': 'portfolio',
            'source': 'update_portfolio',
            'details': {
                'equity': account_info['equity'],
                'positions_count': len(positions)
            }
        }

        redis_client.lpush('frontend:notifications', json.dumps(notification))
        # Keep last 1000 notifications
        redis_client.ltrim('frontend:notifications', 0, 999)

        redis_client.publish('frontend:events', json.dumps({
            'type': 'notification',
            'data': notification
        }))

        # Update API health status
        redis_client.set('api:alpaca:health', 'ok')
        redis_client.set('api:alpaca:last_update', datetime.now().isoformat())

        logger.info(
            f"Updated portfolio data in Redis: {len(positions)} positions, equity: ${account_info['equity']}")
        return True
    except Exception as e:
        logger.error(f"Error updating Redis portfolio: {str(e)}")
        return False


def update_portfolio_data():
    """Update portfolio data from Alpaca"""
    # Connect to Alpaca
    alpaca_client = connect_to_alpaca()
    if not alpaca_client:
        logger.error("Failed to connect to Alpaca")
        return False

    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        logger.error("Failed to connect to Redis")
        return False

    # Get account info
    account_info = get_account_info(alpaca_client)
    if not account_info:
        logger.error("Failed to get account info")
        return False

    # Get positions
    positions = get_positions(alpaca_client)

    # Update Redis
    success = update_redis_portfolio(redis_client, account_info, positions)

    if success:
        logger.info("Portfolio update completed successfully")
    else:
        logger.error("Portfolio update failed")

    return success


def main():
    """Main function"""
    logger.info(
        f"Starting portfolio updater service (interval: {UPDATE_INTERVAL}s)")

    # Initial update
    update_portfolio_data()

    # Main loop
    last_update_time = time.time()

    while running:
        try:
            # Sleep for a short time to allow for graceful shutdown
            time.sleep(1)

            # Check if it's time for an update
            current_time = time.time()
            if current_time - last_update_time >= UPDATE_INTERVAL:
                update_portfolio_data()
                last_update_time = current_time

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # Wait a bit before retrying

    logger.info("Portfolio updater service shutting down")
    sys.exit(0)


if __name__ == "__main__":
    main()
