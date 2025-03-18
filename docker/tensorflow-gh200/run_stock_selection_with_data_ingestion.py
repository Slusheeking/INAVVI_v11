#!/usr/bin/env python3
"""
Run Stock Selection System with Data Ingestion

This script integrates the Stock Selection System with the Data Ingestion System
to provide a complete market data pipeline for algorithmic trading.
"""

import os
import sys
import time
import signal
import logging
import asyncio
import argparse
import redis
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_selection_with_data_ingestion')

# Import API clients
try:
    from data_ingestion_api_clients import (
        PolygonDataClient, 
        UnusualWhalesClient, 
        RedisCache,
        log_memory_usage
    )
    logger.info("API clients imported successfully")
except ImportError as e:
    logger.error(f"Failed to import API clients: {e}")
    sys.exit(1)

# Import data ingestion system
try:
    from test_data_ingestion_system import DataIngestionSystem
    logger.info("Data ingestion system imported successfully")
except ImportError as e:
    logger.error(f"Failed to import data ingestion system: {e}")
    sys.exit(1)

# Import stock selection system
try:
    from stock_selection_system import StockSelectionSystem
    logger.info("Stock selection system imported successfully")
except ImportError as e:
    logger.error(f"Failed to import stock selection system: {e}")
    sys.exit(1)

# Import environment variables
try:
    from load_env import (
        POLYGON_API_KEY, UNUSUAL_WHALES_API_KEY,
        REDIS_HOST, REDIS_PORT, REDIS_DB
    )
    logger.info("Environment variables loaded successfully")
except ImportError:
    logger.warning("Failed to import load_env.py, using default environment variables")
    # Default values from environment variables
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf')
    UNUSUAL_WHALES_API_KEY = os.environ.get('UNUSUAL_WHALES_API_KEY', '4ad71b9e-7ace-4f24-bdfc-532ace219a18')
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Stock Selection with Data Ingestion')
parser.add_argument('--data-only', action='store_true', help='Run only data ingestion system')
parser.add_argument('--selection-only', action='store_true', help='Run only stock selection system')
args = parser.parse_args()

# Integrated system
class IntegratedTradingSystem:
    """
    Integrated Trading System
    
    Combines data ingestion and stock selection systems for a complete
    algorithmic trading pipeline.
    """
    
    def __init__(self, redis_client, polygon_client, unusual_whales_client):
        """
        Initialize the integrated trading system
        
        Args:
            redis_client: Redis client for data storage
            polygon_client: Polygon data client
            unusual_whales_client: Unusual Whales API client
        """
        self.redis = redis_client
        self.polygon = polygon_client
        self.unusual_whales = unusual_whales_client
        
        # Create data ingestion system
        self.data_ingestion = DataIngestionSystem(
            redis_client=self.redis,
            unusual_whales_client=self.unusual_whales,
            polygon_client=self.polygon
        )
        
        # Create stock selection system
        self.stock_selection = StockSelectionSystem(
            redis_client=self.redis,
            polygon_client=self.polygon,
            unusual_whales_client=self.unusual_whales
        )
        
        # Control flags
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info("Integrated trading system initialized")
        
    def _handle_signal(self, signum, frame):
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop()
        
    async def start(self):
        """Start the integrated trading system"""
        if self.running:
            logger.warning("Integrated trading system is already running")
            return
            
        self.running = True
        
        # Start data ingestion system
        if not args.selection_only:
            logger.info("Starting data ingestion system...")
            self.data_ingestion.start()
            
            # Wait for initial data collection
            logger.info("Waiting for initial data collection (30 seconds)...")
            await asyncio.sleep(30)
        
        # Start stock selection system
        if not args.data_only:
            logger.info("Starting stock selection system...")
            await self.stock_selection.start()
        
        logger.info("Integrated trading system started")
        
    async def stop(self):
        """Stop the integrated trading system"""
        if not self.running:
            logger.warning("Integrated trading system is not running")
            return
            
        self.running = False
        
        # Stop stock selection system
        if not args.data_only:
            logger.info("Stopping stock selection system...")
            await self.stock_selection.stop()
        
        # Stop data ingestion system
        if not args.selection_only:
            logger.info("Stopping data ingestion system...")
            self.data_ingestion.stop()
        
        logger.info("Integrated trading system stopped")
        
    async def run(self):
        """Run the integrated trading system"""
        try:
            # Start the system
            await self.start()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Error in integrated trading system: {e}")
        finally:
            # Stop the system
            await self.stop()


# Mock Redis for testing
class MockRedis:
    """Mock Redis implementation for testing"""
    
    def __init__(self):
        self.data = {}
        self.sets = {}
        self.sorted_sets = {}
        self.hashes = {}
        self.expiry = {}
        
    def get(self, key):
        return self.data.get(key)
        
    def set(self, key, value, ex=None):
        self.data[key] = value
        if ex:
            self.expiry[key] = ex
        return True
        
    def setex(self, key, time, value):
        self.data[key] = value
        self.expiry[key] = time
        return True
        
    def sadd(self, key, *values):
        if key not in self.sets:
            self.sets[key] = set()
        for value in values:
            self.sets[key].add(value)
        return len(values)
        
    def smembers(self, key):
        return self.sets.get(key, set())
        
    def zadd(self, key, mapping):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(mapping)
        return len(mapping)
        
    def zrevrange(self, key, start, end, withscores=False):
        if key not in self.sorted_sets:
            return []
        items = sorted(self.sorted_sets[key].items(), key=lambda x: x[1], reverse=True)
        items = items[start:end+1 if end >= 0 else None]
        if withscores:
            return items
        else:
            return [item[0] for item in items]
            
    def hset(self, key, field=None, value=None, mapping=None):
        if key not in self.hashes:
            self.hashes[key] = {}
        if mapping:
            self.hashes[key].update(mapping)
            return len(mapping)
        else:
            self.hashes[key][field] = value
            return 1
            
    def hget(self, key, field):
        if key in self.hashes and field in self.hashes[key]:
            return self.hashes[key][field]
        return None
        
    def hgetall(self, key):
        return self.hashes.get(key, {})
        
    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.sets:
                del self.sets[key]
                count += 1
            if key in self.sorted_sets:
                del self.sorted_sets[key]
                count += 1
            if key in self.hashes:
                del self.hashes[key]
                count += 1
        return count
        
    def expire(self, key, time):
        self.expiry[key] = time
        return True
        
    def pipeline(self):
        return MockRedisPipeline(self)
        
    def publish(self, channel, message):
        # Mock implementation
        return 1


class MockRedisPipeline:
    """Mock Redis Pipeline for testing"""
    
    def __init__(self, redis_instance):
        self.redis = redis_instance
        self.commands = []
        
    def delete(self, key):
        self.commands.append(('delete', key))
        return self
        
    def sadd(self, key, *values):
        self.commands.append(('sadd', key, values))
        return self
        
    def zadd(self, key, mapping):
        self.commands.append(('zadd', key, mapping))
        return self
        
    def set(self, key, value):
        self.commands.append(('set', key, value))
        return self
        
    def expire(self, key, time):
        self.commands.append(('expire', key, time))
        return self
        
    def hset(self, key, field=None, value=None, mapping=None):
        self.commands.append(('hset', key, field, value, mapping))
        return self
        
    def execute(self):
        results = []
        for cmd in self.commands:
            if cmd[0] == 'delete':
                results.append(self.redis.delete(cmd[1]))
            elif cmd[0] == 'sadd':
                results.append(self.redis.sadd(cmd[1], *cmd[2]))
            elif cmd[0] == 'zadd':
                results.append(self.redis.zadd(cmd[1], cmd[2]))
            elif cmd[0] == 'set':
                results.append(self.redis.set(cmd[1], cmd[2]))
            elif cmd[0] == 'expire':
                results.append(self.redis.expire(cmd[1], cmd[2]))
            elif cmd[0] == 'hset':
                results.append(self.redis.hset(cmd[1], field=cmd[2], value=cmd[3], mapping=cmd[4]))
        self.commands = []
        return results


# Main function
async def main():
    """Main function"""
    logger.info("Starting integrated trading system...")
    
    # Create Redis client
    try:
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()  # Test connection
        logger.info("Connected to Redis successfully")
    except redis.RedisError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error("Redis is required for the integrated trading system. Please ensure Redis is running.")
        sys.exit(1)
    
    # Create API clients
    logger.info("Creating API clients...")
    polygon_client = PolygonDataClient(api_key=POLYGON_API_KEY)
    unusual_whales_client = UnusualWhalesClient(api_key=UNUSUAL_WHALES_API_KEY)
    
    try:
        # Create integrated system
        system = IntegratedTradingSystem(
            redis_client=redis_client,
            polygon_client=polygon_client,
            unusual_whales_client=unusual_whales_client
        )
        
        # Run the system
        await system.run()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        # Close API clients
        logger.info("Closing API clients...")
        polygon_client.close()
        unusual_whales_client.close()
        
        logger.info("Integrated trading system shutdown complete")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())