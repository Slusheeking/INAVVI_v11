#!/usr/bin/env python3
"""
Redis Event Listener for Trading System Frontend

This module listens for events published to Redis channels and forwards them
to connected WebSocket clients. It runs as a separate process to avoid blocking
the main Flask application.
"""

import json
import os
import sys
import time
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/events.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("event_listener")

# Import Redis
try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False
    logger.error("Redis not available, cannot start event listener")
    sys.exit(1)


class EventListener:
    """Redis event listener for real-time updates"""

    def __init__(self):
        """Initialize the Redis event listener"""
        # Configure Redis connection
        self.redis = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6380)),
            db=int(os.environ.get("REDIS_DB", 0)),
            password=os.environ.get("REDIS_PASSWORD", ""),
            username=os.environ.get("REDIS_USERNAME", "default"),
            socket_timeout=int(os.environ.get("REDIS_TIMEOUT", 5)),
            decode_responses=True,
        )
        self.pubsub = self.redis.pubsub()
        self.running = False
        self.thread = None

        # Test Redis connection
        try:
            self.redis.ping()
            logger.info("Redis connection established for event listener")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            sys.exit(1)

    def start(self):
        """Start listening for Redis events"""
        if self.running:
            return

        self.running = True
        self.pubsub.subscribe("frontend:events")
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        logger.info("Redis event listener started")

    def stop(self):
        """Stop listening for Redis events"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pubsub.unsubscribe()
        logger.info("Redis event listener stopped")

    def listen(self):
        """Listen for Redis events"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    self.process_message(message)
            except Exception as e:
                logger.error(f"Error in Redis event listener: {e}")
                time.sleep(1.0)

    def process_message(self, message):
        """Process a Redis message"""
        try:
            data = json.loads(message["data"])
            event_type = data.get("type")
            event_data = data.get("data", {})

            # Store the event in Redis for retrieval by the frontend
            if event_type:
                # Store in the appropriate channel
                channel = event_data.get("category", "general")
                event_key = f"frontend:events:{channel}:{event_type}"

                # Store the event with a timestamp
                if "timestamp" not in event_data:
                    event_data["timestamp"] = time.time()

                # Store the full event data
                event_json = json.dumps({
                    "type": event_type,
                    "data": event_data,
                    "received_at": time.time()
                })

                # Store in the event history (keep last 100 events per type)
                self.redis.lpush(event_key, event_json)
                self.redis.ltrim(event_key, 0, 99)

                # Also store in the general event history
                self.redis.lpush("frontend:events:all", event_json)
                self.redis.ltrim("frontend:events:all", 0, 499)

                logger.debug(f"Processed event: {event_type}")
        except Exception as e:
            logger.error(f"Error processing Redis message: {e}")


def main():
    """Main entry point"""
    listener = EventListener()
    listener.start()

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
