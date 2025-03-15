"""
Main entry point for the System Controller service.
"""

import logging
import os
import uvicorn
from src.system_controller.api import create_app

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    logger.info("Starting System Controller service...")
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"API server starting on {host}:{port}")
    uvicorn.run(
        "src.system_controller.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=log_level.lower(),
    )