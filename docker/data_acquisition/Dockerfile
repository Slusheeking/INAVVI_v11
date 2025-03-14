# Use the same NVIDIA TensorFlow image as the existing containers
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

# Set up environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    tzdata \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    psycopg2-binary \
    pandas \
    pytz \
    requests \
    sqlalchemy \
    numpy \
    websocket-client \
    polygon-api-client \
    alpaca-trade-api \
    aiohttp \
    asyncio

# Create app directory
WORKDIR /app

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser autonomous_trading_system/src/utils /app/src/utils
COPY --chown=appuser:appuser autonomous_trading_system/src/data_acquisition /app/src/data_acquisition

# Create a main.py file for the data acquisition service
RUN echo 'import os\n\
import sys\n\
import time\n\
import logging\n\
from src.data_acquisition.pipeline.data_pipeline import DataPipeline\n\
\n\
# Configure logging\n\
logging.basicConfig(\n\
    level=os.environ.get("LOG_LEVEL", "INFO"),\n\
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",\n\
    handlers=[\n\
        logging.StreamHandler(sys.stdout)\n\
    ]\n\
)\n\
\n\
logger = logging.getLogger("data_acquisition")\n\
\n\
def main():\n\
    logger.info("Starting Data Acquisition Service")\n\
    \n\
    # Initialize the data pipeline\n\
    pipeline = DataPipeline()\n\
    \n\
    # Start the data pipeline\n\
    pipeline.start()\n\
    \n\
    try:\n\
        # Keep the main thread alive\n\
        while True:\n\
            time.sleep(1)\n\
    except KeyboardInterrupt:\n\
        logger.info("Shutting down Data Acquisition Service")\n\
        pipeline.stop()\n\
\n\
if __name__ == "__main__":\n\
    main()\n\
' > /app/src/data_acquisition/main.py

# Switch to non-root user
USER appuser

# Set entry point
CMD ["python", "-m", "src.data_acquisition.main"]