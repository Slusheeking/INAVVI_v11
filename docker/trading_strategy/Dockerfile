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
    alpaca-trade-api \
    ccxt \
    redis

# Create app directory
WORKDIR /app

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser autonomous_trading_system/src/utils /app/src/utils
COPY --chown=appuser:appuser autonomous_trading_system/src/trading_strategy /app/src/trading_strategy

# Create a main.py file for the trading strategy service
RUN echo 'import os\n\
import sys\n\
import time\n\
import logging\n\
from src.trading_strategy.alpaca.alpaca_integration import AlpacaIntegration\n\
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
logger = logging.getLogger("trading_strategy")\n\
\n\
def main():\n\
    logger.info("Starting Trading Strategy Service")\n\
    \n\
    # Initialize the trading strategy\n\
    strategy = AlpacaIntegration(\n\
        max_position_size=float(os.environ.get("MAX_POSITION_SIZE", 2500)),\n\
        risk_percentage=float(os.environ.get("RISK_PERCENTAGE", 0.02)),\n\
        max_positions=int(os.environ.get("MAX_POSITIONS", 50))\n\
    )\n\
    \n\
    # Start the trading strategy\n\
    strategy.start()\n\
    \n\
    try:\n\
        # Keep the main thread alive\n\
        while True:\n\
            time.sleep(1)\n\
    except KeyboardInterrupt:\n\
        logger.info("Shutting down Trading Strategy Service")\n\
        strategy.stop()\n\
\n\
if __name__ == "__main__":\n\
    main()\n\
' > /app/src/trading_strategy/main.py

# Switch to non-root user
USER appuser

# Set entry point
CMD ["python", "-m", "src.trading_strategy.main"]