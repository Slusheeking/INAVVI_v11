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
    pyyaml \
    schedule \
    prometheus_client

# Create app directory
WORKDIR /app

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser autonomous_trading_system/src/utils /app/src/utils
COPY --chown=appuser:appuser autonomous_trading_system/src/tests/system_components.py /app/src/system_controller/system_components.py

# Create a main.py file for the system controller
RUN echo 'import os\n\
import sys\n\
import time\n\
import logging\n\
from src.system_controller.system_components import SystemController\n\
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
logger = logging.getLogger("system_controller")\n\
\n\
def main():\n\
    logger.info("Starting System Controller")\n\
    \n\
    # Initialize the system controller\n\
    controller = SystemController()\n\
    \n\
    # Start the system controller\n\
    controller.start()\n\
    \n\
    try:\n\
        # Keep the main thread alive\n\
        while True:\n\
            time.sleep(1)\n\
    except KeyboardInterrupt:\n\
        logger.info("Shutting down System Controller")\n\
        controller.stop()\n\
\n\
if __name__ == "__main__":\n\
    main()\n\
' > /app/src/system_controller/main.py

# Create an empty __init__.py file
RUN mkdir -p /app/src/system_controller && \
    touch /app/src/system_controller/__init__.py

# Switch to non-root user
USER appuser

# Set entry point
CMD ["python", "-m", "src.system_controller.main"]