# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib (Technical Analysis Library)
RUN curl -L https://github.com/mrjbq7/ta-lib/archive/TA_Lib-0.4.0.tar.gz | tar xz && \
    cd ta-lib-TA_Lib-0.4.0 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-TA_Lib-0.4.0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data/historical /app/data/cache /app/logs /app/reports /app/reports/visualizations

# Set permissions
RUN chmod +x /app/main.py

# Set default command
CMD ["python", "main.py", "--mode", "paper"]

# Set entrypoint
ENTRYPOINT ["python", "main.py"]