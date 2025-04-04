# Use a conda-based Python image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    ca-certificates \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib using conda (much more reliable than pip)
RUN conda install -y -c conda-forge ta-lib python=3.10

# Copy requirements file
COPY requirements.txt /app/

# Create a modified requirements file without TA-Lib (as we've installed it via conda)
RUN grep -v "ta-lib" requirements.txt > requirements_modified.txt

# Install Python dependencies (without TA-Lib)
RUN pip install --upgrade pip && \
    pip install -r requirements_modified.txt

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