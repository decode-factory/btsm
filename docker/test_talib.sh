#!/bin/bash

# Exit on error
set -e

# Print commands
set -x

echo "Building Docker image with Conda..."
cd "$(dirname "$0")/.."
docker-compose -f docker/docker-compose.yml build talib-test

echo "Testing TA-Lib installation..."
docker-compose -f docker/docker-compose.yml run --rm talib-test

echo "TA-Lib installation test successful!"