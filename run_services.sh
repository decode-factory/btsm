#!/bin/bash

# Script to run different services of BTSM

# Set the base directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to display usage
usage() {
    echo -e "${BLUE}BTSM Service Runner${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build        - Build all Docker images"
    echo "  test-talib   - Test TA-Lib installation in Docker"
    echo "  trading      - Run paper trading service"
    echo "  backtest     - Run backtesting service"
    echo "  dev          - Run development service with file volume mounts"
    echo "  down         - Stop all running services"
    echo "  logs         - Show logs of running services"
    echo "  help         - Show this help"
    echo ""
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi
}

# Function to build images
build_images() {
    echo -e "${BLUE}Building Docker images...${NC}"
    docker-compose -f docker/docker-compose.yml build
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
    else
        echo -e "${RED}✗ Build failed!${NC}"
        exit 1
    fi
}

# Function to test TA-Lib
test_talib() {
    echo -e "${BLUE}Testing TA-Lib installation...${NC}"
    docker-compose -f docker/docker-compose.yml run --rm talib-test
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ TA-Lib test successful!${NC}"
    else
        echo -e "${RED}✗ TA-Lib test failed!${NC}"
        exit 1
    fi
}

# Function to run trading service
run_trading() {
    echo -e "${BLUE}Starting paper trading service...${NC}"
    docker-compose -f docker/docker-compose.yml up -d trading
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Trading service started!${NC}"
        echo -e "${YELLOW}View logs with: $0 logs${NC}"
    else
        echo -e "${RED}✗ Failed to start trading service!${NC}"
        exit 1
    fi
}

# Function to run backtesting service
run_backtest() {
    echo -e "${BLUE}Running backtesting service...${NC}"
    docker-compose -f docker/docker-compose.yml --profile backtest up backtesting
}

# Function to run dev service
run_dev() {
    echo -e "${BLUE}Starting development service...${NC}"
    docker-compose -f docker/docker-compose.yml --profile dev up -d dev
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Development service started!${NC}"
        echo -e "${YELLOW}View logs with: $0 logs${NC}"
    else
        echo -e "${RED}✗ Failed to start development service!${NC}"
        exit 1
    fi
}

# Function to stop all services
stop_services() {
    echo -e "${BLUE}Stopping all services...${NC}"
    docker-compose -f docker/docker-compose.yml down
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All services stopped!${NC}"
    else
        echo -e "${RED}✗ Failed to stop services!${NC}"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}Showing logs (press Ctrl+C to exit)...${NC}"
    docker-compose -f docker/docker-compose.yml logs -f
}

# Check if docker is running
check_docker

# Process command line arguments
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

case "$1" in
    build)
        build_images
        ;;
    test-talib)
        test_talib
        ;;
    trading)
        run_trading
        ;;
    backtest)
        run_backtest
        ;;
    dev)
        run_dev
        ;;
    down)
        stop_services
        ;;
    logs)
        show_logs
        ;;
    help)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        exit 1
        ;;
esac