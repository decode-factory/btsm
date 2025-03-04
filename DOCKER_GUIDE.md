# BTSM Docker Guide

This guide explains how to run the BTSM system using Docker containers.

## Dockerfile Solution for TA-Lib

The system uses a Conda-based Docker image to ensure proper installation of TA-Lib. This approach resolves the common issues with installing TA-Lib through pip.

### Key Features:

1. **Conda Package Manager**: Uses conda-forge channel for TA-Lib installation
2. **No Manual Compilation**: Avoids compilation errors with pre-built packages
3. **Reliable Dependencies**: Ensures consistent environment across systems
4. **Additional Utilities**: Helper scripts for easy Docker management

## Running with Helper Script

The project includes a `run_services.sh` script to simplify Docker operations:

```bash
# Make script executable
chmod +x run_services.sh

# Build all Docker images
./run_services.sh build

# Test TA-Lib installation
./run_services.sh test-talib

# Start paper trading
./run_services.sh trading

# Run backtesting
./run_services.sh backtest

# Start development mode (with local volume mounts)
./run_services.sh dev

# View logs
./run_services.sh logs

# Stop all services
./run_services.sh down
```

## Docker Compose Services

The `docker-compose.yml` file includes several services:

1. **talib-test**: Tests TA-Lib installation
2. **trading**: Runs the system in paper trading mode
3. **backtesting**: Runs backtest analysis on historical data
4. **dev**: Development environment with local volume mounts

## Manual Docker Commands

You can also use Docker directly:

```bash
# Build the image
docker build -t btsm -f docker/Dockerfile .

# Run with TA-Lib test
docker run --rm btsm python -c "import talib; print('TA-Lib version:', talib.__version__)"

# Run for paper trading
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/reports:/app/reports -v $(pwd)/config:/app/config btsm --mode paper

# Using docker-compose
docker-compose -f docker/docker-compose.yml up trading
```

## Troubleshooting

### TA-Lib Installation Issues

If you encounter issues with TA-Lib in the Docker container:

1. Use the provided test service to verify installation:
   ```bash
   ./run_services.sh test-talib
   ```

2. Check the Docker logs for any errors:
   ```bash
   ./run_services.sh logs
   ```

3. The system uses a Conda-based approach which is more reliable than pip-based installations.

### Volume Mounts

Ensure your local directories exist before running the containers:

```bash
mkdir -p data logs reports config
```

## Custom Docker Configuration

To modify the Docker setup:

1. Edit `docker/Dockerfile` for container configuration
2. Edit `docker/docker-compose.yml` for service configuration
3. Update environment variables in `docker-compose.yml` as needed