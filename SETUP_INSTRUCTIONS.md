# BTSM Setup Instructions

This document provides detailed instructions for setting up and running the BTSM (Backtesting & Trading Strategy Manager) system.

## System Requirements

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for live data

## Installation Steps

### 1. Install Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential git

# Install TA-Lib dependencies
sudo apt-get install -y build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

#### macOS (using Homebrew):
```bash
brew install python ta-lib
```

#### Windows:
- Install Python from the [official website](https://www.python.org/downloads/)
- Download and install the unofficial TA-Lib wheels from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
```bash
pip install PATH_TO_DOWNLOADED_WHEEL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the System

1. Edit `config/config.ini` to set your preferences:
   - API credentials for brokers
   - Trading parameters
   - Risk management settings
   - Data sources

2. For paper trading, no additional setup is required.

3. For live trading, you'll need to:
   - Obtain API keys from your broker
   - Add them to your config file or use environment variables
   - Ensure you have sufficient funds in your trading account

## Running the System

### Basic Commands

#### Run Backtesting:
```bash
python main.py --mode backtest --strategy moving_average --start-date 2023-01-01 --end-date 2023-12-31 --report
```

#### Run Paper Trading:
```bash
python main.py --mode paper --strategy rsi
```

#### Run Live Trading:
```bash
python main.py --mode live --broker zerodha --strategy moving_average
```

### Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--mode` | Trading mode | `paper` | `backtest`, `paper`, `live` |
| `--config` | Config file path | `config/config.ini` | Any path |
| `--strategy` | Strategy to use | `all` | `moving_average`, `rsi`, `all` |
| `--start-date` | Backtest start date | (1 year ago) | Any date in YYYY-MM-DD format |
| `--end-date` | Backtest end date | (today) | Any date in YYYY-MM-DD format |
| `--symbols` | Symbols to trade | (from config) | Comma-separated list |
| `--broker` | Broker to use | `paper` | `zerodha`, `upstox`, `fivepaisa` |
| `--report` | Generate report | Not set | Flag, no value needed |
| `--report-file` | Report filename | `trading_report.html` | Any path |

## Running via Docker

### Build the Docker Image:
```bash
docker build -t btsm -f docker/Dockerfile .
```

### Run with Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up trading
```

### Run Backtesting in Container:
```bash
docker-compose -f docker/docker-compose.yml --profile backtest up backtesting
```

### Run Development Environment:
```bash
docker-compose -f docker/docker-compose.yml --profile dev up dev
```

## Running Tests

### Run All Tests:
```bash
pytest
```

### Run Unit Tests Only:
```bash
pytest tests/unit
```

### Run Integration Tests Only:
```bash
pytest tests/integration
```

### Generate Test Coverage Report:
```bash
pytest --cov=./ --cov-report=html
```

## Troubleshooting

### Common Issues

1. **TA-Lib Installation Errors**:
   - On Ubuntu/Debian, ensure you have `build-essential` installed
   - On Windows, use the precompiled wheel instead of pip install

2. **API Connection Issues**:
   - Check your API credentials in the config file
   - Ensure your broker's API services are operational
   - Check network connectivity and firewall settings

3. **Data Availability**:
   - Some symbols may not have data for the requested time period
   - Financial markets have holidays when no data is available

### Logs

- Check `logs/trading.log` for detailed operation logs
- Performance reports are saved in the `reports/` directory
- Visualizations are saved in `reports/visualizations/`

## Contact for Support

If you encounter issues not covered in this document:
- Open an issue on GitHub
- Email support at [your-email@example.com]
- Join our community Discord/Slack