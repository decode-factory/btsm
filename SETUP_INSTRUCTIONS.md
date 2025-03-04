# BTSM (Backtesting & Trading Strategy Manager) Setup Guide

## Prerequisites
- Python 3.8+ 
- pip or conda for package management
- Git (for cloning the repository)
- For Upstox integration: Upstox API credentials (API key and secret)

## Step 1: Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/BTSM.git
cd BTSM
```

### Using pip (Recommended for most users)
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install python-dotenv if not included in requirements
pip install python-dotenv
```

### Using Docker (Alternative)
```bash
# Build the Docker image
docker build -t btsm -f docker/Dockerfile .

# Run the container
docker run -it --name btsm-container btsm
```

## Step 2: Configuration

### API Credentials Setup
1. Edit `config/config.ini` with your broker API credentials:
   ```ini
   [brokers]
   # Upstox API credentials
   upstox_api_key = your_upstox_api_key_here
   upstox_api_secret = your_upstox_api_secret_here
   ```

2. For Upstox specific setup:
   - Create a `.env` file in the project root with the following content:
   ```
   UPSTOX_API_KEY=your_upstox_api_key_here
   UPSTOX_API_SECRET=your_upstox_api_secret_here
   UPSTOX_REDIRECT_URL=http://localhost:5000/callback
   ```

### Obtaining Upstox API Credentials
1. Register as a developer on the [Upstox Developer Portal](https://developer.upstox.com/)
2. Create a new application to get API credentials
3. Use the following for the redirect URL: `http://localhost:5000/callback`
4. Copy the obtained API key and secret to your config files

## Step 3: Running the Application

### Backtesting Mode
```bash
# Test all strategies
python main.py --mode backtest --broker upstox --strategy all --report

# Test a specific strategy
python main.py --mode backtest --broker upstox --strategy moving_average --report
```

### Paper Trading Mode
```bash
python main.py --mode paper_trading --broker upstox --strategy moving_average
```

### Live Trading Mode (use with caution)
```bash
python main.py --mode live --broker upstox --strategy moving_average
```

## Step 4: Implementing Custom Strategies

1. Create a new file in the `strategies` directory
2. Inherit from the base Strategy class
3. Implement the required methods:
   - `__init__`: Initialize your strategy
   - `generate_signals`: Implement your trading logic
   - `calculate_position_size`: Define position sizing rules

Example:
```python
from strategies.base import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, config, broker):
        super().__init__("MyCustomStrategy", config, broker)
        # Initialize strategy-specific parameters
        
    def generate_signals(self, data):
        # Implement your trading logic
        # Return buy/sell signals
        
    def calculate_position_size(self, signal, price):
        # Implement position sizing logic
        # Return the number of shares/contracts to trade
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'dotenv'**
   - Solution: Install python-dotenv package: `pip install python-dotenv`

2. **API Connection Issues**
   - Verify your API credentials in config.ini and .env files
   - Check if your Upstox redirect URL matches what's registered in the developer portal
   - Ensure your API key has the necessary permissions

3. **NoneType errors in trading loop**
   - This often indicates issues with data processing or missing market data
   - Check your historical data sources and ensure symbols are properly configured

4. **Docker-related issues**
   - See the DOCKER_GUIDE.md file for detailed Docker troubleshooting

### Getting Help
- Check the project's GitHub repository for issues and discussions
- Open an issue for bugs or enhancement requests

## Understanding the System Architecture

The BTSM system consists of several key components:

1. **Core**: Central event system and agent management
2. **Data**: Market data collection and processing
3. **Analysis**: Technical indicators and prediction models
4. **Strategies**: Trading strategy implementation
5. **Execution**: Broker interfaces for different platforms
6. **Reporting**: Performance metrics and visualization

Refer to the README.md for a detailed overview of the system architecture.