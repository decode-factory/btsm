# BTSM - Backtesting & Trading Strategy Manager

BTSM is a comprehensive backtesting and trading strategy management system designed for Indian markets. It supports paper trading and live trading with multiple brokers, advanced strategy development, and detailed performance analytics.

## Features

- **Multiple Broker Support**: Zerodha, Upstox, and 5paisa integration
- **Paper Trading**: Practice trading without risking real money
- **Strategy Development**: Multiple built-in strategies and extensible framework
- **Advanced Backtesting**: Test strategies on historical data
- **Performance Analytics**: Detailed metrics and visualizations
- **Event-Driven Architecture**: React to market events efficiently

## Installation

### Prerequisites

- Python 3.10+
- pip
- TA-Lib (Technical Analysis Library)
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BTSM.git
cd BTSM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your environment:
```bash
# Copy the sample config
cp config/config.sample.ini config/config.ini

# Edit with your broker credentials and preferences
nano config/config.ini
```

## Usage

### Backtesting

Run a backtest on historical data:

```bash
python main.py --mode backtest --strategy moving_average --start-date 2023-01-01 --end-date 2023-12-31 --report
```

### Paper Trading

Start paper trading to practice without risking real money:

```bash
python main.py --mode paper --strategy rsi
```

### Live Trading

Start live trading with a real broker:

```bash
python main.py --mode live --broker zerodha --strategy moving_average
```

### Available Strategies

- **Moving Average Crossover**: Generates signals based on fast/slow MA crossovers
- **RSI Strategy**: Generates signals based on overbought/oversold conditions

### Running Tests

Run unit tests:

```bash
pytest tests/unit
```

Run integration tests:

```bash
pytest tests/integration
```

## Docker Deployment

Build and run using Docker:

```bash
# Build the image
docker build -t btsm -f docker/Dockerfile .

# Run for paper trading
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/reports:/app/reports -v $(pwd)/config:/app/config btsm --mode paper

# Using docker-compose
docker-compose -f docker/docker-compose.yml up trading
```

## Project Structure

```
BTSM/
├── analysis/          # Market analysis modules
├── config/            # Configuration files
├── core/              # Core system components
├── data/              # Data collection and processing
├── docker/            # Docker configuration
├── execution/         # Order execution and broker interfaces
├── reporting/         # Performance reporting and visualization
├── strategies/        # Trading strategies
├── tests/             # Unit and integration tests
└── utils/             # Utility functions
```

## Extending the System

### Adding a New Strategy

1. Create a new file in the `strategies` directory, e.g., `strategies/my_strategy.py`
2. Implement your strategy by extending the base `Strategy` class
3. Implement the `generate_signals` method
4. Add your strategy to the agent in `main.py`

Example:

```python
from strategies.base import Strategy

class MyStrategy(Strategy):
    def __init__(self, params=None):
        super().__init__('My_Strategy', params or {})
        
    def generate_signals(self, data):
        signals = []
        # Your strategy logic here
        return signals
```

### Adding a New Broker

1. Create a new file in the `execution` directory, e.g., `execution/my_broker.py`
2. Implement your broker by extending the `BrokerInterface` class
3. Implement all required abstract methods
4. Add your broker to the available options in `main.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this software.