# Strategy Training and Backtesting Guide for BTSM

This guide explains how to train and optimize trading strategies using the BTSM (Backtesting & Trading Strategy Manager) system with 52 weeks of historical data.

## Overview

The strategy training system allows you to:

1. **Optimize strategy parameters** using historical market data
2. **Train-test split** your data to validate strategies on out-of-sample data
3. **Compare performance** across different strategies
4. **Visualize results** to gain insights into strategy performance

## Getting Started

### Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

To train strategies with default settings (52 weeks of data):

```bash
python train_backtest.py
```

This will:
- Fetch 1 year (52 weeks) of market data
- Split the data into training (70%) and testing (30%) periods
- Train and optimize both Moving Average Crossover and RSI strategies
- Evaluate the optimized strategies on the test data
- Save results to the `reports/optimization` directory

### Advanced Options

The training script supports several command-line options:

```bash
python train_backtest.py --strategy moving_average --symbols RELIANCE,TCS,INFY --visualize
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to configuration file | `config/config.ini` |
| `--start-date` | Start date for backtest (YYYY-MM-DD) | 52 weeks ago |
| `--end-date` | End date for backtest (YYYY-MM-DD) | today |
| `--symbols` | Comma-separated list of symbols | `RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC` |
| `--strategy` | Strategy to optimize (`moving_average`, `rsi`, or `all`) | `all` |
| `--test-size` | Proportion of data to use for testing (0.0-1.0) | `0.3` |
| `--visualize` | Generate performance visualizations | `False` |

## Understanding the Training Process

### Strategy Optimization

The system performs a grid search over various parameter combinations:

#### Moving Average Crossover strategy parameters:
- Fast MA type: SMA, EMA
- Fast MA period: 5, 10, 15, ..., 45
- Slow MA type: SMA, EMA
- Slow MA period: 20, 40, 60, ..., 180
- Position size: 5%, 10%

#### RSI strategy parameters:
- RSI period: 7, 10, 13, 16, 19
- Overbought threshold: 65, 70, 75, 80, 85
- Oversold threshold: 15, 20, 25, 30, 35
- Position size: 5%, 10%

### Evaluation Metrics

Strategies are evaluated using several performance metrics:

- **Total Profit**: Overall profit/loss
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return-to-drawdown ratio

### Visualization

When using the `--visualize` flag, the system creates:

1. **Parameter heat maps** showing profitability across parameter combinations
2. **Strategy comparison charts** comparing training and testing performance
3. **Risk-return scatterplots** showing the trade-off between return and risk

## Integrating Optimized Strategies

After training, the system saves the optimal parameters to JSON files in the `reports/optimization` directory. You can integrate these parameters into your live trading by:

### Method 1: Manual Configuration Update

Copy the optimal parameters from the JSON files to your `config.ini` file:

```ini
[strategies]
# Moving Average Crossover strategy
moving_average.fast_ma_type = EMA
moving_average.fast_ma_period = 20
moving_average.slow_ma_type = SMA 
moving_average.slow_ma_period = 50
moving_average.position_size_pct = 5

# RSI strategy
rsi.rsi_period = 14
rsi.oversold_threshold = 30
rsi.overbought_threshold = 70
rsi.position_size_pct = 5
```

### Method 2: Dynamic Parameter Loading

The system can also load the optimal parameters directly:

```python
def run_with_optimal_params():
    # Load optimal parameters
    with open('reports/optimization/moving_average_optimal_params.json', 'r') as f:
        ma_params = json.load(f)
        
    # Configure strategy with these parameters
    strategy = MovingAverageCrossover('optimized_ma', config)
    strategy.fast_ma_type = ma_params.get('fast_ma_type', 'EMA')
    strategy.fast_ma_period = ma_params.get('fast_ma_period', 20)
    # ... set other parameters
    
    # Run strategy
    agent.add_strategy('optimized_ma', strategy)
    # ... start trading
```

## Troubleshooting

### Common Issues

1. **Not enough data for strategy**
   - Ensure your data source provides enough historical data
   - You may need to adjust the periods for technical indicators

2. **Optimization takes too long**
   - Reduce the number of parameter combinations
   - Test on fewer symbols initially

3. **Strategy performs well in training but poorly in testing**
   - This may indicate overfitting
   - Try simpler parameter combinations
   - Increase your training data size

## Next Steps

After finding optimal parameters:

1. Run a longer backtest to validate the strategy
2. Test the strategy in paper trading mode
3. Monitor performance in live market conditions
4. Periodically retrain the strategy to adapt to changing market conditions

## Additional Resources

- See `strategy_trainer.py` for details on the optimization process
- Examine `train_backtest.py` to understand the workflow
- For custom strategy implementation, check the `strategies` directory