import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.moving_average import MovingAverageCrossover

class TestMovingAverageCrossover(unittest.TestCase):
    """Test cases for the Moving Average Crossover strategy."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample strategy instance
        self.strategy = MovingAverageCrossover({
            'fast_ma_period': 5,
            'slow_ma_period': 10,
            'position_size_pct': 5,
            'stop_loss_pct': 5,
            'take_profit_pct': 10
        })
        
        # Create sample data with 20 days
        self.dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
        
        # Create a dataframe with an uptrend
        self.up_trend_data = pd.DataFrame({
            'timestamp': self.dates,
            'open': [100 + i for i in range(20)],
            'high': [102 + i for i in range(20)],
            'low': [99 + i for i in range(20)],
            'close': [101 + i for i in range(20)],
            'volume': [1000000 for _ in range(20)]
        })
        
        # Add moving averages
        self.up_trend_data['sma_5'] = self.up_trend_data['close'].rolling(window=5).mean()
        self.up_trend_data['sma_10'] = self.up_trend_data['close'].rolling(window=10).mean()
        
        # Create a dataframe with a crossover (buy signal)
        self.crossover_data = self.up_trend_data.copy()
        # Modify the last two rows to create a crossover
        self.crossover_data.loc[18, 'sma_5'] = 117.5  # Fast MA below slow MA
        self.crossover_data.loc[18, 'sma_10'] = 118.0
        self.crossover_data.loc[19, 'sma_5'] = 119.0  # Fast MA above slow MA
        self.crossover_data.loc[19, 'sma_10'] = 118.5
        
        # Create a dataframe with a crossunder (sell signal)
        self.crossunder_data = self.up_trend_data.copy()
        # Modify the last two rows to create a crossunder
        self.crossunder_data.loc[18, 'sma_5'] = 119.0  # Fast MA above slow MA
        self.crossunder_data.loc[18, 'sma_10'] = 118.0
        self.crossunder_data.loc[19, 'sma_5'] = 117.5  # Fast MA below slow MA
        self.crossunder_data.loc[19, 'sma_10'] = 118.0
    
    def test_no_signal_without_crossover(self):
        """Test that no signals are generated without a crossover."""
        # Prepare data
        symbol = 'TEST'
        data = {symbol: self.up_trend_data}
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Check that no signals were generated
        self.assertEqual(len(signals), 0)
    
    def test_buy_signal_on_crossover(self):
        """Test that a buy signal is generated on a crossover."""
        # Prepare data
        symbol = 'TEST'
        data = {symbol: self.crossover_data}
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Check that a buy signal was generated
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['action'], 'BUY')
        self.assertEqual(signals[0]['symbol'], symbol)
        self.assertEqual(signals[0]['price'], self.crossover_data['close'].iloc[-1])
        
        # Verify stop loss and take profit
        expected_stop_loss = signals[0]['price'] * (1 - self.strategy.params['stop_loss_pct'] / 100)
        expected_take_profit = signals[0]['price'] * (1 + self.strategy.params['take_profit_pct'] / 100)
        self.assertAlmostEqual(signals[0]['stop_loss'], expected_stop_loss)
        self.assertAlmostEqual(signals[0]['take_profit'], expected_take_profit)
    
    def test_sell_signal_on_crossunder(self):
        """Test that a sell signal is generated on a crossunder."""
        # Prepare data
        symbol = 'TEST'
        data = {symbol: self.crossunder_data}
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Check that a sell signal was generated
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['action'], 'SELL')
        self.assertEqual(signals[0]['symbol'], symbol)
        self.assertEqual(signals[0]['price'], self.crossunder_data['close'].iloc[-1])
    
    def test_position_size_calculation(self):
        """Test the position size calculation."""
        # Set equity in parameters
        equity = 1000000  # 10 lakhs
        self.strategy.params['equity'] = equity
        
        # Calculate position size
        price = 100
        expected_quantity = int((equity * self.strategy.params['position_size_pct'] / 100) / price)
        calculated_quantity = self.strategy._calculate_position_size(price)
        
        # Verify the calculation
        self.assertEqual(calculated_quantity, expected_quantity)
        
        # Test with a high price that would result in less than 1 share
        high_price = equity * 2
        calculated_quantity = self.strategy._calculate_position_size(high_price)
        
        # Verify minimum quantity is 1
        self.assertEqual(calculated_quantity, 1)

if __name__ == '__main__':
    unittest.main()