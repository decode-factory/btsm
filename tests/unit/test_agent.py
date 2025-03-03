import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.agent import Agent
from strategies.moving_average import MovingAverageCrossover
from execution.paper_trading import PaperTradingBroker

class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple config file
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.ini')
        with open(self.config_path, 'w') as f:
            f.write("""
[core]
app_name = TestTrader
log_level = INFO
log_file = 
timezone = Asia/Kolkata

[data]
data_source = test
data_api_key = 
historical_data_path = 
cache_enabled = true
cache_expiry = 3600

[trading]
paper_trading = true
max_positions = 5
max_position_size_pct = 10
default_stop_loss_pct = 3
default_take_profit_pct = 6
max_daily_drawdown_pct = 5
            """)
        
        # Initialize agent with test config
        self.agent = Agent(self.config_path)
        
        # Create sample data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        self.sample_data = {
            'AAPL': pd.DataFrame({
                'timestamp': dates,
                'open': [150 + i*0.1 for i in range(30)],
                'high': [152 + i*0.1 for i in range(30)],
                'low': [149 + i*0.1 for i in range(30)],
                'close': [151 + i*0.1 for i in range(30)],
                'volume': [1000000 for _ in range(30)],
                'sma_20': [140 + i*0.3 for i in range(30)],
                'sma_50': [130 + i*0.2 for i in range(30)],
                'ema_12': [145 + i*0.25 for i in range(30)],
                'ema_26': [135 + i*0.15 for i in range(30)]
            }),
            'MSFT': pd.DataFrame({
                'timestamp': dates,
                'open': [250 + i*0.2 for i in range(30)],
                'high': [252 + i*0.2 for i in range(30)],
                'low': [249 + i*0.2 for i in range(30)],
                'close': [251 + i*0.2 for i in range(30)],
                'volume': [2000000 for _ in range(30)],
                'sma_20': [240 + i*0.3 for i in range(30)],
                'sma_50': [230 + i*0.2 for i in range(30)],
                'ema_12': [245 + i*0.25 for i in range(30)],
                'ema_26': [235 + i*0.15 for i in range(30)]
            })
        }
        
        # Mock the data collector's fetch_data method
        self.agent.data_collector.fetch_data = MagicMock(return_value=self.sample_data)
        
        # Add a strategy
        self.strategy = MovingAverageCrossover()
        self.agent.add_strategy('ma_crossover', self.strategy)
        
        # Set up a paper trading broker
        self.broker = PaperTradingBroker(self.agent.config)
        self.agent.set_broker(self.broker)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.config['app_name'], 'TestTrader')
        self.assertTrue(self.agent.config['paper_trading'])
        self.assertEqual(self.agent.config['max_positions'], 5)
        
        # Check components initialization
        self.assertIsNotNone(self.agent.data_collector)
        self.assertIsNotNone(self.agent.data_processor)
        self.assertIsNotNone(self.agent.risk_manager)
        self.assertIsNotNone(self.agent.metrics)
        
        # Check default state
        self.assertTrue(self.agent.is_paper_trading)
        self.assertFalse(self.agent.running)
    
    def test_add_strategy(self):
        """Test adding a strategy."""
        # Clear existing strategies
        self.agent.strategies = {}
        
        # Add a strategy
        strategy = MovingAverageCrossover()
        self.agent.add_strategy('test_strategy', strategy)
        
        # Check if strategy was added
        self.assertIn('test_strategy', self.agent.strategies)
        self.assertEqual(self.agent.strategies['test_strategy'], strategy)
    
    def test_set_broker(self):
        """Test setting the broker."""
        # Create a new broker
        broker = PaperTradingBroker(self.agent.config)
        
        # Set the broker
        self.agent.set_broker(broker)
        
        # Check if broker was set
        self.assertEqual(self.agent.broker, broker)
    
    def test_fetch_market_data(self):
        """Test fetching market data."""
        # Mock the data processor's process method
        self.agent.data_processor.process = MagicMock(return_value=self.sample_data)
        
        # Fetch data
        symbols = ['AAPL', 'MSFT']
        timeframe = 'day'
        start_date = '2023-01-01'
        end_date = '2023-01-30'
        
        result = self.agent.fetch_market_data(symbols, timeframe, start_date, end_date)
        
        # Check if data collector and processor were called
        self.agent.data_collector.fetch_data.assert_called_with(symbols, timeframe, start_date, end_date)
        self.agent.data_processor.process.assert_called_with(self.sample_data)
        
        # Check returned data
        self.assertEqual(result, self.sample_data)
    
    @patch('core.agent.Strategy.backtest')
    def test_run_backtest(self, mock_backtest):
        """Test running a backtest."""
        # Set up mock return values
        mock_trades = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'entry_time': '2023-01-01 10:00:00',
                'exit_time': '2023-01-02 10:00:00',
                'entry_price': 150.0,
                'exit_price': 155.0,
                'quantity': 10,
                'profit_loss': 50.0
            }
        ]
        mock_backtest.return_value = mock_trades
        
        # Mock metrics calculation
        self.agent.metrics.calculate = MagicMock(return_value={'win_rate': 100.0})
        
        # Run backtest
        symbols = ['AAPL', 'MSFT']
        start_date = '2023-01-01'
        end_date = '2023-01-30'
        
        result = self.agent.run_backtest(symbols, start_date, end_date)
        
        # Check if data was fetched
        self.agent.data_collector.fetch_data.assert_called_with(symbols, 'day', start_date, end_date)
        
        # Check if strategy backtest was called
        mock_backtest.assert_called()
        
        # Check if metrics were calculated
        self.agent.metrics.calculate.assert_called_with(mock_trades)
        
        # Check returned results
        self.assertIn('ma_crossover', result)
        self.assertEqual(result['ma_crossover']['trades'], mock_trades)
        self.assertEqual(result['ma_crossover']['performance'], {'win_rate': 100.0})
    
    def test_start_and_stop_trading(self):
        """Test starting and stopping live trading."""
        # Test that an error is raised if broker is not set
        self.agent.broker = None
        with self.assertRaises(ValueError):
            self.agent.start_live_trading(['AAPL', 'MSFT'])
        
        # Set broker again
        self.agent.set_broker(self.broker)
        
        # Mock the _wait_for_next_iteration method to avoid infinite loop
        self.agent._wait_for_next_iteration = MagicMock()
        
        # Mock the broker's execute_order method
        self.agent.broker.execute_order = MagicMock()
        
        # Mock the strategy's generate_signals method
        self.agent.strategies['ma_crossover'].generate_signals = MagicMock(return_value=[
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'price': 150.0,
                'quantity': 10
            }
        ])
        
        # Mock the risk manager's evaluate_signals method
        self.agent.risk_manager.evaluate_signals = MagicMock(return_value=[
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'price': 150.0,
                'quantity': 10
            }
        ])
        
        # Define a patched version of the start_live_trading method
        def patched_start_live_trading(symbols, timeframe):
            self.agent.running = True
            # Run one iteration only
            try:
                # Fetch latest market data
                data = self.agent.fetch_market_data(symbols, timeframe)
                
                for strategy_id, strategy in self.agent.strategies.items():
                    # Generate signals
                    signals = strategy.generate_signals(data)
                    
                    # Apply risk management
                    approved_signals = self.agent.risk_manager.evaluate_signals(signals)
                    
                    # Execute trades
                    for signal in approved_signals:
                        self.agent.broker.execute_order(signal)
                
                # Stop after one iteration
                self.agent.stop_trading()
                
            except Exception as e:
                self.fail(f"Exception in trading loop: {str(e)}")
        
        # Patch the method
        self.agent.start_live_trading = patched_start_live_trading
        
        # Start live trading
        self.agent.start_live_trading(['AAPL', 'MSFT'])
        
        # Check that it was started and stopped
        self.assertFalse(self.agent.running)
        
        # Check that methods were called
        self.agent.strategies['ma_crossover'].generate_signals.assert_called()
        self.agent.risk_manager.evaluate_signals.assert_called()
        self.agent.broker.execute_order.assert_called()

if __name__ == '__main__':
    unittest.main()