import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.agent import Agent
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from execution.paper_trading import PaperTradingBroker
from reporting.metrics import PerformanceMetrics
from reporting.visualization import TradeVisualizer

class TestTradingSystem(unittest.TestCase):
    """Integration tests for the trading system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create a temp directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create necessary subdirectories
        os.makedirs(os.path.join(cls.test_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, 'logs'), exist_ok=True)
        
        # Create a simple config file
        cls.config_path = os.path.join(cls.test_dir, 'test_config.ini')
        with open(cls.config_path, 'w') as f:
            f.write(f"""
[core]
app_name = TestTrader
log_level = INFO
log_file = {cls.test_dir}/logs/test.log
timezone = Asia/Kolkata

[data]
data_source = test
data_api_key = 
historical_data_path = {cls.test_dir}/data
cache_enabled = true
cache_expiry = 3600

[trading]
paper_trading = true
max_positions = 5
max_position_size_pct = 10
default_stop_loss_pct = 3
default_take_profit_pct = 6
max_daily_drawdown_pct = 5

[performance]
performance_metrics_file = {cls.test_dir}/reports/performance.json
            """)
        
        # Create sample historical data
        cls.dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        
        # Create a dataframe with price data that will trigger both strategies
        base_price = 100
        cls.prices = []
        # Create a price series with multiple ups and downs
        for i in range(100):
            # Add some oscillation to create trading signals
            oscillation = 10 * np.sin(i / 10) + 5 * np.sin(i / 5)
            cls.prices.append(base_price + oscillation + i * 0.2)  # Slight uptrend
        
        cls.sample_data = pd.DataFrame({
            'timestamp': cls.dates,
            'open': cls.prices,
            'high': [p * 1.01 for p in cls.prices],
            'low': [p * 0.99 for p in cls.prices],
            'close': cls.prices,
            'volume': [1000000 for _ in range(100)]
        })
        
        # Add technical indicators
        # Simple Moving Averages
        cls.sample_data['sma_20'] = cls.sample_data['close'].rolling(window=20).mean()
        cls.sample_data['sma_50'] = cls.sample_data['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        cls.sample_data['ema_12'] = cls.sample_data['close'].ewm(span=12, adjust=False).mean()
        cls.sample_data['ema_26'] = cls.sample_data['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = cls.sample_data['close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        cls.sample_data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Save the data to a CSV file
        cls.data_file = os.path.join(cls.test_dir, 'data', 'TSLA.csv')
        cls.sample_data.to_csv(cls.data_file, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up each test."""
        # Initialize agent with test config
        self.agent = Agent(self.config_path)
        
        # Set up data collector to use the sample data
        def mock_fetch_data(symbols, timeframe='day', start_date=None, end_date=None):
            return {symbol: self.sample_data for symbol in symbols}
        
        self.agent.data_collector.fetch_data = mock_fetch_data
        
        # Add strategies
        self.agent.add_strategy('ma_crossover', MovingAverageCrossover())
        self.agent.add_strategy('rsi', RSIStrategy())
        
        # Set up paper trading broker
        self.broker = PaperTradingBroker(self.agent.config)
        self.agent.set_broker(self.broker)
        
        # Initialize metrics calculator
        self.metrics = PerformanceMetrics(self.agent.config)
        
        # Initialize visualizer
        self.visualizer = TradeVisualizer(self.agent.config)
    
    def test_end_to_end_backtest(self):
        """Test running a full backtest with all components."""
        # Run backtest
        symbols = ['TSLA']
        start_date = self.dates[0].strftime('%Y-%m-%d')
        end_date = self.dates[-1].strftime('%Y-%m-%d')
        
        results = self.agent.run_backtest(symbols, start_date, end_date)
        
        # Check results structure
        self.assertIn('ma_crossover', results)
        self.assertIn('rsi', results)
        
        for strategy_id, strategy_results in results.items():
            # Check that trades were generated
            self.assertIn('trades', strategy_results)
            trades = strategy_results['trades']
            self.assertTrue(len(trades) > 0, f"No trades generated for {strategy_id}")
            
            # Check that performance metrics were calculated
            self.assertIn('performance', strategy_results)
            performance = strategy_results['performance']
            self.assertTrue(len(performance) > 0, f"No performance metrics calculated for {strategy_id}")
            
            # Check specific metrics
            self.assertIn('total_trades', performance)
            self.assertIn('win_rate', performance)
            self.assertIn('profit_factor', performance)
            
            # Check trade data structure
            for trade in trades[:5]:  # Check first few trades
                self.assertIn('symbol', trade)
                self.assertIn('entry_time', trade)
                self.assertIn('exit_time', trade)
                self.assertIn('entry_price', trade)
                self.assertIn('exit_price', trade)
                self.assertIn('profit_loss', trade)
            
            # Generate visualizations
            try:
                # Plot equity curve
                equity_fig = self.visualizer.plot_equity_curve(
                    trades=trades,
                    show=False,
                    save_path=f'{strategy_id}_equity.png'
                )
                self.assertIsNotNone(equity_fig)
                
                # Plot trade distribution
                dist_fig = self.visualizer.plot_trade_distribution(
                    trades=trades,
                    show=False,
                    save_path=f'{strategy_id}_distribution.png'
                )
                self.assertIsNotNone(dist_fig)
                
                # Generate HTML report
                report_success = self.visualizer.generate_html_report(
                    performance_metrics=performance,
                    trades=trades,
                    output_path=f'{strategy_id}_report.html'
                )
                self.assertTrue(report_success)
                
            except Exception as e:
                self.fail(f"Visualization failed: {str(e)}")
    
    def test_paper_trading_execution(self):
        """Test paper trading execution."""
        # Set up the paper trading broker
        broker = self.agent.broker
        
        # Create test signals
        signals = [
            {
                'symbol': 'TSLA',
                'action': 'BUY',
                'price': 100.0,
                'quantity': 10,
                'stop_loss': 95.0,
                'take_profit': 110.0
            },
            {
                'symbol': 'TSLA',
                'action': 'SELL',
                'price': 110.0,
                'quantity': 10
            }
        ]
        
        # Execute signals
        for signal in signals:
            result = broker.execute_order(signal)
            self.assertIn('status', result)
            
            if result['status'] == 'rejected':
                self.fail(f"Order rejected: {result['reason']}")
            elif result['status'] == 'failed':
                self.fail(f"Order failed: {result['reason']}")
            else:
                self.assertEqual(result['status'], 'executed')
        
        # Check account information
        account_info = broker.get_account_info()
        self.assertIn('cash_balance', account_info)
        self.assertIn('equity', account_info)
        
        # Check orders
        orders = broker.get_orders()
        self.assertEqual(len(orders), 0)  # All orders should be executed
        
        # Check order history
        order_history = broker.order_history
        self.assertEqual(len(order_history), 2)  # Both orders should be in history
        
        # Check positions
        positions = broker.get_positions()
        # After buy and sell, there should be no positions
        self.assertEqual(len(positions), 0)

if __name__ == '__main__':
    unittest.main()