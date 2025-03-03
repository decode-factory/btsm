import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from reporting.metrics import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for the PerformanceMetrics class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metrics_file = os.path.join(self.temp_dir.name, 'performance.json')
        
        # Create test configuration
        self.config = {
            'performance_metrics_file': self.metrics_file,
            'risk_free_rate': 4.0,  # 4% annual risk-free rate
            'performance_benchmark': 'TEST_INDEX'
        }
        
        # Create sample trades
        dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
        
        self.sample_trades = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'entry_time': dates[0].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': dates[1].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': 150.0,
                'exit_price': 155.0,
                'quantity': 10,
                'profit_loss': 50.0,
                'profit_loss_pct': 3.33
            },
            {
                'symbol': 'MSFT',
                'action': 'BUY',
                'entry_time': dates[2].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': dates[3].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': 250.0,
                'exit_price': 245.0,
                'quantity': 5,
                'profit_loss': -25.0,
                'profit_loss_pct': -2.0
            },
            {
                'symbol': 'GOOGL',
                'action': 'BUY',
                'entry_time': dates[4].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': dates[5].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': 2700.0,
                'exit_price': 2750.0,
                'quantity': 1,
                'profit_loss': 50.0,
                'profit_loss_pct': 1.85
            },
            {
                'symbol': 'AMZN',
                'action': 'BUY',
                'entry_time': dates[6].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': dates[7].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': 3300.0,
                'exit_price': 3250.0,
                'quantity': 1,
                'profit_loss': -50.0,
                'profit_loss_pct': -1.52
            },
            {
                'symbol': 'NFLX',
                'action': 'BUY',
                'entry_time': dates[8].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': dates[9].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': 500.0,
                'exit_price': 520.0,
                'quantity': 5,
                'profit_loss': 100.0,
                'profit_loss_pct': 4.0
            }
        ]
        
        # Create sample benchmark data
        self.benchmark_data = pd.DataFrame({
            'date': [dates[i] for i in range(10)],
            'open': [100 + i for i in range(10)],
            'high': [102 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [101 + i for i in range(10)],
            'volume': [1000000 for _ in range(10)]
        })
        
        # Initialize metrics calculator
        self.metrics = PerformanceMetrics(self.config)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_calculate_basic_metrics(self):
        """Test calculation of basic performance metrics."""
        # Calculate metrics
        result = self.metrics.calculate(self.sample_trades)
        
        # Check basic metrics
        self.assertEqual(result['total_trades'], 5)
        self.assertEqual(result['winning_trades'], 3)
        self.assertEqual(result['losing_trades'], 2)
        self.assertAlmostEqual(result['win_rate'], 60.0)
        self.assertEqual(result['total_profit'], 125.0)
        self.assertAlmostEqual(result['avg_profit'], 25.0)
        
        # Check average winner and loser
        self.assertAlmostEqual(result['avg_winner'], 200/3)
        self.assertAlmostEqual(result['avg_loser'], -37.5)
        
        # Check profit factor
        self.assertAlmostEqual(result['profit_factor'], 200/75)
    
    def test_calculate_drawdown_metrics(self):
        """Test calculation of drawdown metrics."""
        # Calculate metrics
        result = self.metrics.calculate(self.sample_trades)
        
        # Check for presence of drawdown metrics
        self.assertIn('max_drawdown_pct', result)
        self.assertIn('avg_drawdown_pct', result)
        self.assertIn('max_underwater_period', result)
    
    def test_calculate_risk_metrics(self):
        """Test calculation of risk-adjusted metrics."""
        # Calculate metrics
        result = self.metrics.calculate(self.sample_trades)
        
        # Check for presence of risk metrics
        self.assertIn('sharpe_ratio', result)
        self.assertIn('sortino_ratio', result)
    
    def test_compare_to_benchmark(self):
        """Test comparison to benchmark."""
        # Create some aligned return series
        strategy_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.005, 0.005, 0.01, -0.005, 0.01])
        
        # Compare to benchmark
        result = self.metrics.compare_to_benchmark(strategy_returns, benchmark_returns)
        
        # Check for presence of comparison metrics
        self.assertIn('alpha', result)
        self.assertIn('beta', result)
        self.assertIn('correlation', result)
        self.assertIn('information_ratio', result)
        self.assertIn('tracking_error', result)
    
    def test_metrics_persistence(self):
        """Test saving and loading of metrics."""
        # Calculate and save metrics
        self.metrics.calculate(self.sample_trades)
        
        # Create a new metrics instance to load from file
        new_metrics = PerformanceMetrics(self.config)
        
        # Check if metrics history is loaded
        self.assertEqual(len(new_metrics.metrics_history), 1)
        
        # Check if last metrics match
        for key in self.metrics.last_metrics:
            if isinstance(self.metrics.last_metrics[key], (int, float)):
                self.assertAlmostEqual(new_metrics.last_metrics[key], self.metrics.last_metrics[key])
            else:
                self.assertEqual(new_metrics.last_metrics[key], self.metrics.last_metrics[key])

if __name__ == '__main__':
    unittest.main()