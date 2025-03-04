#!/usr/bin/env python
# strategy_trainer.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import json

from strategies.base import Strategy
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from data.collector import DataCollector
from reporting.metrics import PerformanceMetrics
from execution.risk_management import RiskManager
from config.settings import load_config

class StrategyTrainer:
    """
    Strategy trainer that optimizes strategy parameters using historical data.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the strategy trainer with configuration."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_collector = DataCollector(self.config)
        self.risk_manager = RiskManager(self.config)
        self.metrics = PerformanceMetrics()
        
        # Output directories
        self.results_dir = os.path.join('reports', 'optimization')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def train_moving_average_strategy(self, 
                                     symbols: List[str], 
                                     start_date: str, 
                                     end_date: str) -> Dict[str, Any]:
        """
        Train MovingAverageCrossover strategy by finding optimal parameters.
        
        Args:
            symbols: List of stock symbols to train on
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        self.logger.info(f"Training MovingAverageCrossover strategy from {start_date} to {end_date}")
        
        # Fetch historical data
        data = self.data_collector.fetch_data(symbols, 'day', start_date, end_date)
        
        # Parameter ranges to test
        fast_ma_types = ['SMA', 'EMA']
        fast_ma_periods = range(5, 50, 5)  # 5, 10, 15, ..., 45
        slow_ma_types = ['SMA', 'EMA']
        slow_ma_periods = range(20, 200, 20)  # 20, 40, 60, ..., 180
        position_sizes = [5, 10]  # 5%, 10%
        
        best_params = None
        best_performance = {
            'total_profit': -float('inf'),
            'sharpe_ratio': -float('inf')
        }
        
        results = []
        
        # Grid search
        for fast_type in fast_ma_types:
            for fast_period in fast_ma_periods:
                for slow_type in slow_ma_types:
                    for slow_period in slow_ma_periods:
                        # Skip invalid combinations
                        if fast_period >= slow_period:
                            continue
                            
                        for position_size in position_sizes:
                            # Create strategy with these parameters
                            params = {
                                'fast_ma_type': fast_type,
                                'fast_ma_period': fast_period,
                                'slow_ma_type': slow_type,
                                'slow_ma_period': slow_period,
                                'position_size_pct': position_size
                            }
                            
                            strategy = MovingAverageCrossover(params)
                            
                            # Run backtest
                            try:
                                trades = strategy.backtest(data, self.risk_manager)
                                performance = self.metrics.calculate(trades)
                                
                                # Log progress
                                self.logger.debug(
                                    f"MA({fast_type}-{fast_period}/{slow_type}-{slow_period}): "
                                    f"Profit={performance.get('total_profit', 0):.2f}, "
                                    f"Sharpe={performance.get('sharpe_ratio', 0):.2f}"
                                )
                                
                                # Record results
                                result = {
                                    **params,
                                    **performance
                                }
                                results.append(result)
                                
                                # Check if this is the best strategy so far
                                # We prioritize total profit but also consider risk-adjusted returns
                                if (performance.get('total_profit', 0) > best_performance['total_profit'] and
                                    performance.get('sharpe_ratio', 0) > 0):
                                    best_performance = performance
                                    best_params = params
                                    
                            except Exception as e:
                                self.logger.error(f"Error backtesting MA strategy: {str(e)}")
                                continue
        
        # Save all results
        self._save_results('moving_average', results)
        
        # Return best parameters and performance
        return {
            'strategy': 'MovingAverageCrossover',
            'optimal_params': best_params,
            'performance': best_performance
        }
    
    def train_rsi_strategy(self, 
                          symbols: List[str], 
                          start_date: str, 
                          end_date: str) -> Dict[str, Any]:
        """
        Train RSIStrategy by finding optimal parameters.
        
        Args:
            symbols: List of stock symbols to train on
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        self.logger.info(f"Training RSIStrategy from {start_date} to {end_date}")
        
        # Fetch historical data
        data = self.data_collector.fetch_data(symbols, 'day', start_date, end_date)
        
        # Parameter ranges to test
        rsi_periods = range(7, 22, 3)  # 7, 10, 13, 16, 19
        overbought_thresholds = range(65, 86, 5)  # 65, 70, 75, 80, 85
        oversold_thresholds = range(15, 36, 5)  # 15, 20, 25, 30, 35
        position_sizes = [5, 10]  # 5%, 10%
        
        best_params = None
        best_performance = {
            'total_profit': -float('inf'),
            'sharpe_ratio': -float('inf')
        }
        
        results = []
        
        # Grid search
        for rsi_period in rsi_periods:
            for overbought in overbought_thresholds:
                for oversold in oversold_thresholds:
                    # Skip invalid combinations
                    if oversold >= overbought:
                        continue
                        
                    for position_size in position_sizes:
                        # Create strategy with these parameters
                        params = {
                            'rsi_period': rsi_period,
                            'overbought_threshold': overbought,
                            'oversold_threshold': oversold,
                            'position_size_pct': position_size
                        }
                        
                        strategy = RSIStrategy(params)
                        
                        # Run backtest
                        try:
                            trades = strategy.backtest(data, self.risk_manager)
                            performance = self.metrics.calculate(trades)
                            
                            # Log progress
                            self.logger.debug(
                                f"RSI(period={rsi_period}, oversold={oversold}, overbought={overbought}): "
                                f"Profit={performance.get('total_profit', 0):.2f}, "
                                f"Sharpe={performance.get('sharpe_ratio', 0):.2f}"
                            )
                            
                            # Record results
                            result = {
                                **params,
                                **performance
                            }
                            results.append(result)
                            
                            # Check if this is the best strategy so far
                            if (performance.get('total_profit', 0) > best_performance['total_profit'] and
                                performance.get('sharpe_ratio', 0) > 0):
                                best_performance = performance
                                best_params = params
                                
                        except Exception as e:
                            self.logger.error(f"Error backtesting RSI strategy: {str(e)}")
                            continue
        
        # Save all results
        self._save_results('rsi', results)
        
        # Return best parameters and performance
        return {
            'strategy': 'RSIStrategy',
            'optimal_params': best_params,
            'performance': best_performance
        }
        
    def evaluate_strategy(self, 
                         strategy_type: str, 
                         params: Dict[str, Any], 
                         symbols: List[str], 
                         start_date: str, 
                         end_date: str) -> Dict[str, Any]:
        """
        Evaluate a specific strategy configuration on out-of-sample data.
        
        Args:
            strategy_type: Type of strategy ('moving_average' or 'rsi')
            params: Strategy parameters
            symbols: List of stock symbols to evaluate on
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Evaluating {strategy_type} strategy from {start_date} to {end_date}")
        
        # Fetch historical data
        data = self.data_collector.fetch_data(symbols, 'day', start_date, end_date)
        
        # Create strategy with the given parameters
        if strategy_type == 'moving_average':
            eval_params = {
                'fast_ma_type': params.get('fast_ma_type', 'EMA'),
                'fast_ma_period': params.get('fast_ma_period', 20),
                'slow_ma_type': params.get('slow_ma_type', 'SMA'),
                'slow_ma_period': params.get('slow_ma_period', 50),
                'position_size_pct': params.get('position_size_pct', 5)
            }
            strategy = MovingAverageCrossover(eval_params)
            
        elif strategy_type == 'rsi':
            eval_params = {
                'rsi_period': params.get('rsi_period', 14),
                'overbought_threshold': params.get('overbought_threshold', 70),
                'oversold_threshold': params.get('oversold_threshold', 30),
                'position_size_pct': params.get('position_size_pct', 5)
            }
            strategy = RSIStrategy(eval_params)
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Run backtest
        trades = strategy.backtest(data, self.risk_manager)
        performance = self.metrics.calculate(trades)
        
        return {
            'strategy': strategy_type,
            'params': params,
            'performance': performance,
            'trades': trades
        }
    
    def _save_results(self, strategy_name: str, results: List[Dict[str, Any]]) -> None:
        """Save optimization results to a CSV file."""
        # Convert to DataFrame
        if not results:
            self.logger.warning(f"No results to save for {strategy_name}")
            return
            
        df = pd.DataFrame(results)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to CSV
        file_path = os.path.join(self.results_dir, f"{strategy_name}_optimization_{timestamp}.csv")
        df.to_csv(file_path, index=False)
        
        # Create summary visualization
        self._create_optimization_visualization(df, strategy_name, timestamp)
        
        self.logger.info(f"Results saved to {file_path}")
    
    def _create_optimization_visualization(self, 
                                          df: pd.DataFrame, 
                                          strategy_name: str, 
                                          timestamp: str) -> None:
        """Create visualizations of the optimization results."""
        try:
            plt.figure(figsize=(12, 8))
            
            if strategy_name == 'moving_average':
                # Plot heat map of fast vs. slow period
                pivot = df.pivot_table(
                    values='total_profit', 
                    index='fast_ma_period', 
                    columns='slow_ma_period', 
                    aggfunc='mean'
                )
                plt.subplot(2, 2, 1)
                plt.title('Total Profit by MA Periods')
                sns.heatmap(pivot, annot=False, cmap='viridis')
                
                # Plot bar chart of best combinations
                top_n = df.nlargest(10, 'total_profit')
                plt.subplot(2, 2, 2)
                plt.title('Top 10 Profitable Combinations')
                top_n['combination'] = top_n.apply(
                    lambda x: f"{x['fast_ma_type']}-{x['fast_ma_period']}/{x['slow_ma_type']}-{x['slow_ma_period']}", 
                    axis=1
                )
                sns.barplot(data=top_n, x='combination', y='total_profit')
                plt.xticks(rotation=90)
                
                # Plot profitability by MA type
                plt.subplot(2, 2, 3)
                plt.title('Profit by MA Type Combination')
                type_pivot = df.pivot_table(
                    values='total_profit', 
                    index='fast_ma_type', 
                    columns='slow_ma_type', 
                    aggfunc='mean'
                )
                sns.heatmap(type_pivot, annot=True, cmap='viridis')
                
                # Plot relation between Sharpe and Total Profit
                plt.subplot(2, 2, 4)
                plt.title('Risk-Adjusted Return vs. Total Profit')
                plt.scatter(df['sharpe_ratio'], df['total_profit'], alpha=0.6)
                plt.xlabel('Sharpe Ratio')
                plt.ylabel('Total Profit')
                
            elif strategy_name == 'rsi':
                # Plot heat map of overbought vs. oversold thresholds
                pivot = df.pivot_table(
                    values='total_profit', 
                    index='oversold_threshold', 
                    columns='overbought_threshold', 
                    aggfunc='mean'
                )
                plt.subplot(2, 2, 1)
                plt.title('Total Profit by RSI Thresholds')
                sns.heatmap(pivot, annot=False, cmap='viridis')
                
                # Plot bar chart of best combinations
                top_n = df.nlargest(10, 'total_profit')
                plt.subplot(2, 2, 2)
                plt.title('Top 10 Profitable Combinations')
                top_n['combination'] = top_n.apply(
                    lambda x: f"RSI-{x['rsi_period']} ({x['oversold_threshold']}/{x['overbought_threshold']})", 
                    axis=1
                )
                sns.barplot(data=top_n, x='combination', y='total_profit')
                plt.xticks(rotation=90)
                
                # Plot profitability by RSI period
                plt.subplot(2, 2, 3)
                plt.title('Profit by RSI Period')
                period_data = df.groupby('rsi_period')['total_profit'].mean().reset_index()
                sns.barplot(data=period_data, x='rsi_period', y='total_profit')
                
                # Plot relation between Sharpe and Total Profit
                plt.subplot(2, 2, 4)
                plt.title('Risk-Adjusted Return vs. Total Profit')
                plt.scatter(df['sharpe_ratio'], df['total_profit'], alpha=0.6)
                plt.xlabel('Sharpe Ratio')
                plt.ylabel('Total Profit')
            
            plt.tight_layout()
            
            # Save the figure
            file_path = os.path.join(self.results_dir, f"{strategy_name}_optimization_{timestamp}.png")
            plt.savefig(file_path)
            plt.close()
            
            self.logger.info(f"Visualization saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
    
    def save_optimal_parameters(self, strategy_name: str, optimal_params: Dict[str, Any]) -> None:
        """Save optimal parameters to a JSON file for future use."""
        if not optimal_params:
            self.logger.warning(f"No optimal parameters to save for {strategy_name}")
            return
            
        file_path = os.path.join(self.results_dir, f"{strategy_name}_optimal_params.json")
        
        with open(file_path, 'w') as f:
            json.dump(optimal_params, f, indent=4)
            
        self.logger.info(f"Optimal parameters saved to {file_path}")
    
    def train_test_split(self, start_date: str, end_date: str, test_size: float = 0.3) -> Tuple[str, str, str]:
        """
        Split date range into training and testing periods.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            test_size: Proportion of the date range to use for testing
            
        Returns:
            Tuple of (training_start, training_end, testing_start)
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end - start).days
        
        test_days = int(total_days * test_size)
        train_days = total_days - test_days
        
        train_end = start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        
        return (
            start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            test_start.strftime('%Y-%m-%d')
        )


if __name__ == "__main__":
    # Example usage
    import seaborn as sns
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer
    trainer = StrategyTrainer('config/config.ini')
    
    # Define date range (last 52 weeks)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Split into training and testing periods
    train_start, train_end, test_start = trainer.train_test_split(start_date, end_date)
    
    print(f"Training period: {train_start} to {train_end}")
    print(f"Testing period: {test_start} to {end_date}")
    
    # Select symbols to train on
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    # Train strategies
    print("Training Moving Average Crossover strategy...")
    ma_results = trainer.train_moving_average_strategy(symbols, train_start, train_end)
    
    print("Training RSI strategy...")
    rsi_results = trainer.train_rsi_strategy(symbols, train_start, train_end)
    
    # Save optimal parameters
    trainer.save_optimal_parameters('moving_average', ma_results['optimal_params'])
    trainer.save_optimal_parameters('rsi', rsi_results['optimal_params'])
    
    # Evaluate on test data
    print("Evaluating strategies on test data...")
    ma_eval = trainer.evaluate_strategy(
        'moving_average', 
        ma_results['optimal_params'], 
        symbols, 
        test_start, 
        end_date
    )
    
    rsi_eval = trainer.evaluate_strategy(
        'rsi', 
        rsi_results['optimal_params'], 
        symbols, 
        test_start, 
        end_date
    )
    
    # Print results
    print("\nMoving Average Crossover Strategy:")
    print(f"Optimal Parameters: {ma_results['optimal_params']}")
    print(f"Training Performance: {ma_results['performance']}")
    print(f"Testing Performance: {ma_eval['performance']}")
    
    print("\nRSI Strategy:")
    print(f"Optimal Parameters: {rsi_results['optimal_params']}")
    print(f"Training Performance: {rsi_results['performance']}")
    print(f"Testing Performance: {rsi_eval['performance']}")