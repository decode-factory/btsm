#!/usr/bin/env python
# train_backtest.py
import argparse
import logging
import os
from datetime import datetime, timedelta
from strategies.strategy_trainer import StrategyTrainer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'strategy_training.log')),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='BTSM Strategy Training and Backtesting')
    
    # Configuration
    parser.add_argument('--config', default='config/config.ini',
                       help='Path to configuration file')
    
    # Date range
    parser.add_argument('--start-date', 
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', 
                       help='End date for backtest (YYYY-MM-DD)')
    
    # Symbols to use
    parser.add_argument('--symbols', 
                       help='Comma-separated list of symbols to train on')
    
    # Strategy selection
    parser.add_argument('--strategy', choices=['moving_average', 'rsi', 'all'], default='all',
                       help='Trading strategy to optimize')
    
    # Test-train split
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Proportion of date range to use for testing (0.0-1.0)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate performance visualizations')
    
    return parser.parse_args()

def main():
    """Main function to run strategy training and backtesting."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger('train_backtest')
    
    # Create trainer
    trainer = StrategyTrainer(args.config)
    
    # Determine date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    # For comprehensive backtesting, use 52 weeks (1 year) of data by default
    start_date = args.start_date or (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Split into training and testing periods
    train_start, train_end, test_start = trainer.train_test_split(
        start_date, end_date, test_size=args.test_size
    )
    
    logger.info(f"Training period: {train_start} to {train_end}")
    logger.info(f"Testing period: {test_start} to {end_date}")
    
    # Determine symbols to use
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        # Default symbols from NIFTY50
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ITC']
    
    logger.info(f"Using symbols: {', '.join(symbols)}")
    
    # Train strategies
    ma_results = None
    rsi_results = None
    
    if args.strategy in ['moving_average', 'all']:
        logger.info("Training Moving Average Crossover strategy...")
        ma_results = trainer.train_moving_average_strategy(symbols, train_start, train_end)
        trainer.save_optimal_parameters('moving_average', ma_results['optimal_params'])
    
    if args.strategy in ['rsi', 'all']:
        logger.info("Training RSI strategy...")
        rsi_results = trainer.train_rsi_strategy(symbols, train_start, train_end)
        trainer.save_optimal_parameters('rsi', rsi_results['optimal_params'])
    
    # Evaluate on test data
    logger.info("Evaluating strategies on test data...")
    
    results = []
    
    if ma_results:
        ma_eval = trainer.evaluate_strategy(
            'moving_average', 
            ma_results['optimal_params'], 
            symbols, 
            test_start, 
            end_date
        )
        results.append({
            'Strategy': 'Moving Average Crossover',
            'Training Profit': ma_results['performance'].get('total_profit', 0),
            'Testing Profit': ma_eval['performance'].get('total_profit', 0),
            'Training Sharpe': ma_results['performance'].get('sharpe_ratio', 0),
            'Testing Sharpe': ma_eval['performance'].get('sharpe_ratio', 0),
            'Win Rate': ma_eval['performance'].get('win_rate', 0),
            'Profit Factor': ma_eval['performance'].get('profit_factor', 0),
            'Max Drawdown': ma_eval['performance'].get('max_drawdown_pct', 0)
        })
        
        logger.info("\nMoving Average Crossover Strategy:")
        logger.info(f"Optimal Parameters: {ma_results['optimal_params']}")
        logger.info(f"Training Performance: {ma_results['performance']}")
        logger.info(f"Testing Performance: {ma_eval['performance']}")
    
    if rsi_results:
        rsi_eval = trainer.evaluate_strategy(
            'rsi', 
            rsi_results['optimal_params'], 
            symbols, 
            test_start, 
            end_date
        )
        results.append({
            'Strategy': 'RSI',
            'Training Profit': rsi_results['performance'].get('total_profit', 0),
            'Testing Profit': rsi_eval['performance'].get('total_profit', 0),
            'Training Sharpe': rsi_results['performance'].get('sharpe_ratio', 0),
            'Testing Sharpe': rsi_eval['performance'].get('sharpe_ratio', 0),
            'Win Rate': rsi_eval['performance'].get('win_rate', 0),
            'Profit Factor': rsi_eval['performance'].get('profit_factor', 0),
            'Max Drawdown': rsi_eval['performance'].get('max_drawdown_pct', 0)
        })
        
        logger.info("\nRSI Strategy:")
        logger.info(f"Optimal Parameters: {rsi_results['optimal_params']}")
        logger.info(f"Training Performance: {rsi_results['performance']}")
        logger.info(f"Testing Performance: {rsi_eval['performance']}")
    
    # Generate performance comparison visualization
    if args.visualize and results:
        try:
            # Create results directory
            results_dir = os.path.join('reports', 'optimization')
            os.makedirs(results_dir, exist_ok=True)
            
            # Create comparison table
            df = pd.DataFrame(results)
            
            # Save results to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(results_dir, f"strategy_comparison_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
            
            # Create visualization
            plt.figure(figsize=(16, 12))
            
            # Profit comparison chart
            plt.subplot(2, 2, 1)
            plt.title('Profit Comparison: Training vs Testing')
            df_melted = pd.melt(df, 
                              id_vars=['Strategy'], 
                              value_vars=['Training Profit', 'Testing Profit'],
                              var_name='Dataset', 
                              value_name='Profit')
            sns.barplot(data=df_melted, x='Strategy', y='Profit', hue='Dataset')
            
            # Sharpe ratio comparison
            plt.subplot(2, 2, 2)
            plt.title('Sharpe Ratio Comparison: Training vs Testing')
            df_melted = pd.melt(df, 
                              id_vars=['Strategy'], 
                              value_vars=['Training Sharpe', 'Testing Sharpe'],
                              var_name='Dataset', 
                              value_name='Sharpe Ratio')
            sns.barplot(data=df_melted, x='Strategy', y='Sharpe Ratio', hue='Dataset')
            
            # Win rate and profit factor
            plt.subplot(2, 2, 3)
            plt.title('Win Rate and Profit Factor (Testing)')
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            sns.barplot(data=df, x='Strategy', y='Win Rate', ax=ax1, color='skyblue')
            sns.barplot(data=df, x='Strategy', y='Profit Factor', ax=ax2, color='salmon', alpha=0.5)
            ax1.set_ylabel('Win Rate (%)')
            ax2.set_ylabel('Profit Factor')
            
            # Max drawdown comparison
            plt.subplot(2, 2, 4)
            plt.title('Maximum Drawdown (Testing)')
            sns.barplot(data=df, x='Strategy', y='Max Drawdown')
            plt.ylabel('Maximum Drawdown (%)')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(results_dir, f"strategy_comparison_{timestamp}.png")
            plt.savefig(viz_path)
            plt.close()
            logger.info(f"Visualization saved to {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
    
    logger.info("Strategy training and backtesting completed.")

if __name__ == "__main__":
    main()