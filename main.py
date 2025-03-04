#!/usr/bin/env python
# main.py
import argparse
import sys
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from config.settings import load_config
from core.agent import Agent
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIStrategy
from execution.paper_trading import PaperTradingBroker
from execution.zerodha import ZerodhaInterface
from execution.upstox import UpstoxBroker
from execution.fivepaisa import FivePaisaBroker
from reporting.metrics import PerformanceMetrics
from reporting.visualization import TradeVisualizer
from utils.logger import configure_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BTSM Trading System')
    
    # Mode selection
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default='paper',
                       help='Trading mode: backtest, paper, or live')
    
    # Configuration
    parser.add_argument('--config', default='config/config.ini',
                       help='Path to configuration file')
    
    # Strategy selection
    parser.add_argument('--strategy', choices=['moving_average', 'rsi', 'all'], default='all',
                       help='Trading strategy to use')
    
    # Date range for backtesting
    parser.add_argument('--start-date', 
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', 
                       help='End date for backtest (YYYY-MM-DD)')
    
    # Symbols to trade
    parser.add_argument('--symbols', 
                       help='Comma-separated list of symbols to trade')
    
    # Broker selection for live trading
    parser.add_argument('--broker', choices=['zerodha', 'upstox', 'fivepaisa'], default='paper',
                       help='Broker to use for live trading')
    
    # Reporting options
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report after execution')
    parser.add_argument('--report-file', default='trading_report.html',
                       help='Path to save the performance report')
    
    return parser.parse_args()

def setup_agent(config_path: str) -> Agent:
    """Setup and configure the trading agent."""
    # Load configuration
    config = load_config(config_path)
    
    # Configure logging
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', 'logs/trading.log')
    configure_logging(log_level, log_file)
    
    # Create agent
    agent = Agent(config_path)
    
    return agent

def add_strategies(agent: Agent, strategy_names: List[str], config: Dict[str, Any]) -> None:
    """Add selected strategies to the agent."""
    available_strategies = {
        'moving_average': MovingAverageCrossover,
        'rsi': RSIStrategy
    }
    
    # Extract strategy-specific parameters
    strategy_params = {}
    for key, value in config.items():
        if '.' in key:
            prefix, param = key.split('.', 1)
            if prefix in available_strategies:
                if prefix not in strategy_params:
                    strategy_params[prefix] = {}
                strategy_params[prefix][param] = value
    
    # Add selected strategies
    for name in strategy_names:
        if name in available_strategies:
            strategy_class = available_strategies[name]
            params = strategy_params.get(name, {})
            agent.add_strategy(name, strategy_class(params))

def setup_broker(agent: Agent, broker_name: str, config: Dict[str, Any]) -> None:
    """Setup and configure the broker interface."""
    if broker_name == 'paper':
        broker = PaperTradingBroker(config)
    elif broker_name == 'zerodha':
        broker = ZerodhaInterface(config)
    elif broker_name == 'upstox':
        broker = UpstoxBroker(config)
    elif broker_name == 'fivepaisa':
        broker = FivePaisaBroker(config)
    else:
        raise ValueError(f"Unsupported broker: {broker_name}")
    
    # Connect to broker
    broker.connect()
    
    # Set broker on agent
    agent.set_broker(broker)

def generate_report(metrics: PerformanceMetrics, 
                   data: Dict[str, Any], 
                   config: Dict[str, Any],
                   output_file: str) -> None:
    """Generate performance report."""
    visualizer = TradeVisualizer(config)
    
    # Get metrics and trades
    performance = metrics.get_last_metrics()
    trades = data.get('trades', [])
    
    # Generate HTML report
    visualizer.generate_html_report(
        performance_metrics=performance,
        trades=trades,
        data=data.get('market_data'),
        output_path=output_file
    )
    
    # Generate additional visualizations
    visualizer.plot_equity_curve(
        trades=trades,
        show=False,
        save_path='equity_curve.png'
    )
    
    visualizer.plot_trade_distribution(
        trades=trades,
        show=False,
        save_path='trade_distribution.png'
    )
    
    visualizer.plot_dashboard(
        performance_metrics=performance,
        trades=trades,
        show=False,
        save_path='dashboard.png'
    )
    
    logging.info(f"Performance report generated: {output_file}")

def run_backtest(agent: Agent, symbols: List[str], 
                start_date: str, end_date: str, 
                generate_report_flag: bool = False, 
                report_file: str = 'backtest_report.html') -> None:
    """Run backtest with the configured agent and strategies."""
    logging.info(f"Starting backtest from {start_date} to {end_date} for {', '.join(symbols)}")
    
    # Run backtest
    results = agent.run_backtest(symbols, start_date, end_date)
    
    # Print summary
    for strategy_id, strategy_results in results.items():
        performance = strategy_results['performance']
        trades = strategy_results['trades']
        
        print(f"\nStrategy: {strategy_id}")
        print(f"Total trades: {performance.get('total_trades', 0)}")
        print(f"Win rate: {performance.get('win_rate', 0):.2f}%")
        print(f"Profit factor: {performance.get('profit_factor', 0):.2f}")
        print(f"Total profit: {performance.get('total_profit', 0):.2f}")
        print(f"Max drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
        
        # Generate report if requested
        if generate_report_flag:
            generate_report(
                metrics=agent.metrics,
                data={
                    'trades': trades,
                    'market_data': agent.data_collector.cache
                },
                config=agent.config,
                output_file=f"{strategy_id}_{report_file}"
            )

def run_paper_trading(agent: Agent, symbols: List[str], timeframe: str = 'day') -> None:
    """Run paper trading with the configured agent and strategies."""
    logging.info(f"Starting paper trading for {', '.join(symbols)}")
    
    # Set up signal handler for graceful shutdown
    import signal
    
    def signal_handler(sig, frame):
        logging.info("Shutting down paper trading...")
        agent.stop_trading()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start trading
    agent.start_live_trading(symbols, timeframe)

def run_live_trading(agent: Agent, symbols: List[str], timeframe: str = 'day') -> None:
    """Run live trading with the configured agent and strategies."""
    logging.info(f"Starting live trading for {', '.join(symbols)}")
    
    # Additional risk warning for live trading
    print("\n*** WARNING: Starting LIVE TRADING with real money ***")
    confirmation = input("Type 'CONFIRM' to proceed: ")
    
    if confirmation != "CONFIRM":
        print("Live trading cancelled.")
        return
    
    # Set up signal handler for graceful shutdown
    import signal
    
    def signal_handler(sig, frame):
        logging.info("Shutting down live trading...")
        agent.stop_trading()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start trading
    agent.start_live_trading(symbols, timeframe)

def main():
    """Main function to run the trading system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup agent with configuration
    agent = setup_agent(args.config)
    config = agent.config
    
    # Determine strategies to use
    if args.strategy == 'all':
        strategy_names = ['moving_average', 'rsi']
    else:
        strategy_names = [args.strategy]
    
    # Add strategies to agent
    add_strategies(agent, strategy_names, config)
    
    # Determine symbols to trade
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        # Get symbols from config
        if 'equity_symbols' in config:
            symbols = config['equity_symbols'].split(',')
        else:
            symbols = ['RELIANCE', 'TCS', 'INFY']  # Default symbols
    
    # Setup broker
    broker_name = args.broker if args.mode == 'live' else 'paper'
    setup_broker(agent, broker_name, config)
    
    # Run in selected mode
    if args.mode == 'backtest':
        # Determine date range
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = args.start_date or (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Run backtest
        run_backtest(agent, symbols, start_date, end_date, args.report, args.report_file)
        
    elif args.mode == 'paper':
        # Run paper trading
        run_paper_trading(agent, symbols)
        
    elif args.mode == 'live':
        # Run live trading
        run_live_trading(agent, symbols)
    
    # Cleanup
    if agent.broker:
        agent.broker.disconnect()

if __name__ == "__main__":
    main()