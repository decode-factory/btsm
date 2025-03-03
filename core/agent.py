import logging
from typing import Dict, List, Any
from config.settings import load_config
from data.collector import DataCollector
from data.processor import DataProcessor
from strategies.base import Strategy
from execution.broker_interface import BrokerInterface
from execution.risk_management import RiskManager
from reporting.metrics import PerformanceMetrics

class Agent:
    """Main agent class that orchestrates all trading activities."""
    
    def __init__(self, config_path: str = None):
        """Initialize the trading agent with configuration."""
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_collector = DataCollector(self.config)
        self.data_processor = DataProcessor()
        self.strategies: Dict[str, Strategy] = {}
        self.broker: BrokerInterface = None
        self.risk_manager = RiskManager(self.config)
        self.metrics = PerformanceMetrics()
        
        self.is_paper_trading = self.config.get('paper_trading', True)
        self.running = False
        
    def add_strategy(self, strategy_id: str, strategy: Strategy) -> None:
        """Add a trading strategy to the agent."""
        self.strategies[strategy_id] = strategy
        self.logger.info(f"Strategy '{strategy_id}' added")
        
    def set_broker(self, broker: BrokerInterface) -> None:
        """Set the broker interface for order execution."""
        self.broker = broker
        self.logger.info(f"Broker set to {broker.name}")
        
    def fetch_market_data(self, symbols: List[str], 
                          timeframe: str = 'day', 
                          start_date: str = None, 
                          end_date: str = None) -> Dict[str, Any]:
        """Fetch market data for the given symbols and timeframe."""
        raw_data = self.data_collector.fetch_data(
            symbols, timeframe, start_date, end_date
        )
        processed_data = self.data_processor.process(raw_data)
        return processed_data
        
    def run_backtest(self, symbols: List[str], 
                     start_date: str, 
                     end_date: str, 
                     timeframe: str = 'day') -> Dict[str, Any]:
        """Run backtest for all registered strategies."""
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Fetch historical data
        data = self.fetch_market_data(symbols, timeframe, start_date, end_date)
        
        results = {}
        for strategy_id, strategy in self.strategies.items():
            # Run strategy backtest
            strategy_results = strategy.backtest(data, self.risk_manager)
            
            # Calculate performance metrics
            performance = self.metrics.calculate(strategy_results)
            results[strategy_id] = {
                'trades': strategy_results,
                'performance': performance
            }
            
        return results
        
    def start_live_trading(self, symbols: List[str], timeframe: str = 'day') -> None:
        """Start live trading with all registered strategies."""
        if not self.broker:
            raise ValueError("Broker not set. Use set_broker() before starting trading.")
            
        self.running = True
        self.logger.info(f"Starting {'paper' if self.is_paper_trading else 'live'} trading")
        
        while self.running:
            try:
                # Fetch latest market data
                data = self.fetch_market_data(symbols, timeframe)
                
                for strategy_id, strategy in self.strategies.items():
                    # Generate signals
                    signals = strategy.generate_signals(data)
                    
                    # Apply risk management
                    approved_signals = self.risk_manager.evaluate_signals(signals)
                    
                    # Execute trades
                    for signal in approved_signals:
                        self.broker.execute_order(signal)
                    
                # Sleep until next iteration based on timeframe
                self._wait_for_next_iteration(timeframe)
                    
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                
    def stop_trading(self) -> None:
        """Stop the trading loop."""
        self.running = False
        self.logger.info("Trading stopped")
        
    def _wait_for_next_iteration(self, timeframe: str) -> None:
        """Wait until the next trading iteration based on timeframe."""
        # Implementation varies based on timeframe (seconds, minutes, day)
        # This would use time.sleep() with appropriate duration
        pass