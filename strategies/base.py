# strategies/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
from execution.risk_management import RiskManager

class Strategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            
        Returns:
            List of signal dictionaries with keys:
            - symbol: Stock symbol
            - action: 'BUY', 'SELL', or 'HOLD'
            - price: Suggested entry/exit price
            - quantity: Suggested quantity
            - timestamp: Signal timestamp
            - reason: Reasoning behind the signal
        """
        pass
    
    def backtest(self, data: Dict[str, pd.DataFrame], 
                risk_manager: Optional[RiskManager] = None) -> List[Dict[str, Any]]:
        """
        Run a backtest of the strategy on historical data.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            risk_manager: Optional risk manager to evaluate signals
            
        Returns:
            List of trades executed in the backtest
        """
        trades = []
        positions = {}  # Currently open positions
        
        # Process each symbol's data sequentially
        for symbol, df in data.items():
            # Skip if not enough data
            if len(df) < 50:  # Arbitrary threshold
                self.logger.warning(f"Not enough data for {symbol}, skipping in backtest")
                continue
                
            # Create a copy to avoid modifying original data
            df_copy = df.copy()
            
            # Process each bar
            for i in range(50, len(df_copy)):
                # Get data up to current bar
                current_data = {symbol: df_copy.iloc[:i+1]}
                
                # Generate signals based on current data
                signals = self.generate_signals(current_data)
                
                # Apply risk management if provided
                if risk_manager:
                    signals = risk_manager.evaluate_signals(signals)
                    
                # Process signals
                for signal in signals:
                    if signal['symbol'] != symbol:
                        continue  # Skip signals for other symbols
                        
                    current_bar = df_copy.iloc[i]
                    
                    # Process buy signals
                    if signal['action'] == 'BUY':
                        # Check if we already have a position
                        if symbol in positions:
                            continue  # Skip if already have a position
                        
                        # Calculate execution price with slippage
                        exec_price = self._apply_slippage(signal['price'], 'BUY')
                        
                        # Record the trade
                        trade = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'entry_time': current_bar['timestamp'],
                            'entry_price': exec_price,
                            'quantity': signal['quantity'],
                            'reason': signal['reason']
                        }
                        
                        # Update positions
                        positions[symbol] = trade
                        
                    # Process sell signals
                    elif signal['action'] == 'SELL':
                        # Check if we have a position to sell
                        if symbol not in positions:
                            continue  # Skip if no position
                        
                        # Calculate execution price with slippage
                        exec_price = self._apply_slippage(signal['price'], 'SELL')
                        
                        # Complete the trade record
                        trade = positions[symbol].copy()
                        trade['exit_time'] = current_bar['timestamp']
                        trade['exit_price'] = exec_price
                        trade['exit_reason'] = signal['reason']
                        
                        # Calculate P&L
                        trade['profit_loss'] = (
                            (exec_price - trade['entry_price']) * trade['quantity']
                            if trade['action'] == 'BUY'
                            else (trade['entry_price'] - exec_price) * trade['quantity']
                        )
                        trade['profit_loss_pct'] = (
                            (exec_price / trade['entry_price'] - 1) * 100
                            if trade['action'] == 'BUY'
                            else (trade['entry_price'] / exec_price - 1) * 100
                        )
                        
                        # Add to trades list
                        trades.append(trade)
                        
                        # Remove from positions
                        del positions[symbol]
            
            # Close any open positions at the end of the backtest
            if symbol in positions:
                last_bar = df_copy.iloc[-1]
                
                # Complete the trade record
                trade = positions[symbol].copy()
                trade['exit_time'] = last_bar['timestamp']
                trade['exit_price'] = last_bar['close']
                trade['exit_reason'] = 'End of backtest'
                
                # Calculate P&L
                trade['profit_loss'] = (
                    (last_bar['close'] - trade['entry_price']) * trade['quantity']
                    if trade['action'] == 'BUY'
                    else (trade['entry_price'] - last_bar['close']) * trade['quantity']
                )
                trade['profit_loss_pct'] = (
                    (last_bar['close'] / trade['entry_price'] - 1) * 100
                    if trade['action'] == 'BUY'
                    else (trade['entry_price'] / last_bar['close'] - 1) * 100
                )
                
                # Add to trades list
                trades.append(trade)
                
                # Remove from positions
                del positions[symbol]
                
        return trades
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply realistic slippage to the execution price."""
        # Default slippage: 0.1% adverse for market orders
        slippage_pct = self.params.get('slippage_pct', 0.1)
        
        if action == 'BUY':
            # Higher price for buys
            return price * (1 + slippage_pct / 100)
        else:
            # Lower price for sells
            return price * (1 - slippage_pct / 100)
    
    def optimize(self, data: Dict[str, pd.DataFrame], 
                param_grid: Dict[str, List[Any]], 
                metric: str = 'profit_loss') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            param_grid: Dictionary mapping parameter names to lists of values
            metric: Metric to optimize (e.g., 'profit_loss', 'sharpe_ratio')
            
        Returns:
            Dictionary with best parameters and performance
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_performance = float('-inf')
        best_params = None
        best_trades = None
        
        # Test each parameter combination
        for combination in param_combinations:
            # Update strategy parameters
            params = {name: value for name, value in zip(param_names, combination)}
            self.params.update(params)
            
            # Run backtest with current parameters
            trades = self.backtest(data)
            
            # Calculate performance metric
            performance = self._calculate_performance_metric(trades, metric)
            
            # Update best parameters if better performance
            if performance > best_performance:
                best_performance = performance
                best_params = params.copy()
                best_trades = trades
                
        # Restore best parameters
        if best_params:
            self.params.update(best_params)
            
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'trades': best_trades
        }
    
    def _calculate_performance_metric(self, trades: List[Dict[str, Any]], 
                                    metric: str) -> float:
        """Calculate the specified performance metric from trades."""
        if not trades:
            return float('-inf')
            
        if metric == 'profit_loss':
            return sum(trade['profit_loss'] for trade in trades)
            
        elif metric == 'profit_loss_pct':
            return sum(trade['profit_loss_pct'] for trade in trades)
            
        elif metric == 'win_rate':
            wins = sum(1 for trade in trades if trade['profit_loss'] > 0)
            return wins / len(trades) * 100
            
        elif metric == 'sharpe_ratio':
            returns = [trade['profit_loss_pct'] for trade in trades]
            if not returns or np.std(returns) == 0:
                return float('-inf')
            return (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
            
        else:
            raise ValueError(f"Unknown performance metric: {metric}")