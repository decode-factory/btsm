# metrics.py
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os

class PerformanceMetrics:
    """Calculate and track trading strategy performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.benchmark_symbol = self.config.get('performance_benchmark', 'NIFTY50')
        self.risk_free_rate = float(self.config.get('risk_free_rate', 3.5)) / 100  # Annual risk-free rate
        
        # Storage for metrics
        self.metrics_history = []
        self.last_metrics = {}
        
        # File storage
        self.metrics_file = self.config.get('performance_metrics_file', 'reports/performance.json')
        
        # Load existing metrics if available
        self._load_metrics()
    
    def calculate(self, trades: List[Dict[str, Any]], 
                positions: Optional[Dict[str, Any]] = None,
                benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from trades.
        
        Args:
            trades: List of completed trades
            positions: Optional dictionary of current open positions
            benchmark_data: Optional DataFrame with benchmark data
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            self.logger.warning("No trades to calculate metrics")
            return {}
        
        try:
            # Prepare trades data
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
            losing_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) <= 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Profit metrics
            total_profit = sum(trade.get('profit_loss', 0) for trade in trades)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Calculate average winner and loser
            winning_profits = [trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0]
            losing_profits = [trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) <= 0]
            
            avg_winner = sum(winning_profits) / len(winning_profits) if winning_profits else 0
            avg_loser = sum(losing_profits) / len(losing_profits) if losing_profits else 0
            
            # Calculate profit factor
            gross_profit = sum(winning_profits)
            gross_loss = abs(sum(losing_profits))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate trade duration
            durations = []
            for trade in trades:
                if 'entry_time' in trade and 'exit_time' in trade:
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    duration = (exit_time - entry_time).total_seconds() / 3600  # in hours
                    durations.append(duration)
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Return metrics
            if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
                trades_df['return_pct'] = (trades_df['exit_price'] / trades_df['entry_price'] - 1) * 100
                avg_return_pct = trades_df['return_pct'].mean()
            else:
                avg_return_pct = 0
            
            # Advanced metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'avg_winner': avg_winner,
                'avg_loser': avg_loser,
                'profit_factor': profit_factor,
                'avg_duration_hours': avg_duration,
                'avg_return_pct': avg_return_pct
            }
            
            # Calculate time-based returns if dates are available
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                returns_metrics = self._calculate_returns(trades_df, benchmark_data)
                metrics.update(returns_metrics)
            
            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown(trades)
            metrics.update(drawdown_metrics)
            
            # Calculate risk-adjusted metrics
            risk_metrics = self._calculate_risk_metrics(trades_df)
            metrics.update(risk_metrics)
            
            # Add current positions data if available
            if positions:
                position_metrics = self._calculate_position_metrics(positions)
                metrics.update(position_metrics)
            
            # Store metrics
            self.last_metrics = metrics
            self.metrics_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics
            })
            
            # Save metrics
            self._save_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get historical metrics.
        
        Returns:
            List of metrics snapshots
        """
        return self.metrics_history
    
    def get_last_metrics(self) -> Dict[str, Any]:
        """
        Get most recent metrics.
        
        Returns:
            Dictionary with latest metrics
        """
        return self.last_metrics
    
    def compare_to_benchmark(self, strategy_returns: pd.Series, 
                            benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Compare strategy performance to benchmark.
        
        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Align series by date
            aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
            aligned_returns.columns = ['strategy', 'benchmark']
            aligned_returns = aligned_returns.dropna()
            
            if aligned_returns.empty:
                return {}
            
            # Calculate alpha and beta
            cov_matrix = aligned_returns.cov()
            beta = cov_matrix.loc['strategy', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark']
            
            # Calculate alpha (annualized)
            trading_days = 252
            strategy_mean = aligned_returns['strategy'].mean() * trading_days
            benchmark_mean = aligned_returns['benchmark'].mean() * trading_days
            alpha = strategy_mean - self.risk_free_rate - beta * (benchmark_mean - self.risk_free_rate)
            
            # Calculate information ratio
            excess_returns = aligned_returns['strategy'] - aligned_returns['benchmark']
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
            
            # Calculate tracking error
            tracking_error = excess_returns.std() * np.sqrt(trading_days)
            
            # Calculate up/down capture
            up_months = aligned_returns[aligned_returns['benchmark'] > 0]
            down_months = aligned_returns[aligned_returns['benchmark'] < 0]
            
            if not up_months.empty:
                up_capture = (up_months['strategy'].mean() / up_months['benchmark'].mean()) * 100
            else:
                up_capture = 0
                
            if not down_months.empty:
                down_capture = (down_months['strategy'].mean() / down_months['benchmark'].mean()) * 100
            else:
                down_capture = 0
            
            # Correlation
            correlation = aligned_returns['strategy'].corr(aligned_returns['benchmark'])
            
            return {
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'correlation': correlation,
                'up_capture': up_capture,
                'down_capture': down_capture
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {str(e)}")
            return {}
    
    def _calculate_returns(self, trades_df: pd.DataFrame, 
                          benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate time-based returns metrics.
        
        Args:
            trades_df: DataFrame of trades
            benchmark_data: Optional DataFrame with benchmark data
            
        Returns:
            Dictionary with returns metrics
        """
        # Convert timestamps to datetime if they're strings
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Sort by entry time
        trades_df = trades_df.sort_values('entry_time')
        
        # Get start and end dates
        start_date = trades_df['entry_time'].min()
        end_date = trades_df['exit_time'].max()
        
        # Calculate duration in days
        duration_days = (end_date - start_date).days
        
        if duration_days <= 0:
            return {}
        
        # Calculate total return
        if 'profit_loss' in trades_df.columns and 'entry_price' in trades_df.columns and 'quantity' in trades_df.columns:
            initial_capital = (trades_df['entry_price'] * trades_df['quantity']).sum()
            total_profit = trades_df['profit_loss'].sum()
            total_return = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
        elif 'return_pct' in trades_df.columns:
            # If we have percentage returns, compound them
            total_return = (1 + trades_df['return_pct'] / 100).prod() - 1
            total_return *= 100  # Convert to percentage
        else:
            total_return = 0
        
        # Calculate annualized return
        annual_return = ((1 + total_return / 100) ** (365 / duration_days) - 1) * 100
        
        # Prepare daily returns if possible
        returns_metrics = {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'duration_days': duration_days
        }
        
        # Compare to benchmark if available
        if benchmark_data is not None and not benchmark_data.empty:
            # Make sure benchmark data has datetime index
            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                if 'date' in benchmark_data.columns:
                    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
                    benchmark_data = benchmark_data.set_index('date')
                elif 'timestamp' in benchmark_data.columns:
                    benchmark_data['timestamp'] = pd.to_datetime(benchmark_data['timestamp'])
                    benchmark_data = benchmark_data.set_index('timestamp')
            
            # Filter benchmark data to match trading period
            benchmark_slice = benchmark_data.loc[start_date:end_date]
            
            if not benchmark_slice.empty and 'close' in benchmark_slice.columns:
                # Calculate benchmark return
                benchmark_start = benchmark_slice['close'].iloc[0]
                benchmark_end = benchmark_slice['close'].iloc[-1]
                benchmark_return = ((benchmark_end / benchmark_start) - 1) * 100
                
                # Calculate excess return
                excess_return = total_return - benchmark_return
                
                returns_metrics.update({
                    'benchmark_return_pct': benchmark_return,
                    'excess_return_pct': excess_return
                })
        
        return returns_metrics
    
    def _calculate_drawdown(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Args:
            trades: List of trades
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', ''))
        
        # Create cumulative equity curve
        equity = 100  # Start with base of 100
        peak = equity
        drawdowns = []
        underwater_periods = []
        current_underwater = 0
        
        equity_curve = [equity]
        drawdown_curve = [0]
        
        for trade in sorted_trades:
            # Update equity with trade P&L
            if 'profit_loss_pct' in trade:
                # If percentage P&L is available, use it
                equity *= (1 + trade.get('profit_loss_pct', 0) / 100)
            elif 'profit_loss' in trade and 'entry_price' in trade and 'quantity' in trade:
                # Otherwise calculate from absolute P&L and entry cost
                entry_cost = trade.get('entry_price', 0) * trade.get('quantity', 0)
                if entry_cost > 0:
                    equity *= (1 + trade.get('profit_loss', 0) / entry_cost)
            
            equity_curve.append(equity)
            
            # Update peak and calculate drawdown
            if equity > peak:
                peak = equity
                current_underwater = 0
            else:
                drawdown = (equity / peak - 1) * 100  # Negative value
                drawdowns.append(drawdown)
                drawdown_curve.append(drawdown)
                
                # Track underwater periods
                current_underwater += 1
                underwater_periods.append(current_underwater)
        
        # Calculate max drawdown
        max_drawdown = min(drawdowns) if drawdowns else 0
        
        # Calculate average drawdown
        avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0
        
        # Calculate max underwater period
        max_underwater = max(underwater_periods) if underwater_periods else 0
        
        return {
            'max_drawdown_pct': max_drawdown,
            'avg_drawdown_pct': avg_drawdown,
            'max_underwater_period': max_underwater
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            trades_df: DataFrame of trades
            
        Returns:
            Dictionary with risk metrics
        """
        risk_metrics = {}
        
        # Calculate returns if available
        if 'return_pct' in trades_df.columns:
            returns = trades_df['return_pct'] / 100  # Convert percentage to decimal
            
            # Sharpe Ratio (annualized)
            trading_days = 252
            if len(returns) > 1 and returns.std() > 0:
                sharpe = (returns.mean() - self.risk_free_rate / trading_days) / returns.std() * np.sqrt(trading_days)
                risk_metrics['sharpe_ratio'] = sharpe
            
            # Sortino Ratio (using only negative returns for denominator)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino = (returns.mean() - self.risk_free_rate / trading_days) / negative_returns.std() * np.sqrt(trading_days)
                risk_metrics['sortino_ratio'] = sortino
            
            # Calmar Ratio (annualized return / max drawdown)
            drawdown_metrics = self._calculate_drawdown(trades_df.to_dict('records'))
            max_drawdown = drawdown_metrics.get('max_drawdown_pct', 0)
            
            if max_drawdown < 0:  # Ensure max_drawdown is negative
                # Calculate annualized return
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    start_date = trades_df['entry_time'].min()
                    end_date = trades_df['exit_time'].max()
                    duration_days = (end_date - start_date).days
                    
                    if duration_days > 0:
                        total_return = (1 + returns).prod() - 1
                        annual_return = ((1 + total_return) ** (365 / duration_days) - 1)
                        
                        # Calmar Ratio
                        calmar = -annual_return / (max_drawdown / 100)  # Convert max_drawdown back to decimal
                        risk_metrics['calmar_ratio'] = calmar
        
        return risk_metrics
    
    def _calculate_position_metrics(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics for current open positions.
        
        Args:
            positions: Dictionary of open positions
            
        Returns:
            Dictionary with position metrics
        """
        if not positions:
            return {'open_positions': 0}
        
        # Count positions
        num_positions = len(positions)
        
        # Calculate total position value and cost basis
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        total_cost = sum(pos.get('cost_basis', 0) for pos in positions.values())
        
        # Calculate unrealized P&L
        unrealized_pnl = total_value - total_cost
        unrealized_pnl_pct = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        # Calculate per-position metrics
        winning_positions = sum(1 for pos in positions.values() if pos.get('pnl', 0) > 0)
        losing_positions = sum(1 for pos in positions.values() if pos.get('pnl', 0) <= 0)
        
        return {
            'open_positions': num_positions,
            'total_position_value': total_value,
            'total_position_cost': total_cost,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'winning_positions': winning_positions,
            'losing_positions': losing_positions
        }
    
    def _load_metrics(self) -> None:
        """Load metrics history from file if available."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    
                    if 'metrics_history' in data:
                        self.metrics_history = data['metrics_history']
                        
                        if self.metrics_history:
                            self.last_metrics = self.metrics_history[-1]['metrics']
                
                self.logger.info(f"Loaded {len(self.metrics_history)} metrics records from {self.metrics_file}")
                
        except Exception as e:
            self.logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self) -> None:
        """Save metrics history to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            
            data = {
                'metrics_history': self.metrics_history,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved metrics to {self.metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")