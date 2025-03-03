# risk_management.py
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    """Risk management module for trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.max_positions = int(config.get('max_positions', 10))
        self.max_position_size_pct = float(config.get('max_position_size_pct', 10.0))
        self.max_single_order_size_pct = float(config.get('max_single_order_size_pct', 5.0))
        self.max_daily_drawdown_pct = float(config.get('max_daily_drawdown_pct', 5.0))
        self.max_total_exposure_pct = float(config.get('max_total_exposure_pct', 80.0))
        self.max_single_stock_exposure_pct = float(config.get('max_single_stock_exposure_pct', 20.0))
        self.max_single_sector_exposure_pct = float(config.get('max_single_sector_exposure_pct', 40.0))
        
        # Risk metrics tracking
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.equity_value = float(config.get('initial_balance', 1000000.0))
        self.start_of_day_equity = self.equity_value
        self.sector_exposure = {}
        
        # Order tracking
        self.orders_today = []
        self.trading_locked = False
        self.trading_lock_reason = None
        
        # Initialize risk metrics
        self._reset_daily_metrics()
    
    def evaluate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate trading signals against risk management rules.
        
        Args:
            signals: List of trading signals
            
        Returns:
            List of approved signals
        """
        if self.trading_locked:
            self.logger.warning(f"Trading locked: {self.trading_lock_reason}")
            return []
        
        approved_signals = []
        
        for signal in signals:
            # Skip signals without required information
            if not all(key in signal for key in ['symbol', 'action', 'price', 'quantity']):
                continue
            
            # Apply risk management rules
            approved = True
            reject_reason = None
            
            # Process based on action type
            if signal['action'] == 'BUY':
                # Check against maximum number of positions
                if len(self.current_positions) >= self.max_positions and signal['symbol'] not in self.current_positions:
                    approved = False
                    reject_reason = f"Max positions limit reached: {self.max_positions}"
                
                # Check position size constraint
                order_value = signal['price'] * signal['quantity']
                position_size_pct = (order_value / self.equity_value) * 100
                
                if position_size_pct > self.max_single_order_size_pct:
                    approved = False
                    reject_reason = f"Order size exceeds limit: {position_size_pct:.2f}% > {self.max_single_order_size_pct}%"
                
                # Check current exposure
                if not self._check_exposure_limits(signal):
                    approved = False
                    reject_reason = "Exposure limit would be exceeded"
                
                # Check drawdown limit
                if not self._check_drawdown_limit():
                    approved = False
                    reject_reason = f"Daily drawdown limit reached: {self.max_daily_drawdown_pct}%"
            
            # Apply additional risk rules if needed
            additional_check = self._apply_additional_risk_rules(signal)
            if not additional_check['approved']:
                approved = False
                reject_reason = additional_check['reason']
            
            # Process approved signal
            if approved:
                approved_signals.append(signal)
                
                # Adjust position sizing if needed
                adjusted_signal = self._adjust_position_sizing(signal)
                if adjusted_signal != signal:
                    self.logger.info(f"Position size adjusted for {signal['symbol']}")
                    approved_signals[-1] = adjusted_signal
                
                # Add stop loss and take profit if not present
                if 'stop_loss' not in signal or 'take_profit' not in signal:
                    approved_signals[-1] = self._add_risk_parameters(adjusted_signal)
            else:
                self.logger.warning(f"Signal rejected for {signal['symbol']}: {reject_reason}")
        
        return approved_signals
    
    def update_positions(self, positions: Dict[str, Any]) -> None:
        """
        Update current positions information.
        
        Args:
            positions: Dictionary of current positions
        """
        self.current_positions = positions
        
        # Update sector exposure
        self._update_sector_exposure()
        
        # Update equity value
        positions_value = sum(pos.get('market_value', 0) for pos in positions.values())
        cash_balance = self.equity_value - positions_value
        self.equity_value = cash_balance + positions_value
        
        # Update daily P&L
        self.daily_pnl = self.equity_value - self.start_of_day_equity
        daily_pnl_pct = (self.daily_pnl / self.start_of_day_equity) * 100
        
        # Check for drawdown limit
        if daily_pnl_pct < -self.max_daily_drawdown_pct and not self.trading_locked:
            self.trading_locked = True
            self.trading_lock_reason = f"Daily drawdown limit exceeded: {daily_pnl_pct:.2f}%"
            self.logger.warning(self.trading_lock_reason)
    
    def update_order(self, order: Dict[str, Any]) -> None:
        """
        Update order tracking.
        
        Args:
            order: Order details
        """
        self.orders_today.append(order)
    
    def reset_daily(self) -> None:
        """Reset daily risk metrics at start of trading day."""
        self._reset_daily_metrics()
        self.logger.info("Daily risk metrics reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        total_exposure = sum(pos.get('market_value', 0) for pos in self.current_positions.values())
        total_exposure_pct = (total_exposure / self.equity_value) * 100 if self.equity_value > 0 else 0
        
        return {
            'equity_value': self.equity_value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.start_of_day_equity) * 100 if self.start_of_day_equity > 0 else 0,
            'total_positions': len(self.current_positions),
            'total_exposure': total_exposure,
            'total_exposure_pct': total_exposure_pct,
            'sector_exposure': {sector: (value / self.equity_value) * 100 for sector, value in self.sector_exposure.items()},
            'trading_locked': self.trading_locked,
            'trading_lock_reason': self.trading_lock_reason,
            'total_orders_today': len(self.orders_today)
        }
    
    def _reset_daily_metrics(self) -> None:
        """Reset daily risk metrics."""
        self.start_of_day_equity = self.equity_value
        self.daily_pnl = 0.0
        self.orders_today = []
        self.trading_locked = False
        self.trading_lock_reason = None
    
    def _check_exposure_limits(self, signal: Dict[str, Any]) -> bool:
        """
        Check if a signal would exceed exposure limits.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if within limits, False otherwise
        """
        symbol = signal['symbol']
        order_value = signal['price'] * signal['quantity']
        
        # Calculate current total exposure
        current_exposure = sum(pos.get('market_value', 0) for pos in self.current_positions.values())
        current_exposure_pct = (current_exposure / self.equity_value) * 100
        
        # Calculate new exposure after order
        if symbol in self.current_positions:
            # Adding to existing position
            current_pos_value = self.current_positions[symbol].get('market_value', 0)
            new_position_value = current_pos_value + order_value
            new_total_exposure = current_exposure + order_value
        else:
            # New position
            new_position_value = order_value
            new_total_exposure = current_exposure + order_value
        
        new_exposure_pct = (new_total_exposure / self.equity_value) * 100
        new_position_pct = (new_position_value / self.equity_value) * 100
        
        # Check against limits
        if new_exposure_pct > self.max_total_exposure_pct:
            self.logger.warning(f"Total exposure limit would be exceeded: {new_exposure_pct:.2f}% > {self.max_total_exposure_pct}%")
            return False
        
        if new_position_pct > self.max_single_stock_exposure_pct:
            self.logger.warning(f"Single stock exposure limit would be exceeded: {new_position_pct:.2f}% > {self.max_single_stock_exposure_pct}%")
            return False
        
        # Check sector exposure
        sector = self._get_stock_sector(symbol)
        if sector:
            current_sector_exposure = self.sector_exposure.get(sector, 0)
            new_sector_exposure = current_sector_exposure + order_value
            new_sector_pct = (new_sector_exposure / self.equity_value) * 100
            
            if new_sector_pct > self.max_single_sector_exposure_pct:
                self.logger.warning(f"Sector exposure limit would be exceeded: {new_sector_pct:.2f}% > {self.max_single_sector_exposure_pct}%")
                return False
        
        return True
    
    def _check_drawdown_limit(self) -> bool:
        """
        Check if daily drawdown limit has been reached.
        
        Returns:
            True if within limit, False otherwise
        """
        if self.start_of_day_equity == 0:
            return True
        
        daily_pnl_pct = (self.daily_pnl / self.start_of_day_equity) * 100
        
        if daily_pnl_pct < -self.max_daily_drawdown_pct:
            self.logger.warning(f"Daily drawdown limit reached: {daily_pnl_pct:.2f}% < -{self.max_daily_drawdown_pct}%")
            return False
        
        return True
    
    def _update_sector_exposure(self) -> None:
        """Update sector exposure based on current positions."""
        self.sector_exposure = {}
        
        for symbol, position in self.current_positions.items():
            sector = self._get_stock_sector(symbol)
            if sector:
                market_value = position.get('market_value', 0)
                self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + market_value
    
    def _get_stock_sector(self, symbol: str) -> Optional[str]:
        """
        Get sector for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name or None if not found
        """
        # In a real implementation, this would use a database or API
        # For demonstration, use a simple mapping
        sector_map = {
            'RELIANCE': 'Energy',
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'KOTAKBANK': 'Banking',
            'TCS': 'IT',
            'INFY': 'IT',
            'WIPRO': 'IT',
            'HCLTECH': 'IT',
            'ITC': 'FMCG',
            'HINDUNILVR': 'FMCG',
            'BHARTIARTL': 'Telecom',
            'LT': 'Construction'
        }
        
        return sector_map.get(symbol)
    
    def _adjust_position_sizing(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust position sizing based on risk parameters.
        
        Args:
            signal: Trading signal
            
        Returns:
            Adjusted signal
        """
        adjusted_signal = signal.copy()
        
        # Check if we need to adjust size
        order_value = signal['price'] * signal['quantity']
        position_size_pct = (order_value / self.equity_value) * 100
        
        if position_size_pct > self.max_position_size_pct:
            # Adjust quantity to meet max position size
            max_order_value = (self.max_position_size_pct / 100) * self.equity_value
            adjusted_quantity = int(max_order_value / signal['price'])
            
            adjusted_signal['quantity'] = adjusted_quantity
            adjusted_signal['original_quantity'] = signal['quantity']
            adjusted_signal['adjustment_reason'] = f"Position size reduced to {self.max_position_size_pct}% max"
            
            self.logger.info(f"Position size adjusted for {signal['symbol']}: {signal['quantity']} -> {adjusted_quantity}")
        
        return adjusted_signal
    
    def _add_risk_parameters(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add stop loss and take profit parameters if not present.
        
        Args:
            signal: Trading signal
            
        Returns:
            Signal with risk parameters
        """
        updated_signal = signal.copy()
        
        # Get default parameters from config
        default_stop_loss_pct = float(self.config.get('default_stop_loss_pct', 3.0))
        default_take_profit_pct = float(self.config.get('default_take_profit_pct', 6.0))
        
        # Add stop loss if not present
        if 'stop_loss' not in updated_signal:
            if signal['action'] == 'BUY':
                stop_loss = signal['price'] * (1 - default_stop_loss_pct / 100)
            else:  # SELL (for short positions)
                stop_loss = signal['price'] * (1 + default_stop_loss_pct / 100)
                
            updated_signal['stop_loss'] = stop_loss
            updated_signal['stop_loss_pct'] = default_stop_loss_pct
        
        # Add take profit if not present
        if 'take_profit' not in updated_signal:
            if signal['action'] == 'BUY':
                take_profit = signal['price'] * (1 + default_take_profit_pct / 100)
            else:  # SELL (for short positions)
                take_profit = signal['price'] * (1 - default_take_profit_pct / 100)
                
            updated_signal['take_profit'] = take_profit
            updated_signal['take_profit_pct'] = default_take_profit_pct
        
        return updated_signal
    
    def _apply_additional_risk_rules(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        """
        Apply additional custom risk rules.
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary with approval status and reason
        """
        # Default to approved
        result = {
            'approved': True,
            'reason': None
        }
        
        # Example: Limit number of trades per day
        max_daily_trades = int(self.config.get('max_daily_trades', 20))
        if len(self.orders_today) >= max_daily_trades:
            result['approved'] = False
            result['reason'] = f"Maximum daily trades limit reached: {max_daily_trades}"
            return result
        
        # Example: Time-based restrictions
        if not self._check_trading_hours():
            result['approved'] = False
            result['reason'] = "Outside allowed trading hours"
            return result
        
        # Example: Volatility-based restrictions
        if 'volatility' in signal and signal['volatility'] > float(self.config.get('max_volatility', 5.0)):
            result['approved'] = False
            result['reason'] = f"Volatility too high: {signal['volatility']:.2f}%"
            return result
        
        return result
    
    def _check_trading_hours(self) -> bool:
        """
        Check if current time is within allowed trading hours.
        
        Returns:
            True if within allowed hours, False otherwise
        """
        # Get current time
        now = datetime.now()
        
        # Check if it's a weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Define trading hours (9:15 AM to 3:30 PM, typical NSE hours)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if current time is within trading hours
        return market_open <= now <= market_close