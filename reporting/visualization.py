# visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
from datetime import datetime, timedelta
import mplfinance as mpf
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class TradeVisualizer:
    """
    Visualize trading data, strategy performance, and metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = self.config.get('visualization_output_dir', 'reports/visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up styling
        self.setup_style()
    
    def setup_style(self) -> None:
        """Set up plot styling."""
        sns.set(style="darkgrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        plt.rcParams['font.size'] = 12
        
        # Custom colors
        self.colors = {
            'primary': '#1f77b4',      # Blue
            'secondary': '#ff7f0e',    # Orange
            'positive': '#2ca02c',     # Green
            'negative': '#d62728',     # Red
            'neutral': '#7f7f7f',      # Gray
            'highlight': '#9467bd'     # Purple
        }
    
    def plot_equity_curve(self, trades: List[Dict[str, Any]], 
                         benchmark_data: Optional[pd.DataFrame] = None,
                         title: str = 'Equity Curve',
                         show: bool = False,
                         save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot equity curve from trade history.
        
        Args:
            trades: List of executed trades
            benchmark_data: Optional benchmark data for comparison
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if not trades:
            self.logger.warning("No trades to plot equity curve")
            return None
        
        try:
            # Sort trades by time
            sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', ''))
            
            # Create dataframe with trade timestamps and cumulative P&L
            dates = []
            equity = []
            
            # Start with initial equity (100%)
            current_equity = 100.0
            
            # Calculate equity curve
            for trade in sorted_trades:
                # Add entry point
                entry_time = pd.to_datetime(trade['entry_time'])
                dates.append(entry_time)
                equity.append(current_equity)
                
                # Add exit point with P&L applied
                exit_time = pd.to_datetime(trade['exit_time'])
                
                if 'profit_loss_pct' in trade:
                    current_equity *= (1 + trade['profit_loss_pct'] / 100)
                elif 'profit_loss' in trade and 'entry_price' in trade and 'quantity' in trade:
                    # Calculate percentage from absolute P&L
                    entry_value = trade['entry_price'] * trade['quantity']
                    pct_change = trade['profit_loss'] / entry_value if entry_value > 0 else 0
                    current_equity *= (1 + pct_change)
                
                dates.append(exit_time)
                equity.append(current_equity)
            
            # Create dataframe
            equity_df = pd.DataFrame({
                'date': dates,
                'equity': equity
            })
            
            # Plot
            fig, ax = plt.subplots()
            
            # Plot equity curve
            ax.plot(equity_df['date'], equity_df['equity'], 
                    color=self.colors['primary'], linewidth=2, label='Strategy')
            
            # Add benchmark if available
            if benchmark_data is not None:
                # Ensure benchmark data has datetime index
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    if 'date' in benchmark_data.columns:
                        benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
                        benchmark_data = benchmark_data.set_index('date')
                    elif 'timestamp' in benchmark_data.columns:
                        benchmark_data['timestamp'] = pd.to_datetime(benchmark_data['timestamp'])
                        benchmark_data = benchmark_data.set_index('timestamp')
                
                # Filter benchmark data to match trading period
                start_date = equity_df['date'].min()
                end_date = equity_df['date'].max()
                
                benchmark_slice = benchmark_data.loc[start_date:end_date]
                
                if not benchmark_slice.empty and 'close' in benchmark_slice.columns:
                    # Normalize benchmark to same starting point
                    benchmark_normalized = benchmark_slice['close'] / benchmark_slice['close'].iloc[0] * 100
                    
                    # Plot benchmark
                    ax.plot(benchmark_normalized.index, benchmark_normalized, 
                            color=self.colors['secondary'], linewidth=1.5, 
                            alpha=0.8, linestyle='--', label='Benchmark')
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Add labels and legend
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity (%)', fontsize=12)
            ax.legend()
            
            # Rotate date labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved equity curve to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {str(e)}")
            return None
    
    def plot_trade_distribution(self, trades: List[Dict[str, Any]],
                               title: str = 'Trade Distribution',
                               show: bool = False,
                               save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot distribution of trade returns.
        
        Args:
            trades: List of executed trades
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if not trades:
            self.logger.warning("No trades to plot distribution")
            return None
        
        try:
            # Extract profit/loss percentages
            if 'profit_loss_pct' in trades[0]:
                returns = [trade['profit_loss_pct'] for trade in trades]
            elif 'profit_loss' in trades[0] and 'entry_price' in trades[0] and 'quantity' in trades[0]:
                returns = []
                for trade in trades:
                    entry_value = trade['entry_price'] * trade['quantity']
                    if entry_value > 0:
                        pct_return = (trade['profit_loss'] / entry_value) * 100
                        returns.append(pct_return)
            else:
                self.logger.warning("Trade data missing profit/loss info for distribution plot")
                return None
            
            # Create figure with multiple plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogram of returns
            sns.histplot(returns, bins=30, kde=True, ax=axes[0, 0], color=self.colors['primary'])
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            axes[0, 0].set_title('Distribution of Returns')
            axes[0, 0].set_xlabel('Return (%)')
            axes[0, 0].set_ylabel('Count')
            
            # Box plot
            sns.boxplot(y=returns, ax=axes[0, 1], color=self.colors['primary'])
            axes[0, 1].axhline(y=0, color='red', linestyle='--')
            axes[0, 1].set_title('Return Statistics')
            axes[0, 1].set_ylabel('Return (%)')
            
            # Trade outcomes (win/loss)
            wins = sum(1 for r in returns if r > 0)
            losses = sum(1 for r in returns if r <= 0)
            
            # Win/Loss pie chart
            axes[1, 0].pie([wins, losses], 
                          labels=['Wins', 'Losses'], 
                          autopct='%1.1f%%',
                          colors=[self.colors['positive'], self.colors['negative']])
            axes[1, 0].set_title('Win/Loss Ratio')
            
            # Cumulative return
            cumulative_returns = [100]
            for r in returns:
                cumulative_returns.append(cumulative_returns[-1] * (1 + r/100))
            
            # Calculate drawdowns
            peak = 100
            drawdowns = []
            for equity in cumulative_returns:
                if equity > peak:
                    peak = equity
                drawdown = (equity / peak - 1) * 100  # Negative value
                drawdowns.append(drawdown)
            
            axes[1, 1].plot(drawdowns, color=self.colors['negative'], linewidth=1.5)
            axes[1, 1].fill_between(range(len(drawdowns)), drawdowns, 0, color=self.colors['negative'], alpha=0.3)
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].set_xlabel('Trade Number')
            
            # Overall title
            plt.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved trade distribution to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting trade distribution: {str(e)}")
            return None
    
    def plot_metrics_over_time(self, metrics_history: List[Dict[str, Any]],
                              metrics_to_plot: List[str],
                              title: str = 'Performance Metrics Over Time',
                              show: bool = False,
                              save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot performance metrics over time.
        
        Args:
            metrics_history: List of metrics snapshots
            metrics_to_plot: List of metric names to plot
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if not metrics_history:
            self.logger.warning("No metrics history to plot")
            return None
        
        try:
            # Create dataframe from metrics history
            data = []
            for entry in metrics_history:
                record = {'timestamp': pd.to_datetime(entry['timestamp'])}
                
                for metric in metrics_to_plot:
                    if metric in entry['metrics']:
                        record[metric] = entry['metrics'][metric]
                
                data.append(record)
            
            metrics_df = pd.DataFrame(data)
            metrics_df.set_index('timestamp', inplace=True)
            
            # Create plot
            n_metrics = len(metrics_to_plot)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics))
            
            # Handle case with only one metric
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns:
                    metric_color = list(self.colors.values())[i % len(self.colors)]
                    
                    axes[i].plot(metrics_df.index, metrics_df[metric], color=metric_color, linewidth=2)
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_ylabel(metric.replace('_', ' ').title())
                    
                    # Format x-axis for dates
                    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    axes[i].xaxis.set_major_locator(mdates.AutoDateLocator())
                    
                    # Add horizontal line at zero for percentage metrics
                    if 'pct' in metric or 'rate' in metric:
                        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Overall title
            plt.suptitle(title, fontsize=16)
            
            # Rotate date labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved metrics plot to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics over time: {str(e)}")
            return None
    
    def plot_price_with_indicators(self, data: pd.DataFrame,
                                  symbol: str,
                                  indicators: Dict[str, List[str]] = None,
                                  trades: List[Dict[str, Any]] = None,
                                  title: Optional[str] = None,
                                  show: bool = False,
                                  save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot price chart with indicators and trade markers.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            symbol: Stock symbol
            indicators: Dictionary mapping panel names to lists of column names to plot
            trades: List of trades to mark on the chart
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if data.empty:
            self.logger.warning("No data to plot price chart")
            return None
        
        try:
            # Ensure data has a datetime index
            df = data.copy()
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
                df.set_index('date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Default title if not provided
            if title is None:
                title = f'{symbol} Price Chart'
            
            # Default indicators if not provided
            if indicators is None:
                indicators = {
                    'main': ['sma_20', 'sma_50', 'sma_200'],
                    'secondary': ['volume'],
                    'lower': ['rsi_14']
                }
            
            # Prepare mpf style
            style = mpf.make_mpf_style(
                base_mpf_style='yahoo',
                gridcolor='#e6e6e6',
                gridstyle=':',
                y_on_right=False
            )
            
            # Prepare additional plots for indicators
            add_plots = []
            
            # Main panel indicators (overlay on price)
            if 'main' in indicators:
                for indicator in indicators['main']:
                    if indicator in df.columns:
                        add_plots.append(
                            mpf.make_addplot(df[indicator], panel=0)
                        )
            
            # Secondary panel indicators
            panel_num = 1
            for panel_name, panel_indicators in indicators.items():
                if panel_name != 'main':
                    for indicator in panel_indicators:
                        if indicator in df.columns:
                            add_plots.append(
                                mpf.make_addplot(df[indicator], panel=panel_num)
                            )
                    panel_num += 1
            
            # Add trade markers if provided
            if trades is not None:
                buy_dates = []
                sell_dates = []
                
                for trade in trades:
                    if trade.get('symbol') == symbol:
                        if 'entry_time' in trade:
                            buy_dates.append(pd.to_datetime(trade['entry_time']))
                        if 'exit_time' in trade:
                            sell_dates.append(pd.to_datetime(trade['exit_time']))
                
                # Create marker arrays
                if buy_dates:
                    buy_markers = pd.Series(1, index=buy_dates)
                    add_plots.append(
                        mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='green', panel=0)
                    )
                
                if sell_dates:
                    sell_markers = pd.Series(1, index=sell_dates)
                    add_plots.append(
                        mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='red', panel=0)
                    )
            
            # Convert to mplfinance format
            df_mpf = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Plot
            fig, axes = mpf.plot(
                df_mpf,
                type='candle',
                style=style,
                addplot=add_plots,
                title=title,
                volume=True,
                figsize=(14, 10),
                panel_ratios=(4, 1),
                returnfig=True
            )
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                fig.savefig(full_path, dpi=300)
                self.logger.info(f"Saved price chart to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting price chart: {str(e)}")
            return None
    
    def plot_performance_comparison(self, performance_metrics: Dict[str, Dict[str, Any]],
                                  metrics_to_compare: List[str],
                                  title: str = 'Strategy Performance Comparison',
                                  show: bool = False,
                                  save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Compare performance metrics across different strategies.
        
        Args:
            performance_metrics: Dictionary mapping strategy names to performance metrics
            metrics_to_compare: List of metric names to compare
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if not performance_metrics:
            self.logger.warning("No performance metrics to compare")
            return None
        
        try:
            # Create dataframe for comparison
            comparison_data = []
            
            for strategy_name, metrics in performance_metrics.items():
                record = {'Strategy': strategy_name}
                
                for metric in metrics_to_compare:
                    if metric in metrics:
                        record[metric] = metrics[metric]
                    else:
                        record[metric] = np.nan
                
                comparison_data.append(record)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.set_index('Strategy', inplace=True)
            
            # Create plot
            n_metrics = len(metrics_to_compare)
            fig, axes = plt.subplots(1, n_metrics, figsize=(15, 6))
            
            # Handle case with only one metric
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric in enumerate(metrics_to_compare):
                if metric in comparison_df.columns:
                    # Sort by the metric
                    sorted_df = comparison_df.sort_values(metric)
                    
                    # Determine color based on whether higher is better
                    if 'drawdown' in metric.lower() or 'loss' in metric.lower() or 'risk' in metric.lower():
                        colors = [self.colors['negative'] if val > 0 else self.colors['positive'] for val in sorted_df[metric]]
                    else:
                        colors = [self.colors['positive'] if val > 0 else self.colors['negative'] for val in sorted_df[metric]]
                    
                    # Create bar chart
                    sorted_df[metric].plot(kind='bar', ax=axes[i], color=colors)
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_ylabel(metric.replace('_', ' ').title())
                    
                    # Add value labels on bars
                    for j, v in enumerate(sorted_df[metric]):
                        axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')
                    
                    # Rotate x-axis labels
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add horizontal line at zero
                    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Overall title
            plt.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved performance comparison to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting performance comparison: {str(e)}")
            return None
    
    def plot_correlation_matrix(self, data: Dict[str, pd.DataFrame],
                               title: str = 'Asset Correlation Matrix',
                               show: bool = False,
                               save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot correlation matrix between assets.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            title: Plot title
            show: Whether to display the plot
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if not data:
            self.logger.warning("No data to plot correlation matrix")
            return None
        
        try:
            # Extract close prices for each symbol
            prices = {}
            common_dates = None
            
            for symbol, df in data.items():
                if 'close' in df.columns:
                    # Ensure datetime index
                    if 'timestamp' in df.columns:
                        df = df.set_index(pd.to_datetime(df['timestamp']))
                    elif 'date' in df.columns:
                        df = df.set_index(pd.to_datetime(df['date']))
                    
                    # Save close prices
                    prices[symbol] = df['close']
                    
                    # Track common date range
                    if common_dates is None:
                        common_dates = set(df.index)
                    else:
                        common_dates = common_dates.intersection(set(df.index))
            
            if not prices:
                self.logger.warning("No close price data for correlation matrix")
                return None
            
            # Convert to dataframe aligned by date
            common_dates = sorted(list(common_dates))
            price_df = pd.DataFrame({symbol: prices[symbol].reindex(common_dates) for symbol in prices})
            
            # Calculate returns
            returns_df = price_df.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Plot correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Draw the heatmap
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
            
            ax.set_title(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved correlation matrix to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {str(e)}")
            return None
    
    def plot_dashboard(self, performance_metrics: Dict[str, Any],
                      trades: List[Dict[str, Any]],
                      equity_curve_data: pd.DataFrame = None,
                      title: str = 'Trading Dashboard',
                      show: bool = False,
                      save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Create a comprehensive trading dashboard.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            trades: List of executed trades
            equity_curve_data: Optional dataframe with equity curve data
            title: Dashboard title
            show: Whether to display the dashboard
            save_path: Optional path to save the dashboard
            
        Returns:
            Figure object or None
        """
        if not performance_metrics or not trades:
            self.logger.warning("Insufficient data for dashboard")
            return None
        
        try:
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            
            # Set up grid
            gs = fig.add_gridspec(3, 3)
            
            # Equity curve (top row, spans two columns)
            ax_equity = fig.add_subplot(gs[0, :2])
            
            # Calculate equity curve
            sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', ''))
            equity = [100]  # Start with 100%
            dates = []
            
            for trade in sorted_trades:
                # Skip if missing required fields
                if 'entry_time' not in trade or 'exit_time' not in trade:
                    continue
                
                # Add entry point
                dates.append(pd.to_datetime(trade['entry_time']))
                equity.append(equity[-1])
                
                # Add exit point with P&L applied
                dates.append(pd.to_datetime(trade['exit_time']))
                
                # Calculate equity change
                if 'profit_loss_pct' in trade:
                    equity.append(equity[-1] * (1 + trade['profit_loss_pct'] / 100))
                elif 'profit_loss' in trade and 'entry_price' in trade and 'quantity' in trade:
                    entry_value = trade['entry_price'] * trade['quantity']
                    pct_change = trade['profit_loss'] / entry_value if entry_value > 0 else 0
                    equity.append(equity[-1] * (1 + pct_change))
                else:
                    equity.append(equity[-1])
            
            # Remove first entry (it's just a placeholder)
            if equity and dates:
                equity = equity[1:]
            
            # Plot equity curve
            ax_equity.plot(dates, equity, color=self.colors['primary'], linewidth=2)
            ax_equity.set_title('Equity Curve')
            ax_equity.set_ylabel('Equity (%)')
            ax_equity.set_xlabel('Date')
            
            # Format x-axis for dates
            ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_equity.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Key metrics (top right)
            ax_metrics = fig.add_subplot(gs[0, 2])
            ax_metrics.axis('off')
            
            # Extract key metrics
            metrics_to_display = [
                ('Total Trades', performance_metrics.get('total_trades', 0)),
                ('Win Rate', f"{performance_metrics.get('win_rate', 0):.2f}%"),
                ('Profit Factor', f"{performance_metrics.get('profit_factor', 0):.2f}"),
                ('Max Drawdown', f"{performance_metrics.get('max_drawdown_pct', 0):.2f}%"),
                ('Total Return', f"{performance_metrics.get('total_return_pct', 0):.2f}%"),
                ('Sharpe Ratio', f"{performance_metrics.get('sharpe_ratio', 0):.2f}"),
                ('Sortino Ratio', f"{performance_metrics.get('sortino_ratio', 0):.2f}"),
                ('Avg. Trade', f"{performance_metrics.get('avg_profit', 0):.2f}")
            ]
            
            # Display metrics in a table
            table_data = [[name, value] for name, value in metrics_to_display]
            table = ax_metrics.table(cellText=table_data, colWidths=[0.6, 0.4], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontproperties=plt.font_manager.FontProperties(weight='bold'))
            
            ax_metrics.set_title('Key Metrics')
            
            # Win/Loss breakdown (middle left)
            ax_winloss = fig.add_subplot(gs[1, 0])
            
            # Calculate wins and losses
            wins = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
            losses = sum(1 for trade in trades if trade.get('profit_loss', 0) <= 0)
            
            # Plot pie chart
            ax_winloss.pie([wins, losses], 
                          labels=['Wins', 'Losses'], 
                          autopct='%1.1f%%',
                          colors=[self.colors['positive'], self.colors['negative']])
            ax_winloss.set_title('Win/Loss Ratio')
            
            # Return distribution (middle center)
            ax_dist = fig.add_subplot(gs[1, 1])
            
            # Extract returns
            if 'profit_loss_pct' in trades[0]:
                returns = [trade['profit_loss_pct'] for trade in trades]
            elif 'profit_loss' in trades[0] and 'entry_price' in trades[0] and 'quantity' in trades[0]:
                returns = []
                for trade in trades:
                    entry_value = trade['entry_price'] * trade['quantity']
                    if entry_value > 0:
                        pct_return = (trade['profit_loss'] / entry_value) * 100
                        returns.append(pct_return)
            else:
                returns = []
            
            # Plot histogram
            sns.histplot(returns, bins=20, kde=True, ax=ax_dist, color=self.colors['primary'])
            ax_dist.axvline(x=0, color='red', linestyle='--')
            ax_dist.set_title('Return Distribution')
            ax_dist.set_xlabel('Return (%)')
            ax_dist.set_ylabel('Count')
            
            # Drawdown (middle right)
            ax_drawdown = fig.add_subplot(gs[1, 2])
            
            # Calculate drawdowns
            peak = 100
            drawdowns = []
            for eq in equity:
                if eq > peak:
                    peak = eq
                drawdown = (eq / peak - 1) * 100  # Negative value
                drawdowns.append(drawdown)
            
            # Plot drawdowns
            ax_drawdown.plot(dates, drawdowns, color=self.colors['negative'], linewidth=1.5)
            ax_drawdown.fill_between(dates, drawdowns, 0, color=self.colors['negative'], alpha=0.3)
            ax_drawdown.set_title('Drawdown')
            ax_drawdown.set_ylabel('Drawdown (%)')
            ax_drawdown.set_xlabel('Date')
            
            # Format x-axis for dates
            ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_drawdown.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Monthly returns (bottom row, spans all columns)
            ax_monthly = fig.add_subplot(gs[2, :])
            
            # Convert trades to monthly returns
            monthly_returns = {}
            
            for trade in trades:
                if 'exit_time' in trade and ('profit_loss_pct' in trade or 'profit_loss' in trade):
                    exit_date = pd.to_datetime(trade['exit_time'])
                    month_key = exit_date.strftime('%Y-%m')
                    
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = []
                    
                    # Add return
                    if 'profit_loss_pct' in trade:
                        monthly_returns[month_key].append(trade['profit_loss_pct'])
                    elif 'profit_loss' in trade and 'entry_price' in trade and 'quantity' in trade:
                        entry_value = trade['entry_price'] * trade['quantity']
                        if entry_value > 0:
                            pct_return = (trade['profit_loss'] / entry_value) * 100
                            monthly_returns[month_key].append(pct_return)
            
            # Aggregate monthly returns
            months = []
            returns_by_month = []
            
            for month in sorted(monthly_returns.keys()):
                months.append(month)
                returns_by_month.append(sum(monthly_returns[month]))
            
            # Plot monthly returns
            colors = [self.colors['positive'] if r > 0 else self.colors['negative'] for r in returns_by_month]
            ax_monthly.bar(months, returns_by_month, color=colors)
            ax_monthly.set_title('Monthly Returns')
            ax_monthly.set_ylabel('Return (%)')
            ax_monthly.set_xlabel('Month')
            
            # Add horizontal line at zero
            ax_monthly.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax_monthly.xaxis.get_majorticklabels(), rotation=45)
            
            # Overall title
            plt.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save if path provided
            if save_path:
                full_path = os.path.join(self.output_dir, save_path)
                plt.savefig(full_path, dpi=300)
                self.logger.info(f"Saved dashboard to {full_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            return None
    
    def generate_html_report(self, performance_metrics: Dict[str, Any],
                           trades: List[Dict[str, Any]],
                           data: Dict[str, pd.DataFrame] = None,
                           output_path: str = 'trading_report.html') -> bool:
        """
        Generate HTML report with all visualizations.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            trades: List of executed trades
            data: Optional dictionary mapping symbols to DataFrames
            output_path: Path to save the HTML report
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        try:
            # Generate all plots
            plots = {}
            
            # 1. Equity curve
            equity_fig = self.plot_equity_curve(trades, title='Equity Curve', show=False)
            if equity_fig:
                buf = io.BytesIO()
                equity_fig.savefig(buf, format='png')
                buf.seek(0)
                plots['equity_curve'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(equity_fig)
            
            # 2. Trade distribution
            dist_fig = self.plot_trade_distribution(trades, title='Trade Distribution', show=False)
            if dist_fig:
                buf = io.BytesIO()
                dist_fig.savefig(buf, format='png')
                buf.seek(0)
                plots['trade_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(dist_fig)
            
            # 3. Dashboard
            dashboard_fig = self.plot_dashboard(performance_metrics, trades, title='Trading Dashboard', show=False)
            if dashboard_fig:
                buf = io.BytesIO()
                dashboard_fig.savefig(buf, format='png')
                buf.seek(0)
                plots['dashboard'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(dashboard_fig)
            
            # 4. Price chart with indicators if data is provided
            if data:
                for symbol, df in data.items():
                    price_fig = self.plot_price_with_indicators(
                        df, symbol, 
                        trades=[t for t in trades if t.get('symbol') == symbol],
                        show=False
                    )
                    if price_fig:
                        buf = io.BytesIO()
                        price_fig.savefig(buf, format='png')
                        buf.seek(0)
                        plots[f'price_chart_{symbol}'] = base64.b64encode(buf.read()).decode('utf-8')
                        plt.close(price_fig)
            
            # Generate HTML
            html = self._generate_html_template(performance_metrics, trades, plots)
            
            # Save HTML report
            full_path = os.path.join(self.output_dir, output_path)
            with open(full_path, 'w') as f:
                f.write(html)
                
            self.logger.info(f"Generated HTML report at {full_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return False
    
    def _generate_html_template(self, metrics: Dict[str, Any], 
                              trades: List[Dict[str, Any]], 
                              plots: Dict[str, str]) -> str:
        """Generate HTML report template."""
        # Format metrics for display
        metrics_html = ''
        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                if 'pct' in key or 'rate' in key:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            metrics_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        # Format recent trades
        trades_html = ''
        # Sort trades by exit time in descending order to show recent trades first
        recent_trades = sorted(trades, key=lambda x: x.get('exit_time', ''), reverse=True)[:10]
        
        for trade in recent_trades:
            # Calculate profit/loss for display
            if 'profit_loss' in trade:
                profit_loss = trade['profit_loss']
                profit_loss_str = f"{profit_loss:.2f}"
                css_class = 'positive' if profit_loss > 0 else 'negative'
            elif 'profit_loss_pct' in trade:
                profit_loss_pct = trade['profit_loss_pct']
                profit_loss_str = f"{profit_loss_pct:.2f}%"
                css_class = 'positive' if profit_loss_pct > 0 else 'negative'
            else:
                profit_loss_str = "N/A"
                css_class = 'neutral'
                
            trades_html += f"""
            <tr>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td>{trade.get('action', 'N/A')}</td>
                <td>{trade.get('entry_time', 'N/A')}</td>
                <td>{trade.get('exit_time', 'N/A')}</td>
                <td>{trade.get('entry_price', 'N/A')}</td>
                <td>{trade.get('exit_price', 'N/A')}</td>
                <td class="{css_class}">{profit_loss_str}</td>
            </tr>
            """
        
        # Generate plots HTML
        plots_html = ''
        for plot_name, plot_data in plots.items():
            plots_html += f"""
            <div class="plot-container">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="data:image/png;base64,{plot_data}" alt="{plot_name}">
            </div>
            """
        
        # Main HTML template
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Strategy Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .neutral {{
                    color: gray;
                }}
                .plot-container {{
                    margin-bottom: 30px;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metrics-table {{
                    width: 48%;
                }}
                @media (max-width: 768px) {{
                    .metrics-table {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trading Strategy Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="metrics-container">
                        <table class="metrics-table">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            {metrics_html}
                        </table>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recent Trades</h2>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>Profit/Loss</th>
                        </tr>
                        {trades_html}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {plots_html}
                </div>
                
                <div class="section">
                    <h2>Summary</h2>
                    <p>This report provides a comprehensive analysis of the trading strategy's performance. 
                       It includes key metrics such as win rate, profit factor, and drawdown, as well as 
                       visualizations of the equity curve and trade distribution.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html