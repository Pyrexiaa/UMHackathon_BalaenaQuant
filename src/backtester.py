from .portfolio import Portfolio
from .metrics import Metrics
from tabulate import tabulate
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class Backtester:
    def __init__(self, data: pd.DataFrame, strategy, initial_capital: float = 100000,
                 risk_free_rate: float = 0.0, trading_fee: float = 0.0006):
        """
        Initialize the backtester with strategy, data, and configuration.

        :params data: Historical price data (pandas DataFrame with DateTime index)
        :params strategy: Strategy object containing signal generation logic
        :params initial_capital: Starting capital for the portfolio
        :params risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        :params trading_fee: Fee per trade (default 0.06%)
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.results = None
        self.trades = None
        self.metrics = None

    def run(self, forward_test: bool = False, forward_years: float = 1.0) -> pd.DataFrame:
        """
        Run the backtest with the given strategy.
        
        :params forward_test: Whether to perform forward testing
        :params forward_years: Years of forward testing if enabled
        :return: DataFrame with backtest results
        """
        output = {'results': None, 'metrics': {}}
                  
        if forward_test:
            split_date = self.data.index[-1] - timedelta(days=forward_years*365)
            backtest_data = self.data.loc[:split_date]
            forward_data = self.data.loc[split_date:]
            
            # Run backtest phase
            print("Running backtest phase...")
            backtest_results, backtest_trades = self._run_phase(backtest_data)
            
            # Run forward test phase
            print("\nRunning forward test phase...")
            forward_results, forward_trades = self._run_phase(forward_data)
            
            # Combine results
            self.results = pd.concat([backtest_results, forward_results])
            self.trades = pd.concat([backtest_trades, forward_trades])
            
            self.periods = {
                'backtest': {
                    'start': backtest_results.index[0],
                    'end': backtest_results.index[-1],
                    'results': backtest_results,
                    'trades': backtest_trades
                },
                'forward': {
                    'start': forward_results.index[0],
                    'end': forward_results.index[-1],
                    'results': forward_results,
                    'trades': forward_trades
                }
            }
            
            output['metrics'] = {
                'backtest': self._calculate_metrics(backtest_results, backtest_trades),
                'forward': self._calculate_metrics(forward_results, forward_trades),
                'full': self._calculate_metrics(self.results, self.trades)
            }
            
        else:
            self.results, self.trades = self._run_phase(self.data)
            self.periods = {
                'full': {
                    'start': self.results.index[0],
                    'end': self.results.index[-1],
                    'results': self.results,
                    'trades': self.trades
                }
            }
            output['metrics'] = {
                'full': self._calculate_metrics(self.results, self.trades)
            }
        
        output['results'] = self.results
        self.generate_report()
        return output
    
    def _run_phase(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run a single phase (backtest or forward test) of the strategy.
        
        :params data: DataFrame with historical price data
        :return: Tuple of (portfolio_results, trades_dataframe)
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['cash'] = float(self.initial_capital)
        portfolio['position'] = 0.0
        portfolio['total'] = float(self.initial_capital)
        portfolio['signal'] = self.strategy.generate_signals(data)
        
        trades = []
        
        for i in range(1, len(portfolio)):
            prev_position = portfolio['position'].iloc[i-1]
            signal = portfolio['signal'].iloc[i]
            price = portfolio['price'].iloc[i]
            
            # Determine target position
            if signal == 2:  # Buy
                target_position = (portfolio['cash'].iloc[i-1] * 0.98) / price
            elif signal == 0:  # Sell
                target_position = 0
            else:  # Hold
                target_position = prev_position
            
            # Execute trade
            position_change = target_position - prev_position
            if position_change != 0:
                trade_cost = abs(position_change * price) * self.trading_fee
                
                # Update portfolio
                portfolio.at[portfolio.index[i], 'position'] = target_position
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - (position_change * price) - trade_cost
                
                # Log trade
                trades.append({
                    'entry_time': portfolio.index[i],
                    'exit_time': portfolio.index[i],  # Single bar trade for now
                    'entry_price': price,
                    'exit_price': price,
                    'shares': target_position,
                    'pnl': -trade_cost,  # Just the fee for now
                    'signal': signal,
                    'fee': trade_cost
                })
            else:
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Update total portfolio value
            portfolio.at[portfolio.index[i], 'total'] = portfolio['cash'].iloc[i] + (target_position * price)
        
        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
        portfolio['drawdown'] = self._calculate_drawdown(portfolio['total'])
            
        return portfolio, pd.DataFrame(trades)
        
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown from equity curve."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _calculate_metrics(self, results: pd.DataFrame, trades: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific period."""
        return Metrics(
            equity=results['total'],
            returns=results['returns'],
            trades=trades,
            risk_free_rate=self.risk_free_rate,
            trading_fee=self.trading_fee,
            signals=results['signal']
        ).all_metrics()
    
    def performance_metrics(self, phase: str = 'all') -> Dict:
        """
        Get performance metrics for specific phase.
        
        :params phase: 'all', 'backtest', 'forward', or 'full'
        :return: Dictionary of performance metrics
        """
        if not self.periods:
            raise ValueError("No backtest results available. Run backtest first.")
            
        if phase == 'backtest' and 'backtest' in self.periods:
            return self._calculate_metrics(
                self.periods['backtest']['results'],
                self.periods['backtest']['trades']
            )
        elif phase == 'forward' and 'forward' in self.periods:
            return self._calculate_metrics(
                self.periods['forward']['results'],
                self.periods['forward']['trades']
            )
        else:
            return self._calculate_metrics(self.results, self.trades)
    
    def plot_results(self):
        """Plot backtest results."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Equity curve
        ax1.plot(self.results['total'], label='Portfolio Value', color='blue')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Drawdown
        ax2.fill_between(self.results.index, self.results['drawdown'], 
                        color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Portfolio Drawdown')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Signals and positions
        ax3.plot(self.results['price'], label='Price', color='black', alpha=0.5)
        buy_signals = self.results[self.results['signal'] == 2]
        sell_signals = self.results[self.results['signal'] == 0]
        ax3.scatter(buy_signals.index, buy_signals['price'], 
                   label='Buy', marker='^', color='green', alpha=1)
        ax3.scatter(sell_signals.index, sell_signals['price'], 
                   label='Sell', marker='v', color='red', alpha=1)
        ax3.set_ylabel('Price ($)')
        ax3.set_title('Trading Signals')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def generate_report(self, phase: str = 'all'):
        """
        Generate comprehensive performance report.
        
        :params phase: 'all', 'backtest', 'forward', or 'full'
        """
        if not self.periods:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Get the appropriate metrics
        if phase == 'all' and 'backtest' in self.periods:
            # Show both backtest and forward if available
            print("\n" + "=" * 60)
            print("BACKTEST PHASE PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report('backtest')
            
            print("\n" + "=" * 60)
            print("FORWARD TEST PHASE PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report('forward')
            
            print("\n" + "=" * 60)
            print("COMBINED PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report('full')
        else:
            # Show single phase
            print("\n" + "=" * 60)
            print(f"{phase.upper()} PHASE PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report(phase)
    
    def _print_phase_report(self, phase: str):
        """Print report for a specific phase."""
        if phase not in self.periods and phase != 'full':
            raise ValueError(f"Phase '{phase}' not available")
    
        if phase == 'full':
            results = self.results
            trades = self.trades
            start_date = self.results.index[0]
            end_date = self.results.index[-1]
        else:
            results = self.periods[phase]['results']
            trades = self.periods[phase]['trades']
            start_date = self.periods[phase]['start']
            end_date = self.periods[phase]['end']
        
        metrics = self._calculate_metrics(results, trades)
        
        def fmt_value(name, value):
            if isinstance(value, float):
                if name.endswith(('Ratio', 'Rate', 'Factor')):
                    return f"{value:.2f}"
                elif 'Drawdown' in name or abs(value) < 1:
                    return f"{value:.2%}"
                else:
                    return f"{value:,.2f}"
            return str(value)
        
        # Basic info
        info_data = [
            ["Start Date", start_date.date()],
            ["End Date", end_date.date()],
            ["Duration (days)", len(results)],
            ["Initial Capital", f"${self.initial_capital:,.2f}"],
            ["Final Equity", f"${results['total'].iloc[-1]:,.2f}"]
        ]
        
        # Performance metrics
        perf_data = [[k, fmt_value(k, v)] for k, v in metrics.items()]
        print(tabulate(info_data + perf_data, tablefmt="plain"))