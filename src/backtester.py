from .portfolio import Portfolio
from .metrics import Metrics
from tabulate import tabulate  # If needed later
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


class Backtester:
    def __init__(self, data, strategy, initial_capital: float = 100000):
        """
        Initialize the backtester with strategy, data, and configuration.

        :param strategy: Strategy object containing signal generation logic.
        :param data: Historical price data (pandas DataFrame with DateTime index).
        :param initial_capital: Starting capital for the portfolio.
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None
        self.trade_log = []

        # Ensure the data index is a DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")

    def run(self, forward_test: bool = False, forward_years: float = 1.0) -> pd.DataFrame:
        """
        Run the backtest with the given strategy.
        
        :param forward_test: Whether to perform forward testing
        :param forward_years: Years of forward testing if enabled
        :return: DataFrame with backtest results
        """
        # Split data for forward testing if requested
        if forward_test:
            split_date = self.data.index[-1] - timedelta(days=forward_years*365)
            backtest_data = self.data.loc[:split_date]
            forward_data = self.data.loc[split_date:]
            
            # Run backtest phase
            print("Running backtest phase...")
            backtest_results = self._run_phase(backtest_data)
            
            # Run forward test phase
            print("\nRunning forward test phase...")
            forward_results = self._run_phase(forward_data)
            
            # Combine results
            self.results = pd.concat([backtest_results, forward_results])
            return self.results
        else:
            self.results = self._run_phase(self.data)
            return self.results
    
    def _run_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a single phase (backtest or forward test) of the strategy.
        :param data: DataFrame with historical price data.
        """
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['cash'] = float(self.initial_capital) 
        portfolio['position'] = 0.0
        portfolio['total'] = float(self.initial_capital)
        portfolio['signal'] = self.strategy.generate_signals(data)
        
        # Execute trades based on signals 
        for i in range(1, len(portfolio)):
            prev_position = portfolio['position'].iloc[i-1]
            signal = portfolio['signal'].iloc[i]
            price = portfolio['price'].iloc[i]
            
            if signal == 2:  # Buy
                target_position = (portfolio['cash'].iloc[i-1] * 0.98) / price
            elif signal == 0:  # Sell
                target_position = 0
            else:  # Hold
                target_position = prev_position
            
            # Execute trade with precise 0.06% fee
            position_change = target_position - prev_position
            if position_change != 0:
                trade_cost = abs(position_change * price) * 0.0006  # 0.06% fee
                
                # Update portfolio
                portfolio.at[portfolio.index[i], 'position'] = target_position
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - (position_change * price) - trade_cost
                
                # Log trade details
                self.trade_log.append({
                    'date': portfolio.index[i],
                    'signal': signal,
                    'price': price,
                    'shares': target_position,
                    'trade_value': position_change * price,
                    'fee': trade_cost,
                })
            else:
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Update total portfolio value
            portfolio.at[portfolio.index[i], 'total'] = portfolio['cash'].iloc[i] + (target_position * price)
        
        # Calculate returns and metrics
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
        portfolio['drawdown'] = self._calculate_drawdown(portfolio['total'])
        
        return portfolio
        
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown from equity curve."""
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def performance_metrics(self, phase: str = 'all') -> Dict:
        """Calculate key performance metrics for specified phase."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
            
        if phase == 'backtest':
            results = self.results.iloc[:int(len(self.results)*0.7)]  # First 70% as backtest
        elif phase == 'forward':
            results = self.results.iloc[int(len(self.results)*0.7):]  # Last 30% as forward test
        else:
            results = self.results
            
        returns = results['returns']
        total_return = results['total'].iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252/len(results)) - 1
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = results['drawdown'].min()
        win_rate = (returns > 0).mean()
        
        # Trade analysis
        trade_df = pd.DataFrame(self.trade_log)
        if not trade_df.empty:
            avg_trade_return = trade_df['trade_value'].sum() / trade_df['trade_value'].abs().sum()
            profit_factor = trade_df[trade_df['trade_value'] > 0]['trade_value'].sum() / \
                          abs(trade_df[trade_df['trade_value'] < 0]['trade_value'].sum())
            trades_per_year = len(trade_df) / (len(results) / 252)
        else:
            avg_trade_return = 0
            profit_factor = 0
            trades_per_year = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'trades_per_year': trades_per_year,
            'total_fees': trade_df['fee'].sum()
        }
    

    
    def calculate_trade_frequency(signals):
        trades = np.sum((signals == Config.BUY_SIGNAL) | (signals == Config.SELL_SIGNAL))
        total = len(signals)
        frequency = (trades / total) * 100 if total > 0 else 0
        return frequency

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def plot_results(self):
        """Plot backtest results with enhanced visualization."""
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


    def generate_report(self):
        """Generate comprehensive performance report."""
        metrics_all = self.performance_metrics('all')
        metrics_backtest = self.performance_metrics('backtest')
        metrics_forward = self.performance_metrics('forward')
        
        print("="*50)
        print("BACKTESTING REPORT")
        print("="*50)
        print("\nOVERALL PERFORMANCE:")
        self._print_metrics(metrics_all)
        
        print("\nBACKTEST PHASE PERFORMANCE:")
        self._print_metrics(metrics_backtest)
        
        print("\nFORWARD TEST PHASE PERFORMANCE:")
        self._print_metrics(metrics_forward)
        
        print("\nTRADE ANALYSIS:")
        print(f"Total Trades: {len(self.trade_log)}")
        print(f"Average Trades/Year: {metrics_all['trades_per_year']:.1f}")
        print(f"Total Fees Paid: ${metrics_all['total_fees']:,.2f}")
    
    def _print_metrics(self, metrics: Dict):
        """Helper function to print metrics."""
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Avg Trade Return: {metrics['avg_trade_return']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")