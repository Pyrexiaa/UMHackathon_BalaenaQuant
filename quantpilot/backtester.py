import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from tabulate import tabulate
from datetime import timedelta
from .metrics import Metrics
import os
import warnings

warnings.filterwarnings("ignore")

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy, initial_capital: float = 100000,
                 risk_free_rate: float = 0.0, trading_fee: float = 0.0006, qty_per_trade: int = 1):
        """
        Initialize the backtester with strategy, data, and configuration.

        :param data: Historical price data (pandas DataFrame with DateTime index)
        :param strategy: Strategy object containing signal generation logic
        :param initial_capital: Starting capital for the portfolio
        :param risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        :param trading_fee: Fee per trade (default 0.06%)
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.results = None
        self.trades = None
        self.qty_per_trade = qty_per_trade
        self.records = None
        self.metrics = None
        
        # if output directory does not exist, create it
        if not os.path.exists("output"):
            os.makedirs("output")

    def run(self,
            forward_test: bool = False,
            forward_years: Optional[float] = None,
            forward_start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run the backtest with the given strategy.

        :param forward_test: Whether to perform forward testing
        :param forward_years: Years of forward testing if enabled
        :return: DataFrame with backtest results
        """
        output = {'results': None, 'metrics': {}}

        if forward_test:
            if forward_years is not None:
                split_date = self.data.index[-1] 
                timedelta(days=forward_years*365)
            elif forward_start_date is not None:
                split_date = pd.to_datetime(forward_start_date)
            else:
                # Default to 1 year forward test if neither specified
                split_date = self.data.index[-1] - pd.Timedelta(days=365)

            # Validate split date is within data range
            if split_date < self.data.index[0]:
                raise ValueError(
                    f"Forward start date {split_date} is before backtest start {self.data.index[0]}")
            if split_date >= self.data.index[-1]:
                raise ValueError(
                    f"Forward start date {split_date} is after data end {self.data.index[-1]}")

            # Split data
            backtest_data = self.data.loc[:split_date - pd.Timedelta(days=1)]
            forward_data = self.data.loc[split_date:]

            # Run backtest phase
            print("Running backtest phase...")
            backtest_results, backtest_trades, backtest_records = self._run_phase(backtest_data)

            # Run forward test phase
            print("\nRunning forward test phase...")
            forward_results, forward_trades, forward_records = self._run_phase(forward_data)

            # Combine results
            self.results = pd.concat([backtest_results, forward_results])
            self.trades = pd.concat([backtest_trades, forward_trades])
            self.records = pd.concat([backtest_records, forward_records])
            
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

        :param data: DataFrame with historical price data
        :return: Tuple of (portfolio_results, trades_dataframe)
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['price'] = data['close']
        portfolio['price_change'] = data['close'].pct_change().fillna(0)
        portfolio['signal'] = self.strategy.generate_signals(data)
        portfolio['position'] = portfolio['signal'].replace(0, pd.NA).ffill().fillna(0)  # forward fill previous position
        # portfolio['trades'] = abs(portfolio['position'].diff().fillna(0))
        portfolio['trades'] = abs(portfolio['position'].diff().fillna(abs(portfolio['position'])))
        
        # Capital tracking variables
        capital = self.initial_capital
        # position_size = 0.5
        cash = capital
        shares_held = 0

        equity_history = []
        cash_history = []
        shares_history = []

        # Execute trades and track capital
        for i in range(len(portfolio)):
            price = portfolio.iloc[i]['price']
            position = portfolio.iloc[i]['position']
            trade = portfolio.iloc[i]['trades']

            if trade != 0:
                if position == 1:
                    # Buy
                    buyable = trade * self.qty_per_trade
                    cash -= buyable * price * (1 + self.trading_fee)
                    shares_held += buyable
                elif position == -1:
                    # Sell
                    sellable = trade * self.qty_per_trade
                    cash += sellable * price * (1 - self.trading_fee)
                    shares_held -= sellable
            
            current_equity = cash + shares_held * price
            equity_history.append(current_equity)
            cash_history.append(cash)
            shares_history.append(shares_held)

        portfolio['cash'] = cash_history
        portfolio['equity'] = equity_history
        portfolio['shares'] = shares_history
        portfolio['drawdown'] = (portfolio['equity'] - portfolio['equity'].cummax()) / portfolio['equity'].cummax()
        portfolio['pnl'] = portfolio['equity'].diff().fillna(0)

        # Create transaction record
        records = portfolio.copy()
        # drop all trade == 0
        records = records[records['trades'] != 0]
        records['trade_signal'] = records['position'].replace(
            {1: 'Buy', -1: 'Sell', 0: 'Hold'})
        
        transaction = pd.DataFrame(columns=[
            'mode', 'entry_position', 'trade_shares', 'entry_time', 'exit_time',
            'entry_price', 'exit_price', 'pnl_per_share', 'pnl', 'remaining_shares', 'equity'
        ])

        open_trades = []

        for i in range(len(records)):
            row = records.iloc[i]
            time = records.index[i]
            price = row['price']
            trades = int(row['trades'])              # total trades in row
            net_shares = row['shares']               # resulting position
            position = int(row['position'])          # 1 for long, -1 for short
            equity = row['equity']                   # current equity

            # Determine direction of each individual trade
            if trades == 0:
                continue

            for _ in range(trades):
                # If no open trade or same direction, this is an entry
                if not open_trades or open_trades[-1]['entry_position'] == position:
                    open_trades.append({
                        'entry_position': position,
                        'entry_time': time,
                        'entry_price': price
                    })
                else:
                    # This is an exit of opposite position
                    entry = open_trades.pop()
                    matched_position = entry['entry_position']
                    pnl = (price - entry['entry_price']) * matched_position 

                    transaction = pd.concat([transaction if not transaction.empty else pd.DataFrame(),
                        pd.DataFrame([{
                        'entry_position': matched_position,
                        'trade_shares': 1 * self.qty_per_trade,
                        'entry_time': entry['entry_time'],
                        'exit_time': time,
                        'entry_price': entry['entry_price'],
                        'exit_price': price,
                        'pnl_per_share': pnl,
                        'pnl': pnl * self.qty_per_trade,
                        'remaining_shares': net_shares,
                        'equity': equity
                    }])], ignore_index=True)

        transaction['mode'] = transaction['entry_position'].apply(
            lambda x: 'Buy→Sell' if x == 1 else 'Sell→Buy')
        # drop the entry_position column
        transaction.drop(columns=['entry_position'], inplace=True)
        transaction = transaction[['mode', 'trade_shares', 'entry_time', 'exit_time',
                                   'entry_price', 'exit_price', 'pnl_per_share', 'pnl', 'remaining_shares', 'equity']]
        
        records['trade_shares'] = records['trades'].apply(
            lambda x: x * self.qty_per_trade if x != 0 else 0)
        records = records[['trade_signal', 'trades', 'trade_shares', 'price', 'shares', 'cash', 'equity']]
     
        return portfolio, transaction, records

    def _calculate_metrics(self, results: pd.DataFrame, trades: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific period."""
        return Metrics(
            equity=results['equity'],
            drawdown=results['drawdown'],
            returns=results['pnl'],
            trades=trades,
            risk_free_rate=self.risk_free_rate,
            trading_fee=self.trading_fee,
            signals=results['signal']
        ).all_metrics()

    def performance_metrics(self, phase: str = 'all') -> Dict:
        """
        Get performance metrics for specific phase.

        :param phase: 'all', 'backtest', 'forward', or 'full'
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

    def plot_results(self, use_streamlit: bool = False):
        """Plot backtest results."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")


        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Equity curve
        ax1.plot(self.results['equity'], label='Portfolio Value', color='blue')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Drawdown
        ax2.fill_between(self.results.index, self.results['drawdown'], color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Portfolio Drawdown')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Signals and positions
        ax3.plot(self.results['price'], label='Price', color='black', alpha=0.5)
        buy_signals = self.results[(self.results['position'] == 1) & (self.results['trades'] > 0)]
        sell_signals = self.results[(self.results['position'] == -1) & (self.results['trades'] > 0)]
        ax3.scatter(buy_signals.index, buy_signals['price'],label='Buy', marker='^', color='green', alpha=1)
        ax3.scatter(sell_signals.index, sell_signals['price'],label='Sell', marker='v', color='red', alpha=1)
        ax3.set_ylabel('Price ($)')
        ax3.set_title('Trading Signals')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()

        plt.tight_layout()

        # Save to file
        output_path = "output/result.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

        if use_streamlit:
            import streamlit as st
            st.pyplot(fig)
        else:
            plt.show()


    def generate_report(self, phase: str = 'all'):
        """
        Generate comprehensive performance report.

        :param phase: 'all', 'backtest', 'forward', or 'full'
        """
        if not self.periods:
            raise ValueError(
                "No backtest results available. Run backtest first.")

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
            print("=" * 60)

            print("\n" + "=" * 60)
            print("COMBINED PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report('full')
            print("=" * 60)
        else:
            # Show single phase
            print("\n" + "=" * 60)
            print(f"{phase.upper()} PHASE PERFORMANCE".center(60))
            print("=" * 60)
            self._print_phase_report(phase)
            print("=" * 60)

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
                # elif 'Drawdown' in name or abs(value) < 1:
                elif 'Drawdown' in name or 'Return' in name:
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
            ["Final Equity", f"${results['equity'].iloc[-1]:,.2f}"]
        ]

        # Performance metrics
        perf_data = [[k, fmt_value(k, v)] for k, v in metrics.items()]
        print(tabulate(info_data + perf_data, tablefmt="plain"))

    def export_data(self, results_path='output/portfolio.csv', trades_path='output/transaction.csv', records_path='output/records.csv'):
        """
        Export the results and trades to CSV files.

        :param results_path: File path to save the backtest results
        :param trades_path: File path to save the trade logs
        :param records_path: File path of backtest and forward test records
        """
        if self.results is not None:
            self.results.to_csv(results_path)
            print(f"Results saved to {results_path}")
        else:
            print("No results to export.")

        if self.trades is not None:
            self.trades.to_csv(trades_path)
            print(f"Trades saved to {trades_path}")
        else:
            print("No trades to export.")
        
        if self.records is not None: 
            self.records.to_csv(records_path)
            print(f"Records saved to {records_path}")
        else:
            print("No records to export.")

            
    def export_metrics(self, filepath="output/metrics.csv"):
        """Export calculated performance metrics to a CSV file."""
        if self.metrics is None:
            raise ValueError("No metrics available. Run backtest first.")

        metrics_df = pd.DataFrame([self.metrics])  # Convert dict to DataFrame
        metrics_df.to_csv(filepath, index=False)
