import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from tabulate import tabulate
from datetime import datetime
from .metrics import Metrics
import warnings
from .strategy import BaseStrategy

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, strategy_name: str, initial_capital: float = 100000,
                 risk_free_rate: float = 0.0, trading_fee: float = 0.0006, qty_per_trade: int = 1,
                 mode: str = 'arithmetic', entry_exit_logic: str = 'trend_following', threshold: float = None):
        """
        Initialize the backtester with strategy, data, and configuration.

        :param data: Historical price data (pandas DataFrame with DateTime index)
        :param strategy: Strategy object containing signal generation logic
        :param initial_capital: Starting capital for the portfolio
        :param risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        :param trading_fee: Fee per trade (default 0.06%)
        :param qty_per_trade: Number of shares per trade
        :param mode: Mode of calculation ('arithmetic' or 'geometric')
        :param entry_exit_logic: Entry and exit logic ('trend_following' or 'mean_reversion')
        :param threshold: Optional threshold for signal generation
        """
        self.data = data
        self.strategy = strategy 
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.results = None
        self.trades = None
        self.qty_per_trade = qty_per_trade
        self.records = None
        self.metrics = None
        self.mode = mode
        self.entry_exit_logic = entry_exit_logic
        self.threshold = threshold
        
        # create output directory if not exits
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)

    def run(self,
            backtest_start_date: Optional[str] = None,
            backtest_end_date: Optional[str] = None,
            backtest_years: Optional[float] = None,
            forward_test: bool = False,
            forward_start_date: Optional[str] = None,
            forward_end_date: Optional[str] = None,
            forward_years: Optional[float] = None) -> pd.DataFrame:
        """
        Run the backtest (and optional forward test).

        :param backtest_start_date: Start of backtest period
        :param backtest_end_date: End of backtest period
        :param backtest_years: Duration of backtest (used with start or from beginning)
        :param forward_test: Enable forward testing
        :param forward_start_date: Start of forward test
        :param forward_end_date: End of forward test
        :param forward_years: Duration of forward test (from split date or start)
        :return: Dictionary with results and metrics
        """
        output = {'results': None, 'metrics': {}}
        
        print("STRATEGY NAME:", self.strategy_name)

        def slice_range(data, start=None, end=None, years=None):
            if start:
                start = pd.to_datetime(start)
            if end:
                end = pd.to_datetime(end)
                # default to last hour if no specific time given
                if end.time() == datetime.min.time():
                    same_day_mask = data.index.normalize() == end.normalize()
                    if same_day_mask.any():
                        day_end = data.index[same_day_mask].max()
                        end = day_end

            if start and end:
                return data.loc[start:end]
            elif start and years:
                return data.loc[start: start + pd.Timedelta(days=int(years * 365))]
            elif years:
                start = data.index[0]
                return data.loc[start: start + pd.Timedelta(days=int(years * 365))]
            elif start:
                return data.loc[start:]
            elif end:
                return data.loc[:end]
            else:
                return data.copy()
        
        if forward_test:
            # Determine forward test start date
            if forward_start_date:
                forward_start = pd.to_datetime(forward_start_date)
            elif forward_years:
                forward_start = self.data.index[-1] - pd.Timedelta(days=int(forward_years * 365))
            else:
                forward_start = self.data.index[-1] - pd.Timedelta(days=365)  # default 1 year

            if forward_start < self.data.index[0] or forward_start >= self.data.index[-1]:
                raise ValueError(f"Forward start date {forward_start} is outside data range.")

            backtest_data = self.data.loc[:forward_start - pd.Timedelta(hours=1)]
            forward_data = slice_range(self.data, forward_start, forward_end_date, forward_years)

            print("Running backtest phase...")
            backtest_results, backtest_trades, backtest_records = self._run_phase(backtest_data)

            print("\nRunning forward test phase...")
            forward_results, forward_trades, forward_records = self._run_phase(forward_data)

            self.results = pd.concat([backtest_results, forward_results])
            self.trades = pd.concat([backtest_trades, forward_trades])
            self.records = pd.concat([backtest_records, forward_records])    
            self.periods = {
                'backtest': {
                    'start': backtest_results.index[0],
                    'end': backtest_results.index[-1],
                    'results': backtest_results,
                    'trades': backtest_trades,
                    'records': backtest_records                    
                },
                'forward': {
                    'start': forward_results.index[0],
                    'end': forward_results.index[-1],
                    'results': forward_results,
                    'trades': forward_trades,
                    'records': forward_records
                }
            }

            output['metrics'] = {
                'backtest': self._calculate_metrics(backtest_results, backtest_trades, backtest_records),
                'forward': self._calculate_metrics(forward_results, forward_trades, forward_records),
                'full': self._calculate_metrics(self.results, self.trades, self.records)
            }
            

        else:
            # Backtest only
            backtest_data = slice_range(self.data, backtest_start_date, backtest_end_date, backtest_years)
            print("Running backtest...")
            self.results, self.trades, self.records = self._run_phase(backtest_data)
            
            self.periods = {
                'full': {
                    'start': self.results.index[0],
                    'end': self.results.index[-1],
                    'results': self.results,
                    'trades': self.trades,
                    'records': self.records
                }
            }

            output['metrics'] = {
                'full': self._calculate_metrics(self.results, self.trades, self.records)
            }
            
        output['results'] = self.results
        self.generate_report()
        self.export_data(
            results_path=f'output/portfolio_{self.strategy_name.lower()}.csv',
            trades_path=f'output/trade_{self.strategy_name.lower()}.csv',
            records_path=f'output/records_{self.strategy_name.lower()}.csv'
        )
        
        periods_metadata = {
            key: {
                'start': str(val['start']),
                'end': str(val['end'])
            } for key, val in self.periods.items()
        }

        # Add metrics to metadata before saving
        periods_metadata["metrics"] = output["metrics"]


        with open(f'output/meta_{self.strategy_name.lower()}.json', 'w') as f:
            json.dump(self.convert_json_friendly(periods_metadata), f, indent=4)

        
        return output



    def convert_json_friendly(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_json_friendly(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_json_friendly(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return obj.total_seconds() 
        elif isinstance(obj, pd._libs.tslibs.nattype.NaTType):
            return None
        
        return obj


    def _run_phase(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run a single phase (backtest or forward test) of the strategy.

        :param data: DataFrame with historical price data
        :param threshold: Optional threshold for signal generation
        :return: Tuple of (portfolio_results, trades_dataframe, records_dataframe)
        """
        if self.mode not in ['arithmetic', 'geometric']:
            raise ValueError("Invalid mode. Choose 'arithmetic' or 'geometric'.")
        if self.entry_exit_logic not in ['trend_following', 'mean_reversion']:
            raise ValueError("Invalid entry/exit logic. Choose 'trend_following' or 'mean_reversion'.")
        else:
            portfolio = pd.DataFrame(index=data.index)
            portfolio['price'] = data['close']
            portfolio['price_change'] = data['close'].pct_change().fillna(0)
            if self.threshold is None:
                portfolio['signal'] = self.strategy.generate_signals(data)
            else:
                portfolio['signal'] = self.strategy.generate_signals(data, threshold=self.threshold)
            if self.entry_exit_logic == 'trend_following':
                print("Entry/Exit Logic: Trend Following")
                portfolio['position'] = portfolio['signal'].replace(0, pd.NA).ffill().fillna(0)  # forward fill previous position
                portfolio['trades'] = abs(portfolio['position'].diff().fillna(abs(portfolio['position'])))
            elif self.entry_exit_logic == 'mean_reversion':
                print("Entry/Exit Logic: Mean Reversion")
                portfolio['position'] = portfolio['signal'].copy()
                portfolio['trades'] = abs(portfolio['position'].diff().fillna(abs(portfolio['position']))) 
                    
            if self.mode == 'geometric':
                print("Mode: Geometric")
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
                        elif position == 0:
                            # Close position
                            cash += shares_held * price * (1 - self.trading_fee)
                            shares_held = 0
                    
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
                    {1: 'Buy', -1: 'Sell', 0: 'Close'})
                
                transaction = pd.DataFrame(columns=[
                    'mode', 'entry_position', 'trades', 'trade_shares', 'entry_time', 'exit_time',
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
                                'trades': 1,
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

                # transaction['mode'] = transaction['entry_position'].apply(
                #     lambda x: 'Buy→Sell' if x == 1 else 'Sell→Buy')
                transaction['mode'] = transaction['entry_position'].apply(
                    lambda x: 'Buy→Sell' if x == 1 else 'Sell→Buy' if x == -1 else 'Close')
                # drop the entry_position column
                transaction.drop(columns=['entry_position'], inplace=True)
                transaction = transaction[['mode', 'trades', 'trade_shares', 'entry_time', 'exit_time',
                                        'entry_price', 'exit_price', 'pnl_per_share', 'pnl', 'remaining_shares', 'equity']]
                
                records['trade_shares'] = records['trades'].apply(
                    lambda x: x * self.qty_per_trade if x != 0 else 0)
                records = records[['trade_signal', 'trades', 'trade_shares', 'price', 'shares', 'cash', 'equity']]
            
                return portfolio, transaction, records
            
            elif self.mode == 'arithmetic':
                print("Mode: Arithmetic")
                # PNL = price_pct_change * previous_position - current_trades * fees
                portfolio['pnl'] = (portfolio['price_change'] * (portfolio['position'].shift(1).fillna(0)) 
                                    - portfolio['trades'] * self.trading_fee)      
                # Equity = sum(previous all pnl: current pnl)
                portfolio['equity'] = portfolio['pnl'].cumsum() 
                # Drawdown = current equity - max(previous equity)
                portfolio['drawdown'] = (portfolio['equity'] - portfolio['equity'].cummax()) 

                # Create transaction record
                records = portfolio.copy()
                # drop all trade == 0
                records = records[records['trades'] != 0]
                records['trade_signal'] = records['position'].replace(
                    {1: 'Buy', -1: 'Sell', 0: 'Close'})
                
                transaction = pd.DataFrame(columns=[
                    'mode', 'entry_position', 'trades', 'entry_time', 'exit_time',
                    'entry_price', 'exit_price', 'pnl', 'remaining_shares', 'equity'
                ])

                open_trades = []

                for i in range(len(records)):
                    row = records.iloc[i]
                    time = records.index[i]
                    price = row['price']
                    trades = int(row['trades'])               # total trades in row
                    position = int(row['position'])          # 1 for long, -1 for short
                    equity = row['equity']                    # current equity

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
                            pnl = (price - entry['entry_price']) / entry['entry_price'] * matched_position # use pct change

                            transaction = pd.concat([transaction if not transaction.empty else pd.DataFrame(),
                                pd.DataFrame([{
                                'entry_position': matched_position,
                                'trades': 1,
                                'entry_time': entry['entry_time'],
                                'exit_time': time,
                                'entry_price': entry['entry_price'],
                                'exit_price': price,
                                'pnl': pnl,
                                'remaining_shares': position,
                                'equity': equity
                            }])], ignore_index=True)

                # transaction['mode'] = transaction['entry_position'].apply(
                #     lambda x: 'Buy→Sell' if x == 1 else 'Sell→Buy')
                transaction['mode'] = transaction['entry_position'].apply(
                lambda x: 'Buy→Sell' if x == 1 else 'Sell→Buy' if x == -1 else 'Close')
                # drop the entry_position column
                transaction.drop(columns=['entry_position'], inplace=True)
                transaction = transaction[['mode', 'trades', 'entry_time', 'exit_time',
                                        'entry_price', 'exit_price', 'pnl', 'remaining_shares', 'equity']]
                
                records = records[['trade_signal', 'trades', 'price', 'equity']]
                return portfolio, transaction, records
            
    def _calculate_metrics(self, results: pd.DataFrame, trades: pd.DataFrame, records: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific period."""
        return Metrics(
            equity=results['equity'],
            drawdown=results['drawdown'],
            returns=results['pnl'],
            trades=trades,
            records=records,
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
                self.periods['backtest']['trades'],
                self.periods['backtest']['records']
            )
        elif phase == 'forward' and 'forward' in self.periods:
            return self._calculate_metrics(
                self.periods['forward']['results'],
                self.periods['forward']['trades'],
                self.periods['forward']['records']
            )
        else:
            return self._calculate_metrics(self.results, self.trades, self.records)

    def plot_results(self):
        """Plot backtest results."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")

        for period_name, period_data in self.periods.items():
            results = period_data.get("results")
            if results is None:
                continue
            period_name = "backtest" if period_name == "full" else period_name
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"{self.strategy_name.replace('_', ' ')} - {period_name.capitalize()} Result", fontsize=14)

            # === Plot 1: Price and Equity ===
            ax1.set_title('Close Price and Equity')
            ax1.set_ylabel("Close Price", color='blue')
            ax1.plot(results['price'], color='blue', label='Price')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax1b = ax1.twinx()
            ax1b.set_ylabel("Equity", color='red')
            ax1b.plot(results['equity'], color='red', label='Equity')
            ax1b.tick_params(axis='y', labelcolor='red')

            # Combined legend
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # === Plot 2: Drawdown ===
            ax2.fill_between(results.index, results['drawdown'], color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown')
            ax2.set_title('Portfolio Drawdown')
            ax2.grid(True, linestyle='--', alpha=0.7)

            # === Plot 3: Trading Signals ===
            ax3.plot(results['price'], label='Price', color='black', alpha=0.5)

            buy_signals = results[(results['position'] == 1) & (results['trades'] > 0)]
            sell_signals = results[(results['position'] == -1) & (results['trades'] > 0)]

            ax3.scatter(buy_signals.index, buy_signals['price'], label='Buy', marker='^', color='green', alpha=1)
            ax3.scatter(sell_signals.index, sell_signals['price'], label='Sell', marker='v', color='red', alpha=1)

            ax3.set_ylabel('Price ($)')
            ax3.set_title('Trading Signals')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.97]) 

            # Save to file
            output_path = os.path.join(OUTPUT_DIR, f"{self.strategy_name.lower().replace(' ', '_')}_{period_name}_result.png")
            plt.savefig(output_path)

            plt.show()

    def generate_report(self, phase: str = 'all'):
        """
        Generate comprehensive performance report.

        :param phase: 'all', 'backtest', 'forward', or 'full'
        """
        if not hasattr(self, 'periods') or not self.periods:
            raise ValueError("No backtest results available. Run backtest first.")

        if phase == 'all':
            printed_any = False
            if 'backtest' in self.periods:
                print("\n" + "=" * 60)
                print("BACKTEST PHASE PERFORMANCE".center(60))
                print("=" * 60)
                self._print_phase_report('backtest')
                printed_any = True

            if 'forward' in self.periods:
                print("\n" + "=" * 60)
                print("FORWARD TEST PHASE PERFORMANCE".center(60))
                print("=" * 60)
                self._print_phase_report('forward')
                printed_any = True

            if not printed_any and 'full' in self.periods:
                print("\n" + "=" * 60)
                print("BACKTEST PHASE PERFORMANCE".center(60))
                print("=" * 60)
                self._print_phase_report('full')
            print("=" * 60)

        else:
            if phase not in self.periods:
                raise ValueError(f"Phase '{phase}' not available in results.")
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
            records = self.records
            start_date = self.results.index[0]
            end_date = self.results.index[-1]
        else:
            results = self.periods[phase]['results']
            trades = self.periods[phase]['trades']
            records = self.periods[phase]['records']
            start_date = self.periods[phase]['start']
            end_date = self.periods[phase]['end']

        metrics = self._calculate_metrics(results, trades, records)

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
            ["Start Date", start_date],
            ["End Date", end_date],
            ["Duration (days)", (end_date - start_date).days],
            ["Data rows", len(results)],
        ]
        if self.mode == 'geometric':
            info_data.append(["Initial Capital", f"${self.initial_capital:,.2f}"])
            info_data.append(["Final Equity", f"${results['equity'].iloc[-1]:,.2f}"])

        # Performance metrics
        perf_data = [[k, fmt_value(k, v)] for k, v in metrics.items()]
        print(tabulate(info_data + perf_data, tablefmt="plain"))

    def export_data(self, results_path='output/portfolio.csv', trades_path='output/trade.csv', records_path='output/records.csv'):
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
    
        

    