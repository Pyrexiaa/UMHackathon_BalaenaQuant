from .base_metrics import BaseMetrics
import numpy as np
from ..config import BaseConfig
import pandas as pd

class Metrics(BaseMetrics):
    def total_return(self):
        """Total return over the entire period."""
        if self.mode == "arithmetic":
            return self.equity[-1]
        elif self.mode == "geometric":
            if self.equity.iloc[0] == 0:
                return np.nan
            return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    def annualized_return(self):
        """Compounded Annual Growth Rate (CAGR)."""
        start_date = self.equity.index[0]
        end_date = self.equity.index[-1]
        diff = end_date - start_date
        n_years = diff.days / 365.0
        total_ret = self.total_return()
        
        return ((1 + total_ret) ** (1 / n_years)) - 1

    def annualized_volatility(self):
        """Standard deviation of returns, annualized."""
        return self.returns.std() * np.sqrt(365)

    def sharpe_ratio(self):
        """Risk-adjusted return vs. volatility (uses risk-free rate)."""
        excess_return = self.returns - self.risk_free_rate
        return np.mean(excess_return) / np.std(excess_return) * np.sqrt(365)

    def sortino_ratio(self):
        """Like Sharpe but only penalizes downside volatility."""
        downside = self.returns[self.returns < 0]
        return self.returns.mean() / downside.std() * np.sqrt(365) if not downside.empty else np.nan

    def max_drawdown(self):
        """Maximum peak-to-trough portfolio loss."""
        return self.drawdown.min()

    def calmar_ratio(self):
        """Annual return divided by absolute maximum drawdown."""
        mdd = abs(self.max_drawdown())
        return self.annualized_return() / mdd if mdd != 0 else np.nan

    def drawdown_duration(self):
        """Longest time (in bars) portfolio spent below its peak."""
        high_water_mark = self.equity.cummax()
        underwater = self.equity / high_water_mark - 1
        duration = 0
        max_duration = 0

        for val in underwater:
            if val < 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        return max_duration

    def adjusted_returns(self):
        """Adjust returns for trading fees (0.06% per trade)."""
        return self.returns * (1 - self.trading_fee)

    def trade_metrics(self):
        """Metrics derived from trade-level data."""
        if self.records is None or self.records.empty:
            return {}

        trades = self.trades
        win_trades = trades[trades['pnl'] > 0]
        loss_trades = trades[trades['pnl'] <= 0]

        holding_period = None
        if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'], errors='coerce')
            trades['exit_time'] = pd.to_datetime(trades['exit_time'], errors='coerce')

            holding_period = trades['exit_time'] - trades['entry_time']
            holding_period = holding_period.mean()

        # trading_days = (self.equity.index[-1] - self.equity.index[0]).days
        # trade_frequency = len(trades) / trading_days if trading_days > 0 else np.nan
        trade_frequency = self.records['trades'].sum() / len(self.equity)   # trade_per_interval

        return {
            'Number of Trades': int(self.records['trades'].sum()),
            'Long Trades': self.trades[self.trades['mode'] == "Buy→Sell"]['trades'].sum(),
            'Short Trades': self.trades[self.trades['mode'] == "Sell→Buy"]['trades'].sum(),
            'Win Rate': len(win_trades) / len(trades) if len(trades) > 0 else np.nan,
            'Average PnL': trades['pnl'].mean(),
            'Expectancy': (win_trades['pnl'].mean() * len(win_trades) - abs(loss_trades['pnl'].mean()) * len(loss_trades)) / len(trades) if len(trades) > 0 else np.nan,
            'Profit Factor': float('inf') if loss_trades.empty and win_trades['pnl'].sum() > 0 else 0.0 if loss_trades.empty else win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum()),
            'Average Holding Period': holding_period,
            'Trade Frequency': trade_frequency
        }

    def all_metrics(self):
        metrics = {}
        
        # if self.mode == "geometric":
        metrics['Total Return'] = self.total_return()
        metrics['Annualized Return'] = self.annualized_return()
        metrics['Calmar Ratio'] = self.calmar_ratio()

        # Add other metrics (they will always be included)
        metrics['Sharpe Ratio'] = self.sharpe_ratio()
        metrics['Sortino Ratio'] = self.sortino_ratio()
        metrics['Max Drawdown'] = self.max_drawdown()

        # Merge in trade metrics
        metrics.update(self.trade_metrics())

        return metrics

    def get_metrics(self, names: list = None) -> dict:
        all_metrics = self.all_metrics()
        if names is None:
            return all_metrics
        return {k: all_metrics[k] for k in names if k in all_metrics}
