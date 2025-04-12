from .base_metrics import BaseMetrics
import numpy as np
from ..config import BaseConfig

class Metrics(BaseMetrics):
    def total_return(self):
        """Total return over the entire period."""
        return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    def annualized_return(self):
        """Compounded Annual Growth Rate (CAGR)."""
        periods_per_year = 252  # Use 252 for daily bars, adjust for other timeframes
        total_ret = self.total_return()
        n_years = len(self.returns) / periods_per_year
        return (1 + total_ret) ** (1 / n_years) - 1

    def annualized_volatility(self):
        """Standard deviation of returns, annualized."""
        return self.returns.std() * np.sqrt(252)

    def sharpe_ratio(self):
        """Risk-adjusted return vs. volatility (uses risk-free rate)."""
        excess_return = self.returns - self.risk_free_rate / 252
        return np.mean(excess_return) / np.std(excess_return) * np.sqrt(252)

    def sortino_ratio(self):
        """Like Sharpe but only penalizes downside volatility."""
        downside = self.returns[self.returns < 0]
        return self.returns.mean() / downside.std() * np.sqrt(252) if not downside.empty else np.nan

    def max_drawdown(self):
        """Maximum peak-to-trough portfolio loss."""
        cumulative = self.equity
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

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
    
    def trade_frequency(self, signals):
        """Returns trade frequency (trades per data row)."""
        trades = np.sum((signals == BaseConfig.BUY_SIGNAL) | (signals == BaseConfig.SELL_SIGNAL))  # Count number of signals (trades)
        total = len(signals)
        frequency = (trades / total) * 100 if total > 0 else 0
        return frequency

    def adjusted_returns(self):
        """Adjust returns for trading fees (0.06% per trade)."""
        return self.returns * (1 - self.trading_fee )

    def trade_metrics(self):
        """Metrics derived from trade-level data."""
        if self.trades is None or self.trades.empty:
            return {}

        trades = self.trades
        win_trades = trades[trades['pnl'] > 0]
        loss_trades = trades[trades['pnl'] <= 0]

        holding_period = None
        if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
            holding_period = trades['exit_time'] - trades['entry_time']
            holding_period = holding_period.mean()

        trading_days = (self.equity.index[-1] - self.equity.index[0]).days
        trade_frequency = len(trades) / trading_days if trading_days > 0 else np.nan

        return {
            'Number of Trades': len(trades),
            'Win Rate': len(win_trades) / len(trades),
            'Average PnL': trades['pnl'].mean(),
            'Expectancy': win_trades['pnl'].mean() * (len(win_trades) / len(trades)) - \
                          abs(loss_trades['pnl'].mean()) * (len(loss_trades) / len(trades)),
            'Profit Factor': win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum()) if not loss_trades.empty else np.inf,
            'Average Holding Period': holding_period,
            'Trade Frequency (trades/day)': trade_frequency
        }

    def all_metrics(self):
        return {
            'Total Return': self.total_return(),
            'Annualized Return': self.annualized_return(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(),
            **self.trade_metrics()
        }

    def get_metrics(self, names: list = None) -> dict:
        all_metrics = self.all_metrics()
        if names is None:
            return all_metrics
        return {k: all_metrics[k] for k in names if k in all_metrics}
