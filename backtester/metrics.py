# metrics.py
import pandas as pd
import numpy as np

def calculate_metrics(trades: pd.DataFrame, risk_free_rate: float = 0.01, periods_per_year: int = 252) -> dict:
    if trades.empty:
        return {}

    # Ensure datetime index if you want to use time-based metrics
    trades = trades.copy()
    if not isinstance(trades.index, pd.DatetimeIndex):
        trades.index = pd.date_range(start="2022-01-01", periods=len(trades), freq="D")

    # Daily returns (placeholder if you want to simulate over time)
    daily_returns = trades['return'].resample('D').sum().fillna(0)

    # Total Return
    total_return = trades['return'].sum()

    # Win/Loss
    avg_return = trades['return'].mean()
    win_rate = (trades['return'] > 0).mean()
    loss_rate = (trades['return'] < 0).mean()

    # Standard deviation (volatility)
    std_dev = daily_returns.std()
    downside_std = daily_returns[daily_returns < 0].std()

    # Sharpe Ratio
    sharpe_ratio = (daily_returns.mean() - risk_free_rate / periods_per_year) / std_dev if std_dev != 0 else np.nan

    # Sortino Ratio
    sortino_ratio = (daily_returns.mean() - risk_free_rate / periods_per_year) / downside_std if downside_std != 0 else np.nan

    # Cumulative returns for drawdown
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    max_drawdown = drawdown.min()

    # Recovery Period
    recovery_start = drawdown.idxmin()
    try:
        recovery_end = cumulative[cumulative >= peak[recovery_start]].index[0]
        recovery_period = (recovery_end - recovery_start).days
    except IndexError:
        recovery_period = np.nan  # never recovered

    return {
        'Total Return (%)': total_return * 100,
        'Average Return per Trade (%)': avg_return * 100,
        'Win Rate (%)': win_rate * 100,
        'Loss Rate (%)': loss_rate * 100,
        'Number of Trades': len(trades),
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Recovery Period (days)': recovery_period
    }
