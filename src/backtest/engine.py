from typing import Optional
import pandas as pd
from src.strategy.base_strategy import Strategy
from portfolio import Portfolio
from metrics import calculate_metrics
import numpy as np

class BacktestingEngine:
    def __init__(
        self, 
        strategy: Strategy, 
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,        # 0.05% slippage per trade
        trailing_stop_pct: float = None,  # Optional: trailing stop-loss %
        holding_period: int = None        # Optional: maximum holding period in bars
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trailing_stop_pct = trailing_stop_pct
        self.holding_period = holding_period
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            trailing_stop_pct=trailing_stop_pct,
            holding_period=holding_period
        )

    def run(self, probs: Optional[np.ndarray], prices: pd.Series) -> pd.DataFrame:
        # Generate trade signals
        signals = self.strategy.generate_signals(probs, prices)
        if signals.empty:
            print("No signals generated.")
            return pd.DataFrame()

        # Simulate trades
        trades = self.portfolio.simulate_trades(signals)

        # Calculate metrics
        metrics = calculate_metrics(trades)

        print("Backtest complete.")
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        return trades
