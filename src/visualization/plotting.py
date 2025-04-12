class BacktestVisualizer:
    def __init__(self, equity_curve, trades, price_data):
        self.equity_curve = equity_curve
        self.trades = trades
        self.price_data = price_data

    def plot_equity_curve(self):
        ...

    def plot_drawdown(self):
        ...

    def plot_trades_on_chart(self):
        ...