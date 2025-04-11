import pandas as pd

class Portfolio:
    def __init__(self, initial_capital: float = 100000):
        """
        :param initial_capital: Starting capital for the portfolio.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # e.g. {'AAPL': 100 shares}
        self.equity_curve = pd.Series([initial_capital], name="Equity", dtype=float)
        self.trade_log = []

    def update_equity(self, prices: pd.Series):
        """ Update portfolio equity given latest price data. """
        total_value = self.cash
        for symbol, qty in self.positions.items():
            total_value += prices.get(symbol, 0) * qty
        self.equity_curve = self.equity_curve.append(pd.Series([total_value], index=[prices.index[-1]]))
    
    def record_trade(self, symbol: str, qty: int, price: float, pnl: float, entry_time, exit_time):
        """ Record a trade for analysis (log PnL, entry/exit). """
        self.trade_log.append({
            'symbol': symbol,
            'quantity': qty,
            'price': price,
            'pnl': pnl,
            'entry_time': entry_time,
            'exit_time': exit_time
        })
    
    def get_equity_curve(self):
        """ Get the portfolio equity curve. """
        return self.equity_curve
    
    def get_trade_log(self):
        """ Get a list of trades (used for metrics). """
        return self.trade_log