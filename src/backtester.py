from portfolio import Portfolio
from metrics import Metrics
from tabulate import tabulate

class Backtester:
    def __init__(self, data, strategy, initial_capital: float = 100000, trading_fee: float = 0.0006, risk_free_rate: float = 0.02):
        """
        Initialize the backtester with strategy, data, and configuration.

        :param strategy: Strategy object containing signal generation logic.
        :param data: Historical price data (pandas DataFrame with DateTime index).
        :param initial_capital: Starting capital for the portfolio.
        :param trading_fee: Trading fee as a fraction (e.g., 0.0006 for 0.06%).
        :param risk_free_rate: Risk-free rate used in performance metrics.
        """
        self.data = data
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.trading_fee = trading_fee
        self.risk_free_rate = risk_free_rate

    def execute_trade(self, symbol, qty, price, current_time, entry_time_for_pnl=None, is_buy_trade=True):
        """
        Execute a trade (buy/sell), update portfolio and record trade log.

        :param symbol: Asset symbol.
        :param qty: Quantity of the asset to trade.
        :param price: Price at which the trade is executed.
        :param current_time: The time of trade execution.
        :param entry_time_for_pnl: The entry time for PnL calculation (used for sell).
        :param is_buy_trade: True for buy, False for sell.
        """
        fee = price * abs(qty) * self.trading_fee
        cost = price * qty

        if is_buy_trade:
            self.portfolio.cash -= (cost + fee)
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + qty
        else:
            self.portfolio.cash += (abs(cost) - fee)
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) - abs(qty)

        entry_price = getattr(self.strategy, 'entry_price', price)
        pnl = (price - entry_price) * qty if is_buy_trade else (entry_price - price) * abs(qty)

        self.portfolio.record_trade(
            symbol=symbol,
            quantity=abs(qty),
            price=price,
            pnl=pnl,
            entry_time=entry_time_for_pnl or current_time,
            exit_time=current_time
        )

    def process_signals(self, date, row):
        """
        Fetch trading signals from strategy and process each trade.

        :param date: Current date.
        :param row: Row of market data for that date.
        """
        signals = self.strategy.get_signals(date, row)
        for signal in signals:
            action = signal['action'].upper()
            symbol = signal['symbol']
            qty = signal['quantity']
            price = signal['price']

            if action == 'BUY':
                self.execute_trade(symbol, qty, price, current_time=date, is_buy_trade=True)
            elif action == 'SELL':
                entry_time_for_pnl = signal.get('entry_time')
                self.execute_trade(symbol, qty, price, current_time=date, entry_time_for_pnl=entry_time_for_pnl, is_buy_trade=False)

    def run(self, verbose: bool = True):
        """
        Run the full backtest over historical data and optionally print performance metrics.
        """
        for date, row in self.data.iterrows():
            self.strategy.on_bar(date, row)
            self.process_signals(date, row)
            self.portfolio.update_equity(self.data.loc[:date])

        if verbose:
            metrics = Metrics.get_metrics()
            print(tabulate(metrics.items(), tablefmt="plain"))
