from portfolio import Portfolio
from metrics import Metrics 

class Backtester:
    def __init__(self, strategy, data, initial_capital: float = 100000, trading_fee: float = 0.0006, risk_free_rate: float = 0.02):
        """
        :param strategy: The strategy object that contains the logic for signals.
        :param data: Historical price data (pd.DataFrame with DateTime index).
        :param initial_capital: Starting capital for the portfolio.
        :param risk_free_rate: Risk-free rate for metrics calculation.
        """
        self.strategy = strategy
        self.data = data
        self.portfolio = Portfolio(initial_capital)
        self.trading_fee = trading_fee
        self.risk_free_rate = risk_free_rate
        
    def execute_trade(self, symbol, qty, price, entry_time, exit_time):
        """ Simulate an order execution (buy/sell) and update the portfolio. """
        # Calculate the total cost of the trade (buying/selling)
        cost = price * qty
        if qty > 0:  # Buying
            self.portfolio.cash -= cost  # Deduct from cash
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + qty
        elif qty < 0:  # Selling
            self.portfolio.cash += cost  # Add to cash
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + qty  # Decrease position

        # Calculate PnL for this trade
        pnl = (price - self.strategy.entry_price) * qty  # Simplified PnL calculation
        self.portfolio.record_trade(symbol, abs(qty), price, pnl, entry_time, exit_time)


    # def execute_trade(self, symbol, qty, price, entry_time, exit_time):
    #     """ Simulate an order execution (buy/sell) and update the portfolio. """
    #     # Example: for long trades
    #     cost = price * qty
    #     self.portfolio.cash -= cost  # Deduct from cash
    #     self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + qty
        
    #     pnl = (price - self.strategy.entry_price) * qty  # Calculate PnL for simplicity
        
    #     self.portfolio.record_trade(symbol, qty, price, pnl, entry_time, exit_time)

    def run(self):
        """ Run the backtest loop (each step simulates a new bar of market data). """
        for date, row in self.data.iterrows():
            self.strategy.on_bar(date, row)  # The strategy makes decisions here
            self.portfolio.update_equity(self.data.loc[:date])  # Update portfolio value
            self.execute_trades(date, row)  # Simulate orders (buy/sell)
    
    def execute_trades(self, date, row):
        """ Simulate executing orders from strategy (based on signal). """
        for signal in self.strategy.get_signals(date, row):
            # Example: Buy 100 shares of signal['symbol'] at signal['price']
            if signal['action'] == 'BUY':
                self.execute_trade(signal['symbol'], signal['quantity'], signal['price'], date, None)
            elif signal['action'] == 'SELL':
                self.execute_trade(signal['symbol'], -signal['quantity'], signal['price'], signal['entry_time'], date)

    def evaluate(self):
        metrics = Metrics(self.portfolio.get_equity_curve(), self.portfolio.get_trade_log(), self.trading_fee, self.risk_free_rate)
        return metrics.all_metrics()

