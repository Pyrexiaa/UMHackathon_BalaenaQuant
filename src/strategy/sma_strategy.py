import pandas as pd
import numpy as np

class SimpleMovingAverageStrategy:
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window
        self.entry_price = None
    
    def on_bar(self, date, row):
        """ Strategy logic for generating signals. """
        short_ma = row['close'].rolling(window=self.short_window).mean()
        long_ma = row['close'].rolling(window=self.long_window).mean()
        
        if short_ma > long_ma:
            # Signal to buy
            self.entry_price = row['close']  # Entry price for PnL calculation
            self.buy_signal(date, row['close'])
        elif short_ma < long_ma:
            # Signal to sell
            self.sell_signal(date, row['close'])
    
    def buy_signal(self, date, price):
        """ Generate a buy signal. """
        return [{
            'action': 'BUY',
            'symbol': 'AAPL',
            'quantity': 100,  # Example
            'price': price,
            'entry_time': date
        }]
    
    def sell_signal(self, date, price):
        """ Generate a sell signal. """
        return [{
            'action': 'SELL',
            'symbol': 'AAPL',
            'quantity': 100,  # Example
            'price': price,
            'entry_time': None
        }]
    
    def get_signals(self, date, row):
        """ Get strategy signals for a given bar (date). """
        return self.buy_signal(date, row['close']) + self.sell_signal(date, row['close'])
