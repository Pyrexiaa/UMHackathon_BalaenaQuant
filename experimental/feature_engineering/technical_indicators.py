import pandas as pd
import ta.volume
from src.features.base_feature import BaseFeature
import numpy as np
import ta
from typing import List

class FeatureTechnicalIndicators(BaseFeature):
    def __init__(self, df: pd.DataFrame, price_col: str = 'close'):
        self.df = df
        self.price_col = price_col
        
    def add_sma(self, windows: List[int] = [50, 200]):
        """
        Add Simple Moving Averages (SMA) to the dataframe.
        
        :param windows: List of window sizes for SMA calculation. Defaults to [50, 200] which can be used for Golden Cross & Death Cross strategy.
        """
        for window in windows:
            # Calculate SMA
            if window <= 0:
                raise ValueError("Window size must be positive")
            if window > len(self.df):
                raise ValueError("Window size must be less than or equal to the length of the dataframe")
            else:
                # Calculate SMA
                self.df[f'sma_{window}'] = self.df[self.price_col].rolling(window=window).mean()
                
            # can apply Golden Cross & Death Cross strategy 
            # https://www.investopedia.com/terms/g/goldencross.asp
    
    def add_ema(self, windows: List[int] = [5, 8, 13]):
        """
        Add Exponential Moving Averages (EMA) to the dataframe. 5-8-13 EMA strategy is implemented to identify trends and potential entry/exit points.
        
        :param windows: List of window sizes for EMA calculation. Defaults to [5, 8, 13] which is used for 5-8-13 EMA strategy.
        """
        for window in windows:
            # Calculate EMA
            if window <= 0:
                raise ValueError("Window size must be positive")
            if window > len(self.df):
                raise ValueError("Window size must be less than or equal to the length of the dataframe")
            else:
                # Calculate EMA
                self.df[f'ema_{window}'] = self.df[self.price_col].ewm(span=window, adjust=False).mean()
        
        ## 5-8-13 EMA strategy https://www.investopedia.com/articles/active-trading/010116/perfect-moving-averages-day-trading.asp#:~:text=For%20day%20traders%20seeking%20a,traders%20across%20diverse%20market%20conditions.
        # Buy when the 5 EMA crosses above both the 8 and 13 EMAs
        # Sell when the 5 EMA crosses below both the 8 and 13 EMAs
        if 'ema_5' in self.df.columns and 'ema_8' in self.df.columns and 'ema_13' in self.df.columns:
            # Calculate EMA crossovers
            self.df['ema_5_8_13_cross'] = 0
            self.df.loc[(self.df['ema_5'] > self.df['ema_8']) & (self.df['ema_8'] > self.df['ema_13']), 'ema_5_8_13_cross'] = 1
            self.df.loc[(self.df['ema_5'] < self.df['ema_8']) & (self.df['ema_8'] < self.df['ema_13']), 'ema_5_8_13_cross'] = -1
                         
    def add_rsi(self, windows: List[int] = [14]):
        """
        Add Relative Strength Index (RSI) to the dataframe.
        
        :param windows: List of window sizes for RSI calculation. Defaults to [14]. As the standard number of periods used to calculate the initial RSI value is 14.
        """
        for window in windows:
            # Calculate RSI
            delta = self.df[self.price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Calculate RSI
            rs = gain / loss
            self.df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Add RSI signals when RSI crosses above 70 (overbought = sell) or below 30 (oversold = buy)
            # https://www.investopedia.com/terms/r/rsi.asp
            # self.df[f'rsi_{window}_signal'] = 0
            # self.df.loc[self.df[f'rsi_{window}'] > 70, f'rsi_{window}_signal'] = -1 # Bearish signal 
            # self.df.loc[self.df[f'rsi_{window}'] < 30, f'rsi_{window}_signal'] = 1 # Bullish signal 
            
            # obv
            self.df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=self.df[self.price_col],
                volume=self.df['volume'],
                fillna=True
            ).on_balance_volume()
            
            self.df[f'rsi_obv_signal_{window}'] = 0
            self.df.loc[(self.df[f'rsi_{window}'] > 70) & (self.df['obv'] > self.df['obv'].rolling(window=window).mean())
                        & (self.df['obv'] > self.df['obv'].shift(1))
                        , f'rsi_obv_signal_{window}'] = -1 # Bearish signal
            self.df.loc[(self.df[f'rsi_{window}'] < 30) & (self.df['obv'] < self.df['obv'].rolling(window=window).mean())
                        & (self.df['obv'] < self.df['obv'].shift(1))
                        , f'rsi_obv_signal_{window}'] = 1 # Bullish signal
            
    def add_macd(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        """
        Add Moving Average Convergence Divergence (MACD) to the dataframe.
        
        :param fast_window: Fast EMA window size. Defaults to 12.
        :param slow_window: Slow EMA window size. Defaults to 26.
        :param signal_window: Signal line window size. Defaults to 9.
        """
        # Calculate MACD
        ema_fast = self.df[self.price_col].ewm(span=fast_window, adjust=False).mean()
        ema_slow = self.df[self.price_col].ewm(span=slow_window, adjust=False).mean()
        self.df['macd'] = ema_fast - ema_slow # EMA 12 - EMA 26
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal_window, adjust=False).mean()
        # self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal'] # used for visualize momentum
        
        # macd > macd_signal = bullish signal 1, 
        # macd < macd_signal = bearish signal -1
        self.df['macd_signal_flag'] = 0
        self.df['macd_signal_flag'] = self.df.apply(lambda row: 1 if row['macd'] > row['macd_signal'] else (-1 if row['macd'] < row['macd_signal'] else 0), axis=1)
        
        # trade_signal 2 = golden cross (buy), trade_signal -2 = death cross (sell)
        self.df['macd_trade_signal'] = self.df['macd_signal_flag'].diff()
        # first row is NaN, so fill it with the first row value of macd_signal_flag
        self.df['macd_trade_signal'].fillna(self.df['macd_signal_flag'].iloc[0], inplace=True)
        
        # Add MACD crossover signals
        # Buy when the MACD line crosses above the signal line, Sell when the MACD line crosses below the signal line
        # https://www.investopedia.com/terms/m/macd.asp
        self.df['macd_signal'] = 0
        self.df.loc[(self.df['macd'] > self.df['macd_signal']), 'macd_signal'] = 1
        
        # drop the columns 'macd_signal'
        self.df.drop(columns=['macd_signal'], inplace=True)
        
    def add_price_change(self, windows: List[int] = [1]):
        """
        Add price percentage change features to the dataframe.
        
        :param windows: List of window sizes for price change calculation. Defaults to [1].
        """
        for window in windows:
            # Calculate price change
            self.df[f'price_change_{window}'] = self.df[self.price_col].pct_change(periods=window)    
            
    def add_volatility(self, windows: List[int] = [24, 72, 168]):
        """
        Add volatility features to the dataframe.
        
        :param windows: List of window sizes for volatility calculation. Defaults to [24, 72, 168].
        """
        for window in windows:
            self.df["log_return"] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
            self.df[f'volatility_{window}'] = self.df["log_return"].rolling(window=window).std()
            self.df.drop(columns=["log_return"], inplace=True)
            
    def add_bollinger_bands(self, windows: List[int] = [20], num_std: int = 2):
        """
        Add Bollinger Bands to the dataframe.
        
        :param windows: List of window sizes for Bollinger Bands calculation. Defaults to [20].
        :param num_std: Number of standard deviations for upper and lower bands. Defaults to 2.
        """
        for window in windows:
            # Calculate Bollinger Bands
            rolling_mean = self.df[self.price_col].rolling(window=window).mean() # bb_middle
            rolling_std = self.df[self.price_col].rolling(window=window).std()
            self.df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * num_std)
            self.df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * num_std)

            # Add Bollinger Band signals
            # Buy when price crosses below the lower band, Sell when price crosses above the upper band
            # https://www.investopedia.com/terms/b/bollingerbands.asp
            # self.df[f'bb_signal_{window}'] = 0
            self.df[f'bb_signal_{window}'] = self.df.apply(lambda row: 1 if row[self.price_col] < row[f'bb_lower_{window}'] # buy signal
                                                           else (-1 if row[self.price_col] > row[f'bb_upper_{window}'] else 0), axis=1) # sell signal
            
            # drop the columns 'bb_upper', 'bb_lower'
            self.df.drop(columns=[f'bb_upper_{window}', f'bb_lower_{window}'], inplace=True)

    def add_all_features(self):
        """
        Add all technical indicators to the dataframe.
        """
        self.add_sma()
        self.add_ema()
        self.add_rsi()
        self.add_macd()
        self.add_price_change()
        self.add_volatility()
        self.add_bollinger_bands()         
        
    