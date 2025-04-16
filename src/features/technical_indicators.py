import pandas as pd
import ta.volume
import numpy as np
from typing import List
from .base_feature import BaseFeature


class SMA(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [50, 200]):
        self.column = column
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'sma_{window}'] = df[self.column].rolling(window).mean()
        return df
    
    
class EMA(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [5, 8, 13]):
        self.column = column
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'ema_{window}'] = df[self.column].ewm(span=window, adjust=False).mean()
        
        # Add 5-8-13 crossover signal
        if all(f'ema_{w}' in df.columns for w in [5, 8, 13]):
            df['ema_5_8_13_cross'] = 0
            df.loc[(df['ema_5'] > df['ema_8']) & (df['ema_8'] > df['ema_13']), 'ema_5_8_13_cross'] = 1
            df.loc[(df['ema_5'] < df['ema_8']) & (df['ema_8'] < df['ema_13']), 'ema_5_8_13_cross'] = -1
        return df
    
  
class RSI(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [14]):
        self.column = column
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            delta = df[self.column].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        return df
    
   
class OBV(BaseFeature):
    def __init__(self, price_col: str = 'close', volume_col: str = 'volume'):
        self.price_col = price_col
        self.volume_col = volume_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        obv_indicator = ta.volume.OnBalanceVolumeIndicator(
            close=df[self.price_col],
            volume=df[self.volume_col],
            fillna=True
        )
        df['obv'] = obv_indicator.on_balance_volume()
        return df 
      
      
class RSIObvSignal(BaseFeature):
    def __init__(self, rsi_window: int = 14):
        self.rsi_window = rsi_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi_col = f'rsi_{self.rsi_window}'
        if 'obv' not in df.columns:
            raise ValueError("OBV column not found. Please add OBVFeature before using RSIObvSignalFeature.")
        if rsi_col not in df.columns:
            raise ValueError(f"{rsi_col} column not found. Please add RSIFeature before using RSIObvSignalFeature.")

        signal_col = f'rsi_obv_signal_{self.rsi_window}'
        df[signal_col] = 0

        rsi = df[rsi_col]
        obv = df['obv']
        obv_ma = obv.rolling(self.rsi_window).mean()
        obv_prev = obv.shift(1)

        df.loc[(rsi > 70) & (obv > obv_ma) & (obv > obv_prev), signal_col] = -1  # Bearish
        df.loc[(rsi < 30) & (obv < obv_ma) & (obv < obv_prev), signal_col] = 1   # Bullish

        return df
    

class MACD(BaseFeature):
    def __init__(self, column: str = 'close', fast=12, slow=26, signal=9):
        self.column = column
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df[self.column].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df[self.column].ewm(span=self.slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        macd_signal = df['macd'].ewm(span=self.signal, adjust=False).mean()
        
        df['macd_signal_flag'] = (df['macd'] > macd_signal).astype(int)
        df['macd_signal_flag'] = df['macd_signal_flag'].replace(0, -1)

        df['macd_trade_signal'] = df['macd_signal_flag'].diff().fillna(df['macd_signal_flag'].iloc[0])
        return df
    
    
class PriceChange(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [1]):
        self.column = column
        self.windows = windows
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'price_change_{window}'] = df[self.column].pct_change(periods=window)   
        return df 
            
            
class Volatility(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [24, 72, 168]):
        self.column = column
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log_return = np.log(df[self.column] / df[self.column].shift(1))
        for window in self.windows:
            df[f'volatility_{window}'] = log_return.rolling(window).std()
        return df
    
    
class BollingerBands(BaseFeature):
    def __init__(self, column: str = 'close', windows: List[int] = [20], num_std=2):
        self.column = column
        self.windows = windows
        self.num_std = num_std

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            rolling_mean = df[self.column].rolling(window).mean()
            rolling_std = df[self.column].rolling(window).std()
            upper = rolling_mean + self.num_std * rolling_std
            lower = rolling_mean - self.num_std * rolling_std
            df[f'bb_signal_{window}'] = np.where(df[self.column] < lower, 1,
                                np.where(df[self.column] > upper, -1, 0))
        return df    
    