from quantpilot.features import FeaturePipeline
from quantpilot.features.technical_indicators import *
import pandas as pd

df = pd.read_csv("your_price_data.csv")

pipeline = FeaturePipeline([
    SMA(windows=[50, 200]),
    EMA(),
    RSI(windows=[14]),
    OBV(),
    RSIObvSignal(rsi_window=14),
    MACD(),
    PriceChange(),
    Volatility(),
    BollingerBands(),
])

new_df = pipeline.add_features(df)



# Add a new technical indicators
from ...quantpilot.features.base_feature import BaseFeature
import pandas as pd

class ZScore(BaseFeature):
    def __init__(self, column='close', window=20):
        self.column = column
        self.window = window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df[self.column].rolling(self.window).mean()
        std = df[self.column].rolling(self.window).std()
        df[f'zscore_{self.window}'] = (df[self.column] - mean) / std
        return df
    
pipeline = FeaturePipeline([
    ZScore(),
    RSI(),
    SMA()
])

df = pipeline.transform(df)
