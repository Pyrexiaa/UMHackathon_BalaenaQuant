import pandas as pd
from .base_feature import BaseFeature

class MovingAverageFeature(BaseFeature):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_name] = df[self.column].rolling(self.window).mean()
        return df

class VolatilityFeature(BaseFeature):
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df[self.column].pct_change()
        df[self.feature_name] = returns.rolling(self.window).std()
        return df
