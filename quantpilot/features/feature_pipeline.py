import pandas as pd
from .base_feature import BaseFeature
from .all_features import * 

class FeaturePipeline:
    """
    Applies a list of feature engineers to a DataFrame sequentially.
    """

    def __init__(self, features: list[BaseFeature]):
        self.features = features

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding features...")
        for feature in self.features:
            df = feature.transform(df)
        print("All features added.")
        return df
    
    @staticmethod
    def standard_features() -> "FeaturePipeline":
        """
        Returns a pipeline with the standard set of features commonly used for cryptocurrency data analysis.
        """
        return FeaturePipeline([
            SMA(),                  # Simple Moving Averages
            EMA(),                  # Exponential Moving Average
            RSI(),                  # Relative Strength Index
            OBV(),                  # On-Balance Volume
            RSIObvSignal(),         # RSI + OBV Signal
            MACD(),                 # Moving Average Convergence Divergence
            PriceChange(),          # Price Change percentage
            Volatility(),           # Volatility Indicator
            BollingerBands(),       # Bollinger Bands
            HMM(),                  # Hidden Markov Model for market state prediction
            RollingKMeans(),        # Rolling KMeans clustering for market states
            NLPSentiment()          # Sentiment analysis on news or social media data
        ])
    
    