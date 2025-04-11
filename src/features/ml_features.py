from typing import Optional, Union
from .base_feature import BaseFeature
import pandas as pd

class HMMFeature(BaseFeature):
    def __init__(self, model, column: str = "close", window: Optional[Union[int, str]] = None):
        """
        :param model: Pretrained HMM model that supports a `predict` method.
        :param column: Column to use for feature extraction, default is 'close'.
        :param window: Lookback window (int or str, e.g., '14d', '3h')
        """
        super().__init__(column, window)
        self.model = model  # Assume model has a `predict` method

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add HMM feature to the DataFrame by predicting the hidden states
        using the provided HMM model.
        """
        # Ensure the 'close' column exists
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        
        # Model prediction (Assume model predicts hidden states based on `df[self.column]`)
        hidden_states = self.model.predict(df[[self.column]])
        
        # Add hidden states as a new feature (column)
        df[self.feature_name] = hidden_states
        return df
