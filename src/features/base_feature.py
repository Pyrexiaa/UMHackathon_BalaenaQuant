from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union


class BaseFeature(ABC):
    def __init__(self, column: str = "close", window: Optional[Union[int, str]] = None):
        """
        :param column: Column to operate on (e.g., "close", "volume")
        :param window: Lookback window (int for row-based, str for time-based like '7d' or '3h')
        """
        self.column = column
        self.window = window
        self.feature_name = self.generate_feature_name()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entry point to apply the feature on a DataFrame.
        """
        self._validate(df)
        df = self.add_features(df)
        return df  # ✅ This line is critical

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must be implemented in subclass. Adds the feature column(s) to the DataFrame.
        """
        pass

    def _validate(self, df: pd.DataFrame):
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        if isinstance(self.window, str) and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required for time-based windowing.")

    def generate_feature_name(self) -> str:
        """
        Default naming convention: <feature>_<window>
        Override this method for custom naming.
        """
        window_str = str(self.window) if self.window is not None else "full"
        return f"{self.__class__.__name__.lower()}_{window_str}"
