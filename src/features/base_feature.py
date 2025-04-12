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
        Entry point to apply the feature transformation to the DataFrame.

        :param df: Input DataFrame containing the required column.
        :return: DataFrame with new feature(s) added.
        """
        self._validate(df)
        df = self.add_features(df)
        return df 

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented in subclasses. Adds feature column(s) to the DataFrame.

        :param df: Input DataFrame on which features will be added.
        :return: DataFrame with additional feature column(s).
        """
        pass

    def _validate(self, df: pd.DataFrame):
        """
        Validates that the input DataFrame meets the expected format and contains the required column.

        :param df: Input DataFrame.
        :raises ValueError: If the required column is missing or a time-based window is used without a DatetimeIndex.
        """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        if isinstance(self.window, str) and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex required for time-based windowing.")

    def generate_feature_name(self) -> str:
        """
        Generates a default name for the feature, based on the class name and window size.

        :return: String representing the feature name (e.g., "rollingmean_20").
        """
        window_str = str(self.window) if self.window is not None else "full"
        return f"{self.__class__.__name__.lower()}_{window_str}"
