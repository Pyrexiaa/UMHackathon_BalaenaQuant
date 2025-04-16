from abc import ABC, abstractmethod
import pandas as pd


class BaseFeature(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass