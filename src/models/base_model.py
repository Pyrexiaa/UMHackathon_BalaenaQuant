from abc import ABC, abstractmethod
import pandas as pd
import joblib

class BaseModel(ABC):
    def __init__(self):
        self.model = None  # to be defined by subclasses

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions (signals).
        """
        pass

