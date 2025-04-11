from abc import ABC, abstractmethod
import pandas as pd

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

    @abstractmethod
    def save_model(self, path: str):
        """
        Save the model to a file (.pkl or .pt depending on the backend).
        """
        pass

    @abstractmethod
    def load_model(self, path: str):
        """
        Load the model from a file.
        """
        pass
