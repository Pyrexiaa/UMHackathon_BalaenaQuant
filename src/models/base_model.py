from abc import ABC, abstractmethod
import pandas as pd
import joblib

class BaseModel(ABC):
    def __init__(self):
        self.model = None  # to be defined by subclasses

    @abstractmethod
    def predict(self, data):
        """
        Generate predictions (signals).
        """
        pass

