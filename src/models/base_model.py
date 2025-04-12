from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, data):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
        Generate predictions (signals).
        """
        pass