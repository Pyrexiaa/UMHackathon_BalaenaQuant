from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def predict(self, data):
        """
        Generate predictions (signals).
        """
        pass

