from abc import abstractmethod
import joblib
import os

class ModelUtils:
    def save_model(self, path: str):
        """
        Save the model to a file (.pkl or .pt depending on the backend).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @abstractmethod
    def load_model(self, path: str):
        """
        Load the model from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)