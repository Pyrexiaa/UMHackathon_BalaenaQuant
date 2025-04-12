import os
import joblib
from .tcn_model import TCNModel

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "defaults")

class ModelUtils:
    @staticmethod
    def save_model(model, path: str):
        """
        Save the model to a file (.pkl or .pt depending on the backend).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    @staticmethod
    def load_model(path: str):
        """
        Load the model from a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    @staticmethod
    def get_model(name: str, **kwargs):
        """
        Retrieve a model by name from the defaults directory.

        :param name: Model name 
        :return: Loaded model object
        """
        if name == "tcn":
            return TCNModel(**kwargs)