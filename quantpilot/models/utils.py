import os
import joblib
from .tcn.model import TCNModel
import torch

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "defaults")

class ModelUtils:
    @staticmethod
    def save_model(model, path: str):
        """
        Save the model to a file (.pth for PyTorch, .pkl for others).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
        else:
            joblib.dump(model, path)

    @staticmethod
    def load_model(path: str):
        """
        Load the model from a file. Attempts to load as PyTorch state_dict first,
        then falls back to joblib.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            # Try loading as PyTorch state_dict
            model = None  # Initialize model to None
            if path.endswith(".pth"):
                # Need to know the model architecture to load the state_dict into
                # This requires the calling function to provide the model instance
                raise ValueError("Cannot load PyTorch state_dict without providing the model instance.")
            else:
                model = joblib.load(path)
            return model
        except Exception as e_torch:
            try:
                # Fallback to loading with joblib (pickle)
                return joblib.load(path)
            except Exception as e_joblib:
                raise RuntimeError(f"Failed to load model with both PyTorch ({e_torch}) and joblib ({e_joblib}).")