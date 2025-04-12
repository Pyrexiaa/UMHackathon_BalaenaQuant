from .base_model import BaseModel
from .utils import ModelUtils
from .xgboost.model import XGBoostModel
from .tcn.model import TCNModel
from .cnn.model import CryptoCNN
from .gnn.model import GNNModel

__all__ = [
    "BaseModel",
    "ModelUtils",
]

def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "tcn":
        return TCNModel(**kwargs)
    elif name == "xgboost":
        return XGBoostModel(**kwargs)
    elif name == "cnn":
        return CryptoCNN(**kwargs)
    elif name == "gnn":
        return GNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")