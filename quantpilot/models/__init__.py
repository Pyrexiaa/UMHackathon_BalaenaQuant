from .base_model import BaseModel
from .utils import ModelUtils
from .xgboost.model_train import XGBoostModel
from .tcn.model import TCNModel
from .cnn.model import CNNModel
from .gnn.model import GNNModel
from .tabnet.model import TabNetModel

__all__ = [
    "BaseModel",
    "ModelUtils",
    "TCNModel"
]

def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "tcn":
        return TCNModel(**kwargs)
    elif name == "xgboost":
        return XGBoostModel(**kwargs)
    elif name == "cnn":
        return CNNModel(**kwargs)
    elif name == "gnn":
        return GNNModel(**kwargs)
    elif name == "tabnet":
        return TabNetModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")