from .base_model import BaseModel
from .utils import ModelUtils
from .xgboost_model import XGBoostModel
from .tcn_model import TCNModel

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
    else:
        raise ValueError(f"Unknown model name: {name}")