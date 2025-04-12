import xgboost as xgb
import pandas as pd
import joblib
from ..base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)
    
    