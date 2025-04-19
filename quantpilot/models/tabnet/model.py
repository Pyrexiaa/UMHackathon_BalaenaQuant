import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from quantpilot.config import TabNetConfig  # Adjust this import path based on your project

# In your production ready model script (tabnet/model.py)

SELECTED_FEATURES = [
    'exchange_whale_ratio',
    'taker_buy_ratio',
    'coinbase_premium_gap',
    'coinbase_premium_index',
    'exchange_supply_ratio',
    'miner_supply_ratio',
    'addresses_count_active',
    'addresses_count_outflow',
    'transactions_count_outflow',
    'tokens_transferred_total',
    'short_liquidations',
    'short_liquidations_usd',
    'long_liquidations',
    'long_liquidations_usd'
]

class TabNetModel:
    def __init__(self, model_path=TabNetConfig.TABNET_MODEL_PATH, scaler_path=TabNetConfig.TABNET_SCALER_PATH):
        self.model = TabNetClassifier()
        self.model.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, X):
        """X should be a DataFrame containing all the SELECTED_FEATURES"""
        X = X[SELECTED_FEATURES]  # Select only the features we trained with
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return preds

    def predict_proba(self, X):
        """X should be a DataFrame containing all the SELECTED_FEATURES"""
        X = X[SELECTED_FEATURES]  # Select only the features we trained with
        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)
        return probas
