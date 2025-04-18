import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import json
from ..base_model import BaseModel
from ...config import BaseConfig, XGBConfig
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class XGBoostModel(BaseModel):
    """
    XGBoost Model for trading signals.
    """
    
    def __init__(self, model_path=XGBConfig.XGB_MODEL_PATH, 
                 scaler_path=XGBConfig.XGB_SCALER_PATH,
                 feature_path=XGBConfig.XGB_FEATURE_PATH):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_groups = []
        self.selected_features = []
        self.class_weights = {0: 5, 1: 1, 2: 5}
        self.thresholds = {
            'buy': BaseConfig.BUY_THRESHOLD,
            'sell': BaseConfig.SELL_THRESHOLD
        }
        self.load(model_path)

    def prepare_features(self, df):
        """
        Prepare and engineer features required for model prediction.

        :param df: The raw input DataFrame containing market data
        :return: DataFrame with engineered features
        """
        
        df = df.copy()
        df['close_7d_ma'] = df['close'].rolling(7).mean()
        df['close_30d_std'] = df['close'].rolling(30).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()

        if {'exchange_whale_ratio', 'start_time_exchange_whale_ratio'}.issubset(df.columns):
            df['whale_ratio_diff'] = df['exchange_whale_ratio'] - df['start_time_exchange_whale_ratio']

        if {'estimated_leverage_ratio', 'start_time_estimated_leverage_ratio'}.issubset(df.columns):
            df['delta_estimated_leverage_ratio'] = df['estimated_leverage_ratio'] - df['start_time_estimated_leverage_ratio']

        features = [
            'delta_estimated_leverage_ratio',
            'whale_ratio_diff',
            'close_7d_ma',
            'close_30d_std',
            'volume_zscore',
            'taker_buy_ratio',
            'open_interest',
            'close', 
            'open', 
            'high', 
            'low', 
            'volume'
        ]

        # Select and clean the relevant columns
        df = df[[f for f in features if f in df.columns]].reset_index(drop=True)
        return df

    def normalize(self, data):
        """Apply trained scaler"""
        return self.scaler.transform(data[self.selected_features])

    def preprocess(self, data):
        """Final preprocessing steps"""
        # Add any position-aware features here
        return data

    def predict(self, data):
        """Full prediction pipeline"""
        # Feature pipeline
        feat_df = self.prepare_features(data)
        norm_df = self.normalize(feat_df)
        proc_df = self.preprocess(norm_df)

        # Probability prediction
        probs = self.model.predict_proba(proc_df)
        
        # Signal conversion
        signals = np.ones(len(proc_df), dtype=int)  # Default hold
        signals[probs[:, 2] > self.thresholds['buy']] = BaseConfig.BUY_SIGNAL
        signals[probs[:, 0] > self.thresholds['sell']] = BaseConfig.SELL_SIGNAL
        
        # Apply hold threshold
        max_probs = probs.max(axis=1)
        signals[max_probs < self.thresholds['hold']] = BaseConfig.HOLD_SIGNAL
        
        return pd.Series(signals, index=data.index)

    def train(self, train_data, val_data, test_data):
        """Complete training pipeline"""
        # Feature selection across all datasets
        self.selected_features = self._select_features(
            self.prepare_features(train_data),
            self.prepare_features(val_data),
            self.prepare_features(test_data)
        )

        # Prepare datasets
        X_train = self.normalize(self.prepare_features(train_data))
        X_val = self.normalize(self.prepare_features(val_data))
        y_train = train_data['target']
        y_val = val_data['target']

        # Initialize model

        with open(XGBConfig.XGB_FEATURE_PATH, 'r') as f:
            self.features = json.load(f)
    
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3
        )

        # Train with class weights
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=self._calculate_weights(y_train),
            verbose=10
        )

    def _select_features(self, train, val, test):
        """Feature selection logic"""
        selected = []
        for group, features in self.feature_groups.items():
            available = [f for f in features 
                        if all(f in df.columns for df in [train, val, test])]
            selected.extend(available)
        return selected

    def _calculate_weights(self, y):
        """Class weighting"""
        return np.vectorize(self.class_weights.get)(y)

    def save(self, path):
        """Save full model configuration"""
        path = Path(path)
        joblib.dump(self.model, path / "model.pkl")
        joblib.dump(self.scaler, path / "scaler.pkl")
        with open(XGBConfig.XGB_FEATURE_PATH, 'r') as f:
            json.dump({
                'feature_groups': self.feature_groups,
                'selected_features': self.selected_features
            }, f)

    def load(self, path):
        """Load complete model configuration"""
        path = Path(path)
        self.model = joblib.load(XGBConfig.XGB_MODEL_PATH)
        self.scaler = joblib.load(XGBConfig.XGB_SCALER_PATH)
        with open(XGBConfig.XGB_FEATURE_PATH, 'r') as f:
            features = json.load(f)
            self.feature_groups = features
            self.selected_features = features

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)