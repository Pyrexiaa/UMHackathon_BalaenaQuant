import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import json
from .base_model import BaseModel
from ..config import Config
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class XGBoostModel(BaseModel):
    """
    XGBoost Model with TCN-style interface for signal generation
    """
    
    def __init__(self, model_path=Config.XGB_MODEL_PATH, 
                 scaler_path=Config.XGB_SCALER_PATH,
                 feature_path=Config.XGB_FEATURE_PATH):
        super().__init__()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_groups = {
            'price_technical': [
                'future_return', 'price_change_1', 'ema_5_8_13_cross', 
                'taker_sell_ratio', 'taker_buy_ratio', 'taker_buy_sell_ratio', 
                'rsi_14', 'rsi_obv_signal_14', 'bb_signal_20', 
                'coinbase_premium_index_usdt_adjusted', 'macd_signal_flag', 
                'coinbase_premium_gap_usdt_adjusted', 'macd_trade_signal', 
                'macd', 'addresses_count_sender', 'addresses_count_active', 
                'blockreward', 'tokens_transferred_mean', 'long_liquidations', 
                'addresses_count_receiver'
            ]
        }
        self.selected_features = []
        self.class_weights = {0: 5, 1: 1, 2: 5}
        self.thresholds = {
            'buy': Config.BUY_THRESHOLD,
            'sell': Config.SELL_THRESHOLD,
            'hold': Config.HOLD_THRESHOLD
        }
        
        if model_path.exists():
            self.load(model_path.parent)

    def prepare_features(self, df):
        """Feature engineering pipeline"""
        df = df.copy()
        
        # Technical indicators
        df['close_7d_ma'] = df['close'].rolling(7).mean()
        df['close_30d_std'] = df['close'].rolling(30).std()
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / \
                            df['volume'].rolling(30).std()

        # Delta features
        delta_cols = [c for c in df.columns if c.startswith('start_time_')]
        for col in delta_cols:
            base_col = col.replace('start_time_', '')
            if base_col in df.columns:
                df[f'delta_{base_col}'] = df[base_col] - df[col]

        # Select features present in all datasets
        available_features = []
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            available_features.extend(group_features)
        
        return df[available_features].dropna()

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
        signals[probs[:, 2] > self.thresholds['buy']] = Config.BUY_SIGNAL
        signals[probs[:, 0] > self.thresholds['sell']] = Config.SELL_SIGNAL
        
        # Apply hold threshold
        max_probs = probs.max(axis=1)
        signals[max_probs < self.thresholds['hold']] = Config.HOLD_SIGNAL
        
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
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            **Config.XGB_PARAMS
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
        with open(path / "features.json", 'w') as f:
            json.dump({
                'feature_groups': self.feature_groups,
                'selected_features': self.selected_features
            }, f)

    def load(self, path):
        """Load complete model configuration"""
        path = Path(path)
        self.model = joblib.load(path / "model.pkl")
        self.scaler = joblib.load(path / "scaler.pkl")
        with open(path / "features.json", 'r') as f:
            features = json.load(f)
            self.feature_groups = features['feature_groups']
            self.selected_features = features['selected_features']

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

