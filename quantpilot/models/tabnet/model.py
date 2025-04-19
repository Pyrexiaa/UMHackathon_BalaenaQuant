import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from quantpilot.config import TabNetConfig
from quantpilot.config import BaseConfig

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
    def __init__(self, model_path=TabNetConfig.TABNET_MODEL_PATH, 
                 scaler_path=TabNetConfig.TABNET_SCALER_PATH,
                 debug=False):
        """
        Initialize TabNet model for trading signal prediction
        
        Args:
            model_path: Path to saved TabNet model
            scaler_path: Path to saved scaler
            debug: Whether to print prediction debug info
        """
        self.model = TabNetClassifier()
        self.model.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.debug = debug

    def _validate_features(self, X):
        """Ensure input contains all required features"""
        if isinstance(X, pd.DataFrame):
            missing = set(SELECTED_FEATURES) - set(X.columns)
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            return X[SELECTED_FEATURES]
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(SELECTED_FEATURES):
                raise ValueError(f"Expected {len(SELECTED_FEATURES)} features, got {X.shape[1]}")
            return X
        else:
            raise TypeError("Input must be DataFrame or numpy array")

    def predict_signals(self, X, threshold: float = None):
        """
        Generate trading signals (-1, 0, 1) based on max probability class
        Args:
            X: Input data (DataFrame or array)
            threshold: Probability threshold for buy/sell signals
        Returns:
            Array of trading signals (-1=sell, 0=hold, 1=buy)
        """
        if threshold is None:
            threshold = BaseConfig.THRESHOLD
        
        X = self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
        signals = []
        for p in probs:
        #     max_class = np.argmax(p)
            
        #     if max_class == 0:    # Sell class
        #         signals.append(-1)
        #     elif max_class == 1:  # Hold class
        #         signals.append(0)
        #     elif max_class == 2:  # Buy class
        #         signals.append(1)
        #     else:
        #         signals.append(0)  # Default to hold
            if p[2] > threshold and p[0] <= threshold:
                signals.append(1)  # Buy (class 2 prob high, sell prob low)
            elif p[0] > threshold and p[2] <= threshold:
                signals.append(-1)  # Sell (class 0 prob high, buy prob low)
            else:
                signals.append(0)  # Hold (neither condition met)
        
        if self.debug:
            self._print_prediction_debug(probs, signals)
            
        return np.array(signals)

    def predict_proba(self, X):
        """Get class probabilities"""
        X = self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def _print_prediction_debug(self, probs, signals):
        """Print prediction debug info"""
        print("\nPrediction Probabilities:")
        print("="*50)
        print("Sell (0)\tHold (1)\tBuy (2)\t\tSignal")
        print("-"*50)
        for i, (p, s) in enumerate(zip(probs[-5:], signals[-5:])):  # Last 5 predictions
            max_class = np.argmax(p)
            print(f"{p[0]:.4f}\t\t{p[1]:.4f}\t\t{p[2]:.4f}\t\t"
                  f"{s} ({'Sell' if s==-1 else 'Hold' if s==0 else 'Buy'})")
        print("="*50)

    # For backward compatibility
    def predict(self, X, threshold: float = None):
        """Alias for predict_signals"""
        return self.predict_signals(X, threshold)