import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import xgboost as xgb
import os
from quantpilot.config import BaseConfig, XGBConfig
from quantpilot.models.base_model import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from quantpilot.models.xgboost.model import XGBoostModel

class XGBoostModel(BaseModel):
    """
    XGBoost Model for trading signals.
    """
    
    def __init__(self, model_path=XGBConfig.XGB_MODEL_PATH, scaler_path=XGBConfig.XGB_SCALER_PATH):
        try:
            # Load the full model data (dict)
            model_data = joblib.load(model_path)
            # Extract the actual XGBoost model
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = XGBClassifier(objective='multi:softprob', num_class=3)

        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load scaler from {scaler_path}: {e}")
            self.scaler = StandardScaler()

        # Optimization parameters
        self.WINDOW_SIZES = [24, 48, 72, 96, 108, 120, 132, 144, 168]
        self.THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.optimal_window = None
        self.optimal_buy_thresh = None
        self.optimal_sell_thresh = None

    def prepare_features(self, df):
        required_features = [
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
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        return df[required_features].dropna().reset_index(drop=True)


    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the features using the pre-fitted scaler.

        :param df: DataFrame of features
        :return: Scaled DataFrame
        """
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns)
    
    def optimize_parameters(self, X: pd.DataFrame, y: pd.Series):
        """
        Optimize window size and thresholds using walk-forward validation.
        """
        best_score = -np.inf
        best_params = {}
        
        # Convert to numpy for faster operations
        X_scaled = self.scaler.transform(X)
        y_true = y.values
        
        for window_size in self.WINDOW_SIZES:
            if window_size >= len(X):
                continue
                
            for buy_thresh in self.THRESHOLDS:
                for sell_thresh in self.THRESHOLDS:
                    # Walk-forward validation
                    scores = []
                    for i in range(window_size, len(X)):
                        # Get window data
                        X_window = X_scaled[i-window_size:i]
                        y_window = y_true[i-window_size:i]
                        
                        # Train on window
                        self.model.fit(X_window, y_window)
                        
                        # Predict next step
                        probs = self.model.predict_proba(X_window[-1:])[0]
                        
                        # Apply threshold rules
                        if probs[0] > buy_thresh and probs[2] <= sell_thresh:
                            pred = 1  # Buy
                        elif probs[2] > sell_thresh and probs[0] <= buy_thresh:
                            pred = -1  # Sell
                        else:
                            pred = 0  # Hold
                            
                        # Score prediction
                        scores.append(1 if pred == y_true[i] else 0)
                    
                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'window_size': window_size,
                                'buy_thresh': buy_thresh,
                                'sell_thresh': sell_thresh
                            }
        
        if best_params:
            self.optimal_window = best_params['window_size']
            self.optimal_buy_thresh = best_params['buy_thresh']
            self.optimal_sell_thresh = best_params['sell_thresh']
            print(f"Optimized parameters - Window: {self.optimal_window}, Buy Threshold: {self.optimal_buy_thresh}, Sell Threshold: {self.optimal_sell_thresh}")
        else:
            # Default values if optimization fails
            self.optimal_window = 72
            self.optimal_buy_thresh = 0.5
            self.optimal_sell_thresh = 0.5

    def predict(self, data: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict trading signals from input data using the XGBoost model.

        :param data: Raw input data as a DataFrame
        :param threshold: Optional threshold for buy/sell signals
        :return: Array of trading signals
        """
        if threshold is None:
            threshold = BaseConfig.THRESHOLD
        
        df_feat = self.prepare_features(data)
        if df_feat.empty:
            return np.array([])
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(df_feat)

        df_scaled = self.normalize(df_feat)

        # If parameters not optimized yet, use default threshold
        if self.optimal_buy_thresh is None or self.optimal_sell_thresh is None:
            probs = self.model.predict_proba(df_scaled)
            signals = []
            for p in probs:
                # if p[2] > threshold:
                #     signals.append(1)   # Buy
                # elif p[0] > threshold:
                #     signals.append(-1)  # Sell
                # else:
                #     signals.append(0)   # Hold
                if p[2] > threshold and p[0] <= threshold:
                    signals.append(1)  # Buy (class 2 prob high, sell prob low)
                elif p[0] > threshold and p[2] <= threshold:
                    signals.append(-1)  # Sell (class 0 prob high, buy prob low)
                else:
                    signals.append(0)  # Hold (neither condition met)
        else:
            # Use optimized parameters
            if self.optimal_window is not None and len(df_scaled) > self.optimal_window:
                # Get the most recent window of data
                window_data = df_scaled[-self.optimal_window:]
                # Retrain on the window (optional - could just predict)
                # self.model.fit(window_data, ...) if we had targets
                probs = self.model.predict_proba(window_data)
            else:
                probs = self.model.predict_proba(df_scaled)
                
            signals = []
            for p in probs:
                # if p[0] > self.optimal_buy_thresh and p[2] <= self.optimal_sell_thresh:
                #     signals.append(1)   # Buy
                # elif p[2] > self.optimal_sell_thresh and p[0] <= self.optimal_buy_thresh:
                #     signals.append(-1)  # Sell
                # else:
                #     signals.append(0)   # Hold
                
                if p[2] > self.optimal_buy_thresh and p[0] <= self.optimal_sell_thresh:
                    signals.append(1)   # Buy
                elif p[0] > self.optimal_sell_thresh and p[2] <= self.optimal_buy_thresh:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)   # Hold

        # Padding for warmup period (if needed)
        padding = len(data) - len(signals)
        if padding > 0:
            signals = [0] * padding + signals

        return np.array(signals)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model and optimize parameters.
        """
        self.scaler.fit(X)             
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        
        # Optimize parameters after initial fit
        self.optimize_parameters(X, y)


    '''
    sample script 
    '''

    # if __name__ == "__main__":
    #     import sys
    #     import warnings
    #     warnings.filterwarnings("ignore")

    # # Load the dataset
    # csv_path = Path(__file__).resolve().parents[3] / "experimental" / "datasets" / "btc_data_with_target_modified.csv"
    # if not csv_path.exists():
    #     print(f"CSV file not found at {csv_path}")
    #     sys.exit(1)

    # df = pd.read_csv(csv_path)

    # # Define config values manually (or mock BaseConfig for testing)
    # class MockBaseConfig:
    #     BUY_THRESHOLD = 0.7
    #     SELL_THRESHOLD = 0.3
    #     BUY_SIGNAL = 1
    #     SELL_SIGNAL = -1
    #     HOLD_SIGNAL = 0


    # # Inject temporary config for testing
    # BaseConfig.BUY_THRESHOLD = MockBaseConfig.BUY_THRESHOLD
    # BaseConfig.SELL_THRESHOLD = MockBaseConfig.SELL_THRESHOLD
    # BaseConfig.BUY_SIGNAL = MockBaseConfig.BUY_SIGNAL
    # BaseConfig.SELL_SIGNAL = MockBaseConfig.SELL_SIGNAL
    # BaseConfig.HOLD_SIGNAL = MockBaseConfig.HOLD_SIGNAL

    # # Define test features and target manually if needed
    # if not hasattr(XGBConfig, 'XGB_FEATURE_PATH') or isinstance(XGBConfig.XGB_FEATURE_PATH, str):
    #     # Assuming a column named 'target' exists
    #     XGBConfig.XGB_FEATURE_PATH = [col for col in df.columns if col != "target"]

    # # Initialize model
    # model = XGBoostModel()

    # # Run training & prediction test
    # print("Running basic training and prediction test...")

    # # Use a small subset for fast testing
    # df_subset = df.dropna().tail(300)
    # X, y = model.prepare_features(df_subset, "target")
    # X_scaled = model.preprocess(X)
    # model.train(X_scaled, y)
    # probs = model.predict(X)
    # signals = model.apply_thresholds(probs)

    # print("Sample predictions (first 10 signals):", signals[:10])

