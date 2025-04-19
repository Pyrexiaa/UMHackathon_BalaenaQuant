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
            self.features = [
    'taker_buy_ratio',
    'taker_sell_ratio',
    'coinbase_premium_gap_usdt_adjusted',
    'coinbase_premium_index_usdt_adjusted',
    'exchange_supply_ratio',
    'addresses_count_active',
    'tokens_transferred_mean',
    'short_liquidations_usd',
    'long_liquidations',
    'long_liquidations_usd'
]
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
    'taker_buy_ratio',
    'taker_sell_ratio',
    'coinbase_premium_gap_usdt_adjusted',
    'coinbase_premium_index_usdt_adjusted',
    'exchange_supply_ratio',
    'addresses_count_active',
    'tokens_transferred_mean',
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

    # def predict(self, data: pd.DataFrame, threshold: float = None) -> np.ndarray:
    #     """
    #     Predict trading signals from input data using the XGBoost model.

    #     :param data: Raw input data as a DataFrame
    #     :param threshold: Optional threshold for buy/sell signals
    #     :return: Array of trading signals
    #     """
    #     if threshold is None:
    #         threshold = BaseConfig.THRESHOLD
        
    #     df_feat = self.prepare_features(data)
    #     if df_feat.empty:
    #         return np.array([])
    #     if not hasattr(self.scaler, 'mean_'):
    #         self.scaler.fit(df_feat)

    #     df_scaled = self.normalize(df_feat)

    #     # If parameters not optimized yet, use default threshold
    #     if self.optimal_buy_thresh is None or self.optimal_sell_thresh is None:
    #         probs = self.model.predict_proba(df_scaled)
    #         signals = []
    #         for p in probs:
    #             # if p[2] > threshold:
    #             #     signals.append(1)   # Buy
    #             # elif p[0] > threshold:
    #             #     signals.append(-1)  # Sell
    #             # else:
    #             #     signals.append(0)   # Hold
    #             if p[2] > threshold and p[0] <= threshold:
    #                 signals.append(1)  # Buy (class 2 prob high, sell prob low)
    #             elif p[0] > threshold and p[2] <= threshold:
    #                 signals.append(-1)  # Sell (class 0 prob high, buy prob low)
    #             else:
    #                 signals.append(0)  # Hold (neither condition met)
    #     else:
    #         # Use optimized parameters
    #         if self.optimal_window is not None and len(df_scaled) > self.optimal_window:
    #             # Get the most recent window of data
    #             window_data = df_scaled[-self.optimal_window:]
    #             # Retrain on the window (optional - could just predict)
    #             # self.model.fit(window_data, ...) if we had targets
    #             probs = self.model.predict_proba(window_data)
    #         else:
    #             probs = self.model.predict_proba(df_scaled)
                
    #         signals = []
    #         for p in probs:
    #             # if p[0] > self.optimal_buy_thresh and p[2] <= self.optimal_sell_thresh:
    #             #     signals.append(1)   # Buy
    #             # elif p[2] > self.optimal_sell_thresh and p[0] <= self.optimal_buy_thresh:
    #             #     signals.append(-1)  # Sell
    #             # else:
    #             #     signals.append(0)   # Hold
                
    #             if p[2] > 0.5 and p[0] <= 0.05:
    #                 signals.append(1)   # Buy
    #             elif p[0] > 0.4 and p[2] <= 0.05:
    #                 signals.append(-1)  # Sell
    #             else:
    #                 signals.append(0)   # Hold

    #     # Padding for warmup period (if needed)
    #     padding = len(data) - len(signals)
    #     if padding > 0:
    #         signals = [0] * padding + signals

    #     return np.array(signals)

    # ver 3
    def predict(self, data: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict trading signals from input data using the XGBoost model.
        Uses maximum probability class for signal generation:
            -1 (Sell) if max is class 0
            0 (Hold) if max is class 1 
            1 (Buy) if max is class 2

        :param data: Raw input data as a DataFrame
        :param threshold: Unused parameter (kept for backward compatibility)
        :return: Array of trading signals
        """
        df_feat = self.prepare_features(data)
        if df_feat.empty:
            return np.array([])
        
        # Ensure scaler is fitted
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(df_feat)

        df_scaled = self.normalize(df_feat)
        
        # Get probabilities and find max class
        probs = self.model.predict_proba(df_scaled)
        signals = []
        
        for p in probs:
            max_class = np.argmax(p)  # Get index of highest probability
            
            if max_class == 0:    # Sell class
                signals.append(-1)
            elif max_class == 1:  # Hold class
                signals.append(0)
            elif max_class == 2:  # Buy class
                signals.append(1)
            else:
                signals.append(0)  # Fallback to hold
        
        # # Optional: Print probabilities for debugging
        # if self.debug:
        #     print("\nPrediction Probabilities:")
        #     print("="*40)
        #     print("Sell (0)\tHold (1)\tBuy (2)\t\tPredicted Class")
        #     print("-"*60)
        #     for i, p in enumerate(probs[-5:]):  # Print last 5 predictions
        #         print(f"{p[0]:.4f}\t\t{p[1]:.4f}\t\t{p[2]:.4f}\t\t{max_class} ({'Sell' if max_class==0 else 'Hold' if max_class==1 else 'Buy'})")
        #     print("="*40 + "\n")
        
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
