"""
XGBoost Trading Strategy Implementation (Final Production-Ready)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump
from sklearn.metrics import classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_latest_v2.csv"
MODEL_DIR = Path("quantpilot/models_weights/xgboost2")
SCALING_PATH = MODEL_DIR / "scaler.pkl"

# Configuration
class BaseConfig:
    THRESHOLD = 0.3
    WINDOW_SIZE = 120  # Added rolling window size

# Ensure output directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTradingModel:
    def __init__(self):
        # Fixed 14-feature set matching ASSUMPTION_9
        self.features = [
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
        self.model = None
        self.scaler = StandardScaler()
        self.class_weights = {0: 5, 1: 1, 2: 5}
        self.optimal_buy_thresh = BaseConfig.THRESHOLD
        self.optimal_sell_thresh = BaseConfig.THRESHOLD

    def load_data(self, df_path):
        """Load and preprocess data with datetime index"""
        df = pd.read_csv(df_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        
        # Split data according to specified date ranges
        train = df.loc["2020-01-01":"2022-12-31"].copy()
        val = df.loc["2023-01-01":"2023-12-31"].copy()
        test = df.loc["2024-01-01":"2025-03-31"].copy()
        
        return train, val, test

    def prepare_data(self, df):
        """Select only the required features and target"""
        required_cols = self.features + ['target']
        return df[required_cols].copy()

    def normalize_data(self, df):
        """Normalize features only"""
        features = df[self.features]
        target = df["target"]
        
        scaled = self.scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled, columns=self.features, index=df.index)
        scaled_df["target"] = target.values
        
        return scaled_df

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train with fixed parameters"""
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": 0.01,
            "max_depth": 4,
            "n_estimators": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
            "min_child_weight": 3,
            "gamma": 0.1,
            "eval_metric": ["mlogloss", "merror"],
            "early_stopping_rounds": 50,
        }

        # Class weights
        class_counts = np.bincount(y_train)
        weights = len(y_train) / (3 * class_counts)
        sample_weights = np.array([weights[label] for label in y_train])

        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=10,
        )

    def predict_signals(self, X, window_size=BaseConfig.WINDOW_SIZE):
        """
        Generate trading signals with rolling window prediction
        Uses strict threshold checks to avoid conflicting signals
        """
        signals = []
        probs = self.model.predict_proba(X)
        
        for p in probs:
            if p[2] > BaseConfig.THRESHOLD and p[0] <= BaseConfig.THRESHOLD:
                signals.append(1)  # Buy
            elif p[0] > BaseConfig.THRESHOLD and p[2] <= BaseConfig.THRESHOLD:
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Hold
        
        return np.array(signals)

    def rolling_window_predict(self, df):
        """Rolling window prediction with 120-period window"""
        signals = []
        for i in range(BaseConfig.WINDOW_SIZE, len(df)):
            window_data = df.iloc[i-BaseConfig.WINDOW_SIZE:i]
            X_window = window_data[self.features].values
            signals.append(self.predict_signals(X_window[-1:])[0])
        
        # Pad with holds for initial period
        return np.concatenate([np.zeros(BaseConfig.WINDOW_SIZE), signals])

    def calculate_performance_metrics(self, y_true, y_pred):
        """Calculate and print performance metrics"""
        print("\nModel Performance Metrics:")
        print("="*50)
        
        metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted')
        }
        
        for metric_name, value in metrics.items():
            print(f"{metric_name:20}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return metrics

    def save_model(self):
        """Save model with all required components"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features,
            "optimal_buy_thresh": self.optimal_buy_thresh,
            "optimal_sell_thresh": self.optimal_sell_thresh,
            "window_size": BaseConfig.WINDOW_SIZE  # Save window size
        }
        
        dump(model_data, MODEL_DIR / "xgboost_model.pkl")
        dump(self.scaler, SCALING_PATH)
        print(f"Model saved with {len(self.features)} features")

def main():
    print("Starting training process...")
    model = XGBoostTradingModel()
    
    # Load and prepare data
    print("Loading data...")
    train_df, val_df, test_df = model.load_data(DATA_PATH)
    
    print("Preparing data...")
    train_ready = model.prepare_data(train_df)
    val_ready = model.prepare_data(val_df)
    test_ready = model.prepare_data(test_df)
    
    print("Normalizing data...")
    train_norm = model.normalize_data(train_ready)
    val_norm = model.normalize_data(val_ready)
    test_norm = model.normalize_data(test_ready)
    
    # Prepare arrays
    X_train = train_norm[model.features].values
    y_train = train_norm["target"].values
    X_val = val_norm[model.features].values
    y_val = val_norm["target"].values
    X_test = test_norm[model.features].values
    y_test = test_norm["target"].values
    
    # Train model
    print("\nTraining model...")
    model.train_model(X_train, y_train, X_val, y_val)
    
    # Test predictions
    print("\nEvaluating on validation set...")
    val_pred = model.model.predict(X_val)
    model.calculate_performance_metrics(y_val, val_pred)
    
    print("\nEvaluating on test set...")
    test_pred = model.model.predict(X_test)
    model.calculate_performance_metrics(y_test, test_pred)
    
    # Test rolling window predictions
    print("\nTesting rolling window predictions...")
    train_signals = model.rolling_window_predict(train_norm)
    test_signals = model.rolling_window_predict(test_norm)
    
    print(f"\nTraining signals: {np.unique(train_signals, return_counts=True)}")
    print(f"Test signals: {np.unique(test_signals, return_counts=True)}")
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()