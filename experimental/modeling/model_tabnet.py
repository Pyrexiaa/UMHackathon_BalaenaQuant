import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os
import joblib
from experimental.modeling.constants import ASSUMPTION_10

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_latest_v2.csv"
MODEL_SAVE_PATH = Path("quantpilot/models_weights/tabnet")

WINDOW_SIZE = 120
SELECTED_FEATURES = ASSUMPTION_10.copy()
SELECTED_FEATURES.remove("target")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Parse and sort datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

    # Train / Val / Test split by datetime
    train_df = df[(df["datetime"] >= "2020-01-01") & (df["datetime"] <= "2022-12-31")]
    val_df = df[(df["datetime"] >= "2023-01-01") & (df["datetime"] <= "2023-12-31")]
    test_df = df[(df["datetime"] >= "2024-01-01") & (df["datetime"] <= "2025-03-31")]

    # Features and labels
    def split_X_y(dataframe):
        y = dataframe["target"].astype(int)
        X = dataframe[SELECTED_FEATURES]
        return X, y

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    # Normalize with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, MODEL_SAVE_PATH / "scaler.pkl")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

def predict_signals(model, X):

    probs = model.predict_proba(X)
    signals = []
    
    for p in probs:
        max_class = np.argmax(p)
        
        if max_class == 0:    # Sell
            signals.append(-1)
        elif max_class == 1:  # Hold
            signals.append(0)
        elif max_class == 2:  # Buy
            signals.append(1)
        else:
            signals.append(0)  # Default to hold
    
    return np.array(signals)

def rolling_window_predict(model, X, window_size=WINDOW_SIZE):
    """Rolling window prediction"""
    signals = []
    for i in range(window_size, len(X)):
        window_data = X[i-window_size:i]
        signal = predict_signals(model, window_data[-1:])[0]
        signals.append(signal)
    
    return np.concatenate([np.zeros(window_size), signals])

def train_tabnet():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    splits = TimeSeriesSplit(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(splits.split(X)):
        print(f"\nTraining Fold {fold + 1}")

        model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            verbose=10
        )

        model.fit(
            X_train=X[train_idx],
            y_train=y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128
        )

        # Evaluate
        print("\nValidation Set Performance:")
        val_pred = predict_signals(model, X[val_idx])
        print(classification_report(y[val_idx], val_pred))
        
        # Test rolling window predictions
        print("\nRolling Window Predictions:")
        val_rolling_pred = rolling_window_predict(model, X[val_idx])
        print(f"Signal distribution: {np.unique(val_rolling_pred, return_counts=True)}")

        # Save model
        model.save_model(str(MODEL_SAVE_PATH / f"tabnet_fold{fold + 1}"))
        
        break

if __name__ == "__main__":
    train_tabnet()