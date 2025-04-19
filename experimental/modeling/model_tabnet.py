# model_tabnet.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os
import joblib

# Set paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_latest_v2.csv"
MODEL_SAVE_PATH = Path("quantpilot/models_weights/tabnet")

# Define the features to use
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

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Parse and sort datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)

    # Drop all columns that are just "start_time_..."
    df = df[df.columns.drop(list(df.filter(regex="^start_time_")))]

    # Train / Val / Test split by datetime
    train_df = df[(df["datetime"] >= "2020-01-01") & (df["datetime"] <= "2022-12-31")]
    val_df   = df[(df["datetime"] >= "2023-01-01") & (df["datetime"] <= "2023-12-31")]
    test_df  = df[(df["datetime"] >= "2024-01-01") & (df["datetime"] <= "2025-03-31")]

    # Features and labels - now using only selected features
    def split_X_y(dataframe):
        y = dataframe["target"].astype(int)
        X = dataframe[SELECTED_FEATURES]  # Only use selected features
        return X, y

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    # Normalize with StandardScaler (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, MODEL_SAVE_PATH / "scaler.pkl")

    return X_train_scaled, y_train.to_numpy(), X_val_scaled, y_val.to_numpy(), X_test_scaled, y_test.to_numpy()

def train_tabnet():
    # Change this line to receive all 6 returned values
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    # Combine train and validation for cross-validation
    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    splits = TimeSeriesSplit(n_splits=5)
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
        preds = model.predict(X[val_idx])
        print(classification_report(y[val_idx], preds))

        # Save model
        model.save_model(str(MODEL_SAVE_PATH / f"tabnet_fold{fold + 1}"))

        break  # Remove this if you want to train all folds

if __name__ == "__main__":
    train_tabnet()