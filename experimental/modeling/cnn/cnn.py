import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    WINDOW_SIZE,
    LOSS_FUNCTION,
    SCALING_PATH,
    OUTPUT_PATH,
)
from .model_architecture import CryptoCNN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump, load

# --- Load CSV ---
def load_csv(df_path):
    df = pd.read_csv(df_path)
    
    # Keep and parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    
    # Generate label: future close price difference after WINDOW_SIZE steps
    df["label"] = df["close"].shift(-WINDOW_SIZE) - df["close"]
    df = df.dropna()

    # Split by year
    df_train = df[df["year"].between(2020, 2022)].copy()
    df_val = df[df["year"] == 2023].copy()
    df_test = df[df["year"] >= 2024].copy()

    df_train = df_train.drop(columns=["datetime", "year"], axis=1)
    df_val = df_val.drop(columns=["datetime", "year"], axis=1)
    df_test = df_test.drop(columns=["datetime", "year"], axis=1)

    return df_train, df_val, df_test

# --- Normalize ---
def normalize_data(df, previous_scaling=None):
    # Separate features and label
    features = df.drop(columns=["label"])
    label = df["label"].reset_index(drop=True)

    if previous_scaling:
        # Load previously saved scaler
        scaler = load(previous_scaling)
        scaled_features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        dump(scaler, SCALING_PATH, compress=True)

    # Combine scaled features and label
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["label"] = label

    return scaled_df

# --- Preprocess data ---
def preprocess_data(df):
    X = []
    y = []
    for i in range(WINDOW_SIZE, len(df)):
        X.append(df.iloc[i - WINDOW_SIZE:i].drop(columns=["label"]).values)
        y.append(df["label"].iloc[i])
    
    return np.array(X), np.array(y)

# --- Convert to Torch tensors ---
def convert_to_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Wrap into datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_test, y_test

# --- Training Loop ---
def train_model(model, train_loader, val_loader, optimizer):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = LOSS_FUNCTION(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        # TODO: Add early stopping etc
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_preds = model(val_x)
                val_loss += LOSS_FUNCTION(val_preds, val_y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return model

# --- Evaluation ---
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze()
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Test MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return preds.numpy(), y_test.numpy()
    
def plot_predictions(preds, actual, title="Model Prediction vs Actual"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", linewidth=2)
    plt.plot(preds, label="Predicted", linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = "experimental/datasets/btc_data.csv"
    # --- Load CSV ---
    df_train, df_val, df_test = load_csv(dataset_path)
    # # --- Normalize ---
    scaled_train = normalize_data(df_train)
    scaled_val = normalize_data(df_val, previous_scaling=SCALING_PATH)
    scaled_test = normalize_data(df_test, previous_scaling=SCALING_PATH)
    # # --- Preprocess data ---
    X_train, y_train = preprocess_data(scaled_train)
    X_val, y_val = preprocess_data(scaled_val)
    X_test, y_test = preprocess_data(scaled_test)
    # --- Convert to Torch tensors ---
    train_loader, val_loader, X_test, y_test = convert_to_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)
    # --- Initialize model ---
    model = CryptoCNN(input_features=X_train.shape[2], num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # --- Train model ---
    train_model(model, train_loader, val_loader, optimizer)
    # --- Evaluate model ---
    evaluate_model(model, X_test, y_test)