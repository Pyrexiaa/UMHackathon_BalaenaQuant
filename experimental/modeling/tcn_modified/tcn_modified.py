import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_BEST_CONFIG,
    MODEL_CHECKPOINT_PATH,
    MODEL_METRICS_PATH,
    RAYTUNE_SAMPLES,
    SCALING_PATH,
    MODEL_OUTPUT_FILE_PATH,
    EVALUATE_PATH,
    FEE_RATE,
    SELL_SIGNAL,
    BUY_SIGNAL,
    HOLD_SIGNAL,
)
from .model_architecture import CryptoSignalModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from joblib import dump, load
from .datasets import GroupedFeatureTimeSeriesDataset
from torch.utils.data import DataLoader
from ..constants import (
    ASSUMPTION_9
)

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import os
from sklearn.model_selection import TimeSeriesSplit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TCN_DIR = os.path.dirname(CURRENT_DIR)
MODELING_DIR = os.path.dirname(TCN_DIR)
BASE_DIR = os.path.dirname(MODELING_DIR)

# --- Read from env if available ---
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 30))
OUTPUT_DIR = os.path.join(BASE_DIR, "output/tcn_modified", "ASSUMPTION_9", str(WINDOW_SIZE))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define search space for hyperparameters
raytune_config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "encoder_dim": tune.choice([16, 32, 64]),
    "num_channels_0": tune.choice([128, 64, 256]),
    "num_channels_1": tune.choice([64, 32, 128]),
    "num_channels_2": tune.choice([32, 16, 64]),
    "tcn_dropout": tune.uniform(0.1, 0.5),
    "head_dropout": tune.uniform(0.1, 0.5),
    "head_hidden_dim": tune.choice([16, 32, 64]),
    "weight_decay": tune.uniform(0.0, 1e-4),
    "batch_size": tune.choice([32, 64]),
    "epochs": tune.choice([10, 20])
}

def prepare_features(df):
    df = df[ASSUMPTION_9].copy()
    df = df.dropna()  # Drop rows with NaN values
    df = df.reset_index(drop=True)  # Reset index after dropping rows

    return df


def calculate_class_distribution(y_train):
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(
        y_train_tensor
    )  # Counts of each class in the training labels
    total_samples = len(y_train_tensor)
    class_freq = (
        class_counts.float() / total_samples
    )  # Normalize counts by total samples

    # Calculate class weights (inverse of class frequency)
    class_weights = 1.0 / class_freq  # Inverse of frequency
    class_weights = class_weights / class_weights.sum()  # Normalize so they sum to 1

    # Convert to a tensor and ensure it's of float type
    class_weights = class_weights.float()
    return class_weights


# --- Load CSV ---
def load_csv(df_path):
    df = pd.read_csv(df_path)

    # Keep and parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year

    # Drop nan rows
    df = df.dropna()

    # Split by year
    df_train = df[df["year"] < 2024].copy()
    df_test = df[df["year"] >= 2024].copy()

    df_train = df_train.drop(columns=["datetime", "year"], axis=1)
    df_test = df_test.drop(columns=["datetime", "year"], axis=1)

    return df_train, df_test


# --- Normalize ---
def normalize_data(df, previous_scaling=None, scaling_path=SCALING_PATH):
    # Separate features and label
    features = df.drop(columns=["target"])
    target = df["target"].reset_index(drop=True)

    if previous_scaling:
        # Load previously saved scaler
        scaler = load(previous_scaling)
        scaled_features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        dump(scaler, scaling_path, compress=True)

    # Combine scaled features and label
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["target"] = target

    return scaled_df


# --- Preprocess data ---
def preprocess_data(df):
    X = []
    y = []
    for i in range(WINDOW_SIZE, len(df)):
        X.append(df.iloc[i - WINDOW_SIZE : i].drop(columns=["target"]).values)
        y.append(df["target"].iloc[i])

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, X_test, y_test


# --- Training Loop ---
def train_model(model, train_loader, val_loader, optimizer, loss_criterion):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_preds = model(val_x)
                val_loss += loss_criterion(val_preds, val_y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )

    return model


def predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30):
    """
    Convert class probabilities into trading signals:
    - class 0 (sell): prob > buy_thresh → 0.30
    - class 1 (hold): prob > hold_thresh → 0.40
    - class 2 (sell): prob > sell_thresh → 0.30
    """
    signals = []
    for p in probs:
        p = np.array(p)
        # Apply threshold rules
        if p[0] > buy_thresh and p[2] <= sell_thresh:
            signals.append(SELL_SIGNAL)  # Buy
        elif p[2] > sell_thresh and p[0] <= buy_thresh:
            signals.append(BUY_SIGNAL)  # Sell
        else:
            signals.append(HOLD_SIGNAL)  # Hold
    return np.array(signals)


def backtest(df, signals):
    capital = 1_000_000
    position = 0
    entry_price = None
    equity = []
    trades = []
    trade_dates = []

    # Fixed risk parameters (removed volatility dependency)
    stop_loss = 0.02  # Fixed 2% stop loss
    take_profit = 0.03  # Fixed 3% take profit
    position_size_pct = 0.05  # Fixed 5% position size

    for i in range(1, len(df)):
        price = df.iloc[i]["close"]

        # Exit conditions
        if position != 0:
            pnl = position * (price - entry_price)
            returns = pnl / abs(position * entry_price)

            # Simplified exit logic
            if (
                (returns < -stop_loss)
                or (returns > take_profit)
                or (signals[i] != (2 if position > 0 else 0))
            ):
                # Apply fees and close position
                pnl -= abs(position * price) * FEE_RATE
                capital += pnl
                trades.append(pnl)
                trade_dates.append(df.index[i])
                position = 0

        # Entry conditions - simplified
        if position == 0 and signals[i] != 1:  # Just check if not hold signal
            entry_price = price
            position_size = int((capital * position_size_pct) // price)
            position = position_size if signals[i] == 2 else -position_size
            capital -= abs(position * price) * (1 + FEE_RATE)
            trade_dates.append(df.index[i])

        equity.append(capital + position * price)

    return np.array(equity), trades, trade_dates


def evaluate_performance(equity, trades, set_name="Validation"):
    """Calculate performance metrics"""
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    max_dd = max(
        (peak - val) / peak for peak, val in zip(np.maximum.accumulate(equity), equity)
    )
    win_rate = np.mean(np.array(trades) > 0) if trades else 0

    print(f"\n{set_name} Performance:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": win_rate,
    }


# --- Evaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_array = np.array(all_probs)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["Buy", "Hold", "Sell"], output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Overall Test Accuracy: {acc:.4f}")
    print(
        "Overall Classification Report:\n",
        classification_report(y_true, y_pred, target_names=["Buy", "Hold", "Sell"]),
    )
    print("Overall Confusion Matrix:\n", cm)

    # Convert to numpy and save probabilities and predictions
    df_probs = pd.DataFrame(probs_array, columns=["Prob_Buy", "Prob_Hold", "Prob_Sell"])
    df_probs["True_Label"] = y_true
    df_probs["Predicted_Label"] = y_pred
    df_probs.to_csv(os.path.join(OUTPUT_DIR, MODEL_OUTPUT_FILE_PATH), index=False)
    print(
        f"Saved overall model probabilities and predictions to: {os.path.join(OUTPUT_DIR, MODEL_OUTPUT_FILE_PATH)}"
    )

    # Prepare metrics for saving
    metrics["accuracy"] = acc
    for label, scores in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics[f"precision_{label}"] = scores["precision"]
            metrics[f"recall_{label}"] = scores["recall"]
            metrics[f"f1-score_{label}"] = scores["f1-score"]
            metrics[f"support_{label}"] = scores["support"]
        elif label in ["macro avg", "weighted avg"]:
            metrics[f"precision_{label.replace(' ', '_')}"] = scores["precision"]
            metrics[f"recall_{label.replace(' ', '_')}"] = scores["recall"]
            metrics[f"f1-score_{label.replace(' ', '_')}"] = scores["f1-score"]
            metrics[f"support_{label.replace(' ', '_')}"] = scores["support"]

    cm_df = pd.DataFrame(
        cm,
        index=["Buy", "Hold", "Sell"],
        columns=["Predicted_Buy", "Predicted_Hold", "Predicted_Sell"],
    )

    # Save metrics and confusion matrix to a single CSV file
    metrics_df = pd.DataFrame([metrics])
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, MODEL_METRICS_PATH)) as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print(
        f"Saved overall model metrics and confusion matrix to: {os.path.join(OUTPUT_DIR, MODEL_METRICS_PATH)}"
    )

    return probs_array, y_true


def plot_results(df, signals):
    """Visualize backtest results"""
    plt.figure(figsize=(14, 6))
    plt.plot(df["close"], label="Price", alpha=0.6)

    buy_signals = df.iloc[np.where(signals == 2)]
    sell_signals = df.iloc[np.where(signals == 0)]

    plt.scatter(
        buy_signals.index,
        buy_signals["close"],
        marker="^",
        color="green",
        label="Buy Signal",
        s=50,
    )
    plt.scatter(
        sell_signals.index,
        sell_signals["close"],
        marker="v",
        color="red",
        label="Sell Signal",
        s=50,
    )

    plt.title("Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, EVALUATE_PATH))
    plt.close()


def train_tune(config):
    dataset_path = os.path.join(
        MODELING_DIR, "datasets/btc_data_with_target_latest_v2.csv"
    )
    # --- Load CSV ---
    raw_df_train, _ = load_csv(dataset_path)
    # --- Prepare full features and normalize ---
    df_full = prepare_features(raw_df_train)
    scaled_full = normalize_data(
        df_full, previous_scaling=False,
        scaling_path=os.path.join(OUTPUT_DIR, SCALING_PATH),
    )
    X, y = preprocess_data(scaled_full)
    # --- TimeSeriesSplit Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Create datasets and dataloaders
        train_loader = DataLoader(GroupedFeatureTimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(GroupedFeatureTimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        # Init model
        model = CryptoSignalModel(
            num_classes=3,
            encoder_dim=config["encoder_dim"],
            tcn_channels=[config[f"num_channels_{i}"] for i in range(3)],
            tcn_dropout=config["tcn_dropout"],
            head_dropout=config["head_dropout"],
            head_hidden_dim=config["head_hidden_dim"]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        class_weights = calculate_class_distribution(y_train)
        loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                model.load_state_dict(checkpoint_dict["model_state"])

        # Training
        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                batch_x = {k: v for k, v in batch_x.items()}
                logits = model(batch_x)
                loss = loss_criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Validation + Metrics
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = {k: v for k, v in val_x.items()}
                val_preds = model(val_x)
                _, predicted = torch.max(val_preds.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(val_y.cpu().numpy())

        y_true, y_pred = np.array(all_targets), np.array(all_preds)

        fold_metrics = {
            "fold": fold_idx + 1,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        fold_metrics_list.append(fold_metrics)
        print(f"[Fold {fold_idx+1}] Metrics: {fold_metrics}")

        # Save checkpoint for this fold
        fold_checkpoint_dir = os.path.join(
            OUTPUT_DIR, MODEL_CHECKPOINT_PATH, f"fold_{fold_idx+1}"
        )
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(fold_checkpoint_dir, "checkpoint.pt")
        torch.save(
            {"epoch": EPOCHS, "model_state": model.state_dict()}, checkpoint_file
        )

    # --- Final Reporting ---
    metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_df["config_id"] = config.get("trial_id", "unknown")

    # Save metrics DataFrame
    metrics_csv_path = os.path.join(OUTPUT_DIR, MODEL_CHECKPOINT_PATH, "crossval_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved cross-validation metrics to {metrics_csv_path}")

    # Report average metrics to Ray Tune
    mean_metrics = metrics_df.drop(columns=["fold", "config_id"]).mean().to_dict()
    # Optional: Save last model again for Ray Tune checkpoint
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, MODEL_CHECKPOINT_PATH, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    torch.save(
        {"epoch": EPOCHS, "model_state": model.state_dict()},
        os.path.join(final_checkpoint_dir, "checkpoint.pt"),
    )
    train.report(metrics=mean_metrics, checkpoint=Checkpoint.from_directory(final_checkpoint_dir))

if __name__ == "__main__":
    dataset_path = os.path.join(
        MODELING_DIR, "datasets/btc_data_with_target_latest_v2.csv"
    )
    # --- Load CSV ---
    raw_df_train, raw_df_test = load_csv(dataset_path)
    # --- Prepare features ---
    df_train = prepare_features(raw_df_train)
    df_test = prepare_features(raw_df_test)
    # # --- Normalize ---
    scaled_train = normalize_data(
        df_train,
        previous_scaling=False,
        scaling_path=os.path.join(OUTPUT_DIR, SCALING_PATH),
    )
    scaled_test = normalize_data(
        df_test, previous_scaling=os.path.join(OUTPUT_DIR, SCALING_PATH)
    )
    # # --- Preprocess data ---
    X_train, y_train = preprocess_data(scaled_train)
    X_test, y_test = preprocess_data(scaled_test)
    # --- Convert to Torch tensors and DataLoaders ---
    train_dataset = GroupedFeatureTimeSeriesDataset(X_train, y_train)
    test_dataset = GroupedFeatureTimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    # Configure the Ray Tune experiment

    scheduler = ASHAScheduler(
        metric="balanced_accuracy",
        mode="max",
        max_t=50,
    )

    result = tune.run(
        train_tune,
        resources_per_trial={"cpu": 2, "gpu": 0},
        config=raytune_config,
        num_samples=RAYTUNE_SAMPLES,
        scheduler=scheduler,
    )

    # Load the net with best configuration
    # Best Config focuses only on the best-performing hyperparameter configuration.
    best_config = result.get_best_config(metric="balanced_accuracy", mode="max")
    print("Best config: ", best_config)
    # Best Trial refers to the entire trial run with the best performance, including both the hyperparameters and the model results (e.g., validation accuracy, checkpoint, etc.).
    best_trial = result.get_best_trial(metric="balanced_accuracy", mode="max", scope="all")
    print("Best Trial: ", best_trial.config)
    best_checkpoint = result.get_best_checkpoint(
        best_trial, metric="balanced_accuracy", mode="max"
    )
    print("Best Checkpoint: ", best_checkpoint)
    best_checkpoint_path = os.path.join(best_checkpoint.path, "checkpoint.pt")
    print("Best Checkpoint Path: ", best_checkpoint_path)

    # Save the best config, trial and checkpoint to a dictionary
    raytune_data = {
        "Best_Config": [best_config],
        "Best_Trial_Config": [best_trial.config],
        "Best_Checkpoint_Path": [best_checkpoint_path],
    }
    raytune_df = pd.DataFrame(raytune_data)
    raytune_df.to_csv(os.path.join(OUTPUT_DIR, MODEL_BEST_CONFIG), index=False)

    input_features = X_train.shape[2]  # Get input features dynamically
    num_channels = [best_config[f"num_channels_{i}"] for i in range(3)]
    best_model = CryptoSignalModel(input_features, 3, num_channels)
    checkpoint = torch.load(best_checkpoint_path)
    best_model.load_state_dict(checkpoint["model_state"])

    # Get predicted probabilities
    probs, y_true = evaluate_model(best_model, test_loader)

    # Generate trading signals
    signals = predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30)

    # Backtest on raw (non-scaled) test set
    df_test_raw = raw_df_test.reset_index(drop=True).iloc[WINDOW_SIZE:].copy()
    equity, trades, trade_dates = backtest(df_test_raw, signals)

    # Evaluate performance
    evaluate_performance(equity, trades, set_name="Test")

    # Plot result
    plot_results(df_test_raw, signals)
