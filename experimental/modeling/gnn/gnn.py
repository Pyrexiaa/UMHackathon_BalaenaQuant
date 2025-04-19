import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from .config import (
    BATCH_SIZE,
    EPOCHS,
    RAYTUNE_SAMPLES,
)
from ..general_config import (
    MODEL_BEST_CONFIG,
    MODEL_CHECKPOINT_PATH,
    SCALING_PATH,
)
from .model_architecture import TimeSeriesGNN
from .datasets import TimeSeriesDataset
from torch.utils.data import DataLoader
import os
from ..utils import (
    evaluate_gnn,
    predict_signals_from_probs,
    backtest,
    evaluate_performance,
    evaluate_model,
    plot_results,
    calculate_class_distribution,
    load_csv,
    prepare_features,
    preprocess_data,
    normalize_data,
)
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# raytune_config = {
#     "lr": tune.loguniform(1e-4, 1e-2),
#     "weight_decay": tune.loguniform(1e-5, 1e-3),
#     "num_channels_0": tune.choice([32, 64, 128]),
#     "num_channels_1": tune.choice([64, 128, 256]),
#     "num_channels_2": tune.choice([32, 64, 128]),
# }

raytune_config = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_features": 16,
    "num_layers": 2
}

# raytune_config = {
#     "lr": tune.loguniform(1e-4, 1e-2),
#     "weight_decay": tune.loguniform(1e-5, 1e-3),
#     "hidden_features": tune.choice([16, 32, 64, 128]),
#     "num_layers": tune.choice([1, 2, 3, 4])
# }


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TCN_DIR = os.path.dirname(CURRENT_DIR)
MODELING_DIR = os.path.dirname(TCN_DIR)
BASE_DIR = os.path.dirname(MODELING_DIR)


# --- Read from env if available ---
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 120))
OUTPUT_DIR = os.path.join(BASE_DIR, "output/gnn", "ASSUMPTION_9", str(WINDOW_SIZE))
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
            # batch_x = batch_x.to(device)
            # batch_y = batch_y.to(device)
            optimizer.zero_grad()
            # Assuming a static adjacency matrix (identity for now, customize as needed)
            num_nodes = batch_x.shape[2]
            adj = torch.eye(num_nodes).unsqueeze(0).repeat(batch_x.size(0), 1, 1)

            logits = model(batch_x, adj)
            loss = loss_criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                # val_x = val_x.to(device)
                # val_y = val_y.to(device)
                num_nodes = val_x.shape[2]
                adj = torch.eye(num_nodes).unsqueeze(0).repeat(val_x.size(0), 1, 1)

                val_preds = model(val_x, adj)
                loss = loss_criterion(val_preds, val_y)
                val_loss += loss.item() * val_x.size(0)
                _, predicted = torch.max(val_preds.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%"
        )

    return model


def train_tune(config):
    dataset_path = os.path.join(
        MODELING_DIR, "datasets/btc_data_with_target_latest_v2.csv"
    )
    # --- Load CSV ---
    raw_df_train, _ = load_csv(dataset_path)
    # --- Prepare full features and normalize ---
    df_full = prepare_features(raw_df_train)
    scaled_full = normalize_data(
        df_full,
        previous_scaling=False,
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
        train_loader = DataLoader(
            TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False
        )
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
        )

        # Init model
        model = TimeSeriesGNN(
            num_nodes=X_train.shape[2],
            in_features=1,
            hidden_features=config["hidden_features"],
            out_features=3,
            num_layers=config["num_layers"],
            sequence_length=X_train.shape[1]
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        class_weights = calculate_class_distribution(y_train)
        loss_criterion = nn.CrossEntropyLoss(weight=class_weights)

        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint.pt")
                )
                model.load_state_dict(checkpoint_dict["model_state"])

        # Training
        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                num_nodes = batch_x.shape[2]
                adj = torch.eye(num_nodes).unsqueeze(0).repeat(batch_x.size(0), 1, 1)

                logits = model(batch_x, adj)
                loss = loss_criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Validation + Metrics
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                num_nodes = val_x.shape[2]
                adj = torch.eye(num_nodes).unsqueeze(0).repeat(val_x.size(0), 1, 1)
                val_preds = model(val_x, adj)
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
        print(f"[Fold {fold_idx + 1}] Metrics: {fold_metrics}")

        # Save checkpoint for this fold
        fold_checkpoint_dir = os.path.join(
            OUTPUT_DIR, MODEL_CHECKPOINT_PATH, f"fold_{fold_idx + 1}"
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
    metrics_csv_path = os.path.join(
        OUTPUT_DIR, MODEL_CHECKPOINT_PATH, "crossval_metrics.csv"
    )
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved cross-validation metrics to {metrics_csv_path}")

    # Report average metrics to Ray Tune
    mean_metrics = metrics_df.drop(columns=["fold", "config_id"]).mean().to_dict()

    # Optional: Save last model again for Ray Tune checkpoint
    final_checkpoint_dir = os.path.join(
        OUTPUT_DIR, MODEL_CHECKPOINT_PATH, "final_checkpoint"
    )
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    torch.save(
        {"epoch": EPOCHS, "model_state": model.state_dict()},
        os.path.join(final_checkpoint_dir, "checkpoint.pt"),
    )
    train.report(
        metrics=mean_metrics, checkpoint=Checkpoint.from_directory(final_checkpoint_dir)
    )


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
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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
    best_trial = result.get_best_trial(
        metric="balanced_accuracy", mode="max", scope="all"
    )
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

    best_model = TimeSeriesGNN(
        num_nodes=X_train.shape[2],
        in_features=1,
        hidden_features=best_config["hidden_features"],
        out_features=3,
        num_layers=best_config["num_layers"],
        sequence_length=X_train.shape[1],
    )
    checkpoint = torch.load(best_checkpoint_path)
    best_model.load_state_dict(checkpoint["model_state"])

    # Get predicted probabilities
    probs, y_true = evaluate_gnn(best_model, test_loader, OUTPUT_DIR)

    # Generate trading signals
    signals = predict_signals_from_probs(probs, buy_thresh=0.30, sell_thresh=0.30)

    # Backtest on raw (non-scaled) test set
    df_test_raw = raw_df_test.reset_index(drop=True).iloc[WINDOW_SIZE:].copy()
    equity, trades, trade_dates = backtest(df_test_raw, signals)

    # Evaluate performance
    evaluate_performance(equity, trades, set_name="Test")

    # Plot result
    plot_results(df_test_raw, signals, OUTPUT_DIR)
