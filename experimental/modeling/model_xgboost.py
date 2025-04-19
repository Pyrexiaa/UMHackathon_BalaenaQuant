"""
XGBoost Trading Strategy Implementation
"""
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report
import joblib
from joblib import dump, load
import json
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from .constants import (ASSUMPTION_1,ASSUMPTION_2,ASSUMPTION_3,ASSUMPTION_4,ASSUMPTION_5,ASSUMPTION_6, ASSUMPTION_7, ASSUMPTION_8,ASSUMPTION_9)
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import f_oneway

import sys

# Add the path to your 'quantpilot' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../quantpilot')))

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_latest_v2.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost"
MODEL_DIR = Path("quantpilot/models_weights/xgboost")
FEE_RATE = 0.0006
SCALING_PATH = "quantpilot/models_weights/xgboost/scaler.pkl"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTradingModel:
    def __init__(self):
        self.ASSUMPTION_9 = [
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
    'long_liquidations_usd',
    'target'
]
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.class_weights = {0: 5, 1: 1, 2: 5}

        # Optimization parameters
        self.WINDOW_SIZES = [48, 120]
        self.THRESHOLDS = [0.3, 0.4]
        self.optimal_window = None
        self.optimal_buy_thresh = 0.6  # Default higher threshold for buys
        self.optimal_sell_thresh = 0.4  # Default lower threshold for sells
    
    def optimize_parameters(self, X: np.ndarray, y: np.ndarray):
        """
        Tests rolling-window performance with FIXED window=120 and threshold=0.3.
        Uses the PRE-TRAINED model (self.model) instead of retraining.
        """
        window_size = 120
        threshold = 0.3
        
        if len(X) < window_size:
            print(f"Error: Data too short for window size {window_size}.")
            return

        signals, true_labels = [], []

        for i in range(window_size, len(X)):
            X_window = X[i-window_size:i]
            y_window = y[i-window_size:i]

            # Skip if window lacks all classes
            if len(np.unique(y_window)) != 3:
                continue

            # Predict using the pre-trained model
            p = self.model.predict_proba(X[i:i+1])[0]  # Shape: (3,)
            
            # Generate signal
            if p[2] > threshold and p[0] <= threshold:
                signal = 2  # Buy
            elif p[0] > threshold and p[2] <= threshold:
                signal = 0  # Sell
            else:
                signal = 1  # Hold
            
            signals.append(signal)
            true_labels.append(y[i])

        # Metrics
        if not signals:
            print("No valid windows tested.")
            return

        accuracy = np.mean(np.array(signals) == np.array(true_labels))
        print("\n=== Rolling Window Test (Pre-Trained Model) ===")
        print(f"Window Size: {window_size} | Threshold: {threshold}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Signals: Buy={sum(np.array(signals) == 2)}, Sell={sum(np.array(signals) == 0)}, Hold={sum(np.array(signals) == 1)}")


    def train_model(self, X_train, y_train, X_val, y_val):
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Check class distribution
        unique_classes = np.unique(y_train)
        if len(unique_classes) != 3:
            raise ValueError(f"Training data must contain all 3 classes. Found: {unique_classes}")
    
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.01,
            'max_depth': 4,
            'n_estimators': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'min_child_weight': 3,
            'gamma': 0.1,
            'eval_metric': ['mlogloss', 'merror'],
            'early_stopping_rounds': 50
        }
        
        # Class weights
        class_counts = np.bincount(y_train)
        weights = len(y_train) / (3 * class_counts)
        sample_weights = np.array([weights[label] for label in y_train])
        
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=10
        )
        
        # After training, validate rolling-window performance
        print("\n=== Validation Set Rolling-Window Test ===")
        self.optimize_parameters(X_val, y_val)  # Now uses pre-trained model


    def predict_signals(self, X):
        """
        Converts model probabilities to trading signals (-1, 0, 1).
        Uses strict threshold checks to avoid conflicting signals.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            np.ndarray: Trading signals (-1=Sell, 0=Hold, 1=Buy)
        """
        # Get class probabilities (shape: [n_samples, 3])
        probs = self.model.predict_proba(X)
        
        # Initialize with Hold (0) - using -1,0,1 convention from start
        signals = np.zeros(len(probs), dtype=int)
        
        # Strict Buy condition: 
        # - Buy prob > threshold AND Sell prob not elevated
        buy_mask = (
            (probs[:, 2] > self.optimal_buy_thresh) & 
            (probs[:, 0] <= self.optimal_sell_thresh)
        )
        signals[buy_mask] = 1  # Buy
        
        # Strict Sell condition:
        # - Sell prob > threshold AND Buy prob not elevated
        sell_mask = (
            (probs[:, 0] > self.optimal_sell_thresh) & 
            (probs[:, 2] <= self.optimal_buy_thresh)
        )
        signals[sell_mask] = -1  # Sell
        
        # All other cases remain Hold (0)
        return signals
    
    def save_model(self):
        """Save model with optimization parameters"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'optimal_window': self.optimal_window,
            'optimal_buy_thresh': self.optimal_buy_thresh,
            'optimal_sell_thresh': self.optimal_sell_thresh,
            'features': self.features
        }
        
        joblib.dump(model_data, MODEL_DIR / "xgboost_model.pkl")    
        print(f"Model saved to {MODEL_DIR / 'xgboost_model.pkl'}")

    def evaluate_model_performance(self, y_true, y_pred, set_name="Validation"):
            """
            Evaluate classification performance metrics including balanced accuracy,
            precision, recall, and F1 score for each class and averaged.
            """
            # Convert signals back to 0,1,2 if they're in -1,0,1 format
            if set(np.unique(y_pred)) == {-1, 0, 1}:
                y_pred = y_pred + 1
            
            print(f"\n{classification_report(y_true, y_pred)}")
            
            # Calculate metrics for each class
            metrics = {
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'recall_macro': recall_score(y_true, y_pred, average='macro'),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Print the results
            print(f"\n{set_name} Classification Performance:")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"Macro Precision: {metrics['precision_macro']:.4f}")
            print(f"Macro Recall: {metrics['recall_macro']:.4f}")
            print(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
            print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
            print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
            print(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}")
            
            return metrics

    def load_csv(self,df_path):
        df = pd.read_csv(df_path)
        
        # Keep and parse datetime
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year

        # Split by year
        df_train = df[df["year"] < 2024].copy()
        df_test = df[df["year"] >= 2024].copy()

        df_train = df_train.drop(columns=["datetime", "year"], axis=1)
        df_test = df_test.drop(columns=["datetime", "year"], axis=1)

        return df_train, df_test

    def train_val_test_split(self, df):
        return (
            df['2020-01-01':'2022-12-31'],
            df['2023-01-01':'2023-12-31'],
            df['2024-01-01':'2025-03-31']
        )

    def prepare_features(self, df, plot_dir="plots"):
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")

        # Create directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)

        features = [f for f in self.ASSUMPTION_9 if f in df.columns]
        anova_results = {}

        for feature in features:
            groups = []
            for target_value in sorted(df['target'].unique()):
                groups.append(df[df['target'] == target_value][feature].dropna())

            f_val, p_val = f_oneway(*groups)
            anova_results[feature] = {'F-value': f_val, 'p-value': p_val}

            plt.figure(figsize=(10, 6))
            sns.boxplot(x='target', y=feature, data=df)

            # Format p-value in scientific notation if too small
            if p_val < 0.0001:
                p_text = f"ANOVA p-value: {p_val:.2e}"
            else:
                p_text = f"ANOVA p-value: {p_val:.4f}"

            plt.title(f'Distribution of {feature} by Target Group\n{p_text}')
            plt.xlabel('Target Class')
            plt.ylabel(feature)

            # Save the plot to a file (before plt.show())
            plot_filename = os.path.join(plot_dir, f"{feature}_boxplot.png")
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to free memory

        anova_df = pd.DataFrame.from_dict(anova_results, orient='index')
        anova_df.sort_values('p-value', inplace=True)
        return anova_df

    
    def normalize_data(self,df, previous_scaling=None):
        # Separate features and label
        features = df.drop(columns=["target"])
        target = df["target"].reset_index(drop=True)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        dump(scaler, SCALING_PATH, compress=True)

        # Combine scaled features and label
        scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
        scaled_df["target"] = target

        return scaled_df

    def get_param_grid(self):
        return {
            'learning_rate': [0.01, 0.02],  
            'max_depth': [4, 5, 6],            
            'n_estimators': [50, 100],      
            'subsample': [0.7, 0.6],        
            'colsample_bytree': [0.7, 0.6],
            'min_child_weight': [3],
            'gamma': [0.1],
            'reg_alpha': [0.1],
            'reg_lambda': [0.5]
        }
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        # Create time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Initialize model with fixed params
        base_model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            tree_method='hist',
            eval_metric=['mlogloss'],
            early_stopping_rounds=50,
            random_state=42
        )
        
        # Setup GridSearch
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.get_param_grid(),
            scoring='neg_log_loss',  # For probabilistic classification
            cv=tscv,
            verbose=3,
            n_jobs=-1,
            refit=True
        )
        
        # Sample weights for class imbalance
        sample_weights = np.where(
            y_train == 0, 1.5, 
            np.where(y_train == 1, 1.0, 1.3)
        )
        
        # Fit with validation data as early stopping
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=10
        )
        
        # Save best model
        self.model = grid_search.best_estimator_
        
        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print(f"Best validation score: {grid_search.best_score_:.4f}")
        
        return grid_search
    

# def main():
#     dataset_path = "experimental/datasets/btc_data_with_target_latest_v2.csv"
#     model = XGBoostTradingModel()
#     train , test = model.load_csv(dataset_path)
#     # test_feat = model.prepare_features(test)
#     test_feat = test
#     test_feat = model.normalize_data(test_feat, SCALING_PATH)
#     X_test = test_feat.drop(columns=['target'])
#     y_test = test_feat["target"]
#      # Prepare metrics storage
#     all_fold_results = []

#     # Use TimeSeriesSplit
#     tscv = TimeSeriesSplit(n_splits=5)
#     for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train)):
#         print(f"\n=== Fold {fold_idx + 1} ===")
#         train_df = train.iloc[train_idx].copy()
#         val_df = train.iloc[val_idx].copy()

#         # Feature prep
#         # train_feat = model.prepare_features(train_df)
#         # val_feat = model.prepare_features(val_df)

#         train_feat = train_df
#         val_feat = val_df

#         # Normalize
#         train_feat = model.normalize_data(train_feat)
#         val_feat = model.normalize_data(val_feat, SCALING_PATH)

#         # Setup for training
#         model.features = train_feat.columns.tolist()
#         X_train = train_feat.drop(columns=["target"])
#         y_train = train_feat["target"]
#         X_val = val_feat.drop(columns=["target"])
#         y_val = val_feat["target"]

#         # Train
#         model.train_model(X_train.values, y_train.values, X_val.values, y_val.values)

#         # Predict and Evaluate
#         signals = model.predict_signals(X_val.values)
#         model.evaluate_model_performance(y_val, signals, f"Validation Fold {fold_idx+1}")
#         # equity, trades, trade_dates = model.backtest(val_df, signals)
#         fold_results = model.evaluate_model_performance(equity, trades, f"Validation Fold {fold_idx+1}")
#         all_fold_results.append(fold_results)

#         # Optional: Save model per fold
#         model.save_model(fold=fold_idx + 1)

#     # Convert results to DataFrame
#     results_df = pd.DataFrame(all_fold_results)
#     print("\n=== Cross-Validation Summary ===")
#     print(results_df.describe())

#     # Save overall results
#     results_df.to_csv(f"{RESULTS_DIR}/cv_fold_metrics.csv", index=False)

#     # Test set
#     signals = model.predict_signals(X_test.values)
#     model.evaluate_model_performance(y_test, signals, "Test")
#     equity, trades, trade_dates = model.backtest(test, signals)
#     test_results = model.evaluate_performance( equity, trades,"Test")
    
#     # Save model
#     model.save_model()

#      # Evaluate performance
#     model.evaluate_performance(equity, trades,"Test")

def main():
    # Initialize model and load data
    model = XGBoostTradingModel()
    dataset_path = "experimental/datasets/btc_data_with_target_latest_v2.csv"
    
    print("\nLoading data...")
    train_df, test_df = model.load_csv(dataset_path)
    
    # Debug: Check columns
    print("\nColumns in train_df:", train_df.columns.tolist())
    print("Columns in test_df:", test_df.columns.tolist())
    
    # Skip prepare_features() and directly normalize
    print("\nNormalizing data...")
    train_normalized = model.normalize_data(train_df)
    test_normalized = model.normalize_data(test_df, SCALING_PATH)
    
    # Split into features/target
    X_train = train_normalized.drop(columns=["target"])
    y_train = train_normalized["target"]
    X_test = test_normalized.drop(columns=["target"])
    y_test = test_normalized["target"]
    
    # Time-based validation split (adjust as needed)
    print("\nSplitting into train/validation...")
    val_cutoff = int(0.8 * len(X_train))  # 80% train, 20% validation
    X_train, X_val = X_train[:val_cutoff], X_train[val_cutoff:]
    y_train, y_val = y_train[:val_cutoff], y_train[val_cutoff:]
    
    # Train model
    print("\nTraining model...")
    model.train_model(X_train.values, y_train.values, X_val.values, y_val.values)
    
    # Evaluate on validation set
    print("\nEvaluating validation set...")
    val_preds = model.predict_signals(X_val.values)
    val_metrics = model.evaluate_model_performance(y_val.values, val_preds, "Validation")
    
    # Test set evaluation
    print("\nEvaluating test set...")
    test_preds = model.predict_signals(X_test.values)
    test_metrics = model.evaluate_model_performance(y_test.values, test_preds, "Test")
    
    # Save model and results
    print("\nSaving results...")
    model.save_model()
    
    print("\n=== Training Complete ===")
    print(f"Validation Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}") 

if __name__ == "__main__":
    main()


'''for features anova test use'''
# def main():
#     dataset_path = "experimental/datasets/btc_data_with_target_latest_v2.csv"
#     model = XGBoostTradingModel()
#     df_train, df_test = model.load_csv(dataset_path)

#     # Save plots to "plots/" folder
#     anova_results = model.prepare_features(df_train, plot_dir="plots")

#     print("\nANOVA Results (sorted by p-value):")
#     print(anova_results)
#     anova_results.to_csv("anova_results.csv")

