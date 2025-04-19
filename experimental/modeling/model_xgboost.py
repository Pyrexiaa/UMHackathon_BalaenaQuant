"""
XGBoost Trading Strategy Implementation
"""
import os
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
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
from tabulate import tabulate
from .constants import (ASSUMPTION_1,ASSUMPTION_2,ASSUMPTION_3,ASSUMPTION_4,ASSUMPTION_5,ASSUMPTION_6, ASSUMPTION_7, ASSUMPTION_8)

import sys

# Add the path to your 'quantpilot' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../quantpilot')))

# from features.technical_indicators import MovingAverageFeature, VolatilityFeature

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data_with_target_latest.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost_assumption_8"
MODEL_DIR = Path("quantpilot/models_weights/xgboost_assumption_8")
FEE_RATE = 0.0006
SCALING_PATH = "quantpilot/models_weights/xgboost_assumption_8/scaler.pkl"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.class_weights = {0: 5, 1: 1, 2: 5}

        # Optimization parameters
        self.WINDOW_SIZES = [48, 72]
        self.THRESHOLDS = [0.3, 0.4]
        # self.WINDOW_SIZES = [24, 48, 72, 96, 108, 120, 132, 144, 168]
        # self.THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.optimal_window = None
        self.optimal_buy_thresh = 0.6  # Default higher threshold for buys
        self.optimal_sell_thresh = 0.4  # Default lower threshold for sells
    
    '''
    method to loop window
    '''
    # def optimize_parameters(self, X: np.ndarray, y: np.ndarray):
    #     """
    #     Optimize window size and thresholds using walk-forward validation.
    #     Called internally during training.
    #     """
    #     best_score = -np.inf
    #     best_params = {}
        
    #     # Convert y to numpy array if it's not already
    #     y = np.array(y)
        
    #     # Get unique classes in the full dataset
    #     unique_classes = np.unique(y)
    #     num_classes = len(unique_classes)
        
    #     # Ensure we have all 3 classes (0, 1, 2)
    #     if num_classes != 3:
    #         print(f"Warning: Expected 3 classes but found {num_classes}. Using default parameters.")
    #         return
            
    #     for window_size in self.WINDOW_SIZES:
    #         if window_size >= len(X):
    #             continue
                
    #         for buy_thresh in self.THRESHOLDS:
    #             for sell_thresh in [t for t in self.THRESHOLDS if t < buy_thresh]:
    #                 scores = []
    #                 valid_windows = 0

    #                 print(f"Trying window={window_size}, buy={buy_thresh}, sell={sell_thresh}")

                    
    #                 for i in range(window_size, len(X)):
    #                     # Get current window data
    #                     X_window = X[i-window_size:i]
    #                     y_window = y[i-window_size:i]
                        
    #                     # Check if window has all 3 classes
    #                     window_classes = np.unique(y_window)
    #                     if len(window_classes) != 3:
    #                         continue  # Skip windows missing classes
                            
    #                     valid_windows += 1
                        
    #                     # Clone the base model for this window
    #                     window_model = XGBClassifier(
    #                         objective='multi:softprob',
    #                         num_class=3,
    #                         n_estimators=50,  # Smaller for faster training
    #                         random_state=42
    #                     )
                        
    #                     # Train on window
    #                     window_model.fit(X_window, y_window)
                        
    #                     # Predict next step
    #                     probs = window_model.predict_proba(X[i-1:i])[0]
                        
    #                     # Apply threshold rules
    #                     if probs[0] > buy_thresh and probs[2] <= sell_thresh:
    #                         pred = 1  # Buy
    #                     elif probs[2] > sell_thresh and probs[0] <= buy_thresh:
    #                         pred = -1  # Sell
    #                     else:
    #                         pred = 0  # Hold
                            
    #                     # Score prediction
    #                     scores.append(1 if pred == y[i] else 0)
                    
    #                 if valid_windows > 0 and scores:
    #                     avg_score = np.mean(scores)
    #                     if avg_score > best_score:
    #                         best_score = avg_score
    #                         best_params = {
    #                             'window_size': window_size,
    #                             'buy_thresh': buy_thresh,
    #                             'sell_thresh': sell_thresh
    #                         }
    #                         print(f"New best: Window={window_size}, Buy={buy_thresh}, Sell={sell_thresh}, Score={avg_score:.4f}")
        
    #     if best_params:
    #         self.optimal_window = best_params['window_size']
    #         self.optimal_buy_thresh = best_params['buy_thresh']
    #         self.optimal_sell_thresh = best_params['sell_thresh']
    #         print(f"\nOptimized parameters - Window: {self.optimal_window}, Buy Threshold: {self.optimal_buy_thresh}, Sell Threshold: {self.optimal_sell_thresh}")
    #     else:
    #         print("Warning: Parameter optimization failed. Using defaults.")

    def optimize_parameters(self, X: np.ndarray, y: np.ndarray):
        """Use window size 48 and threshold 0.3 for quick testing."""
        # Convert y to numpy array if it's not already
        y = np.array(y)

        # Get unique classes in the full dataset
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        # Ensure we have all 3 classes (0, 1, 2)
        if num_classes != 3:
            print(f"Warning: Expected 3 classes but found {num_classes}. Using default parameters.")
            return
        
        # Set window size to 48 and threshold to 0.3
        window_size = 168
        if window_size >= len(X):
            print("Window size too large for data. Using defaults.")
            return
            
        scores = []
        valid_windows = 0
        total_windows = len(X) - window_size

        print(f"Testing with window={window_size}")
        print(f"Total windows to process: {total_windows}")

        # Use threshold values of 0.3 for both buy and sell
        buy_thresh = 0.4
        sell_thresh = 0.6

        # Iterate over each window individually
        for i in range(window_size, min(len(X), window_size + 100)):  # Limit to 100 windows for testing
            # Get current window data
            X_window = X[i - window_size:i]
            y_window = y[i - window_size:i]

            # Check if window has all 3 classes
            window_classes = np.unique(y_window)
            if len(window_classes) != 3:
                print(f"Skipping window {i-window_size}-{i} (missing classes)")
                continue  # Skip windows missing classes

            valid_windows += 1
            print(f"Processing valid window {i-window_size}-{i} ({valid_windows}/{total_windows})")

            # Clone the base model for this window
            window_model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=50,  # Smaller for faster training
                random_state=42
            )

            # Train on window
            window_model.fit(X_window, y_window)

            # Predict next step
            probs = window_model.predict_proba(X[i - 1:i])[0]

            # Apply threshold rules
            if probs[0] > buy_thresh and probs[2] <= sell_thresh:
                pred = 1  # Buy
            elif probs[2] > sell_thresh and probs[0] <= buy_thresh:
                pred = -1  # Sell
            else:
                pred = 0  # Hold

            # Score prediction
            scores.append(1 if pred == y[i] else 0)

        if valid_windows > 0 and scores:
            avg_score = np.mean(scores)
            print(f"\nTest results - Window: {window_size}, Buy Threshold: {buy_thresh}, Sell Threshold: {sell_thresh}, Score: {avg_score:.4f}")
        else:
            print("No valid windows found for testing.")


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
            'max_depth': 5,
            'n_estimators': 500,
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
        
        # Add optimization after initial training
        print("\nOptimizing trading parameters...")
        self.optimize_parameters(X_train, y_train)


    def predict_signals(self, X):
        probs = self.model.predict_proba(X)
        signals = np.ones(len(probs), dtype=int)  # Default: Hold (0)
        
        # Apply thresholds (using 0,1,2 internally)
        signals[probs[:, 2] > self.optimal_buy_thresh] = 2    # Buy
        signals[probs[:, 0] > self.optimal_sell_thresh] = 0   # Sell
        
        # Convert to -1, 0, 1 for external use (optional)
        signals = signals - 1  # 0→-1, 1→0, 2→1
        
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
        
        # Drop nan rows
        # df = df.dropna()

        # Split by year
        df_train = df[df["year"].between(2020, 2022)].copy()
        df_val = df[df["year"] == 2023].copy()
        df_test = df[df["year"] >= 2024].copy()

        df_train = df_train.drop(columns=["datetime", "year"], axis=1)
        df_val = df_val.drop(columns=["datetime", "year"], axis=1)
        df_test = df_test.drop(columns=["datetime", "year"], axis=1)

        return df_train, df_val, df_test

    def train_val_test_split(self, df):
        return (
            df['2020-01-01':'2022-12-31'],
            df['2023-01-01':'2023-12-31'],
            df['2024-01-01':'2025-03-31']
        )

    # def prepare_features(self, df):

    #     df = df[ASSUMPTION_8].copy()
    #     df = df.dropna()
    #     df = df.reset_index(drop=True)

    #     return df 


    def prepare_features(self, df):
        # Create directories for saving plots
        os.makedirs("feature_distributions", exist_ok=True)
        os.makedirs("assumption_distributions", exist_ok=True)
        
        # Dictionary to store test results
        assumption_results = {}
        
        # Loop through all assumptions
        for assumption_name, features in [
            ("Assumption 1", ASSUMPTION_1),
            ("Assumption 2", ASSUMPTION_2),
            ("Assumption 3", ASSUMPTION_3),
            ("Assumption 4", ASSUMPTION_4),
            ("Assumption 5", ASSUMPTION_5),
            ("Assumption 6", ASSUMPTION_6),
            ("Assumption 7", ASSUMPTION_7),
            ("Assumption 8", ASSUMPTION_8),
        ]:
            # Check if target is in features
            if "target" not in features:
                print(f"Warning: 'target' not found in {assumption_name} features")
                continue
                
            # Get the feature columns (excluding target)
            feature_cols = [f for f in features if f != "target"]
            
            # Check for missing features
            missing_features = [f for f in feature_cols if f not in df.columns]
            if missing_features:
                print(f"Skipping {assumption_name} - Missing features: {missing_features}")
                continue
                
            # Prepare data - drop rows with any NA values in these features
            subset_df = df[features].dropna()
            
            if len(subset_df) == 0:
                print(f"Skipping {assumption_name} - No data after dropping NA values")
                continue
                
            # Split into positive and negative target groups
            pos_group = subset_df[subset_df["target"] > 0][feature_cols]
            neg_group = subset_df[subset_df["target"] <= 0][feature_cols]
            
            if len(pos_group) < 2 or len(neg_group) < 2:
                print(f"Skipping {assumption_name} - Not enough samples for t-test")
                continue
                
            # Perform t-tests for each feature and create plots
            feature_results = {}
            p_values = []
            
            # Create a figure for the assumption overview
            n_features = len(feature_cols)
            n_cols = min(3, n_features)
            n_rows = int(np.ceil(n_features / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for i, (feature, ax) in enumerate(zip(feature_cols, axes)):
                try:
                    # Statistical test
                    t_stat, p_value = ttest_ind(pos_group[feature], neg_group[feature])
                    p_values.append(p_value)
                    
                    feature_results[feature] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'pos_mean': pos_group[feature].mean(),
                        'neg_mean': neg_group[feature].mean(),
                        'pos_std': pos_group[feature].std(),
                        'neg_std': neg_group[feature].std(),
                        'mean_diff': pos_group[feature].mean() - neg_group[feature].mean(),
                    }
                    
                    # Individual feature plot
                    plt.figure(figsize=(10, 6))
                    sns.kdeplot(pos_group[feature], label="Positive Target", color="green", fill=True)
                    sns.kdeplot(neg_group[feature], label="Negative Target", color="red", fill=True, alpha=0.5)
                    plt.axvline(feature_results[feature]['pos_mean'], color='green', linestyle='--', linewidth=1)
                    plt.axvline(feature_results[feature]['neg_mean'], color='red', linestyle='--', linewidth=1)
                    plt.title(f"{feature}\n(p-value: {p_value:.4f})")
                    plt.xlabel(feature)
                    plt.ylabel("Density")
                    plt.legend()
                    
                    plot_filename = f"feature_distributions/{assumption_name.replace(' ', '_')}_{feature}.png"
                    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
                    plt.close()
                    feature_results[feature]['plot_path'] = plot_filename
                    
                    # Assumption overview plot
                    sns.kdeplot(pos_group[feature], label="Positive", color="green", ax=ax, fill=True)
                    sns.kdeplot(neg_group[feature], label="Negative", color="red", ax=ax, fill=True, alpha=0.5)
                    ax.axvline(feature_results[feature]['pos_mean'], color='green', linestyle='--', linewidth=1)
                    ax.axvline(feature_results[feature]['neg_mean'], color='red', linestyle='--', linewidth=1)
                    ax.set_title(f"{feature}\n(p={p_value:.3f})")
                    ax.set_xlabel("")
                    
                except Exception as e:
                    print(f"Error testing {feature} in {assumption_name}: {str(e)}")
                    feature_results[feature] = {'error': str(e)}
                    ax.set_title(f"{feature}\n(Error)")
            
            # Remove empty subplots
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])
            
            # Finalize assumption overview plot
            plt.suptitle(f"{assumption_name} Feature Distributions", y=1.02, fontsize=14)
            plt.tight_layout()
            assumption_plot_path = f"assumption_distributions/{assumption_name.replace(' ', '_')}_overview.png"
            plt.savefig(assumption_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Calculate assumption-level metrics
            significant_features = [
                f for f in feature_results 
                if 'p_value' in feature_results[f] and feature_results[f]['p_value'] < 0.05
            ]
            mean_p_value = np.mean(p_values) if p_values else None
            
            assumption_results[assumption_name] = {
                'features_tested': feature_cols,
                'feature_results': feature_results,
                'num_significant': len(significant_features),
                'percent_significant': len(significant_features)/len(feature_cols) if feature_cols else 0,
                'mean_p_value': mean_p_value,
                'sample_size_pos': len(pos_group),
                'sample_size_neg': len(neg_group),
                'assumption_plot_path': assumption_plot_path,
            }
        
        # Sort assumptions by mean p-value (best first)
        sorted_assumptions = sorted(
            assumption_results.items(),
            key=lambda x: x[1]['mean_p_value'] if x[1]['mean_p_value'] is not None else float('inf')
        )
        
        # Print summary of results
        print("\n" + "="*50)
        print("ASSUMPTION TESTING SUMMARY (Sorted by Mean p-value)")
        print("="*50)
        summary_data = []
        for assumption, results in sorted_assumptions:
            summary_data.append([
                assumption,
                len(results['features_tested']),
                results['num_significant'],
                f"{results['percent_significant']:.1%}",
                f"{results['mean_p_value']:.4f}" if results['mean_p_value'] is not None else "N/A",
                results['sample_size_pos'],
                results['sample_size_neg'],
                results['assumption_plot_path']
            ])
        
        print(tabulate(summary_data, 
                    headers=["Assumption", "Features", "Sig. Features", "% Sig.", 
                            "Mean p-value", "Pos Samples", "Neg Samples", "Plot Path"],
                    tablefmt="grid",
                    floatfmt=".4f"))
        
        # Print detailed results for each assumption
        print("\n" + "="*50)
        print("DETAILED FEATURE TEST RESULTS")
        print("="*50)
        
        detailed_results = []
        for assumption, results in sorted_assumptions:
            print(f"\n{assumption.upper()} (Mean p-value: {results['mean_p_value']:.4f})")
            print("-"*(len(assumption) + 20))
            
            # Prepare data for table
            table_data = []
            for feature, stats in results['feature_results'].items():
                if 'p_value' in stats:
                    table_data.append([
                        feature,
                        stats['t_statistic'],
                        stats['p_value'],
                        stats['pos_mean'],
                        stats['neg_mean'],
                        stats['mean_diff'],
                        "YES" if stats['p_value'] < 0.05 else "no",
                        stats['plot_path'] if 'plot_path' in stats else ""
                    ])
            
            # Print the table
            print(tabulate(table_data,
                        headers=["Feature", "t-stat", "p-value", "Pos Mean", "Neg Mean", 
                                "Mean Diff", "Significant", "Plot Path"],
                        tablefmt="grid",
                        floatfmt=".4f"))
            
            # Store for saving
            for feature, stats in results['feature_results'].items():
                if 'p_value' in stats:
                    detailed_results.append({
                        'assumption': assumption,
                        'feature': feature,
                        't_statistic': stats['t_statistic'],
                        'p_value': stats['p_value'],
                        'pos_mean': stats['pos_mean'],
                        'neg_mean': stats['neg_mean'],
                        'pos_std': stats['pos_std'],
                        'neg_std': stats['neg_std'],
                        'mean_diff': stats['mean_diff'],
                        'significant': stats['p_value'] < 0.05,
                        'plot_path': stats.get('plot_path', ''),
                        'assumption_mean_p_value': results['mean_p_value'],
                        'assumption_plot_path': results['assumption_plot_path']
                    })
        
        # Save detailed results to DataFrame
        self.assumption_test_results = pd.DataFrame(detailed_results)
        
        # Print message about saved plots
        print("\n" + "="*50)
        print(f"Feature plots saved to: {os.path.abspath('feature_distributions')}")
        print(f"Assumption overview plots saved to: {os.path.abspath('assumption_distributions')}")
        print("="*50)
        
        # Return features from best performing assumption
        if sorted_assumptions:
            best_assumption = sorted_assumptions[0][0]
            best_features = [f for f in sorted_assumptions[0][1]['features_tested'] if f in df.columns]
            print(f"\nUsing features from best performing assumption: {best_assumption}")
            return df[best_features].dropna().reset_index(drop=True)
        
        # Fallback to original features if no assumptions worked
        required_features = [
            "tokens_transferred_mean",
            "tokens_transferred_median",
            "tokens_transferred_total",
        ]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        return df[required_features].dropna().reset_index(drop=True)
    
    def normalize_data(self,df, previous_scaling=None):
        # Separate features and label
        features = df.drop(columns=["target"])
        target = df["target"].reset_index(drop=True)

        # if previous_scaling:
        #     # Load previously saved scaler
        #     scaler = load(previous_scaling)
        #     scaled_features = scaler.transform(features)
        # else:
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
    
    def analyze_clusters(self, df, predictions, cluster_column='kmeans_cluster'):
        """
        Analyzes model predictions by different clusters.
        
        Args:
            df (pd.DataFrame): The original dataset with a cluster column.
            predictions (np.array): Model predictions aligned with df.
            cluster_column (str): The name of the clustering column in df.
        """
        df = df.copy()
        df['predicted_signal'] = predictions

        if cluster_column not in df.columns:
            print(f"Cluster column '{cluster_column}' not found in dataframe.")
            return

        cluster_groups = df.groupby(cluster_column)

        for cluster, group in cluster_groups:
            print(f"\nCluster {cluster} - Sample Size: {len(group)}")
            print("Predicted Signal Distribution:")
            print(group['predicted_signal'].value_counts())
            print("True Target Distribution:")
            print(group['target'].value_counts())

            print("\nClassification Report:")
            print(classification_report(group['target'], group['predicted_signal']))
            

    def backtest(self, df, signals):
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
            price = df.iloc[i]['close']
            
            # Exit conditions
            if position != 0:
                pnl = position * (price - entry_price)
                returns = pnl / abs(position * entry_price)
                
                # Simplified exit logic
                if (returns < -stop_loss) or (returns > take_profit) or \
                (signals[i] != (2 if position > 0 else 0)):
                    
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

    def evaluate_performance(self, df, equity, trades, trade_dates, set_name="Validation"):

        # print(f"\n--- {set_name} Trade Dates ---")
        # for date in trade_dates:
        #     print(date)
        # print(f"Total trades: {len(trade_dates)}\n")

        # Calculate trading days
        total_days = (df.index[-1] - df.index[0]).days
        unique_trade_days = len(set([d.date() for d in trade_dates]))  # Count unique trading days
        trade_frequency = (unique_trade_days / total_days) * 100  # % of days with trades

        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe Ratio (annualized)
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        
        # Max Drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Win Rate
        win_rate = np.mean(np.array(trades) > 0) if trades else 0
        
        # Rest of your metrics...
        print(f"Trade Frequency: {trade_frequency:.2f}% of days had trades")
        
        print(f"\n{set_name} Performance:")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Trade Frequency: {trade_frequency:.2f}% of days")
        
        # Strategy Validation
        passed = (sharpe >= 1.8) and (max_dd <= 0.4) and (trade_frequency >= 3.0)
        print(f"\nStrategy Passed: {'YES' if passed else 'NO'}")
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'trade_frequency': trade_frequency,
            'passed': passed
        }

    def evaluate_performance(self,equity, trades, set_name="Validation"):
        """Calculate performance metrics"""
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        max_dd = max((peak - val) / peak for peak, val in zip(np.maximum.accumulate(equity), equity))
        win_rate = np.mean(np.array(trades) > 0) if trades else 0

        print(f"\n{set_name} Performance:")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")

        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate
        }
        

def plot_signals(df, signals):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Price', alpha=0.6)

    buy_signals = df.iloc[np.where(signals == 2)]
    sell_signals = df.iloc[np.where(signals == 0)]

    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', s=50)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', s=50)

    plt.title("Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    dataset_path = "experimental/datasets/btc_data_with_target_latest_v2.csv"
    model = XGBoostTradingModel()
    train , val , test = model.load_csv(dataset_path)
    # train, val, test = model.train_val_test_split(df)
    
    # Prepare features (now returns DataFrames)
    train_feat = model.prepare_features(train)

    '''
    val_feat = model.prepare_features(val)
    test_feat = model.prepare_features(test)

    # Set the features list for scaling
    # model.features = train_feat.columns.tolist()

    # Normalize data
    train_feat = model.normalize_data(train_feat)
    val_feat = model.normalize_data(val_feat, SCALING_PATH)
    test_feat = model.normalize_data(test_feat, SCALING_PATH)
    
    # Run tuning
    # grid_search = model.tune_hyperparameters(X_train.values, y_train, X_val.values, y_val)
    
    # Train with proper feature names
    model.features = train_feat.columns.tolist()
    X_train = train_feat.drop(columns=['target'])
    y_train = train_feat["target"]
    X_val = val_feat.drop(columns=['target'])
    y_val = val_feat["target"]
    X_test = test_feat.drop(columns=['target'])
    y_test = test_feat["target"]
    model.train_model(X_train.values, y_train.values, X_val.values, y_val.values)

    # Validation set
    signals = model.predict_signals(X_val.values)
    model.evaluate_model_performance(y_val, signals, "Validation")
    equity, trades, trade_dates = model.backtest(val, signals)
    val_results = model.evaluate_performance(equity, trades,"Validation")
    
    # Test set
    signals = model.predict_signals(X_test.values)
    model.evaluate_model_performance(y_test, signals, "Test")
    equity, trades, trade_dates = model.backtest(test, signals)
    test_results = model.evaluate_performance( equity, trades,"Test")
    
    # Save model
    model.save_model()

     # Evaluate performance
    model.evaluate_performance(equity, trades,"Test")

    '''

if __name__ == "__main__":
    main()
