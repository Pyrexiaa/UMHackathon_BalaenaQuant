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
from quantpilot.models.xgboost.model import XGBoostModel

class XGBoostModel(BaseModel):
    """
    XGBoost Model for trading signals.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = XGBConfig.XGB_MODEL_PATH
        self.scaler_path = XGBConfig.XGB_SCALER_PATH
        self.features = XGBConfig.XGB_FEATURE_PATH

    def prepare_features(self,  df: pd.DataFrame, target_column: str):
        X = df[self.features]
        y = df[target_column]
        return X, y

    def preprocess(self, X: pd.DataFrame):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train(self, X, y, params=None):
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss'
            }
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)
        return prob

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path):
        joblib.dump((self.model, self.scaler), self.model_path)

    def load(self, path):
        if os.path.exists(self.model_path):
            self.model, self.scaler = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("Saved model not found.")

    def tune_hyperparameters(self, X, y, param_grid):
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)
        grid = GridSearchCV(xgb_model, param_grid, cv=3, verbose=1, n_jobs=-1)
        grid.fit(X, y)
        print("Best Params:", grid.best_params_)
        self.model = grid.best_estimator_

    def rolling_train(self, df, target_column: str, windows, thresholds):
        results = []
        for window in windows:
            for buy_thresh in thresholds:
                for sell_thresh in thresholds:
                    df_window = df.tail(window)
                    X, y = self.prepare_features(df_window, target_column)
                    X_scaled = self.preprocess(X)
                    self.train(X_scaled, y)
                    probs = self.predict(X)
                    signals = self.apply_thresholds(probs, buy_thresh, sell_thresh)
                    results.append({
                        'window': window,
                        'buy_thresh': buy_thresh,
                        'sell_thresh': sell_thresh,
                        'signals': signals
                    })
        return results

    def apply_thresholds(self, probs, buy_thresh=None, sell_thresh=None):
        buy_thresh = buy_thresh if buy_thresh is not None else BaseConfig.BUY_THRESHOLD
        sell_thresh = sell_thresh if sell_thresh is not None else BaseConfig.SELL_THRESHOLD

        signals = []
        for p in probs:
            if p[BaseConfig.BUY_SIGNAL] > buy_thresh and p[BaseConfig.SELL_SIGNAL] <= sell_thresh:
                signals.append(BaseConfig.BUY_SIGNAL)
            elif p[BaseConfig.SELL_SIGNAL] > sell_thresh and p[BaseConfig.BUY_SIGNAL] <= buy_thresh:
                signals.append(BaseConfig.SELL_SIGNAL)
            else:
                signals.append(BaseConfig.HOLD_SIGNAL)
        return signals

# def predict(self, data):
    #     """Full prediction pipeline"""
    #     # Feature pipeline
    #     feat_df = self.prepare_features(data)
    #     norm_df = self.normalize(feat_df)
    #     proc_df = self.preprocess(norm_df)

    #     # Probability prediction
    #     probs = self.model.predict_proba(proc_df)
        
    #     # Signal conversion
    #     signals = np.ones(len(proc_df), dtype=int)  # Default hold
    #     signals[probs[:, 2] > self.thresholds['buy']] = BaseConfig.BUY_SIGNAL
    #     signals[probs[:, 0] > self.thresholds['sell']] = BaseConfig.SELL_SIGNAL
        
    #     # Apply hold threshold
    #     max_probs = probs.max(axis=1)
    #     signals[max_probs < self.thresholds['hold']] = BaseConfig.HOLD_SIGNAL
        
    #     return pd.Series(signals, index=data.index)

    # def train(self, train_data, val_data, test_data):
    #     """Complete training pipeline"""
    #     # Feature selection across all datasets
    #     self.selected_features = self._select_features(
    #         self.prepare_features(train_data),
    #         self.prepare_features(val_data),
    #         self.prepare_features(test_data)
    #     )

    #     # Prepare datasets
    #     X_train = self.normalize(self.prepare_features(train_data))
    #     X_val = self.normalize(self.prepare_features(val_data))
    #     y_train = train_data['target']
    #     y_val = val_data['target']

    #     # Initialize model

    #     with open(XGBConfig.XGB_FEATURE_PATH, 'r') as f:
    #         self.features = json.load(f)
    
    #     self.model = XGBClassifier(
    #         objective='multi:softprob',
    #         num_class=3
    #     )

    #     # Train with class weights
    #     self.model.fit(
    #         X_train, y_train,
    #         eval_set=[(X_val, y_val)],
    #         sample_weight=self._calculate_weights(y_train),
    #         verbose=10
    #     )

    '''
    sample script 
    '''

    if __name__ == "__main__":
        import sys
        import warnings
        warnings.filterwarnings("ignore")

    # Load the dataset
    csv_path = Path(__file__).resolve().parents[3] / "experimental" / "datasets" / "btc_data_with_target_modified.csv"
    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Define config values manually (or mock BaseConfig for testing)
    class MockBaseConfig:
        BUY_THRESHOLD = 0.7
        SELL_THRESHOLD = 0.3
        BUY_SIGNAL = 1
        SELL_SIGNAL = -1
        HOLD_SIGNAL = 0


    # Inject temporary config for testing
    BaseConfig.BUY_THRESHOLD = MockBaseConfig.BUY_THRESHOLD
    BaseConfig.SELL_THRESHOLD = MockBaseConfig.SELL_THRESHOLD
    BaseConfig.BUY_SIGNAL = MockBaseConfig.BUY_SIGNAL
    BaseConfig.SELL_SIGNAL = MockBaseConfig.SELL_SIGNAL
    BaseConfig.HOLD_SIGNAL = MockBaseConfig.HOLD_SIGNAL

    # Define test features and target manually if needed
    if not hasattr(XGBConfig, 'XGB_FEATURE_PATH') or isinstance(XGBConfig.XGB_FEATURE_PATH, str):
        # Assuming a column named 'target' exists
        XGBConfig.XGB_FEATURE_PATH = [col for col in df.columns if col != "target"]

    # Initialize model
    model = XGBoostModel()

    # Run training & prediction test
    print("Running basic training and prediction test...")

    # Use a small subset for fast testing
    df_subset = df.dropna().tail(300)
    X, y = model.prepare_features(df_subset, "target")
    X_scaled = model.preprocess(X)
    model.train(X_scaled, y)
    probs = model.predict(X)
    signals = model.apply_thresholds(probs)

    print("Sample predictions (first 10 signals):", signals[:10])

