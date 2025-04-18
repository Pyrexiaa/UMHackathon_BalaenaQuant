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
    
    def __init__(self, model_path=XGBConfig.XGB_MODEL_PATH, scaler_path=XGBConfig.XGB_SCALER_PATH):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            self.model = XGBClassifier(objective='multi:softprob', num_class=3)

        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load scaler from {scaler_path}: {e}")
            self.scaler = StandardScaler()


    # def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Feature engineering for XGBoost
    #     """
    #     df = df.copy()
    #     df['close_7d_ma'] = df['close'].rolling(7).mean()
    #     df['close_30d_std'] = df['close'].rolling(30).std()
    #     df['volume_zscore'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()

    #     if {'exchange_whale_ratio', 'start_time_exchange_whale_ratio'}.issubset(df.columns):
    #         df['whale_ratio_diff'] = df['exchange_whale_ratio'] - df['start_time_exchange_whale_ratio']

    #     if {'estimated_leverage_ratio', 'start_time_estimated_leverage_ratio'}.issubset(df.columns):
    #         df['delta_estimated_leverage_ratio'] = df['estimated_leverage_ratio'] - df['start_time_estimated_leverage_ratio']

    #     features = [
    #         'delta_estimated_leverage_ratio',
    #         'whale_ratio_diff',
    #         'close_7d_ma',
    #         'close_30d_std',
    #         'volume_zscore',
    #         'taker_buy_ratio',
    #         'open_interest',
    #         'close',
    #         'open',
    #         'high',
    #         'low',
    #         'volume'
    #     ]
    #     return df[[f for f in features if f in df.columns]].reset_index(drop=True)

    def prepare_features(self, df):
        required_features = [
            'future_return', 'price_change_1', 'ema_5_8_13_cross', 'taker_sell_ratio', 
            'taker_buy_ratio', 'taker_buy_sell_ratio', 'rsi_14', 'rsi_obv_signal_14', 
            'bb_signal_20', 'coinbase_premium_index_usdt_adjusted', 'macd_signal_flag', 
            'coinbase_premium_gap_usdt_adjusted', 'macd_trade_signal', 'macd', 
            'addresses_count_sender', 'addresses_count_active', 'blockreward', 
            'tokens_transferred_mean', 'long_liquidations', 'addresses_count_receiver', 'target'
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

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict trading signals from input data using the XGBoost model.

        :param data: Raw input data as a DataFrame
        :return: Array of trading signals
        """
        df_feat = self.prepare_features(data)
        if df_feat.empty:
            return np.array([])
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(df_feat)

        df_scaled = self.normalize(df_feat)

        probs = self.model.predict_proba(df_scaled)
        signals = []

        for p in probs:
            if p[2] > BaseConfig.THRESHOLD:
                signals.append(1)   # Buy
            elif p[0] > BaseConfig.THRESHOLD:
                signals.append(-1)  # Sell
            else:
                signals.append(0)   # Hold

        # Padding for warmup period (if needed)
        padding = len(data) - len(signals)
        if padding > 0:
            signals = [0] * padding + signals

        return np.array(signals)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.scaler.fit(X)             
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

        
    
    '''
    previous version of model used in prelims
    '''

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

    # if __name__ == "__main__":
    #     import sys
    #     import warnings
    #     warnings.filterwarnings("ignore")

    # # Load the dataset
    # csv_path = Path(__file__).resolve().parents[3] / "experimental" / "datasets" / "btc_data_with_target_modified.csv"
    # if not csv_path.exists():
    #     print(f"CSV file not found at {csv_path}")
    #     sys.exit(1)

    # df = pd.read_csv(csv_path)

    # # Define config values manually (or mock BaseConfig for testing)
    # class MockBaseConfig:
    #     BUY_THRESHOLD = 0.7
    #     SELL_THRESHOLD = 0.3
    #     BUY_SIGNAL = 1
    #     SELL_SIGNAL = -1
    #     HOLD_SIGNAL = 0


    # # Inject temporary config for testing
    # BaseConfig.BUY_THRESHOLD = MockBaseConfig.BUY_THRESHOLD
    # BaseConfig.SELL_THRESHOLD = MockBaseConfig.SELL_THRESHOLD
    # BaseConfig.BUY_SIGNAL = MockBaseConfig.BUY_SIGNAL
    # BaseConfig.SELL_SIGNAL = MockBaseConfig.SELL_SIGNAL
    # BaseConfig.HOLD_SIGNAL = MockBaseConfig.HOLD_SIGNAL

    # # Define test features and target manually if needed
    # if not hasattr(XGBConfig, 'XGB_FEATURE_PATH') or isinstance(XGBConfig.XGB_FEATURE_PATH, str):
    #     # Assuming a column named 'target' exists
    #     XGBConfig.XGB_FEATURE_PATH = [col for col in df.columns if col != "target"]

    # # Initialize model
    # model = XGBoostModel()

    # # Run training & prediction test
    # print("Running basic training and prediction test...")

    # # Use a small subset for fast testing
    # df_subset = df.dropna().tail(300)
    # X, y = model.prepare_features(df_subset, "target")
    # X_scaled = model.preprocess(X)
    # model.train(X_scaled, y)
    # probs = model.predict(X)
    # signals = model.apply_thresholds(probs)

    # print("Sample predictions (first 10 signals):", signals[:10])

