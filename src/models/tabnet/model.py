import pandas as pd
import torch
import joblib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from ..base_model import BaseModel
from ...config import BaseConfig, TabNetConfig


class TabNetModel(BaseModel):
    """
    A TabNet model used to generate classification signals from tabular data.
    """

    def __init__(self, model_path=TabNetConfig.TABNET_MODEL_PATH, scaler_path=TabNetConfig.TABNET_SCALER_PATH, feature_names_path=TabNetConfig.TABNET_FEATURE_NAMES_PATH, device=None):
        """
        Initialize the TabNet model by loading the model, scaler, and feature names.

        :param model_path: Path to the trained TabNet model
        :param scaler_path: Path to the feature scaler
        :param feature_names_path: Path to the saved feature names
        :param device: Device to use for inference ('cuda' or 'cpu')
        """
        self.scaler = joblib.load(scaler_path)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TabNetClassifier()
        self.model.load_model(model_path)

        # Load feature names from the saved file
        try:
            with open(TabNetConfig.TABNET_FEATURE_NAMES_PATH, "rb") as f:
                self.feature_names = json.load(f)
        except Exception as e:
            print(f"Error loading feature names: {e}")
            raise e

    def prepare_features(self, df):
        """
        Engineer and select features required for model prediction.

        :param df: Raw input DataFrame
        :return: Processed DataFrame
        """
        df = df.copy()
    
        features = [
            'delta_estimated_leverage_ratio',
            'whale_ratio_diff',
            'close_7d_ma',
            'close_30d_std',
            'volume_zscore',
            'taker_buy_ratio',
            'open_interest',
            'close', 
            'open', 
            'high', 
            'low', 
            'volume'
        ]

        df = df[[f for f in features if f in df.columns]].dropna().reset_index(drop=True)
        return df

    def normalize(self, df):
        """
        Normalize input features using the pre-trained scaler.

        :param df: Input DataFrame
        :return: Scaled DataFrame
        """
        scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(scaled, columns=df.columns)
        return df_scaled

    def predict(self, data):
        """
        Predict probabilities using the TabNet model, without feature preparation and scaling.

        :param data: Raw input DataFrame
        :return: Numpy array of predicted probabilities
        """
        # Ensure the data contains the correct features (feature selection)
        df_feat = data[self.feature_names]

        if df_feat.empty:
            return np.array([]), np.array([])

        # Pass the raw features to the model for prediction
        X = df_feat.values  # Use the raw feature values (no scaling)

        # Perform the prediction
        probs = self.model.predict_proba(X)
        return probs


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the TabNet model and save both model and scaler.
        """
        # Prepare features and normalize the data
        X_scaled = self.normalize(X)

        self.model.fit(
            X_train=X_scaled.values,
            y_train=y.values,
            eval_set=[(X_scaled.values, y.values)],
            eval_name=["train"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=512,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        # Save model and scaler
        self.model.save_model(TabNetConfig.TABNET_MODEL_PATH)
        joblib.dump(self.scaler, TabNetConfig.TABNET_SCALER_PATH)

        # Save the feature names for consistency
        try:
            with open(TabNetConfig.TABNET_FEATURE_NAMES_PATH, "wb") as f:
                joblib.dump(X.columns.tolist(), f)
        except Exception as e:
            print(f"Error saving feature names: {e}")
            raise e


# import os
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from pytorch_tabnet.tab_model import TabNetClassifier

# # === Configs ===
# FEE_RATE = 0.006
# MODEL_DIR = "src/models_weights/tabnet"
# MODEL_PATH = os.path.join(MODEL_DIR, "model")
# SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# # === Load Data ===
# df = pd.read_csv("experimental/datasets/btc_data_with_target_technical_hmm_kmeans.csv")

# # === Feature Engineering ===
# if 'target' not in df.columns:
#     df['future_return'] = df['close'].pct_change().shift(-1)
#     df['target'] = 1  # Default: neutral
#     df.loc[df['future_return'] > FEE_RATE, 'target'] = 2  # Long
#     df.loc[df['future_return'] < -FEE_RATE, 'target'] = 0  # Short

# # === Feature Columns ===
# features = [
#     'future_return', 'price_change_1', 'ema_5_8_13_cross', 'taker_sell_ratio',
#     'taker_buy_ratio', 'taker_buy_sell_ratio', 'rsi_14', 'rsi_obv_signal_14',
#     'bb_signal_20', 'coinbase_premium_index_usdt_adjusted', 'macd_signal_flag',
#     'coinbase_premium_gap_usdt_adjusted', 'macd_trade_signal', 'macd',
#     'addresses_count_sender', 'addresses_count_active', 'blockreward',
#     'tokens_transferred_mean', 'long_liquidations', 'addresses_count_receiver'
# ]

# # === Prepare Data ===
# df = df.dropna(subset=features + ['target'])
# X = df[features]
# y = df['target'].astype(int)

# # === Normalize ===
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # === Split ===
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # === Train TabNet ===
# clf = TabNetClassifier()
# clf.fit(
#     X_train=X_train, y_train=y_train.values,
#     eval_set=[(X_val, y_val.values)],
#     max_epochs=100, patience=10,
#     batch_size=256, virtual_batch_size=128
# )

# # === Save Model & Scaler ===
# os.makedirs(MODEL_DIR, exist_ok=True)
# clf.save_model(MODEL_PATH)  # Saves to model.pth
# joblib.dump(scaler, SCALER_PATH)

# print("✅ TabNet model and scaler saved successfully!")
