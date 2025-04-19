import pandas as pd
import torch
from experimental.modeling.constants import ASSUMPTION_10, ASSUMPTION_9
import joblib
import numpy as np
from ..base_model import BaseModel
from ...config import BaseConfig, TCNConfig
from .model_architecture import TCNClassifier


class TCNModel(BaseModel):
    """
    A Temporal Convolution Network (TCN) model used to generate signals.
    """

    def __init__(
        self,
        model_path=TCNConfig.TCN_MODEL_PATH,
        scaler_path=TCNConfig.TCN_SCALER_PATH,
        device=None,
    ):
        """
        Initialize the TCN model by loading the model and scaler.

        :param model_path: Path to the trained model
        :param scaler_path: Path to the scaler used for feature normalization
        :param device: The device on which to run the model (e.g., 'cuda' or 'cpu')
        """
        self.scaler = joblib.load(scaler_path)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        input_features = 10  # Specify your input features
        num_channels = [128, 64, 128]  # Example for TCN channels
        self.model = TCNClassifier(input_features, 3, num_channels).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state"]
        )
        self.model.eval()  # Set the model to evaluation mode

    def prepare_features(self, df):
        """
        Prepare and engineer features required for model prediction.

        :param df: The raw input DataFrame containing market data
        :return: DataFrame with engineered features
        """

        features = ASSUMPTION_10.copy()
        if "target" in features:
            features.remove("target")

        # Select and clean the relevant columns
        df = df[[f for f in features if f in df.columns]].reset_index(drop=True)
        return df

    def normalize(self, df):
        """
        Normalize the input DataFrame using the pre-trained scaler.

        :param df: The input DataFrame
        :return: The normalized DataFrame
        """
        scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(scaled, columns=df.columns)
        return df_scaled

    def preprocess(self, df):
        """
        Preprocess the input DataFrame into a time-series window format.

        :param df: The normalized DataFrame
        :return: Numpy array representing time-series windows
        """
        X = []
        for i in range(BaseConfig.WINDOW_SIZE, len(df)):
            window = df.iloc[i - BaseConfig.WINDOW_SIZE : i].values
            X.append(window)
        return np.array(X)

    def predict(self, data, threshold: float = None):
        """
        Make predictions based on raw input data.

        :param data: Raw input data as a DataFrame
        :param threshold: Optional threshold for buy/sell signals
        :return: Numpy array of predicted signals
        """
        if threshold is None:
            threshold = BaseConfig.THRESHOLD
        
        df_feat = self.prepare_features(data)
        df_scaled = self.normalize(df_feat)
        X = self.preprocess(df_scaled)

        if len(X) == 0:
            return np.array([]), np.array([])

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            print(logits.shape)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        signals = []

        # The model's predictions will be shorter than input due to windowing
        window_size = getattr(self.model, "window_size", BaseConfig.WINDOW_SIZE)
        start_idx = window_size

        # Pad signals to match input data length
        total_len = len(df_scaled)

        # Initialize with HOLD signals for the warmup period
        for i in range(start_idx):
            signals.append(0)

        for i, p in enumerate(probs, start=start_idx):
            if i >= total_len:
                break  # Handle case where we have extra predictions
            p = np.array(p)

            # if p[2] > threshold:
            #     signals.append(1)  # Buy
            # elif p[0] > threshold:
            #     signals.append(-1) # Sell

            # Apply threshold rules
            if p[0] > BaseConfig.THRESHOLD and p[2] <= BaseConfig.THRESHOLD:
                signals.append(1)  # Buy
            elif p[2] > BaseConfig.THRESHOLD and p[0] <= BaseConfig.THRESHOLD:
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Hold

        # for i, p in enumerate(probs):
        #     max_class = np.argmax(p)  # Get the class with highest probability
            
        #     # Print probabilities with class labels
        #     print(f"{p[0]:.4f}\t\t{p[1]:.4f}\t\t{p[2]:.4f}\t\t{max_class} ({'Sell' if max_class==0 else 'Hold' if max_class==1 else 'Buy'})")
            
        #     if max_class == 0:    # Sell class
        #         signals.append(-1)
        #     elif max_class == 1:  # Hold class
        #         signals.append(0)
        #     elif max_class == 2:  # Buy class
        #         signals.append(1)

            
        df_probs = pd.DataFrame(probs, columns=['prob_sell', 'prob_hold', 'prob_buy'])
        df_probs['predicted_signal'] = signals[start_idx:]
        df_probs.to_csv("latest_tcn_predicted_signals.csv", index=False)
        return np.array(signals)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
