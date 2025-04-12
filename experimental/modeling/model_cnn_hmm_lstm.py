"""
CNN-LSTM-HMM Hybrid Model
File: experimental/modeling/cnn_lstm_hmm.py
"""

# Pipeline Overview:
# Raw Time Series Data
#     ↓
# Preprocessing (e.g., normalization, sliding window)
#     ↓
# CNN: Extracts local features/patterns from the data
#     ↓
# LSTM: Learns temporal dependencies across the sequence
#     ↓
# HMM: Models hidden states and captures regime switching behavior
#     ↓
# Final Output: Prediction or interpretation based on learned states

import os
from pathlib import Path
import joblib 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Reshape, LSTM
from tensorflow.keras.models import Model

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent 

DATA_PATH = PROJECT_ROOT / "experimental/datasets/btc_data.csv"
RESULTS_DIR = PROJECT_ROOT / "experimental/modeling/results/xgboost"
MODEL_DIR = PROJECT_ROOT / "experimental/modeling/models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEE_RATE = 0.0006  # 0.06% trading fee

class CNN_HMM_LSTM_Model:
    """
    A hybrid model combining CNN, LSTM, and HMM for time series prediction.
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def preprocess_data(self, df):
        cnn_data = df[['close', 'high', 'low', 'open', 'volume']]  # For CNN
        hmm_data = df[['exchange_whale_ratio', 'miner_supply_ratio', 'coinbase_premium_gap']]  # For HMM
        lstm_data = df[['addresses_count_active', 'long_liquidations', 'open_interest']]  # For LSTM

        # Normalize each group of features separately
        cnn_data = StandardScaler().fit_transform(cnn_data)
        hmm_data = StandardScaler().fit_transform(hmm_data)
        lstm_data = StandardScaler().fit_transform(lstm_data)

        # Train HMM and get hidden states
        hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag")
        hmm_model.fit(hmm_data)
        hmm_features = hmm_model.predict(hmm_data)

        return cnn_data, hmm_features, lstm_data

    def build_model(self):
        # Define the CNN model
        cnn_input = Input(shape=self.input_shape)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)

        # Use a custom layer to integrate HMM outputs (HMM is trained externally)
        hmm_input = Input(shape=(self.input_shape[0],))  # Assuming HMM gives 1D hidden states
        hmm_dense = Dense(10, activation='relu')(hmm_input)  # Transform HMM features if needed

        # Combine CNN and HMM outputs
        combined = Concatenate()([x, hmm_dense])

        combined = Dense(900)(combined)  # Project concatenated features into 900
        combined = Reshape((15, 60))(combined)  # Now reshape is valid

        # Define the output layer
        output = Dense(self.num_classes, activation='softmax')(combined)

        # Compile the model
        model = Model(inputs=[cnn_input, hmm_input], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the model.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return self.model
    
    def save_model(self):
        self.model.save(MODEL_DIR / "cnn_hmm_lstm_model.h5")

    def calculate_performance_metrics(self, signals, prices):
        returns = np.diff(prices) * signals[:-1]  # Assuming binary signals (1 for buy, 0 for hold)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        return {"Sharpe Ratio": sharpe_ratio}


    def split_data_by_date(df):
        # Convert datetime column to pandas datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Split into train, validation, and test sets
        train_data = df[(df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2022-12-31')]
        val_data = df[(df['datetime'] >= '2023-01-01') & (df['datetime'] <= '2023-12-31')]
        test_data = df[(df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2025-03-31')]
        
        print(f"Train data: {train_data.shape}")
        print(f"Validation data: {val_data.shape}")
        print(f"Test data: {test_data.shape}")

        return train_data, val_data, test_data

def main():
    # Set input shape and number of classes
    input_shape = (30, 5)  # Example input shape
    num_classes = 3        # Example: 3 output classes
    
    # Initialize the model
    model_instance = CNN_HMM_LSTM_Model(input_shape=input_shape, num_classes=num_classes)
    
    # Load data
    data = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

    # Preprocess features and labels
    cnn_data, hmm_features, lstm_data = model_instance.preprocess_data(data)
    labels = df['target'].values  # Replace 'target' with your actual label column
    
    # Train the model
    model_instance.train([cnn_data, hmm_features, lstm_data], labels)

    # Evaluate the model
    loss, accuracy = model_instance.model.evaluate([cnn_data, hmm_features, lstm_data], labels)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Save the model
    model_instance.save_model()

if __name__ == "__main__":
    main()






