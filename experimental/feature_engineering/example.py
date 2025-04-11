import pandas as pd
import numpy as np

import sys
import os

# Add the path to your 'src' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from features.technical_indicators import MovingAverageFeature, VolatilityFeature
from features.ml_features import HMMFeature
from hmmlearn.hmm import GaussianHMM

# Create a simple DataFrame with 'close' prices
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=100, freq="D")
df = pd.DataFrame({
    "close": np.random.normal(100, 5, size=100)
}, index=dates)

# Display first few rows of df
print(df.head())

# Define technical indicators (Moving Average, Volatility)
ma_feature = MovingAverageFeature(window="7d")  # 7-day moving average
volatility_feature = VolatilityFeature(window="7d")  # 7-day volatility

# Define HMM feature with a simple model (fit to close prices)
hmm_model = GaussianHMM(n_components=3, covariance_type="diag")
hmm_model.fit(df[['close']])  # Fit the model on close prices
hmm_feature = HMMFeature(model=hmm_model)

# Apply features to the DataFrame
print("Before MA feature:", df.shape)
df = ma_feature.transform(df)
print("After MA feature:", df.shape)

print("Before Volatility feature:", df.shape)
df = volatility_feature.transform(df)
print("After Volatility feature:", df.shape)

print("Before HMM feature:", df.shape)
df = hmm_feature.transform(df)
print("After HMM feature:", df.shape)

# Display first few rows of the updated df with all features
print(df.head())
