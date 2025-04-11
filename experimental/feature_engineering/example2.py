#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoAlphaLab - Feature Engineering Demo

This script demonstrates the feature engineering capabilities with HMM regime detection.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add the path to your 'src' folder dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.feature_engineering import FeatureTechnicalIndicators


# Load sample data
df = pd.read_csv('merged_features.csv', parse_dates=['datetime'], index_col='datetime')

# Display data info
print("\nData preview:")
print(df.head())

# 2. Basic technical indicators
print("\n2. Adding basic technical indicators...")
df_tech = df.copy()
feature_technical = FeatureTechnicalIndicators(df_tech, price_col='close')

# Add technical indicators
feature_technical.add_ema([5, 8, 13])
feature_technical.add_sma([5, 20])
feature_technical.add_rsi([14])
feature_technical.add_macd(fast_window=12, slow_window=26, signal_window=9)
feature_technical.add_price_change([1])

# Show added features by comparing with original DataFrame
added_features = set(df_tech.columns) - set(df.columns)
print(f"Added {len(added_features)} features: {', '.join(added_features)}")

print(df_tech.head())
print(df_tech[['close'] + list(added_features)].head())

