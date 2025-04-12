import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.xgboost.xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split

# ----- Step 1: Create toy features and labels -----
np.random.seed(42)
df = pd.DataFrame({
    "feature_1": np.random.randn(100),
    "feature_2": np.random.randn(100),
    "feature_3": np.random.randn(100),
})

# Let's say our target is binary classification (buy=1, hold/sell=0)
df["target"] = np.random.choice([0, 1], size=100)

# ----- Step 2: Split into features and label -----
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

# ----- Step 3: Train/test split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Step 4: Use your custom XGBoost model -----
model = XGBoostModel(max_depth=3, n_estimators=100)
model.fit(X_train, y_train)

# ----- Step 5: Predict and save -----
preds = model.predict(X_test)
print("Predictions:", preds.tolist())

model.save_model("experimental/modeling/saved_model/xgb_model.pkl")

model2 = XGBoostModel()
model2.load_model("experimental/modeling/saved_model/xgb_model.pkl")
preds2 = model2.predict(X_test)

