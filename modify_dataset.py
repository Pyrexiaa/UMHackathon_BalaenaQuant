import numpy as np
import pandas as pd

df = pd.read_csv("experimental/datasets/btc_data.csv")

fee = 0.0006
df['future_return'] = df['close'].pct_change().shift(-1)
df['target'] = np.select(
    [df['future_return'] > fee, df['future_return'] < -fee],
    [2, 0],  # Buy=2, Sell=0
    default=1  # Hold=1
)

df.to_csv("experimental/datasets/btc_data_with_target.csv", index=False)