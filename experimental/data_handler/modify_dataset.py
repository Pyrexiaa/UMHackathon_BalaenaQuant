import numpy as np
import pandas as pd

df = pd.read_csv("experimental/datasets/btc_data.csv")

fee = 0.0006
df['future_return'] = df['close'].pct_change().shift(-1)
df['positions'] = np.select(
    [df['future_return'] > fee, df['future_return'] < -fee],
    [2, 0],  # Buy=2, Sell=0
    default=1  # Hold=1
)
df['trades'] = df['positions']

previous_trade = 1
for i in range(0, len(df)):
    if previous_trade == 2:
        if df.loc[i, 'trades'] == 2:
            df.loc[i, 'trades'] = 1
        # Found the change
        if df.loc[i, 'trades'] == 0:
            previous_trade = 0
            
    elif previous_trade == 0:
        if df.loc[i, 'trades'] == 0:
            df.loc[i, 'trades'] = 1
        # Found the change
        if df.loc[i, 'trades'] == 2:
            previous_trade = 2

    # Find the first target
    elif previous_trade == 1:
        if df.loc[i, 'trades'] == 2:
            previous_trade = 2
        if df.loc[i, 'trades'] == 0:
            previous_trade = 0
    

df.to_csv("experimental/datasets/btc_data_with_target_modified.csv", index=False)