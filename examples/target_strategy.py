import pandas as pd
# from sample_data import BTC_DATA

BTC_DATA = pd.read_csv('correct/path/to/your/btc_data_with_target_latest_v2_copy.csv')

# Assuming BTC_DATA is already loaded and has a 'datetime' column
BTC_DATA['datetime'] = pd.to_datetime(BTC_DATA['datetime'], dayfirst=True)

# Create a mask for June, July, August
summer_months_mask = BTC_DATA['datetime'].dt.month.isin([6, 7, 8])

# Define the remapping function
def remap_target(val):
    if val == 2:
        return 1
    elif val == 0:
        return 2
    elif val == 1:
        return 0
    return val

# Apply the remapping only where the date is in summer months
BTC_DATA.loc[summer_months_mask, 'target'] = BTC_DATA.loc[summer_months_mask, 'target'].apply(remap_target)

# (Optional) Save it if you want
BTC_DATA.to_csv('btc_data_updated_target_strategy.csv', index=False)
