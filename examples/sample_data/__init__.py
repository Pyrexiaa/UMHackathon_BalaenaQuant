import pandas as pd
from os.path import join, dirname
       
def _read_file(filename):
    return pd.read_csv(join(dirname(__file__), filename), index_col=0, parse_dates=True)

<<<<<<< HEAD
BTC_DATA = _read_file('btc_data_with_target_latest_v2.csv')
=======
# BTC_DATA = _read_file('btc_data.csv')
# BTC_DATA = _read_file('btc_data_with_target_technical_hmm_kmeans.csv')
# BTC_DATA = _read_file('btc_data_with_target_latest_v2.csv')
# BTC_DATA = _read_file('btc_data_with_target_latest_v2_copy.csv')
BTC_DATA = _read_file('btc_data_with_target_latest_v3.csv')
>>>>>>> 3fe2a1d673196adbc0912aa70abf5314abcb2abe
