import pandas as pd
from typing import Any
from .base_strategy import BaseStrategy


class HMMStrategy(BaseStrategy):
    """
    Strategy that maps HMM state to trading signals:
    - States 0 or 2 → Sell (-1)
    - States 1 or 3 → Buy (+1)
    - State 4 → Hold (0)
    """

    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on HMM state classification.

        :param X: DataFrame that includes a 'hmm_state' column
        :return: pd.Series of trading signals with values -1, 0, 1
        """
        hmm_state = X['hmm_state']
        
        # signal = hmm_state.map({
        #     0: -1,  # Sell
        #     2: -1,  # Sell
        #     1:  1,  # Buy
        #     3:  1,  # Buy 
        #     4:  0   # Hold
        # })
        signals = pd.Series(0, index=X.index)
        
        signals[(hmm_state == 0) & (X['miner_supply_ratio'] < -1) & (X['transactions_count_inflow'] < 0) & (X['exchange_whale_ratio'] > 1) & (X['exchange_supply_ratio'] > 1)] = -1
        signals[(hmm_state == 1) & (X['miner_supply_ratio'] > 1)] = 1
        signals[(hmm_state == 2) & (X['miner_supply_ratio'] < 0) & (X['transactions_count_inflow'] < 0) ] = -1
        signals[(hmm_state == 3) & (X['exchange_supply_ratio'] < 0)] = 1
        signals[hmm_state == 4] = 0
        
        # Ensure index matches original input
        # return pd.Series(signal.values, index=X.index)
        return signals
