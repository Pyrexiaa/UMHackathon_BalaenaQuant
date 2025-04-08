import pandas as pd

class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        commission_rate: float,  
        slippage: float,        
        trailing_stop_pct: float,  
        holding_period: int  
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trailing_stop_pct = trailing_stop_pct
        self.holding_period = holding_period

    def simulate_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        trades = signals.copy()
        trades = trades.reset_index(drop=True)

        num_trades = len(trades)
        if num_trades == 0:
            print("⚠️ No trades to simulate.")
            return trades

        trades['position_size'] = self.initial_capital / num_trades

        exit_prices = []
        for idx, row in trades.iterrows():
            entry = row['entry_price']
            sl = row['stop_loss']
            tp = row['take_profit']
            direction = row['trade_type']

            # Simulate idealized price path (assumes SL/TP can be hit in candle)
            # You can replace this with actual OHLC data logic if available
            exit_price = tp if direction == 'BUY' else tp
            exit_reason = "TP"

            if self.trailing_stop_pct is not None:
                # Trailing stop-loss simulated simply
                if direction == 'BUY':
                    trailing_sl = entry * (1 - self.trailing_stop_pct)
                    if trailing_sl > sl:
                        sl = trailing_sl
                elif direction == 'SELL':
                    trailing_sl = entry * (1 + self.trailing_stop_pct)
                    if trailing_sl < sl:
                        sl = trailing_sl

            # Let's simulate SL hit before TP for conservativeness
            if direction == 'BUY' and sl >= exit_price:
                exit_price = sl
                exit_reason = "SL"
            elif direction == 'SELL' and sl <= exit_price:
                exit_price = sl
                exit_reason = "SL"

            # Holding period enforced (simulate flat exit)
            if self.holding_period is not None:
                # Simulate flat exit after holding period
                hold_exit = entry * (1 + 0.002) if direction == 'BUY' else entry * (1 - 0.002)
                exit_price = hold_exit
                exit_reason = "TIME"

            # Apply commission and slippage
            entry_adj = entry * (1 + self.slippage + self.commission_rate)
            exit_adj = exit_price * (1 - self.slippage - self.commission_rate)

            # PnL calculation
            if direction == 'BUY':
                pnl = exit_adj - entry_adj
            elif direction == 'SELL':
                pnl = entry_adj - exit_adj
            else:
                pnl = 0

            trades.at[idx, 'exit_price'] = exit_price
            trades.at[idx, 'exit_reason'] = exit_reason
            trades.at[idx, 'entry_adj'] = entry_adj
            trades.at[idx, 'exit_adj'] = exit_adj
            trades.at[idx, 'pnl'] = pnl
            trades.at[idx, 'return'] = pnl / trades.at[idx, 'position_size']

        return trades
