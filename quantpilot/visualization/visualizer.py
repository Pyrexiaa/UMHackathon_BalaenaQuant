# visualize_streamlit.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

st.set_page_config(layout="wide")
st.title("Multistrategy Backtest Visualization")

# Find all portfolio CSVs in the output folder
portfolio_files = sorted(glob.glob("output/portfolio_*.csv"))

if not portfolio_files:
    st.error("No portfolio CSV files found in the 'output' directory.")
    st.stop()

# Create one tab per strategy
strategy_tabs = st.tabs([os.path.basename(f).replace("portfolio_", "").replace(".csv", "") for f in portfolio_files])

for tab, filepath in zip(strategy_tabs, portfolio_files):
    with tab:
        strategy_name = os.path.basename(filepath).replace("portfolio_", "").replace(".csv", "")
        st.subheader(f"Strategy: {strategy_name}")

        df = pd.read_csv(filepath, parse_dates=True, index_col=0)

        # Equity Curve
        st.markdown("### 📈 Equity Curve")
        st.line_chart(df['equity'])

        # Drawdown
        if 'drawdown' in df.columns:
            st.markdown("### 📉 Drawdown")
            st.area_chart(df['drawdown'])

        # Trading Signals
        if 'price' in df.columns and 'signal' in df.columns:
            st.markdown("### 🔄 Trading Signals")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df['price'], label='Price', color='black', alpha=0.5)

            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]

            ax.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', label='Buy', alpha=1)
            ax.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', label='Sell', alpha=1)

            ax.set_title("Trading Signals")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
