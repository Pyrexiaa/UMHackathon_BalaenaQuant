import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import json

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

        # Load metadata and slice periods
        meta_path = filepath.replace("portfolio_", "meta_").replace(".csv", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            backtest_start = pd.to_datetime(meta['backtest']['start'])
            backtest_end = pd.to_datetime(meta['backtest']['end'])
            forward_start = pd.to_datetime(meta['forward']['start']) if 'forward' in meta else None
            forward_end = pd.to_datetime(meta['forward']['end']) if 'forward' in meta else None

            df_backtest = df.loc[backtest_start:backtest_end]
            df_forward = df.loc[forward_start:forward_end] if forward_start else pd.DataFrame()
        else:
            df_backtest = df
            df_forward = pd.DataFrame()
            st.warning("Metadata file not found. Showing only full-period data.")

        # Equity Curve
        if 'equity' in df.columns:
            st.markdown("### 📈 Equity Curve (Backtest)")
            if not df_backtest.empty:
                st.line_chart(df_backtest['equity'])

            st.markdown("### 📈 Equity Curve (Forward Test)")
            if not df_forward.empty:
                st.line_chart(df_forward['equity'])

        # Drawdown
        if 'drawdown' in df.columns:
            st.markdown("### 📉 Drawdown (Backtest)")
            if not df_backtest.empty:
                st.area_chart(df_backtest['drawdown'])

            st.markdown("### 📉 Drawdown (Forward Test)")
            if not df_forward.empty:
                st.area_chart(df_forward['drawdown'])

        # Trading Signals (full period view)
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
