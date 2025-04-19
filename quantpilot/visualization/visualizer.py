import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import glob

st.set_page_config(layout="wide")
st.title("Multistrategy Backtest Visualization")

# Read all meta_*.json files to get run metadata
meta_files = sorted(glob.glob("output/meta_*.json"))

if not meta_files:
    st.error("No metadata files found. Please run a backtest first.")
    st.stop()

# Extract portfolio paths from metadata files
strategies = []
for meta_file in meta_files:
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        
        strategy_name = os.path.basename(meta_file).replace("meta_", "").replace(".json", "")
        portfolio_file = f"output/portfolio_{strategy_name}.csv"
        if not os.path.exists(portfolio_file):
            continue  # skip if portfolio file is missing
        
        strategies.append({
            "name": strategy_name,
            "meta_path": meta_file,
            "portfolio_path": portfolio_file,
            "meta": meta
        })
    except Exception as e:
        st.warning(f"Error reading metadata file {meta_file}: {e}")

if not strategies:
    st.error("No valid strategy data found.")
    st.stop()

# Create a tab for each strategy
strategy_tabs = st.tabs([s["name"] for s in strategies])

for tab, strat in zip(strategy_tabs, strategies):
    with tab:
        st.subheader(f"Strategy: {strat['name']}")

        df = pd.read_csv(strat["portfolio_path"], parse_dates=True, index_col=0)
        meta = strat["meta"]

        # Backtest and forward periods
        backtest_start = pd.to_datetime(meta['backtest']['start'])
        backtest_end = pd.to_datetime(meta['backtest']['end'])
        forward_start = pd.to_datetime(meta['forward']['start']) if 'forward' in meta else None
        forward_end = pd.to_datetime(meta['forward']['end']) if 'forward' in meta else None

        df_backtest = df.loc[backtest_start:backtest_end]
        df_forward = df.loc[forward_start:forward_end] if forward_start else pd.DataFrame()

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

# Cleanup: delete all meta_*.json files after displaying
for meta_file in meta_files:
    try:
        os.remove(meta_file)
    except Exception as e:
        st.warning(f"Could not delete {meta_file}: {e}")
