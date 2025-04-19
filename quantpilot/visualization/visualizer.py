import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import glob

import seaborn as sns

def generate_metric_heatmap(strategies, metric: str = "Sharpe Ratio"):
    """
    Generates a heatmap comparing the selected metric across strategies.
    Returns the matplotlib figure object.
    """
    metric_rows = []
    for strat in strategies:
        try:
            value = strat['meta']['metrics']['full'].get(metric)
            if value is not None:
                metric_rows.append({
                    "Strategy": strat['name'],
                    metric: value
                })
        except Exception as e:
            continue

    if not metric_rows:
        return None

    df = pd.DataFrame(metric_rows)
    df.set_index("Strategy", inplace=True)

    fig, ax = plt.subplots(figsize=(8, len(df) * 0.6 + 1))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title(f"{metric} Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Strategy")
    fig.tight_layout()
    return fig


def run_dashboard():
    st.set_page_config(layout="wide")
    st.title("Multistrategy Backtest Visualization")

    meta_files = sorted(glob.glob("output/meta_*.json"))
    if not meta_files:
        st.error("No metadata files found. Please run a backtest first.")
        st.stop()

    strategies = []
    for meta_file in meta_files:
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)

            strategy_name = os.path.basename(meta_file).replace("meta_", "").replace(".json", "")
            portfolio_file = f"output/portfolio_{strategy_name}.csv"
            if not os.path.exists(portfolio_file):
                continue

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

    strategy_tabs = st.tabs([s["name"] for s in strategies])
    for tab, strat in zip(strategy_tabs, strategies):
        with tab:
            st.subheader(f"Strategy: {strat['name']}")
            df = pd.read_csv(strat["portfolio_path"], parse_dates=True, index_col=0)
            meta = strat["meta"]

            backtest_start = pd.to_datetime(meta['backtest']['start'])
            backtest_end = pd.to_datetime(meta['backtest']['end'])
            forward_start = pd.to_datetime(meta.get('forward', {}).get('start')) if 'forward' in meta else None
            forward_end = pd.to_datetime(meta.get('forward', {}).get('end')) if 'forward' in meta else None

            df_backtest = df.loc[backtest_start:backtest_end]
            df_forward = df.loc[forward_start:forward_end] if forward_start else pd.DataFrame()

            if 'equity' in df.columns:
                st.markdown("### 📈 Equity Curve (Backtest)")
                if not df_backtest.empty:
                    st.line_chart(df_backtest['equity'])

                st.markdown("### 📈 Equity Curve (Forward Test)")
                if not df_forward.empty:
                    st.line_chart(df_forward['equity'])

            if 'drawdown' in df.columns:
                st.markdown("### 📉 Drawdown (Backtest)")
                if not df_backtest.empty:
                    st.area_chart(df_backtest['drawdown'])

                st.markdown("### 📉 Drawdown (Forward Test)")
                if not df_forward.empty:
                    st.area_chart(df_forward['drawdown'])

            if 'price' in df.columns and 'signal' in df.columns:
                st.markdown("### 🔄 Trading Signals")
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df['price'], label='Price', color='black', alpha=0.5)
                ax.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['price'], color='green', marker='^', label='Buy')
                ax.scatter(df[df['signal'] == -1].index, df[df['signal'] == -1]['price'], color='red', marker='v', label='Sell')
                ax.set_title("Trading Signals")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)
                
    # 🔥 Metric Heatmap Comparison
    st.markdown("## 📊 Strategy Metric Comparison")

    # Gather all unique metric keys from available strategies
    available_metrics = set()
    print("Strat dict: ", strategies)
    for strat in strategies:
        try:
            available_metrics.update(strat['meta']['metrics']['full'].keys())
        except Exception:
            continue

    if not available_metrics:
        st.info("No metrics found for heatmap.")
    else:
        selected_metric = st.selectbox("Choose metric to compare:", sorted(available_metrics))

        fig = generate_metric_heatmap(strategies, selected_metric)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Could not generate heatmap for selected metric.")

    for meta_file in meta_files:
        try:
            os.remove(meta_file)
        except Exception as e:
            st.warning(f"Could not delete {meta_file}: {e}")

if __name__ == "__main__":
    run_dashboard()
