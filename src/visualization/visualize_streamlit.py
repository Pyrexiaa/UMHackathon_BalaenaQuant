# visualize_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Backtest Visualization")

# Load backtest results
try:
    df = pd.read_csv("output/backtest_results.csv", parse_dates=True, index_col=0)
except FileNotFoundError:
    st.error("No backtest results found. Run the backtest first.")
    st.stop()

# Plot equity curve
st.subheader("Equity Curve")
st.line_chart(df['total'])

# Plot drawdown
st.subheader("Drawdown")
st.area_chart(df['drawdown'])

# Plot trading signals
st.subheader("Trading Signals")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['price'], label='Price', color='black', alpha=0.5)

buy_signals = df[df['signal'] == 2]
sell_signals = df[df['signal'] == 0]

ax.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', label='Buy', alpha=1)
ax.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', label='Sell', alpha=1)
ax.legend()
ax.set_title("Trading Signals")
ax.set_ylabel("Price")
st.pyplot(fig)
