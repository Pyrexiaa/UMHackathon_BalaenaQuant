from quantpilot.visualization.run import run_dashboard
from sample_data import BTC_DATA
from quantpilot.strategy import MLStrategy, HMMStrategy, CombinedStrategy
from quantpilot.multibacktester import Multibacktester
from quantpilot.models import get_model

# bt = Multibacktester(
#     data=BTC_DATA,
#     strategies=[MLStrategy(get_model("TCN")), MLStrategy(get_model("XGBoost"))],
#     strategy_names=["TCN", "XGBoost"],
#     mode="geometric", 
#     entry_exit_logic="mean_reversion"
# )

bt = Multibacktester(
    data=BTC_DATA,
    strategies=[
        MLStrategy(get_model("XGBoost")),
        HMMStrategy(),
        CombinedStrategy(
            ml_strategy=MLStrategy(get_model("XGBoost")),
            hmm_strategy=HMMStrategy(),
            method="and"
        ),
        CombinedStrategy(
            ml_strategy=MLStrategy(get_model("XGBoost")),
            hmm_strategy=HMMStrategy(),
            method="vote"
        ),
    ],
    strategy_names=["XGBoost", "HMM", "Combined (AND)", "Combined (Vote)"],
    mode="geometric", 
    entry_exit_logic="mean_reversion",
    threshold_values=[0.35, None, 0.35, 0.35],
)

bt.run_all(backtest_end_date="2023-12-31")  # Run backtest only
# bt.run_all(forward_test=True, forward_start_date="2024-01-01")
bt.plot_permutation_heatmap(
    metric="Sharpe Ratio",
    show_plot=True,
    save_path="output/multi_backtest_heatmap.png"
)
# Launch Streamlit dashboard
# run_dashboard()
