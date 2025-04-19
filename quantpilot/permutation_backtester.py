import pandas as pd
from typing import List, Dict, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
from quantpilot.backtester import Backtester
from quantpilot.strategy import BaseStrategy, MLStrategy

class BacktesterPermutationTest:
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: BaseStrategy,
                 threshold_values: List[float],
                 strategy_name: Optional[str] = None,
                 initial_capital: float = 100000,
                 risk_free_rate: float = 0.0,
                 trading_fee: float = 0.0006,
                 qty_per_trade: int = 1,
                 mode: str = 'arithmetic',
                 entry_exit_logic: str = 'mean_reversion',
                 output_dir: str = "output"):
        """
        Initialize BacktesterPermutationTest to loop through various thresholds and run backtests.
        """
        self.data = data
        self.strategy = strategy
        self.strategy_name = strategy_name if strategy_name else f"{strategy.__class__.__name__}_Permutation_Test"
        self.threshold_values = threshold_values
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.qty_per_trade = qty_per_trade
        self.mode = mode
        self.entry_exit_logic = entry_exit_logic
        self.output_dir = output_dir
        self.results_summary = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_all(self):
        """
        Run the backtest for each threshold value.
        """
        for threshold in self.threshold_values:
            strategy_name = f"{self.strategy_name}_Threshold_{threshold}"
            print(f"\n=== Running backtest with threshold: {threshold} ===")
            
            backtester = Backtester(
                data=self.data,
                strategy=self.strategy,
                strategy_name=strategy_name,
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
                trading_fee=self.trading_fee,
                qty_per_trade=self.qty_per_trade,
                mode=self.mode,
                entry_exit_logic=self.entry_exit_logic,
                threshold=threshold
            )

            backtester.run()

            # Plot and save results
            # backtester.plot_results()

            self.results_summary[threshold] = backtester.performance_metrics()

    def export_summary(self, summary_filename: str = "threshold_comparison_summary.csv"):
        """
        Export summary of performance metrics across all thresholds.
        """
        summary_df = pd.DataFrame.from_dict(self.results_summary, orient='index')
        summary_df.index.name = 'Threshold'
        path = os.path.join(self.output_dir, summary_filename)
        summary_df.to_csv(path)
        print(f"\nSummary of all thresholds saved to {path}")
        return summary_df

    def plot_permutation_heatmap(self, metric: str = "Sharpe Ratio", save_path: Optional[str] = None, show_plot: bool = True) -> pd.DataFrame:
        """
        Extract a metric (e.g. Sharpe Ratio), plot it as a heatmap, and return a labeled DataFrame for merging.

        :param metric: The metric to visualize (default = Sharpe Ratio).
        :param save_path: Optional custom path to save the heatmap image.
        :param show_plot: Whether to display the plot interactively.
        :return: A DataFrame with Threshold as index and strategy_name as column.
        """
        if not self.results_summary:
            raise ValueError("No results available. Please run `run_all()` first.")

        # Extract metric (e.g. Sharpe Ratio) per threshold
        metric_data = {
            threshold: metrics.get(metric, None)
            for threshold, metrics in self.results_summary.items()
        }

        # Create single-column DataFrame and sort
        df = pd.DataFrame.from_dict(metric_data, orient='index', columns=[self.strategy_name])
        df.index.name = 'Threshold'
        df = df.sort_index()

        # Plot heatmap for this strategy
        plt.figure(figsize=(6, len(df) * 0.5 + 1))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"{self.strategy_name}_Permutation_Test - {metric} Heatmap for Thresholds")
        plt.xlabel("Metric")
        plt.ylabel("Thresholds")
        plt.tight_layout()

        # Save figure
        heatmap_path = save_path or os.path.join(
            self.output_dir, f"{self.strategy_name.lower()}_{metric.lower().replace(' ', '_')}_heatmap.png"
        )
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to: {heatmap_path}")

        if show_plot:
            plt.show()

        return df  # For combining into a full strategy x threshold heatmap


