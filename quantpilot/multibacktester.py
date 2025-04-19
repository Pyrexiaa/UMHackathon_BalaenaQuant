import os
from typing import List, Dict, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .backtester import Backtester  

class Multibacktester:
    def __init__(self, data: pd.DataFrame, strategies: List, strategy_names: Optional[List[str]] = None, threshold_values: Optional[List[float]] = None,
                 initial_capital: float = 100000, risk_free_rate: float = 0.0, trading_fee: float = 0.0006,
                 qty_per_trade: int = 1, mode: str = 'arithmetic', entry_exit_logic: str = 'trend_following',
                 output_dir: str = "output"):
        """
        Initialize the strategy tester with multiple strategies.
        
        :param data: The price data to backtest on.
        :param strategies: A list of strategy instances (each must implement generate_signals()).
        :param strategy_names: Optional list of names corresponding to each strategy.
        :param initial_capital: Initial capital for the backtest.
        :param risk_free_rate: Risk-free rate for performance metrics.
        :param trading_fee: Trading fee for each transaction.
        :param qty_per_trade: Quantity per trade.
        :param mode: Mode of the strategy ('arithmetic' or 'geometric').
        :param entry_exit_logic: Logic for entry and exit signals ('trend_following' or 'mean_reversion').
        :param output_dir: Directory to save output files.
        """
        self.data = data
        self.strategies = strategies
        self.strategy_names = strategy_names or [f"Strategy_{i+1}" for i in range(len(strategies))]
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.qty_per_trade = qty_per_trade
        self.mode = mode
        self.entry_exit_logic = entry_exit_logic
        self.results_summary = {}
        self.output_dir = output_dir
        self.threshold_values = threshold_values

        if len(self.strategies) != len(self.strategy_names):
            raise ValueError("strategies and strategy_names must have the same length")

    def run_all(self, 
                backtest_start_date: Optional[str] = None,
                backtest_end_date: Optional[str] = None,
                backtest_years: Optional[float] = None,
                forward_test: bool = False,
                forward_start_date: Optional[str] = None,
                forward_end_date: Optional[str] = None,
                forward_years: Optional[float] = None,
                show_plot: bool = False
                ) -> Dict[str, Dict]:
        """
        Run backtests for all strategies and return their results.
        
        :return: Dictionary of results keyed by strategy name.
        """
        results_summary = {}

        # for name, strategy in zip(self.strategy_names, self.strategies):
        if self.threshold_values is None:
            self.threshold_values = [None] * len(self.strategies)
        for name, strategy, threshold in zip(self.strategy_names, self.strategies, self.threshold_values):
            print(f"\nRunning backtest for: {name}")
            bt = Backtester(
                data=self.data,
                strategy=strategy,
                strategy_name= name,
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
                trading_fee=self.trading_fee,
                qty_per_trade=self.qty_per_trade,
                mode = self.mode,
                entry_exit_logic = self.entry_exit_logic,
                threshold=threshold,
            )
            result = bt.run(
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                backtest_years=backtest_years,
                forward_test=forward_test,
                forward_years=forward_years,
                forward_start_date=forward_start_date,
                forward_end_date=forward_end_date
            )

            if show_plot:
                bt.plot_results()

            results_summary[name] = {
                'backtester': bt,
                'results': result['results'],
                'metrics': result['metrics'],
            }
            self.results_summary = results_summary
                
            
        
        return results_summary
    

    def plot_permutation_heatmap(self, metric: str = "Sharpe Ratio", save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot a permutation heatmap for the strategies.
        
        :param metric: The metric to use for the heatmap.
        :param save_path: Optional path to save the heatmap.
        :param show_plot: Whether to show the plot.
        
        :return: 
        """

        # Create a DataFrame for the metrics
        metrics_list = [
            {'Strategy': name, metric: result['metrics']['full'][metric]}
            for name, result in self.results_summary.items()
        ]
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index('Strategy', inplace=True)

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"{metric} Heatmap for Strategies")
        plt.xlabel("Metric")
        plt.ylabel("Strategies")
        plt.tight_layout()
        heatmap_path = os.path.join(self.output_dir, f"multibacktest_{metric.lower().replace(' ', '_')}_heatmap.png") if save_path is None else save_path
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to: {heatmap_path}")
        
        if show_plot:
            plt.show()
        
        return metrics_df
