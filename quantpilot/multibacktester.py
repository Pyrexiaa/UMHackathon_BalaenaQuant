from typing import List, Dict, Optional
import pandas as pd
from .backtester import Backtester  

class Multibacktester:
    def __init__(self, data: pd.DataFrame, strategies: List, strategy_names: Optional[List[str]] = None,
                 initial_capital: float = 100000, risk_free_rate: float = 0.0, trading_fee: float = 0.0006,
                 qty_per_trade: int = 1):
        """
        Initialize the strategy tester with multiple strategies.
        
        :param data: The price data to backtest on.
        :param strategies: A list of strategy instances (each must implement generate_signals()).
        :param strategy_names: Optional list of names corresponding to each strategy.
        """
        self.data = data
        self.strategies = strategies
        self.strategy_names = strategy_names or [f"Strategy_{i+1}" for i in range(len(strategies))]
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.qty_per_trade = qty_per_trade

        if len(self.strategies) != len(self.strategy_names):
            raise ValueError("strategies and strategy_names must have the same length")

    def run_all(self, forward_test: bool = False, forward_years: Optional[float] = None,
                forward_start_date: Optional[str] = None) -> Dict[str, Dict]:
        """
        Run backtests for all strategies and return their results.
        
        :return: Dictionary of results keyed by strategy name.
        """
        results_summary = {}

        for name, strategy in zip(self.strategy_names, self.strategies):
            print(f"\nRunning backtest for: {name}")
            tester = Backtester(
                data=self.data,
                strategy=strategy,
                strategy_name= name,
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
                trading_fee=self.trading_fee,
                qty_per_trade=self.qty_per_trade
            )
            result = tester.run(
                forward_test=forward_test,
                forward_years=forward_years,
                forward_start_date=forward_start_date
            )

            results_summary[name] = {
                'backtester': tester,
                'results': result['results'],
                'metrics': result['metrics'],
            }
            
        # for name, result in results_summary.items():
        #     print(f"\n{name} metrics:")
        #     print(result['metrics']['full']) 
    
        return results_summary
