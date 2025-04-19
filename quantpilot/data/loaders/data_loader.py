from typing import Optional
import pandas as pd
from datetime import datetime, timezone
from .base_loader import BaseLoader
import cybotrade_datasource
import asyncio

class DataLoader(BaseLoader):
    """
    DataLoader is a subclass of BaseLoader that loads data from an external data source.
    """
    STANDARD_METRICS = [
        "price-ohlcv", "taker-buy-sell-stats", "addresses_count_inflow", "addresses_count",
        "coinbase_premium_index", "estimated_leverage_ratio", "exchange_whale_ratio",
        "tokens_transferred", "blockreward", "fees_transaction", "miner_supply_ratio", 
        "exchange_supply_ratio", "transactions_count_inflow", "liquidations", "open_interest", "difficulty"
    ]
    
    async def load_data(
        self,
        metrics: list[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_data: Optional[bool] = True,
        merged: Optional[bool] = True,
    ) -> None:
        """
        Loads data for the specified metrics within the given time range.

        :param metrics: A list of metrics to load.
        :param start_time: The start time for the data query (default is January 1, 2020).
        :param end_time: The end time for the data query (default is April 1, 2025).
        :param save_data: Whether to save the data to a CSV file (default is True).
        :param merged: Whether to merge the data for multiple metrics (default is True).
        """
        # Default start and end time if not provided
        start_time = start_time or datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)
        end_time = end_time or datetime(year=2025, month=4, day=1, tzinfo=timezone.utc)
        
        # If no metrics provided, use standard metrics
        metrics = metrics or self.STANDARD_METRICS
        
        # Fetch data concurrently for all metrics
        tasks = [self._fetch_metric_data(metric, start_time, end_time) for metric in metrics]
        results = await asyncio.gather(*tasks)

        # Store the fetched data for each metric
        for metric, df in zip(metrics, results):
            if not df.empty:
                self.dataframes[metric] = df

                # Save the data to CSV if required
                if save_data:
                    self.save_data_to_csv(self.datasource_key, metric)

        # If more than one metric is provided and merged is True, merge the data
        if len(metrics) > 1 and merged:
            print("Merging data...")
            merged_df = self.merge_csv(self.datasource_key, metrics)
            self.data = merged_df
    
    async def _fetch_metric_data(self, metric: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Helper method to fetch data for a specific metric.
        
        :param metric: The metric to fetch data for.
        :param start_time: The start time for the data query.
        :param end_time: The end time for the data query.
        :return: A pandas DataFrame containing the data for the metric.
        """
        if metric not in self.datasource["topics"]:
            raise ValueError(f"Metric '{metric}' not found in datasource topics.")

        try:
            endpoint = self.datasource["topics"][metric]
            data = await cybotrade_datasource.query_paginated(
                api_key=self.datasource["api_key"],
                topic=endpoint,
                start_time=start_time,
                end_time=end_time,
            )

            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data)

            # If 'datetime' column exists, convert it to datetime and set as index
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)

            # Sort the data by the timestamp
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for metric '{metric}': {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    async def run(
        self,
        metrics: Optional[list] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_data: Optional[bool] = True,
        merged: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Runs the data loading process and returns the cleaned data.

        :param metrics: A list of metrics to load.
        :param start_time: The start time for the data query (default is January 1, 2020).
        :param end_time: The end time for the data query (default is April 1, 2025).
        :param save_data: Whether to save the data to a CSV file (default is True).
        :param merged: Whether to merge the data for multiple metrics (default is False).
        :return: The cleaned DataFrame containing the data.
        """
        print("Fetching data...")
        # Load the data for the given metrics
        await self.load_data(metrics, start_time, end_time, save_data, merged)

        # Clean the loaded data (remove duplicates, fill missing values)
        self.clean_data()
        
        # Return the cleaned data
        return self.get_data()