from typing import Optional
import pandas as pd
from datetime import datetime, timezone

from .base_loader import BaseLoader
import cybotrade_datasource

class CustomLoader(BaseLoader):
    """
    CustomLoader is a subclass of BaseLoader that loads data from an external data source.
    """

    async def load_data(
        self,
        metrics: list[str],
        start_time: Optional[datetime] = datetime(
            year=2020, month=1, day=1, tzinfo=timezone.utc
        ),
        end_time: Optional[datetime] = datetime(
            year=2025, month=4, day=1, tzinfo=timezone.utc
        ),
        save_data: Optional[bool] = True,
        merged: Optional[bool] = False,
    ) -> None:
        """
        Loads data for the specified metrics within the given time range.

        :param metrics: A list of metrics to load.
        :param start_time: The start time for the data query (default is January 1, 2020).
        :param end_time: The end time for the data query (default is April 1, 2025).
        :param save_data: Whether to save the data to a CSV file (default is True).
        :param merged: Whether to merge the data for multiple metrics (default is False).
        """
        for metric in metrics:
            # Check if the metric exists in the data source's topics
            if metric not in self.datasource["topics"]:
                raise ValueError(f"Metric '{metric}' not found in datasource topics.")
            
            try:
                # Retrieve the endpoint for the metric and fetch paginated data
                endpoint = self.datasource["topics"][metric]
                data = await cybotrade_datasource.query_paginated(
                    api_key=self.datasource["api_key"],
                    topic=endpoint,
                    start_time=start_time,
                    end_time=end_time,
                )
                
                # Convert the data to a pandas DataFrame
                self.data = pd.DataFrame(data)
                
                # If 'timestamp' column exists, convert it to datetime and set as index
                if "timestamp" in self.data.columns:
                    self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
                    self.data.set_index("timestamp", inplace=True)
                self.data.sort_index(inplace=True)  # Sort the data by the timestamp
                print("Data loaded and indexed.")
                
                # Store the DataFrame for the metric
                self.dataframes[metric] = self.data

                # Save the data to a CSV file if save_data is True
                if save_data:
                    self.save_data_to_csv(self.datasource_key, metric)

            except Exception as e:
                print(f"Error loading data: {e}")

        # If more than one metric is provided and merged is True, merge the data
        if len(metrics) > 1 and merged:
            self.merge_csv(self.datasource_key, metrics)

    async def run(
        self,
        metrics: list[str],
        start_time: Optional[datetime] = datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_time: Optional[datetime] = datetime(2025, 4, 1, tzinfo=timezone.utc),
        save_data: Optional[bool] = True,
        merged: Optional[bool] = False,
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
        # Load the data for the given metrics
        await self.load_data(metrics, start_time, end_time, save_data, merged)
        
        # Clean the loaded data (remove duplicates, fill missing values)
        self.clean_data()
        
        # Return the cleaned data
        return self.get_data()
