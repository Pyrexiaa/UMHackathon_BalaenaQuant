from typing import Optional
import pandas as pd
from datetime import datetime, timezone

from .base_loader import BaseLoader
import cybotrade_datasource


class CustomLoader(BaseLoader):

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

        for metric in metrics:
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
                self.data = pd.DataFrame(data)
                if "timestamp" in self.data.columns:
                    self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
                    self.data.set_index("timestamp", inplace=True)
                self.data.sort_index(inplace=True)
                print("✅ Data loaded and indexed.")
                
                self.dataframes[metric] = self.data

                if save_data:
                    self.save_data_to_csv(self.datasource_key, metric)

            except Exception as e:
                print(f"❌ Error loading data: {e}")

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
        await self.load_data(metrics, start_time, end_time, save_data, merged)
        self.clean_data()
        return self.get_data()
