import pandas as pd
from datetime import datetime

from .base_loader import BaseLoader
import cybotrade_datasource


class CustomLoader(BaseLoader):

    async def load_data(
        self, start_time: datetime, end_time: datetime, metrics: str, save_data: bool
    ) -> None:
        try:
            endpoint = self.datasource["topics"][metrics]
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

            if save_data:
                self.save_data_to_csv(self.datasource_key, metrics)
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    async def run(
        self, metrics: str, window: int, limit: int, save_data: bool
    ) -> pd.DataFrame:
        await self.load_data(metrics, window, limit, save_data)
        self.clean_data()
        return self.get_data()
