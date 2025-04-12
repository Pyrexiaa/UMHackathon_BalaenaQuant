from typing import Optional
import pandas as pd
from data.loaders.base_loader import BaseLoader


class APILoader(BaseLoader):
        
    async def load_data(self, metrics: list[str], window: Optional[str] = "hour", limit: Optional[int] = 46211, save_data: Optional[bool] = True, merged: Optional[bool] = False) -> None:
        for metric in metrics:
            
            try:
                endpoint = self.datasource["endpoints"][metric]
                if "?" not in endpoint:
                    endpoint_with_query_param = f"{endpoint}?window={window}&limit={limit}"
                else:
                    endpoint_with_query_param = f"{endpoint}&window={window}&limit={limit}"
                response = self.api_client.get(endpoint_with_query_param)

                self.data = pd.DataFrame(response['data']) if 'data' in response else pd.DataFrame(response)

                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data.set_index('timestamp', inplace=True)
                self.data.sort_index(inplace=True)
                self.dataframes[metric] = self.data
                print(" Data loaded and indexed.")
                
                if save_data:
                        self.save_data_to_csv(self.datasource_key, metric)
            
            except Exception as e:
                print(f" Error loading data: {e}")
                    

        if len(metrics) > 1 and merged:
            self.merge_csv(self.datasource_key, metrics)
        
                
            
    async def run(self, metrics: list[str], window: Optional[str] = "hour", limit: Optional[int] = 46211, save_data: Optional[bool] = True, merged: Optional[bool] = False) -> pd.DataFrame:
        await self.load_data(metrics, window, limit, save_data, merged)
        self.clean_data()
        return self.get_data()