import pandas as pd
from data.loaders.base_loader import BaseLoader


class APILoader(BaseLoader):
        
    async def load_data(self, metrics: str, window: int, limit: int, save_data: bool):
        try:
            endpoint = self.datasource["endpoints"][metrics]
            endpoint_with_query_param = f"{endpoint}?window={window}&limit={limit}"
            response = self.api_client.get(endpoint_with_query_param)

            self.data = pd.DataFrame(response['data']) if 'data' in response else pd.DataFrame(response)

            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)

            print(" Data loaded and indexed.")
            
            if save_data:
                self.save_data_to_csv(self.datasource_key, metrics)
                
        except Exception as e:
            print(f" Error loading data: {e}")
            
    async def run(self, metrics: str, window: int, limit: int, save_data: bool) -> pd.DataFrame:
        await self.load_data(metrics, window, limit, save_data)
        self.clean_data()
        return self.get_data()