from abc import ABC, abstractmethod
import os
import pandas as pd

from data.api.api_client import APIClient
from data.constant.data_source import data_source

class BaseLoader(ABC):
    def __init__(self, datasource_key: str):
        self.datasource_key = datasource_key
        self.datasource = data_source[datasource_key]
        self.api_client = APIClient(
            datasource_key
        )
        self.data = pd.DataFrame()
        
    @abstractmethod
    async def load_data(self) -> None:
        pass

    def clean_data(self) -> None:
        self.data = self.data.drop_duplicates()
        self.data = self.data.fillna(method='ffill')

    def get_data(self) -> pd.DataFrame:
        return self.data

    def reset_data(self):
        """
        Reset the data to an empty DataFrame.
        """
        self.data = pd.DataFrame()
        print(self.data)
        print("Data reset to an empty DataFrame.")
        
    def save_data_to_csv(self, datasource: str, metrics: str) -> None:
        output_path = f"output/{datasource}_{metrics}_processed.csv"
        os.makedirs("output", exist_ok=True)
        self.get_data().to_csv(output_path)
        print(f"Processed data saved to {output_path}")

    @abstractmethod
    async def run(self) -> pd.DataFrame:
        pass
    
    
