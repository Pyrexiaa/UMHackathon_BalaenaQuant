from abc import ABC, abstractmethod
from datetime import datetime
import os
from typing import Dict
import numpy as np
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
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
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
    
    def merge_csv(self, datasource: str, metrics: list[str]) -> None:
        
        
        csv_paths = [f"output/{datasource}_{metric}_processed.csv" for metric in metrics]
        print(f"CSV paths: {csv_paths}")
        merged_df = None
        for df_name in self.dataframes.keys():
            try:
                # df = pd.read_csv(path, parse_dates=["datetime"])
                df = self.dataframes[df_name]
                df.set_index("datetime", inplace=True)

                if merged_df is None:
                    merged_df = df
                else:
                    suffix = os.path.basename(df_name)
                    merged_df = merged_df.join(df, how="inner", rsuffix=f"_{suffix}")
            except Exception as e:
                print(f"❌ Failed to process {df_name}: {e}")

        if merged_df is None:
            raise ValueError("No DataFrames were successfully joined.")

        merged_df.reset_index(inplace=True)

        merged_output_path = f"output/{datasource}_{datetime.now()}_merged.csv"
        # merged_output_path = "output/merged_features.csv"
        merged_df.to_csv(merged_output_path, index=False)
        print(f"✅ Joined CSV saved to {merged_output_path}")
        
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = merged_df.where(pd.notnull(merged_df), None)
        
        merged_df["datetime"] = merged_df["datetime"].astype(str)
        return merged_df.head().to_dict(orient="records")
        

    @abstractmethod
    async def run(self) -> pd.DataFrame:
        pass
    
    
